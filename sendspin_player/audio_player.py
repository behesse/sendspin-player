"""Audio playback for Sendspin player with time synchronization.

This module provides an AudioPlayer that handles time-synchronized audio playback
using sounddevice. It manages buffering and scheduled playback times to maintain
sync between server and client timelines.

Copyright (C) 2025  behesse

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import asyncio
import collections
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Any

import numpy as np
from scipy import signal

# Import PCMFormat for runtime use (not just type checking)
from aiosendspin.client import PCMFormat

if TYPE_CHECKING:
    pass  # PCMFormat imported above for runtime use

from sendspin_player.audio_device import AudioDeviceManager, AudioDevice, sounddevice

logger = logging.getLogger('audio_player')


@dataclass
class _QueuedChunk:
    """Represents a queued audio chunk with timing information."""
    server_timestamp_us: int
    """Server timestamp when this chunk should start playing."""
    audio_data: bytes
    """Raw PCM audio bytes."""


class AudioPlayer:
    """
    Audio player for Sendspin player with time synchronization support.
    
    This player accepts audio chunks with server timestamps and dynamically
    computes playback times using time synchronization functions from aiosendspin.
    
    Based on the implementation in sendspin-cli:
    https://github.com/Sendspin/sendspin-cli/blob/main/sendspin/audio.py
    """
    
    _MIN_CHUNKS_TO_START: int = 16
    """Minimum chunks buffered before starting playback to absorb network jitter."""
    _MIN_CHUNKS_TO_MAINTAIN: int = 8
    """Minimum chunks to maintain during playback to avoid underruns."""
    _BLOCKSIZE: int = 2048
    """Audio block size (~46ms at 44.1kHz)."""
    _MICROSECONDS_PER_SECOND: int = 1_000_000
    
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        compute_client_time: Callable[[int], int],
        compute_server_time: Callable[[int], int],
        device_manager: Optional[AudioDeviceManager] = None,
    ) -> None:
        """
        Initialize the audio player.
        
        Args:
            loop: The asyncio event loop to use for scheduling.
            compute_client_time: Function that converts server timestamps to client
                timestamps (monotonic loop time), accounting for clock drift, offset,
                and static delay.
            compute_server_time: Function that converts client timestamps to server
                timestamps (inverse of compute_client_time).
            device_manager: AudioDeviceManager instance. If None, creates a new one.
        """
        self._loop = loop
        self._compute_client_time = compute_client_time
        self._compute_server_time = compute_server_time
        self._device_manager = device_manager if device_manager is not None else AudioDeviceManager()
        self._audio_device: Optional[AudioDevice] = None  # AudioDevice instance (set when format is configured)
        self._format: Optional['PCMFormat'] = None  # Server's original format (for timestamp calculations)
        self._playback_format: Optional['PCMFormat'] = None  # Actual playback format (may differ if fallback)
        self._format_mismatch: bool = False  # True if playback format differs from server format
        self._resampling_warned: bool = False  # Track if we've warned about active resampling
        self._queue: asyncio.Queue[_QueuedChunk] = asyncio.Queue()
        self._stream: Optional[sounddevice.RawOutputStream] = None
        self._closed = False
        self._stream_started = False
        
        self._volume: int = 100  # 0-100 range
        self._muted: bool = False
        
        # Partial chunk tracking
        self._current_chunk: Optional[_QueuedChunk] = None
        self._current_chunk_offset = 0
        
        # Track expected next chunk timestamp for intelligent gap/overlap handling (matches sendspin-cli)
        self._expected_next_timestamp: Optional[int] = None
        
        # Track queued audio duration
        self._queued_duration_us = 0
        
        # DAC timing for accurate playback position tracking
        self._dac_loop_calibrations: collections.deque[tuple[int, int]] = collections.deque(maxlen=100)
        """Recent [(dac_time_us, loop_time_us), ...] pairs for DAC-Loop mapping."""
        self._last_known_playback_position_us: int = 0
        """Current playback position in server timestamp space."""
        self._last_dac_calibration_time_us: int = 0
        """Last loop time when we calibrated DAC-Loop mapping."""
        
        # Server timeline cursor
        self._server_ts_cursor_us: int = 0
        self._server_ts_cursor_remainder: int = 0
        
        # First chunk tracking
        self._first_server_timestamp_us: Optional[int] = None
        self._scheduled_start_loop_time_us: Optional[int] = None
        self._scheduled_start_dac_time_us: Optional[int] = None
        
        # Underrun tracking
        self._underrun_count = 0
        
        # Last output frame for potential duplication (for sync corrections)
        self._last_output_frame: bytes = b""
    
    def set_format(self, pcm_format: 'PCMFormat', device: Optional[int] = None) -> None:
        """Configure the audio output format.
        
        Args:
            pcm_format: PCM audio format specification.
            device: Audio device ID to use. None for default device.
        """
        # Store server's original format for timestamp calculations
        self._format = pcm_format
        # Playback format starts as server format, may change if we fall back
        self._playback_format = pcm_format
        self._format_mismatch = False
        self._close_stream()
        
        # Reset state on format change
        self._stream_started = False
        self._resampling_warned = False  # Reset warning flag on format change
        
        # Get AudioDevice instance for this device
        self._audio_device = self._device_manager.get_device(device)
        
        # Low latency settings for accurate playback
        # Always try the server's sample rate first - devices often support multiple rates
        # even if their "default" is different
        sample_rate = pcm_format.sample_rate
        device_default_rate = self._audio_device.sample_rate
        if device is not None:
            logger.debug(
                f"Device {device} default sample rate: {device_default_rate}Hz, "
                f"server requested: {sample_rate}Hz. Will try server rate first."
            )
        
        try:
            self._stream = self._audio_device.create_output_stream(
                samplerate=sample_rate,
                channels=pcm_format.channels,
                dtype="int16",
                blocksize=self._BLOCKSIZE,
                callback=self._audio_callback,
                latency="high",
            )
            # Start the stream immediately
            self._stream.start()
            self._closed = False
            logger.info(
                f"Audio stream configured and started: blocksize={self._BLOCKSIZE}, latency=high{self._audio_device.format_info_string()}, "
                f"samplerate={sample_rate}Hz, channels={pcm_format.channels}"
            )
        except Exception as e:
            # Log the actual error for debugging
            error_str = str(e).lower()
            logger.debug(f"Failed to open audio stream at {sample_rate}Hz: {type(e).__name__}: {e}")
            
            # Only fall back if it's actually a sample rate error
            # Check for common sample rate error messages
            is_sample_rate_error = (
                "sample rate" in error_str or 
                "samplerate" in error_str or
                "invalid sample rate" in error_str or
                "unsupported sample rate" in error_str
            )
            
            if is_sample_rate_error:
                logger.warning(
                    f"Device does not support server's sample rate {sample_rate}Hz "
                    f"(error: {type(e).__name__}: {e}). "
                    f"Resampling to device default {device_default_rate}Hz. "
                )
                try:
                    self._stream = self._audio_device.create_output_stream(
                        samplerate=device_default_rate,
                        channels=pcm_format.channels,
                        dtype="int16",
                        blocksize=self._BLOCKSIZE,
                        callback=self._audio_callback,
                        latency="high",
                    )
                    self._stream.start()
                    self._closed = False
                    # Update playback format to reflect actual sample rate
                    # BUT keep _format as server's original format for timestamp calculations
                    self._playback_format = PCMFormat(
                        sample_rate=device_default_rate,
                        channels=pcm_format.channels,
                        bit_depth=pcm_format.bit_depth
                    )
                    self._format_mismatch = True  # Mark that formats don't match - will trigger resampling
                    logger.info(
                        f"Audio resampling enabled: server {sample_rate}Hz -> device {device_default_rate}Hz"
                    )
                    logger.info(
                        f"Audio stream configured and started: blocksize={self._BLOCKSIZE}, latency=high{self._audio_device.format_info_string()}, "
                        f"samplerate={device_default_rate}Hz, channels={pcm_format.channels}"
                    )
                except Exception as e2:
                    logger.error(f"Failed to start audio stream even with device default rate: {e2}", exc_info=True)
                    raise
            else:
                logger.error(f"Failed to start audio stream: {e}", exc_info=True)
                raise
    
    def set_volume(self, volume: int, *, muted: bool) -> None:
        """Set volume and mute state.
        
        Args:
            volume: Volume level (0-100).
            muted: Whether audio is muted.
        """
        old_volume = self._volume
        old_muted = self._muted
        self._volume = max(0, min(100, volume))
        self._muted = muted
        logger.debug(
            f"AudioPlayer.set_volume called: volume={self._volume}% (was {old_volume}%), "
            f"muted={self._muted} (was {old_muted})"
        )
    
    def clear(self) -> None:
        """Drop all queued audio chunks and reset for new stream."""
        # Drain all queued chunks
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Reset state - but keep stream running, just stop playback until buffered
        self._stream_started = False  # This will make callback output silence until we buffer enough
        self._current_chunk = None
        self._current_chunk_offset = 0
        self._expected_next_timestamp = None  # Reset expected timestamp tracking
        self._underrun_count = 0
        self._queued_duration_us = 0
        
        # Reset timing calibration for new stream
        self._dac_loop_calibrations.clear()
        self._last_known_playback_position_us = 0
        self._last_dac_calibration_time_us = 0
        self._scheduled_start_loop_time_us = None
        self._scheduled_start_dac_time_us = None
        self._server_ts_cursor_us = 0  # Reset cursor - will be initialized from first chunk
        self._server_ts_cursor_remainder = 0
        self._first_server_timestamp_us = None  # Reset so we can track first chunk of new stream
        
        # Reset callback counter for debugging
        if hasattr(self, '_callback_count'):
            self._callback_count = 0
        
        logger.debug("Audio queue cleared, waiting for new stream to buffer")
    
    def submit(self, server_timestamp_us: int, payload: bytes) -> None:
        """
        Queue an audio payload for playback.
        
        Args:
            server_timestamp_us: Server timestamp when this audio should play.
            payload: Raw PCM audio bytes.
        """
        if self._format is None:
            logger.debug("Audio format missing; dropping audio chunk")
            return
        if self._format.frame_size == 0:
            return
        if len(payload) % self._format.frame_size != 0:
            logger.warning(
                f"Dropping audio chunk with invalid size: {len(payload)} bytes "
                f"(frame size {self._format.frame_size})"
            )
            return
        
        # Handle gaps and overlaps (matches sendspin-cli exactly)
        # Initialize expected next timestamp on first chunk
        if self._expected_next_timestamp is None:
            self._expected_next_timestamp = server_timestamp_us
        # Handle gap: insert silence to fill the gap
        elif server_timestamp_us > self._expected_next_timestamp:
            gap_us = server_timestamp_us - self._expected_next_timestamp
            gap_frames = (gap_us * self._format.sample_rate) // 1_000_000
            silence_bytes = int(gap_frames * self._format.frame_size)
            silence = b"\x00" * silence_bytes
            try:
                self._queue.put_nowait(
                    _QueuedChunk(
                        server_timestamp_us=self._expected_next_timestamp,
                        audio_data=silence,
                    )
                )
                # Account for inserted silence in buffer duration
                silence_duration_us = (gap_frames * 1_000_000) // self._format.sample_rate
                self._queued_duration_us += silence_duration_us
                logger.debug(
                    f"Gap: {gap_us / 1000.0:.1f} ms filled with silence"
                )
                self._expected_next_timestamp = server_timestamp_us
            except asyncio.QueueFull:
                logger.warning(f"Queue full while inserting gap silence")
                return
        # Handle overlap: trim the start of the chunk
        elif server_timestamp_us < self._expected_next_timestamp:
            overlap_us = self._expected_next_timestamp - server_timestamp_us
            overlap_frames = (overlap_us * self._format.sample_rate) // 1_000_000
            trim_bytes = overlap_frames * self._format.frame_size
            if trim_bytes < len(payload):
                payload = payload[trim_bytes:]
                server_timestamp_us = self._expected_next_timestamp
                logger.debug(
                    f"Overlap: {overlap_us / 1000.0:.1f} ms trimmed"
                )
            else:
                # Entire chunk is overlap, skip it
                logger.debug(
                    f"Overlap: {overlap_us / 1000.0:.1f} ms (chunk skipped, already played)"
                )
                return
        
        # Resample if playback format differs from server format
        # Calculate duration BEFORE resampling (based on server time)
        chunk_duration_us = 0
        original_chunk_frames = 0
        if len(payload) > 0:
            original_chunk_frames = len(payload) // self._format.frame_size
            chunk_duration_us = (original_chunk_frames * 1_000_000) // self._format.sample_rate
        
        if self._format_mismatch and self._playback_format is not None and len(payload) > 0:
            # Resample audio from server rate to playback rate
            server_rate = self._format.sample_rate
            playback_rate = self._playback_format.sample_rate
            
            # Warn once when resampling first occurs
            if not self._resampling_warned:
                logger.warning(
                    f"Resampling audio: server {server_rate}Hz -> device {playback_rate}Hz. "
                    f"This may slightly increase CPU usage but ensures correct playback speed."
                )
                self._resampling_warned = True
            
            # Convert bytes to numpy array (int16)
            audio_array = np.frombuffer(payload, dtype=np.int16)
            
            # Reshape for multi-channel (if needed)
            if self._format.channels > 1:
                audio_array = audio_array.reshape(-1, self._format.channels)
            
            # Resample using scipy
            # Note: signal.resample uses FFT which is high quality but can be slow for real-time
            # For better performance, consider using resampy or soxr in the future
            num_output_samples = int(original_chunk_frames * playback_rate / server_rate)
            resampled = signal.resample(audio_array, num_output_samples, axis=0)
            
            # Convert back to int16 and flatten
            resampled_int16 = resampled.astype(np.int16)
            if self._format.channels > 1:
                resampled_int16 = resampled_int16.flatten()
            
            # Convert back to bytes
            payload = resampled_int16.tobytes()
            logger.debug(
                f"Resampled audio: {original_chunk_frames} frames @ {server_rate}Hz -> "
                f"{num_output_samples} frames @ {playback_rate}Hz (duration: {chunk_duration_us/1000:.1f}ms)"
            )
        
        # Queue the chunk (matches sendspin-cli - uses put_nowait which is non-blocking)
        # Duration is already calculated above (based on server time, before resampling)
        if len(payload) > 0:
            chunk = _QueuedChunk(server_timestamp_us=server_timestamp_us, audio_data=payload)
            try:
                self._queue.put_nowait(chunk)
                # Track duration of queued audio
                self._queued_duration_us += chunk_duration_us
                # Update expected position for next chunk
                self._expected_next_timestamp = server_timestamp_us + chunk_duration_us
            except asyncio.QueueFull:
                # Queue is full - this shouldn't happen with proper buffering, but log if it does
                logger.warning(
                    f"Audio queue full! Dropping chunk (ts: {server_timestamp_us}, "
                    f"queue size: {self._queue.qsize()})"
                )
                return
        
        queue_size = self._queue.qsize()
        
        # Track first chunk and schedule start time (matches sendspin-cli)
        # sendspin-cli schedules start on first chunk, not when enough chunks are buffered
        queue_size = self._queue.qsize()
        if self._first_server_timestamp_us is None:
            self._first_server_timestamp_us = server_timestamp_us
            buffered_seconds = self._queued_duration_us / 1_000_000.0
            if len(payload) > 0:
                chunk_frames = len(payload) // self._format.frame_size
                logger.debug(
                    f"First audio chunk received: {server_timestamp_us} us, {chunk_frames} frames, "
                    f"total chunks: {queue_size}, buffered: {buffered_seconds:.2f}s"
                )
            else:
                logger.debug(
                    f"First audio chunk received: {server_timestamp_us} us, total chunks: {queue_size}, "
                    f"buffered: {buffered_seconds:.2f}s"
                )
            
            # Schedule start time aligned to server timeline (matches sendspin-cli)
            if self._scheduled_start_loop_time_us is None:
                self._schedule_start()
        else:
            buffered_seconds = self._queued_duration_us / 1_000_000.0
            if len(payload) > 0:
                chunk_frames = len(payload) // self._format.frame_size
                logger.debug(
                    f"Audio chunk queued: {chunk_frames} frames, total chunks: {queue_size}, "
                    f"buffered: {buffered_seconds:.2f}s"
                )
            else:
                logger.debug(
                    f"Audio chunk queued: total chunks: {queue_size}, buffered: {buffered_seconds:.2f}s"
                )
            
            # While waiting to start, keep the scheduled loop start updated as time sync improves
            # This is critical - time sync accuracy improves as more messages arrive
            if (self._scheduled_start_loop_time_us is not None and 
                self._first_server_timestamp_us is not None and
                not self._stream_started):
                try:
                    updated_loop_start = self._compute_client_time(self._first_server_timestamp_us)
                    # Only update if it moves significantly to avoid churn (matches sendspin-cli)
                    _START_TIME_UPDATE_THRESHOLD_US = 10_000  # 10ms threshold
                    if abs(updated_loop_start - self._scheduled_start_loop_time_us) > _START_TIME_UPDATE_THRESHOLD_US:
                        self._scheduled_start_loop_time_us = updated_loop_start
                        logger.debug(
                            f"Updated scheduled start time: {updated_loop_start} "
                            f"(was: {self._scheduled_start_loop_time_us - updated_loop_start}us different)"
                        )
                except Exception:
                    logger.exception("Failed to update start time")
        
        # Check if we have enough chunks to start (for logging only - start is already scheduled)
        if not self._stream_started and queue_size >= self._MIN_CHUNKS_TO_START:
            logger.info(f"Enough chunks buffered ({queue_size} >= {self._MIN_CHUNKS_TO_START}), ready to start")
        elif not self._stream_started:
            logger.debug(f"Buffering... ({queue_size}/{self._MIN_CHUNKS_TO_START} chunks)")
        
        # Warn if buffer is getting low during playback
        if self._stream_started and queue_size < self._MIN_CHUNKS_TO_MAINTAIN:
            logger.warning(
                f"Low audio buffer during playback: {queue_size} chunks "
                f"(minimum to maintain: {self._MIN_CHUNKS_TO_MAINTAIN}, "
                f"queued duration: {self._queued_duration_us / 1000:.1f}ms)"
            )
    
    def _schedule_start(self) -> None:
        """Schedule playback start time based on buffered chunks.
        
        Matches sendspin-cli's _compute_and_set_loop_start implementation.
        """
        if self._first_server_timestamp_us is None or self._format is None:
            logger.warning("Cannot schedule start: missing first timestamp or format")
            return
        
        # Convert server timestamp to client time (matches sendspin-cli exactly)
        try:
            self._scheduled_start_loop_time_us = self._compute_client_time(self._first_server_timestamp_us)
        except Exception:
            logger.exception("Failed to compute client time for start")
            # Fallback to current time if computation fails
            self._scheduled_start_loop_time_us = int(
                self._loop.time() * self._MICROSECONDS_PER_SECOND
            )
        
        self._stream_started = True
        
        now_us = int(self._loop.time() * self._MICROSECONDS_PER_SECOND)
        delay_ms = (self._scheduled_start_loop_time_us - now_us) / 1000.0
        
        logger.info(
            f"Scheduled audio start: {delay_ms:.1f}ms delay, "
            f"queue size: {self._queue.qsize()}, "
            f"scheduled_time_us: {self._scheduled_start_loop_time_us}, "
            f"now_us: {now_us}"
        )
    
    def _fill_silence(self, buffer: memoryview, offset: int, bytes_count: int) -> None:
        """Fill buffer with silence (zeros) starting at offset."""
        buffer[offset:offset + bytes_count] = b'\x00' * bytes_count
    
    def _audio_callback(
        self,
        outdata: memoryview,
        frames: int,
        time_info: Any,  # CFFI structure with outputBufferDacTime, inputBufferAdcTime, currentTime
        status: sounddevice.CallbackFlags,
    ) -> None:
        """Callback function for sounddevice audio stream."""
        assert self._format is not None
        
        bytes_needed = frames * self._format.frame_size
        output_buffer = memoryview(outdata).cast("B")
        
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Handle underruns
        if status.output_underflow:
            self._underrun_count += 1
            logger.warning(f"Audio underrun #{self._underrun_count}")
        
        # Check if we should start playback
        if not self._stream_started:
            self._fill_silence(output_buffer, 0, bytes_needed)
            return
        
        # Log first few callbacks for debugging (reduced frequency)
        if not hasattr(self, '_callback_count'):
            self._callback_count = 0
        self._callback_count += 1
        # Only log first 3 callbacks, then every 1000th callback
        if self._callback_count <= 3 or self._callback_count % 1000 == 0:
            logger.debug(
                f"Audio callback #{self._callback_count}, "
                f"queue: {self._queue.qsize()}, stream_started: {self._stream_started}, "
                f"scheduled_start: {self._scheduled_start_loop_time_us}"
            )
        
        # Check scheduled start time
        if self._scheduled_start_loop_time_us is not None:
            now_us = int(self._loop.time() * self._MICROSECONDS_PER_SECOND)
            if now_us < self._scheduled_start_loop_time_us:
                # Not time to start yet, output silence
                if self._callback_count <= 5:
                    logger.debug(
                        f"Waiting for start time: now_us={now_us}, "
                        f"scheduled_us={self._scheduled_start_loop_time_us}, "
                        f"wait_ms={(self._scheduled_start_loop_time_us - now_us) / 1000:.1f}"
                    )
                self._fill_silence(output_buffer, 0, bytes_needed)
                return
            else:
                # Check if we have enough chunks before starting (matches sendspin-cli behavior)
                queue_size = self._queue.qsize()
                if queue_size >= self._MIN_CHUNKS_TO_START:
                    # Start playback
                    # Initialize cursor to first chunk's timestamp when we actually start
                    # This ensures cursor matches where we are in the server timeline
                    if self._server_ts_cursor_us == 0 and self._first_server_timestamp_us is not None:
                        self._server_ts_cursor_us = self._first_server_timestamp_us
                        logger.debug(f"Initialized server_ts_cursor to first chunk timestamp: {self._server_ts_cursor_us}")
                    
                    self._scheduled_start_loop_time_us = None
                    logger.info(f"Audio playback started! Callback #{self._callback_count}, queue: {queue_size}, cursor: {self._server_ts_cursor_us}")
                else:
                    # Not enough chunks yet, keep waiting
                    if self._callback_count % 100 == 0:
                        logger.debug(
                            f"Waiting for more chunks: {queue_size}/{self._MIN_CHUNKS_TO_START} "
                            f"at callback #{self._callback_count}"
                        )
                    self._fill_silence(output_buffer, 0, bytes_needed)
                    return
        
        # If we're still waiting for start time, output silence
        if self._scheduled_start_loop_time_us is not None:
            self._fill_silence(output_buffer, 0, bytes_needed)
            return
        
        # Check buffer level
        queue_size = self._queue.qsize()
        if queue_size < self._MIN_CHUNKS_TO_MAINTAIN:
            logger.warning(
                f"Low buffer warning: {queue_size} chunks remaining "
                f"(minimum: {self._MIN_CHUNKS_TO_MAINTAIN}) at callback #{self._callback_count}, "
                f"server_ts_cursor: {self._server_ts_cursor_us}"
            )
        
        # Read frames using bulk read (matches sendspin-cli's fast path)
        # This reads chunks in order and advances cursor as frames are read
        frames_data = self._read_input_frames_bulk(frames)
        frames_bytes = len(frames_data)
        
        # Copy to output buffer
        output_buffer[0:frames_bytes] = frames_data
        
        # Fill remaining with silence if needed
        if frames_bytes < bytes_needed:
            self._fill_silence(output_buffer, frames_bytes, bytes_needed - frames_bytes)
        
        # Apply volume scaling to the output (matches sendspin-cli)
        self._apply_volume(output_buffer, bytes_needed)
    
    def _initialize_current_chunk(self) -> None:
        """Load next chunk from queue and initialize read position.
        
        Updates server timestamp cursor if needed.
        """
        self._current_chunk = self._queue.get_nowait()
        self._current_chunk_offset = 0
        # Initialize server cursor if needed
        if self._server_ts_cursor_us == 0:
            self._server_ts_cursor_us = self._current_chunk.server_timestamp_us
    
    def _read_input_frames_bulk(self, n_frames: int) -> bytes:
        """Read N frames efficiently in bulk, handling chunk boundaries.
        
        Returns concatenated frame data. Much faster than calling
        _read_one_input_frame() N times due to reduced overhead.
        Matches sendspin-cli's implementation.
        """
        if self._format is None or n_frames <= 0:
            return b""
        
        frame_size = self._format.frame_size
        total_bytes_needed = n_frames * frame_size
        result = bytearray(total_bytes_needed)
        bytes_written = 0
        
        while bytes_written < total_bytes_needed:
            # Get frames from current chunk
            if self._current_chunk is None:
                if self._queue.empty():
                    # No more data - pad with silence
                    silence_bytes = total_bytes_needed - bytes_written
                    result[bytes_written:] = b"\x00" * silence_bytes
                    break
                self._initialize_current_chunk()
            
            # Calculate how much we can read from current chunk
            assert self._current_chunk is not None
            chunk_data = self._current_chunk.audio_data
            available_bytes = len(chunk_data) - self._current_chunk_offset
            bytes_to_read = min(available_bytes, total_bytes_needed - bytes_written)
            
            # Bulk copy from chunk to result
            result[bytes_written : bytes_written + bytes_to_read] = chunk_data[
                self._current_chunk_offset : self._current_chunk_offset + bytes_to_read
            ]
            
            # Update state
            self._current_chunk_offset += bytes_to_read
            bytes_written += bytes_to_read
            frames_read = bytes_to_read // frame_size
            self._advance_server_cursor_frames(frames_read)
            
            # Check if chunk finished
            if self._current_chunk_offset >= len(chunk_data):
                self._advance_finished_chunk()
        
        # Save last frame for potential duplication (matches sendspin-cli)
        if bytes_written >= frame_size:
            self._last_output_frame = bytes(result[bytes_written - frame_size : bytes_written])
        
        return bytes(result)
    
    def _advance_finished_chunk(self) -> None:
        """Update durations and state when current chunk is fully consumed."""
        assert self._format is not None
        if self._current_chunk is None:
            return
        data = self._current_chunk.audio_data
        chunk_frames = len(data) // self._format.frame_size
        chunk_duration_us = (chunk_frames * 1_000_000) // self._format.sample_rate
        self._queued_duration_us = max(0, self._queued_duration_us - chunk_duration_us)
        self._current_chunk = None
        self._current_chunk_offset = 0
    
    def _apply_volume(self, output_buffer: memoryview, num_bytes: int) -> None:
        """Apply volume scaling to the output buffer.
        
        Matches sendspin-cli's implementation.
        """
        volume = 0 if self._muted else self._volume
        
        if volume == 0:
            # Fill with silence
            output_buffer[:num_bytes] = b"\x00" * num_bytes
            return
        
        if volume == 100:
            return
        
        # Create view of buffer as int16 samples (copy needed for modification)
        samples = np.frombuffer(output_buffer[:num_bytes], dtype=np.int16).copy()
        # Power curve for natural volume control (gentler at high volumes)
        amplitude = (volume / 100.0) ** 1.5
        samples = (samples * amplitude).astype(np.int16)
        # Write modified samples back to buffer
        output_buffer[:num_bytes] = samples.tobytes()
        # Write back to buffer
        output_buffer[:num_bytes] = samples.tobytes()
    
    def _advance_server_cursor_frames(self, frames: int) -> None:
        """Advance server timeline cursor by a number of frames.
        
        This advances the cursor based on frames actually output, ensuring
        we only consume chunks when it's time to play them.
        """
        if self._format is None or frames <= 0:
            return
        # Accumulate microseconds precisely: add 1e6 per frame, carry by sample_rate
        # This matches sendspin-cli's implementation
        self._server_ts_cursor_remainder += frames * 1_000_000
        sr = self._format.sample_rate
        if self._server_ts_cursor_remainder >= sr:
            inc_us = self._server_ts_cursor_remainder // sr
            self._server_ts_cursor_remainder = self._server_ts_cursor_remainder % sr
            self._server_ts_cursor_us += int(inc_us)
    
    def _update_playback_position(self, frames: int, time_info: Any) -> None:
        """Update playback position tracking using DAC timing."""
        if self._format is None:
            return
        
        # Get DAC time (when this buffer will be played)
        dac_time_sec = time_info.outputBufferDacTime
        dac_time_us = int(dac_time_sec * self._MICROSECONDS_PER_SECOND)
        
        # Get current loop time
        loop_time_us = int(self._loop.time() * self._MICROSECONDS_PER_SECOND)
        
        # Calibrate DAC-to-loop mapping
        self._dac_loop_calibrations.append((dac_time_us, loop_time_us))
        
        # Update last known playback position
        self._last_known_playback_position_us = self._server_ts_cursor_us
        self._last_dac_calibration_time_us = loop_time_us
    
    def _close_stream(self) -> None:
        """Close the audio stream."""
        if self._stream is not None:
            try:
                if not self._closed:
                    self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.debug(f"Error closing audio stream: {e}")
            self._stream = None
            self._closed = True
            self._stream_started = False
    
    def __del__(self) -> None:
        """Cleanup on deletion."""
        self._close_stream()

