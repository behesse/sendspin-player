"""Async sendspin client using aiosendspin with audio playback.

This implementation uses aiosendspin for the Sendspin protocol and includes
audio playback using sounddevice, based on the sendspin-cli implementation.

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
import logging
import subprocess
import uuid
from typing import Optional, List, Dict, Any
from sendspin_player.config import AppConfig

from aiosendspin.client import SendspinClient, PCMFormat
from aiosendspin.models.core import Roles, StreamStartMessage, ServerStatePayload, ServerCommandPayload
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import (
    AudioCodec,
    PlayerCommand,
    PlaybackStateType,
    UndefinedField,
)

from sendspin_player.audio_player import AudioPlayer
from sendspin_player.audio_device import AudioDeviceManager

logger = logging.getLogger('sendspin_client')


class SendspinClientWrapper:
    """Manages the sendspin client connection using aiosendspin with audio playback.
    
    This implementation handles both the Sendspin protocol and audio playback,
    based on the sendspin-cli reference implementation.
    """
    
    def __init__(self, config: AppConfig, audio_device: Optional[int] = None):
        """
        Initialize the sendspin client wrapper.
        
        Args:
            config: Application configuration.
            audio_device: Optional audio device ID to use. None for default device.
        """
        self.config = config
        self.client: Optional[SendspinClient] = None
        self._connection_task: Optional[asyncio.Task] = None
        self.is_running = False
        self._connection_status = "disconnected"
        self._last_error: Optional[str] = None
        self._client_id = str(uuid.uuid4())
        self._device_manager = AudioDeviceManager()
        # Resolve audio_device name to index if provided
        if audio_device is not None:
            # audio_device can be int (legacy) or str (device name)
            if isinstance(audio_device, int):
                self._audio_device = audio_device
            else:
                self._audio_device = self._device_manager.resolve_device(audio_device)
        elif config.audio_device is not None:
            # config.audio_device is now a string (device name) or None
            self._audio_device = self._device_manager.resolve_device(config.audio_device)
        else:
            self._audio_device = None
        self.audio_player: Optional[AudioPlayer] = None
        self._current_format: Optional[PCMFormat] = None
        
        # Track server state (metadata, playback state, etc.)
        self._playback_state: Optional[PlaybackStateType] = None
        self._title: Optional[str] = None
        self._artist: Optional[str] = None
        self._album: Optional[str] = None
        self._volume: Optional[int] = None
        self._muted: Optional[bool] = None
        self._track_progress: Optional[int] = None
        self._track_duration: Optional[int] = None
    
    def _extract_format_info(self, pcm_format: PCMFormat) -> Dict[str, Any]:
        """Extract format information from a PCMFormat object.
        
        Args:
            pcm_format: PCM format object.
            
        Returns:
            Dictionary with sample_rate, channels, and bit_depth.
        """
        return {
            "sample_rate": pcm_format.sample_rate,
            "channels": pcm_format.channels,
            "bit_depth": getattr(pcm_format, 'bit_depth', 16)
        }
    
    async def start(self) -> bool:
        """Start the sendspin client connection."""
        if self.is_running:
            logger.debug("Sendspin client is already running, cannot start again")
            return False
        
        if not self.config.sendspin_server_url:
            logger.debug("Sendspin server URL not configured")
            raise ValueError("Sendspin server URL not configured")
        
        ws_url = self.config.sendspin_server_url
        logger.debug(f"Connecting to websocket URL: {ws_url}")
        logger.info(f"Starting sendspin client connection to {ws_url}")
        
        try:
            # Determine supported sample rates based on audio device
            # Query device to get its default/preferred sample rate
            supported_sample_rates = [44_100]  # Always support 44.1kHz (standard)
            audio_device = self._device_manager.get_device(self._audio_device)
            device_rate = audio_device.sample_rate
            # Add device's preferred rate if different from 44.1kHz
            if device_rate != 44100 and device_rate not in supported_sample_rates:
                supported_sample_rates.append(device_rate)
                logger.info(f"Detected device sample rate: {device_rate}Hz, will request server to use this format")
            
            # Build supported formats list - include both 44.1kHz and device rate
            supported_formats = []
            for sample_rate in supported_sample_rates:
                # Support both mono and stereo
                supported_formats.append(
                    SupportedAudioFormat(
                        codec=AudioCodec.PCM,
                        channels=2,
                        sample_rate=sample_rate,
                        bit_depth=16
                    )
                )
                supported_formats.append(
                    SupportedAudioFormat(
                        codec=AudioCodec.PCM,
                        channels=1,
                        sample_rate=sample_rate,
                        bit_depth=16
                    )
                )
            
            # Create aiosendspin client
            # Roles: PLAYER for audio playback, METADATA for track info
            # player_support is required when PLAYER role is specified
            # The server will send audio in one of the supported formats, eliminating need for client-side resampling
            self.client = SendspinClient(
                client_id=self._client_id,
                client_name=self.config.client_name,
                roles=[Roles.PLAYER, Roles.METADATA],  # Basic roles for audio client
                player_support=ClientHelloPlayerSupport(
                    supported_formats=supported_formats,
                    buffer_capacity=32_000_000,  # 32MB buffer capacity
                    supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
                ),
                static_delay_ms=0.0,
                initial_volume=100,
                initial_muted=False
            )
            
            # Set up audio chunk callback
            def audio_chunk_callback(server_timestamp_us: int, audio_data: bytes, fmt: PCMFormat) -> None:
                """Handle incoming audio chunks."""
                # Initialize or reconfigure audio player if format changed
                if self.audio_player is None or self._current_format != fmt:
                    if self.audio_player is not None:
                        self.audio_player.clear()
                    
                    loop = asyncio.get_running_loop()
                    self.audio_player = AudioPlayer(
                        loop=loop,
                        compute_client_time=self.client.compute_play_time,
                        compute_server_time=self.client.compute_server_time,
                        device_manager=self._device_manager,
                    )
                    self.audio_player.set_format(fmt, device=self._audio_device)
                    # Set initial volume if we have it
                    if self._volume is not None:
                        self.audio_player.set_volume(self._volume, muted=self._muted)
                        logger.debug(f"Set initial volume on audio player: {self._volume}% {'(muted)' if self._muted else ''}")
                    self._current_format = fmt
                    logger.info(f"Audio player configured: {fmt.sample_rate}Hz, {fmt.channels}ch")
                
                # Submit audio chunk - AudioPlayer handles timing
                if self.audio_player is not None:
                    self.audio_player.submit(server_timestamp_us, audio_data)
            
            self.client.set_audio_chunk_listener(audio_chunk_callback)
            
            # Set up stream start callback
            def stream_start_callback(message: StreamStartMessage) -> None:
                """Handle stream start by clearing stale audio chunks."""
                if self.audio_player is not None:
                    self.audio_player.clear()
                    logger.debug("Cleared audio queue on stream start")
                logger.info("Stream started")
            
            self.client.set_stream_start_listener(stream_start_callback)
            
            # Set up stream end callback
            def stream_end_callback(roles: Optional[List[Roles]]) -> None:
                """Handle stream end by clearing audio queue."""
                if (roles is None or Roles.PLAYER in roles) and self.audio_player is not None:
                    self.audio_player.clear()
                    logger.debug("Cleared audio queue on stream end")
                logger.info("Stream ended")
            
            self.client.set_stream_end_listener(stream_end_callback)
            
            # Set up stream clear callback
            def stream_clear_callback() -> None:
                """Handle stream clear by clearing audio queue."""
                if self.audio_player is not None:
                    self.audio_player.clear()
                    logger.debug("Cleared audio queue on stream clear")
            
            self.client.set_stream_clear_listener(stream_clear_callback)
            
            # Set up disconnect callback
            def disconnect_callback():
                """Handle disconnection."""
                logger.info("Disconnected from sendspin server")
                self._connection_status = "disconnected"
                self.is_running = False
                if self.audio_player is not None:
                    self.audio_player.clear()
            
            self.client.set_disconnect_listener(disconnect_callback)
            
            # Set up metadata listener to track track information
            def metadata_listener(payload: ServerStatePayload) -> None:
                """Handle metadata updates from server."""
                if payload.metadata is not None:
                    # Update title, artist, album
                    if not isinstance(payload.metadata.title, UndefinedField):
                        self._title = payload.metadata.title if payload.metadata.title else None
                    if not isinstance(payload.metadata.artist, UndefinedField):
                        self._artist = payload.metadata.artist if payload.metadata.artist else None
                    if not isinstance(payload.metadata.album, UndefinedField):
                        self._album = payload.metadata.album if payload.metadata.album else None
                    
                    # Update progress
                    if not isinstance(payload.metadata.progress, UndefinedField):
                        if payload.metadata.progress is None:
                            self._track_progress = None
                            self._track_duration = None
                        else:
                            self._track_progress = payload.metadata.progress.track_progress
                            self._track_duration = payload.metadata.progress.track_duration
            
            self.client.set_metadata_listener(metadata_listener)
            
            # Handle server commands - volume changes come through here!
            def server_command_listener(command: ServerCommandPayload) -> None:
                """Handle server commands (volume, mute, etc.)."""
                logger.debug(f"Server command received: {command}")
                if command.player is not None:
                    player_cmd = command.player
                    if player_cmd.command == PlayerCommand.VOLUME:
                        # Volume command received
                        if player_cmd.volume is not None:
                            old_volume = self._volume
                            self._volume = player_cmd.volume
                            logger.info(f"Volume command received: {self._volume}% (was {old_volume}%)")
                            if self.audio_player is not None:
                                self.audio_player.set_volume(self._volume, muted=self._muted)
                                logger.info(f"Volume updated via server command: {self._volume}%")
                            else:
                                logger.debug(f"Volume command received but audio player not ready: {self._volume}%")
                        if player_cmd.mute is not None:
                            old_muted = self._muted
                            self._muted = player_cmd.mute
                            logger.info(f"Mute command received: {self._muted} (was {old_muted})")
                            if self.audio_player is not None:
                                self.audio_player.set_volume(self._volume, muted=self._muted)
                                logger.info(f"Mute updated via server command: {self._muted}")
                            else:
                                logger.debug(f"Mute command received but audio player not ready: {self._muted}")
                    elif player_cmd.command == PlayerCommand.MUTE:
                        # Mute command received - server sends this to toggle mute state
                        # The mute field might always be True regardless of desired state,
                        # so we toggle based on current state
                        old_muted = self._muted if self._muted is not None else False
                        # Toggle the mute state
                        self._muted = not old_muted
                        logger.info(f"Mute toggle command received: {self._muted} (was {old_muted})")
                        if self.audio_player is not None:
                            self.audio_player.set_volume(self._volume, muted=self._muted)
                            logger.info(f"Mute toggled via server command: {self._muted}")
                        else:
                            logger.debug(f"Mute toggle command received but audio player not ready: {self._muted}")
            
            self.client.set_server_command_listener(server_command_listener)
            logger.debug("Server command listener registered")
            
            # Start connection in background task
            self._connection_task = asyncio.create_task(self._run_client(ws_url))
            self.is_running = True
            self._connection_status = "connecting"
            logger.info("Sendspin client connection started")
            return True
            
        except Exception as e:
            error_msg = f"Failed to start sendspin client: {e}"
            logger.debug(f"Exception during start: {type(e).__name__}: {e}", exc_info=True)
            logger.error(error_msg)
            self._last_error = str(e)
            self._connection_status = "error"
            raise RuntimeError(error_msg)
    
    async def _run_client(self, url: str):
        """Run the client connection in a background task."""
        try:
            self._connection_status = "connecting"
            logger.debug("Establishing connection to sendspin server...")
            
            # Connect to server (connect is async)
            await self.client.connect(url)
            
            self._connection_status = "connected"
            logger.info("Successfully connected to sendspin server")
            
            # Keep the connection alive
            # The client handles protocol communication and will call our callbacks
            # We just need to keep this task running
            try:
                await asyncio.sleep(float('inf'))
            except asyncio.CancelledError:
                raise
                
        except asyncio.CancelledError:
            logger.debug("Client connection task cancelled")
            self._connection_status = "disconnected"
            raise
        except Exception as e:
            error_msg = f"Connection error: {e}"
            logger.error(error_msg)
            logger.debug(f"Connection error details: {type(e).__name__}: {e}", exc_info=True)
            self._last_error = str(e)
            self._connection_status = "error"
        finally:
            self.is_running = False
            self._connection_status = "disconnected"
            logger.info("Sendspin client connection closed")
    
    async def stop(self) -> bool:
        """Stop the sendspin client connection."""
        if not self.is_running:
            logger.debug("Sendspin client is not running, cannot stop")
            return False
        
        logger.info("Stopping sendspin client connection")
        
        try:
            # Cancel the connection task
            if self._connection_task and not self._connection_task.done():
                self._connection_task.cancel()
                try:
                    await self._connection_task
                except asyncio.CancelledError:
                    pass
            
            # Clear audio player
            if self.audio_player is not None:
                self.audio_player.clear()
                self.audio_player = None
            
            # Disconnect the client
            if self.client:
                try:
                    # disconnect is async
                    await self.client.disconnect()
                except Exception as e:
                    logger.debug(f"Error disconnecting client: {e}")
                self.client = None
            
            self.is_running = False
            self._connection_status = "disconnected"
            logger.info("Sendspin client stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping client: {e}")
            return False
    
    async def restart(self) -> bool:
        """Restart the sendspin client."""
        await self.stop()
        return await self.start()
    
    @staticmethod
    async def discover_servers() -> List[Dict[str, Any]]:
        """Discover available sendspin servers using sendspin --list-servers."""
        try:
            cmd = ["sendspin", "--list-servers"]
            logger.debug(f"Running command: {' '.join(cmd)}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=10.0
            )
            
            logger.debug(f"Command return code: {process.returncode}")
            logger.debug(f"Command stdout: {stdout.decode()}")
            if stderr:
                logger.debug(f"Command stderr: {stderr.decode()}")
            
            if process.returncode != 0:
                logger.warning(f"sendspin --list-servers failed: {stderr.decode()}")
                return []
            
            servers = []
            raw_output = stdout.decode()
            logger.debug(f"Parsing output: {repr(raw_output)}")
            
            # Parse structured output format
            lines = raw_output.split('\n')
            
            for line in lines:
                line_stripped = line.strip()
                
                # Look for "URL:" line
                if line_stripped.startswith('URL:'):
                    url_part = line_stripped[4:].strip()
                    if not url_part or '://' not in url_part:
                        continue
                    
                    logger.debug(f"Found URL line: {url_part}")
                    
                    if not (url_part.startswith('ws://') or url_part.startswith('wss://')):
                        logger.debug(f"URL missing protocol: {url_part}")
                        continue
                    
                    display = url_part
                    
                    server_info = {
                        "url": url_part,
                        "display": display
                    }
                    
                    if not any(s.get("url") == url_part for s in servers):
                        servers.append(server_info)
                        logger.debug(f"Parsed server: {server_info}")
            
            logger.debug(f"Parsed {len(servers)} servers: {servers}")
            logger.info(f"Discovered {len(servers)} sendspin servers")
            return servers
            
        except FileNotFoundError:
            logger.debug("sendspin command not found in PATH")
            logger.warning("sendspin not found, cannot discover servers")
            return []
        except asyncio.TimeoutError:
            logger.debug("sendspin --list-servers command timed out after 10 seconds")
            logger.warning("sendspin --list-servers timed out")
            return []
        except Exception as e:
            logger.debug(f"Exception during server discovery: {type(e).__name__}: {e}", exc_info=True)
            logger.error(f"Error discovering servers: {e}")
            return []
    
    def get_status(self) -> dict:
        """Get current status of the client."""
        # Get audio device info
        audio_device_obj = self._device_manager.get_device(self._audio_device)
        audio_device_info = audio_device_obj.get_info()
        # Only include index, name, and channels for status (not sample_rate)
        audio_device_info = {
            "index": audio_device_info["index"],
            "name": audio_device_info["name"],
            "channels": audio_device_info["channels"]
        }
        
        # Get audio format info - check both _current_format and audio_player
        audio_format_info = None
        format_mismatch = False
        playback_format_info = None
        
        # Prefer audio_player format if available (more accurate)
        if self.audio_player is not None and self.audio_player._format is not None:
            audio_format_info = self._extract_format_info(self.audio_player._format)
            # Check if playback format differs (fallback occurred)
            if self.audio_player._playback_format is not None and self.audio_player._format_mismatch:
                format_mismatch = True
                playback_format_info = self._extract_format_info(self.audio_player._playback_format)
        elif self._current_format is not None:
            # Fallback to _current_format if audio_player not initialized yet
            audio_format_info = self._extract_format_info(self._current_format)
        
        # Get audio player queue info
        queue_info = None
        if self.audio_player is not None:
            queue_info = {
                "queue_size": self.audio_player._queue.qsize(),
                "stream_started": self.audio_player._stream_started
            }
        
        return {
            "running": self.is_running,
            "format_mismatch": format_mismatch,
            "playback_format": playback_format_info,
            "server_url": self.config.sendspin_server_url,
            "client_name": self.config.client_name,
            "connection_status": self._connection_status,
            "last_error": self._last_error,
            # Metadata
            "title": self._title,
            "artist": self._artist,
            "album": self._album,
            "playback_state": self._playback_state.value if self._playback_state else None,
            "volume": self._volume,
            "muted": self._muted,
            "track_progress": self._track_progress,
            "track_duration": self._track_duration,
            # Audio device and format
            "audio_device": audio_device_info,
            "audio_format": audio_format_info,
            "audio_queue": queue_info
        }
    
    @staticmethod
    def list_audio_devices() -> List[Dict[str, Any]]:
        """List available audio output devices."""
        device_manager = AudioDeviceManager()
        return device_manager.list_devices()
