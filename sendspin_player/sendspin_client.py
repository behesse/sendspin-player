"""Async sendspin client using aiosendspin-sounddevice library.

This implementation uses aiosendspin-sounddevice which provides a complete
solution for connecting to Sendspin servers and playing audio.

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
import uuid
from typing import Optional, List, Dict, Any
from sendspin_player.config import AppConfig

from aiosendspin_sounddevice import SendspinAudioClient, SendspinAudioClientConfig
from aiosendspin_sounddevice.audio_device import AudioDeviceManager
from aiosendspin_sounddevice.discovery import ServiceDiscovery

logger = logging.getLogger('sendspin_client')


class SendspinClientWrapper:
    """Manages the sendspin client connection using aiosendspin-sounddevice.
    
    This wrapper provides a simplified interface around SendspinAudioClient
    for use in the web application.
    """
    
    def __init__(self, config: AppConfig, audio_device: Optional[int] = None):
        """
        Initialize the sendspin client wrapper.
        
        Args:
            config: Application configuration.
            audio_device: Optional audio device ID to use. None for default device.
        """
        self.config = config
        self.client: Optional[SendspinAudioClient] = None
        self._connection_task: Optional[asyncio.Task] = None
        self.is_running = False
        self._connection_status = "disconnected"
        self._last_error: Optional[str] = None
        self._client_id = str(uuid.uuid4())
        self._device_manager = AudioDeviceManager()
        
        # Resolve audio_device name to AudioDevice object if provided
        self._audio_device = None
        if audio_device is not None:
            # audio_device can be int (legacy) or str (device name)
            if isinstance(audio_device, int):
                self._audio_device = self._device_manager.find_by_index(audio_device)
            else:
                self._audio_device = self._device_manager.find_by_name(audio_device, exact=False)
        elif config.audio_device is not None:
            # config.audio_device is now a string (device name) or None
            self._audio_device = self._device_manager.find_by_name(config.audio_device, exact=False)
    
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
            # Set up callbacks for state updates
            def on_metadata_update(metadata: Dict[str, Any]) -> None:
                """Handle metadata updates from server."""
                logger.debug(f"Metadata update: {metadata}")
            
            def on_group_update(group: Dict[str, Any]) -> None:
                """Handle group updates from server."""
                logger.debug(f"Group update: {group}")
            
            def on_controller_state_update(state: Dict[str, Any]) -> None:
                """Handle controller state updates from server."""
                logger.debug(f"Controller state update: {state}")
            
            def on_event(event: str) -> None:
                """Handle events from server."""
                logger.debug(f"Event: {event}")
            
            # Create client config
            # audio_device can be AudioDevice, str (name), int (index), or None
            audio_device_config = None
            if self._audio_device:
                audio_device_config = self._audio_device
            elif self.config.audio_device:
                # Try to resolve by name
                audio_device_config = self._device_manager.find_by_name(self.config.audio_device, exact=False)
                if not audio_device_config:
                    # If not found, pass the name string directly
                    audio_device_config = self.config.audio_device
            
            client_config = SendspinAudioClientConfig(
                url=ws_url,
                client_id=self._client_id,
                client_name=self.config.client_name,
                static_delay_ms=0.0,
                audio_device=audio_device_config,
                on_metadata_update=on_metadata_update,
                on_group_update=on_group_update,
                on_controller_state_update=on_controller_state_update,
                on_event=on_event,
            )
            
            # Create client
            self.client = SendspinAudioClient(client_config)
            
            # Start connection in background task
            self._connection_task = asyncio.create_task(self._run_client())
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
    
    async def _run_client(self):
        """Run the client connection in a background task."""
        try:
            self._connection_status = "connecting"
            logger.debug("Establishing connection to sendspin server...")
            
            # Connect to server
            await self.client.connect()
            
            self._connection_status = "connected"
            logger.info("Successfully connected to sendspin server")
            
            # Wait for disconnect
            try:
                await self.client.wait_for_disconnect()
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
            
            # Disconnect the client
            if self.client:
                try:
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
        """Discover available sendspin servers using mDNS."""
        try:
            servers = await ServiceDiscovery.discover_servers(discovery_time=3.0)
            return [
                {
                    "url": server.url,
                    "display": server.name or server.url
                }
                for server in servers
            ]
        except Exception as e:
            logger.debug(f"Exception during server discovery: {type(e).__name__}: {e}", exc_info=True)
            logger.error(f"Error discovering servers: {e}")
            return []
    
    def get_status(self) -> dict:
        """Get current status of the client."""
        # Get audio device info
        audio_device_info = None
        if self._audio_device:
            audio_device_info = {
                "index": self._audio_device.index,
                "name": self._audio_device.name,
                "channels": self._audio_device.max_output_channels
            }
        else:
            # Get default device info
            default_device = self._device_manager.get_default_device()
            if default_device:
                audio_device_info = {
                    "index": default_device.index,
                    "name": default_device.name,
                    "channels": default_device.max_output_channels
                }
        
        # Get metadata from client
        metadata = {}
        playback_state = None
        volume = None
        muted = None
        track_progress = None
        track_duration = None
        audio_format_info = None
        format_mismatch = False
        playback_format_info = None
        queue_info = None
        
        if self.client:
            try:
                # Try to get state - the library should handle connection state internally
                if hasattr(self.client, 'is_connected') and self.client.is_connected:
                    # Get metadata (includes title, artist, album, track_progress, track_duration)
                    metadata = self.client.get_metadata() or {}
                    playback_state = self.client.get_playback_state()
                    volume, muted = self.client.get_player_volume()
                    
                    # get_track_progress() returns interpolated progress if playing
                    # This is more accurate than metadata.track_progress for live updates
                    track_progress, track_duration = self.client.get_track_progress()
                    
                    # Use track_progress from get_track_progress() if available (more accurate)
                    # Otherwise fall back to metadata
                    if track_progress is None and 'track_progress' in metadata:
                        track_progress = metadata.get('track_progress')
                    if track_duration is None and 'track_duration' in metadata:
                        track_duration = metadata.get('track_duration')
                    
                    logger.debug(
                        f"Status: state={playback_state}, progress={track_progress}/{track_duration}, "
                        f"volume={volume}, muted={muted}, metadata={metadata}"
                    )
                    
                    # Get timing metrics which may include format info
                    timing_metrics = self.client.get_timing_metrics()
                    if timing_metrics:
                        # The library handles format internally, so we don't have direct access
                        # but we can infer from timing metrics if needed
                        pass
                else:
                    # Client exists but not connected yet
                    logger.debug(f"Client exists but not connected. is_connected={getattr(self.client, 'is_connected', 'N/A')}, wrapper_status={self._connection_status}")
            except AttributeError as e:
                # Client might not be fully initialized yet
                logger.debug(f"Client not fully initialized: {e}")
            except Exception as e:
                logger.debug(f"Error getting client status: {e}", exc_info=True)
        
        return {
            "running": self.is_running,
            "format_mismatch": format_mismatch,
            "playback_format": playback_format_info,
            "server_url": self.config.sendspin_server_url,
            "client_name": self.config.client_name,
            "connection_status": self._connection_status,
            "last_error": self._last_error,
            # Metadata
            "title": metadata.get("title"),
            "artist": metadata.get("artist"),
            "album": metadata.get("album"),
            "playback_state": playback_state.value if playback_state else None,
            "volume": volume,
            "muted": muted,
            "track_progress": track_progress,
            "track_duration": track_duration,
            # Audio device and format
            "audio_device": audio_device_info,
            "audio_format": audio_format_info,
            "audio_queue": queue_info
        }
    
    @staticmethod
    def list_audio_devices() -> List[Dict[str, Any]]:
        """List available audio output devices."""
        devices = AudioDeviceManager.list_audio_devices()
        return [
            {
                "index": device.index,
                "name": device.name,
                "channels": device.max_output_channels,
                "sample_rate": int(device.default_samplerate)
            }
            for device in devices
        ]
