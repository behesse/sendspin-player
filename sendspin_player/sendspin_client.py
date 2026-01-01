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

from aiosendspin_sounddevice import SendspinAudioClient, SendspinAudioClientConfig, SupportedAudioFormat, AudioCodec
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
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self.is_running = False
        self._connection_status = "disconnected"
        self._last_error: Optional[str] = None
        self._client_id = str(uuid.uuid4())
        self._device_manager = AudioDeviceManager()
        
        # Resolve audio_device name to AudioDevice object if provided
        self._audio_device = self._resolve_audio_device(audio_device, config.audio_device)
    
    def _resolve_audio_device(self, audio_device: Optional[int] = None, config_device: Optional[str] = None) -> Optional[Any]:
        """Resolve audio device from various input types to AudioDevice object."""
        if audio_device is not None:
            if isinstance(audio_device, int):
                return self._device_manager.find_by_index(audio_device)
            else:
                return self._device_manager.find_by_name(audio_device, exact=False)
        elif config_device is not None:
            return self._device_manager.find_by_name(config_device, exact=False)
        return None
    
    async def start(self) -> bool:
        """Start the sendspin client connection."""
        if self.is_running:
            logger.debug("Sendspin client is already running, cannot start again")
            return False
        
        if not self.config.sendspin_server_url:
            logger.debug("Sendspin server URL not configured")
            raise ValueError("Sendspin server URL not configured")
        
        # Clear any previous error when starting
        self._last_error = None
        self._connection_status = "connecting"
        
        ws_url = self.config.sendspin_server_url
        logger.debug(f"Connecting to websocket URL: {ws_url}")
        logger.info(f"Starting sendspin client connection to {ws_url}")
        
        try:
            # Set up callbacks for state updates (all just log for now)
            def create_log_callback(name: str):
                """Create a callback that logs updates."""
                def callback(data: Any) -> None:
                    logger.debug(f"{name}: {data}")
                return callback
            
            on_metadata_update = create_log_callback("Metadata update")
            on_group_update = create_log_callback("Group update")
            on_controller_state_update = create_log_callback("Controller state update")
            on_event = create_log_callback("Event")
            
            def on_audio_error(error: Exception, error_msg: str) -> None:
                """Handle audio playback errors (e.g., unsupported format).
                
                Args:
                    error: The exception that occurred
                    error_msg: Formatted error message from the library
                """
                try:
                    logger.error(error_msg, exc_info=error)
                    self._last_error = error_msg
                    self._connection_status = "error"
                    # Stop the client connection asynchronously
                    # We can't await here since this is a callback, so we schedule it
                    if self.client:
                        # Try to schedule the stop task using the stored event loop
                        if self._event_loop and self._event_loop.is_running():
                            # Schedule task in the event loop thread-safely
                            def schedule_stop():
                                try:
                                    self._event_loop.create_task(self._stop_on_error())
                                except Exception as e:
                                    logger.error(f"Error creating stop task: {e}", exc_info=True)
                            self._event_loop.call_soon_threadsafe(schedule_stop)
                        else:
                            # Try to get the current running loop as fallback
                            try:
                                loop = asyncio.get_running_loop()
                                loop.create_task(self._stop_on_error())
                            except RuntimeError:
                                logger.warning("Could not schedule client stop: no running event loop available")
                except Exception as e:
                    # Make sure the callback itself doesn't raise exceptions
                    logger.error(f"Error in on_audio_error callback: {e}", exc_info=True)
            
            # Resolve audio device config (AudioDevice, str, int, or None)
            audio_device_config = self._audio_device
            if not audio_device_config and self.config.audio_device:
                # Try to resolve by name, fallback to name string if not found
                audio_device_config = self._device_manager.find_by_name(self.config.audio_device, exact=False) or self.config.audio_device
            
            # Create supported audio format from config
            try:
                codec = AudioCodec[self.config.audio_codec.upper()]
            except (KeyError, AttributeError):
                codec = AudioCodec.PCM  # Default to PCM
            
            supported_format = SupportedAudioFormat(
                codec=codec,
                channels=self.config.audio_channels,
                sample_rate=self.config.audio_sample_rate,
                bit_depth=self.config.audio_bit_depth
            )
            
            client_config = SendspinAudioClientConfig(
                url=ws_url,
                client_id=self._client_id,
                client_name=self.config.client_name,
                static_delay_ms=0.0,
                audio_device=audio_device_config,
                supported_formats=[supported_format],
                on_metadata_update=on_metadata_update,
                on_group_update=on_group_update,
                on_controller_state_update=on_controller_state_update,
                on_event=on_event,
                on_audio_error=on_audio_error,
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
    
    async def _disconnect_and_cleanup(self, error_context: str = "") -> None:
        """Disconnect client and cancel connection task."""
        # Disconnect the client
        if self.client:
            try:
                await self.client.disconnect()
                logger.debug(f"Client disconnected{error_context}")
            except Exception as e:
                log_level = logger.error if error_context else logger.debug
                log_level(f"Error disconnecting client: {e}", exc_info=bool(error_context))
            finally:
                self.client = None
        
        # Cancel the connection task
        if self._connection_task and not self._connection_task.done():
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.debug(f"Error waiting for cancelled task: {e}", exc_info=True)
    
    async def _stop_on_error(self):
        """Stop the client when an audio error occurs."""
        try:
            logger.info("Stopping client due to audio error")
            await self._disconnect_and_cleanup(" due to audio error")
            self.is_running = False
            self._connection_status = "disconnected"
            logger.info("Client stopped due to audio error")
        except Exception as e:
            logger.error(f"Error stopping client after audio error: {e}", exc_info=True)
            self.is_running = False
            self._connection_status = "error"
    
    async def stop(self) -> bool:
        """Stop the sendspin client connection."""
        if not self.is_running:
            logger.debug("Sendspin client is not running, cannot stop")
            return False
        
        logger.info("Stopping sendspin client connection")
        
        try:
            await self._disconnect_and_cleanup()
            self.is_running = False
            self._connection_status = "disconnected"
            logger.info("Sendspin client stopped successfully")
            return True
        except Exception as e:
            logger.error(f"Error stopping client: {e}", exc_info=True)
            self.is_running = False
            self._connection_status = "disconnected"
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
    
    def _get_audio_device_info(self) -> Optional[Dict[str, Any]]:
        """Get audio device information dictionary."""
        device = self._audio_device or self._device_manager.get_default_device()
        if device:
            return {
                "index": device.index,
                "name": device.name,
                "channels": device.max_output_channels
            }
        return None
    
    def get_status(self) -> dict:
        """Get current status of the client."""
        audio_device_info = self._get_audio_device_info()
        
        # Get metadata from client
        metadata = {}
        playback_state = None
        volume = None
        muted = None
        track_progress = None
        track_duration = None
        
        if self.client and hasattr(self.client, 'is_connected') and self.client.is_connected:
            try:
                metadata = self.client.get_metadata() or {}
                playback_state = self.client.get_playback_state()
                volume, muted = self.client.get_player_volume()
                track_progress, track_duration = self.client.get_track_progress()
                
                # Fallback to metadata if get_track_progress() returns None
                if track_progress is None:
                    track_progress = metadata.get('track_progress')
                if track_duration is None:
                    track_duration = metadata.get('track_duration')
                
                logger.debug(
                    f"Status: state={playback_state}, progress={track_progress}/{track_duration}, "
                    f"volume={volume}, muted={muted}"
                )
            except (AttributeError, Exception) as e:
                logger.debug(f"Error getting client status: {e}", exc_info=True)
        
        return {
            "running": self.is_running,
            "server_url": self.config.sendspin_server_url,
            "client_name": self.config.client_name,
            "connection_status": self._connection_status,
            "last_error": self._last_error,
            "title": metadata.get("title"),
            "artist": metadata.get("artist"),
            "album": metadata.get("album"),
            "playback_state": playback_state.value if playback_state else None,
            "volume": volume,
            "muted": muted,
            "track_progress": track_progress,
            "track_duration": track_duration,
            "audio_device": audio_device_info,
        }
    
    @staticmethod
    def _device_to_dict(device) -> Dict[str, Any]:
        """Convert AudioDevice to dictionary."""
        return {
            "index": device.index,
            "name": device.name,
            "channels": device.max_output_channels,
            "sample_rate": int(device.default_samplerate),
            "is_default": device.is_default
        }
    
    @staticmethod
    def list_audio_devices() -> List[Dict[str, Any]]:
        """List available audio output devices."""
        devices = AudioDeviceManager.list_audio_devices()
        return [SendspinClientWrapper._device_to_dict(device) for device in devices]
