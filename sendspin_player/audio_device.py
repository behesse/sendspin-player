"""Audio device management and querying.

This module provides functionality to query and manage audio output devices
using sounddevice.
"""

import logging
from typing import Optional, List, Dict, Any

import sounddevice

logger = logging.getLogger('audio_device')

# Expose sounddevice module for type annotations
__all__ = ['AudioDevice', 'AudioDeviceManager', 'sounddevice']


class AudioDevice:
    """Represents a single audio output device.
    
    Each instance wraps sounddevice operations for a specific device.
    """
    
    def __init__(self, device_id: Optional[int], device_manager: 'AudioDeviceManager'):
        """
        Initialize an audio device instance.
        
        Args:
            device_id: Audio device index, or None for default device.
            device_manager: Reference to the AudioDeviceManager that created this instance.
        """
        self._device_id = device_id
        self._device_manager = device_manager
        self._info: Optional[Dict[str, Any]] = None
        self._load_info()
    
    def _load_info(self) -> None:
        """Load device information from sounddevice."""
        self._info = self._device_manager._get_device_info_raw(self._device_id)
    
    @property
    def device_id(self) -> Optional[int]:
        """Get the device index."""
        return self._device_id
    
    @property
    def name(self) -> str:
        """Get the device name."""
        if self._info:
            return self._info.get("name", "Unknown")
        return "Unknown"
    
    @property
    def channels(self) -> int:
        """Get the maximum number of output channels."""
        if self._info:
            return self._info.get("channels", 0)
        return 0
    
    @property
    def sample_rate(self) -> int:
        """Get the default sample rate for this device."""
        if self._info:
            return self._info.get("sample_rate", 44100)
        return 44100
    
    def get_info(self) -> Dict[str, Any]:
        """Get device information dictionary.
        
        Returns:
            Dictionary with index, name, channels, and sample_rate.
        """
        if self._info:
            return {
                "index": self._device_id,
                "name": self.name,
                "channels": self.channels,
                "sample_rate": self.sample_rate
            }
        return {
            "index": self._device_id,
            "name": "Unknown",
            "channels": 0,
            "sample_rate": 44100
        }
    
    def create_output_stream(
        self,
        samplerate: int,
        channels: int,
        dtype: str = "int16",
        blocksize: int = 2048,
        callback: Optional[Any] = None,
        latency: str = "high",
    ) -> sounddevice.RawOutputStream:
        """Create a RawOutputStream for this device.
        
        Args:
            samplerate: Sample rate in Hz.
            channels: Number of channels.
            dtype: Data type (default: "int16").
            blocksize: Block size in frames.
            callback: Callback function for audio processing.
            latency: Latency setting (default: "high").
            
        Returns:
            RawOutputStream instance configured for this device.
        """
        return sounddevice.RawOutputStream(
            samplerate=samplerate,
            channels=channels,
            dtype=dtype,
            blocksize=blocksize,
            callback=callback,
            latency=latency,
            device=self._device_id,
        )
    
    def format_info_string(self) -> str:
        """Format device information string for logging.
        
        Returns:
            Formatted string like ", device=29" or empty string for default.
        """
        return f", device={self._device_id}" if self._device_id is not None else ""


class AudioDeviceManager:
    """Manages audio device queries and creates AudioDevice instances."""
    
    def __init__(self):
        """Initialize the audio device manager."""
        pass
    
    def _get_device_info_raw(self, device_id: Optional[int]) -> Optional[Dict[str, Any]]:
        """Get raw device information from sounddevice.
        
        Internal method used by AudioDevice instances.
        
        Args:
            device_id: Audio device index, or None for default device.
            
        Returns:
            Dictionary with device info or None if not available.
        """
        try:
            if device_id is not None:
                device_info = sounddevice.query_devices(device_id, kind='output')
                return {
                    "index": device_id,
                    "name": device_info.get("name", "Unknown"),
                    "channels": device_info.get("max_output_channels", 0),
                    "sample_rate": int(device_info.get("default_samplerate", 44100))
                }
            else:
                # Default device
                devices = sounddevice.query_devices()
                default_idx = sounddevice.default.device[1] if sounddevice.default.device else None
                if default_idx is not None and 0 <= default_idx < len(devices):
                    device_info = devices[default_idx]
                    return {
                        "index": default_idx,
                        "name": device_info.get("name", "Unknown"),
                        "channels": device_info.get("max_output_channels", 0),
                        "sample_rate": int(device_info.get("default_samplerate", 44100))
                    }
        except Exception as e:
            logger.debug(f"Error querying device {device_id}: {e}")
        return None
    
    def get_device(self, device_id: Optional[int]) -> AudioDevice:
        """Get an AudioDevice instance for the specified device.
        
        Args:
            device_id: Audio device index, or None for default device.
            
        Returns:
            AudioDevice instance for the specified device.
        """
        return AudioDevice(device_id, self)
    
    def get_device_info(self, device_id: Optional[int]) -> Optional[Dict[str, Any]]:
        """Get information about an audio device.
        
        Args:
            device_id: Audio device index, or None for default device.
            
        Returns:
            Dictionary with device info (index, name, channels, sample_rate) or None if not available.
        """
        device = self.get_device(device_id)
        return device.get_info()
    
    def get_device_sample_rate(self, device_id: Optional[int]) -> int:
        """Get the default sample rate for an audio device.
        
        Args:
            device_id: Audio device index, or None for default device.
            
        Returns:
            Default sample rate in Hz, or 44100 if unavailable.
        """
        device = self.get_device(device_id)
        return device.sample_rate
    
    def list_devices(self) -> List[Dict[str, Any]]:
        """List all available audio output devices.
        
        Returns:
            List of dictionaries with device information (index, name, channels, sample_rate, is_default).
            Returns empty list if no devices are available or if an error occurs.
        """
        try:
            devices = sounddevice.query_devices()
            default_device = sounddevice.default.device[1] if sounddevice.default.device else None
            
            result = []
            for i, device in enumerate(devices):
                if device.get("max_output_channels", 0) > 0:
                    result.append({
                        "index": i,
                        "name": device.get("name", "Unknown"),
                        "channels": device.get("max_output_channels", 0),
                        "sample_rate": int(device.get("default_samplerate", 44100)),
                        "is_default": i == default_device
                    })
            return result
        except Exception as e:
            logger.error(f"Error listing audio devices: {e}")
            return []
    
    def format_device_info_string(self, device: Optional[int]) -> str:
        """Format device information string for logging.
        
        Args:
            device: Audio device index, or None for default device.
            
        Returns:
            Formatted string like ", device=29" or empty string for default.
        """
        return f", device={device}" if device is not None else ""
    
    def find_device_by_name(self, device_name: str) -> Optional[int]:
        """Find a device index by its name.
        
        Args:
            device_name: Name of the audio device to find.
            
        Returns:
            Device index if found, None otherwise.
        """
        devices = self.list_devices()
        for device in devices:
            if device.get("name") == device_name:
                return device.get("index")
        return None
    
    def resolve_device(self, device_identifier: Optional[str]) -> Optional[int]:
        """Resolve a device identifier (name or "default") to an index.
        
        Args:
            device_identifier: Device name, "default", or None for default device.
            
        Returns:
            Device index if found, None for default device.
        """
        if device_identifier is None or device_identifier == "" or device_identifier.lower() == "default":
            return None
        return self.find_device_by_name(device_identifier)
