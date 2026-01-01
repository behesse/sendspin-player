"""Configuration management for the sendspin player application.

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
import yaml
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict


@dataclass
class AppConfig:
    """Application configuration model."""
    sendspin_server_url: str = ""
    client_name: str = "sendspin-player"
    audio_device: Optional[str] = None  # None for default device, str for device name
    audio_codec: str = "PCM"  # Audio codec: PCM, OPUS, FLAC
    audio_channels: int = 2  # Number of audio channels
    audio_sample_rate: int = 44100  # Sample rate in Hz
    audio_bit_depth: int = 16  # Bit depth


class ConfigManager:
    """Manages application configuration persistence."""
    
    def __init__(self, config_file: str = "config.yml"):
        self.config_file = Path(config_file)
        self._config: Optional[AppConfig] = None
    
    def load(self) -> AppConfig:
        """Load configuration from file or return defaults."""
        if self._config is not None:
            return self._config
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = yaml.safe_load(f) or {}
                    # Migrate from old format (address + port) to new format (URL)
                    server_url = data.get('sendspin_server_url', '')
                    if not server_url and data.get('sendspin_server_address'):
                        # Migrate old format to new format
                        address = data.get('sendspin_server_address', '')
                        port = data.get('sendspin_server_port', 8080)
                        if address:
                            server_url = f"ws://{address}:{port}"
                    
                    audio_device = data.get('audio_device')
                    # Ensure audio_device is a string or None
                    if audio_device is not None and not isinstance(audio_device, str):
                        audio_device = None
                    
                    config_dict = {
                        'sendspin_server_url': server_url,
                        'client_name': data.get('client_name', 'sendspin-player'),
                        'audio_device': audio_device,
                        'audio_codec': data.get('audio_codec', 'PCM'),
                        'audio_channels': data.get('audio_channels', 2),
                        'audio_sample_rate': data.get('audio_sample_rate', 44100),
                        'audio_bit_depth': data.get('audio_bit_depth', 16)
                    }
                    self._config = AppConfig(**config_dict)
            except (yaml.YAMLError, ValueError, TypeError) as e:
                print(f"Error loading config: {e}. Using defaults.")
                self._config = AppConfig()
        else:
            self._config = AppConfig()
        
        return self._config
    
    def save(self, config: AppConfig) -> None:
        """Save configuration to file."""
        self._config = config
        with open(self.config_file, 'w') as f:
            yaml.dump(asdict(config), f, default_flow_style=False, sort_keys=False)
    
    def get_config(self) -> AppConfig:
        """Get current configuration."""
        return self.load()
    
    def update_config(self, **kwargs) -> AppConfig:
        """Update configuration with new values."""
        config = self.load()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        self.save(config)
        return config
