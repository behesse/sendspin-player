# Sendspin Player

A Python web application for configuring and managing a sendspin player on headless Linux devices with audio output. The app provides a web interface for configuring the sendspin server connection and manages the sendspin player connection using aiosendspin.

> **⚠️ Work in Progress**: This project is still under active development. Features may change or are not working at all.

> **📢 Disclaimer**: This project is not directly affiliated with sendspin. It is a community project that implements the sendspin protocol for audio streaming. For official sendspin products and support, please visit the [sendspin website](https://www.sendspin-audio.com/).

## Supported Devices

This player works on any headless Linux device with audio output capabilities. Common setups include:
- Raspberry Pi with HiFiBerry DAC (e.g., HiFiBerry DAC+)
- Any Linux system with USB audio devices
- Linux systems with built-in audio output

## Features

- Web-based configuration interface using FastAPI
- Modern UI with Pico.css
- Interactive forms with HTMX
- Configuration management for:
  - Sendspin server address and port
  - Client name
- Client control (start, stop, restart)
- Real-time status updates

## Requirements

- Python 3.8+
- Node.js and npm (for frontend dependencies)
- sendspin-cli (for server discovery only, installed separately)

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install frontend dependencies (Pico.css and HTMX):
```bash
npm install
```

The `postinstall` script will automatically copy the assets to the `static/` directory.

3. Install sendspin-cli (for server discovery - follow the official sendspin-cli installation instructions)

4. Run the application:
```bash
python main.py
```

The web interface will be available at `http://localhost:8000`

## Configuration

The application stores configuration in `config.yml` in the application directory. You can configure:

- **Sendspin Server**: The address and port of your sendspin server
- **Client Name**: Identifier for this client

## Usage

1. Access the web interface at `http://<device-ip>:8000`
2. Configure the sendspin server address and port
3. Start the sendspin player to begin streaming

## Project Structure

```
client/
├── main.py              # FastAPI application
├── sendspin_player/     # Sendspin player package
│   ├── __init__.py
│   ├── config.py        # Configuration management
│   ├── sendspin_client.py  # Sendspin client wrapper
│   ├── audio_device.py  # Audio device management
│   └── audio_player.py   # Audio playback
├── requirements.txt     # Python dependencies
├── package.json         # Frontend dependencies
├── templates/           # Jinja2 templates
│   ├── base.html
│   └── index.html
└── static/             # Static files (CSS/JS)
```

## Development

The application uses:
- **FastAPI**: Web framework
- **Jinja2**: Template engine
- **Pico.css**: Minimal CSS framework
- **HTMX**: For dynamic interactions without JavaScript

## Notes

- The sendspin player will automatically restart when server configuration changes
- Configuration is persisted in `config.yml`
- Audio device selection is based on device names for stability across reboots

