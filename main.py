"""Main FastAPI application for sendspin player configuration.

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
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from pathlib import Path
import uvicorn
import logging
import sys

from sendspin_player.config import ConfigManager, AppConfig
from sendspin_player.sendspin_client import SendspinClientWrapper as SendspinClient
from sendspin_player.audio_device import AudioDeviceManager

# Configure unified logging format for all components
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    DEBUG = '\033[37m'      # White
    INFO = '\033[96m'       # Light cyan
    WARNING = '\033[93m'    # Yellow
    ERROR = '\033[91m'      # Red
    
    @staticmethod
    def get_color(level):
        """Get color code for log level."""
        if level >= logging.ERROR:
            return Colors.ERROR
        elif level >= logging.WARNING:
            return Colors.WARNING
        elif level >= logging.INFO:
            return Colors.INFO
        else:
            return Colors.DEBUG

# Custom formatter that maps logger names and adds colors
class LoggerNameFormatter(logging.Formatter):
    """Formatter that maps logger names to more intuitive names and adds colors."""
    
    LOGGER_NAME_MAP = {
        'uvicorn.access': 'web',
        'uvicorn.error': 'app',
        'uvicorn': 'app',
        '__main__': 'app',
        'main': 'app',
    }
    
    def __init__(self, fmt=None, datefmt=None, use_colors=None):
        super().__init__(fmt, datefmt)
        # Auto-detect if colors should be used (if output is a TTY)
        if use_colors is None:
            self.use_colors = sys.stderr.isatty() if hasattr(sys.stderr, 'isatty') else False
        else:
            self.use_colors = use_colors
    
    def format(self, record):
        # Map logger name to more intuitive name
        original_name = record.name
        mapped_name = self.LOGGER_NAME_MAP.get(original_name, original_name)
        record.name = mapped_name
        
        # Format the message
        formatted = super().format(record)
        
        # Add color if enabled
        if self.use_colors:
            color = Colors.get_color(record.levelno)
            return f"{color}{formatted}{Colors.RESET}"
        
        return formatted

# Configure root logger with custom formatter
root_logger = logging.getLogger()
root_logger.handlers.clear()
root_handler = logging.StreamHandler()
root_handler.setFormatter(LoggerNameFormatter(LOG_FORMAT, LOG_DATE_FORMAT))
root_logger.addHandler(root_handler)
root_logger.setLevel(logging.DEBUG)

# Configure uvicorn loggers to use the same format with intuitive names
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_error_logger = logging.getLogger("uvicorn.error")
uvicorn_logger = logging.getLogger("uvicorn")

# Set all uvicorn loggers to use the same handler format
formatter = LoggerNameFormatter(LOG_FORMAT, LOG_DATE_FORMAT)
for logger_obj in [uvicorn_access_logger, uvicorn_error_logger, uvicorn_logger]:
    logger_obj.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger_obj.addHandler(handler)
    logger_obj.setLevel(logging.DEBUG)
    logger_obj.propagate = False

# Configure application logger
logger = logging.getLogger(__name__)
logger.handlers.clear()
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logger.propagate = False

app = FastAPI(title="Sendspin Player Configuration")

# Setup templates and static files
templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(templates_dir))
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Initialize configuration and client
config_manager = ConfigManager()
config = config_manager.get_config()
# Initialize client - device name will be resolved to index internally
sendspin_client = SendspinClient(config)


@app.on_event("startup")
async def startup_event():
    """Auto-start sendspin player on application startup if server is configured."""
    if config.sendspin_server_url and config.sendspin_server_url.strip():
        logger.info(f"Server configured: {config.sendspin_server_url}. Auto-starting client...")
        try:
            await sendspin_client.start()
            logger.info("Client started successfully on startup")
        except Exception as e:
            logger.warning(f"Failed to auto-start client on startup: {e}")
    else:
        logger.info("No server configured. Client will not auto-start.")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up sendspin player on application shutdown."""
    if sendspin_client.is_running:
        logger.info("Shutting down sendspin player...")
        await sendspin_client.stop()

# Cache for discovered servers
_discovered_servers_cache = {
    "servers": [],
    "timestamp": None
}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main configuration page."""
    config = config_manager.get_config()
    status = sendspin_client.get_status()
    # Use cached servers if available, otherwise empty list
    discovered_servers = _discovered_servers_cache["servers"]
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "config": config,
            "status": status,
            "discovered_servers": discovered_servers
        }
    )


@app.get("/tab/status", response_class=HTMLResponse)
async def tab_status(request: Request):
    """Status tab content."""
    status = sendspin_client.get_status()
    return templates.TemplateResponse(
        "tab_status.html",
        {
            "request": request,
            "status": status
        }
    )


@app.get("/tab/sendspin", response_class=HTMLResponse)
async def tab_sendspin(request: Request):
    """Sendspin config tab content."""
    config = config_manager.get_config()
    discovered_servers = _discovered_servers_cache["servers"]
    audio_devices = SendspinClient.list_audio_devices()
    return templates.TemplateResponse(
        "tab_sendspin.html",
        {
            "request": request,
            "config": config,
            "discovered_servers": discovered_servers,
            "audio_devices": audio_devices
        }
    )


@app.get("/tab/wifi", response_class=HTMLResponse)
async def tab_wifi(request: Request):
    """WiFi config tab content."""
    config = config_manager.get_config()
    return templates.TemplateResponse(
        "tab_wifi.html",
        {
            "request": request,
            "config": config
        }
    )


@app.get("/api/status/partial", response_class=HTMLResponse)
async def status_partial(request: Request):
    """Partial status update for auto-refresh."""
    status = sendspin_client.get_status()
    return templates.TemplateResponse(
        "status_partial.html",
        {
            "request": request,
            "status": status
        }
    )


@app.post("/api/config/sendspin", response_class=HTMLResponse)
async def update_sendspin_config(
    request: Request,
    server_url: str = Form(...),
    client_name: str = Form("sendspin-player"),
    audio_device: str = Form("")
):
    """Update sendspin server configuration."""
    try:
        # Parse audio_device (empty string means None/default)
        audio_device_name = None
        if audio_device and audio_device.strip():
            # audio_device is now a device name, not an index
            audio_device_name = audio_device.strip()
        
        config_manager.update_config(
            sendspin_server_url=server_url,
            client_name=client_name,
            audio_device=audio_device_name
        )
        
        # Resolve device name to index and update client
        device_manager = AudioDeviceManager()
        sendspin_client._audio_device = device_manager.resolve_device(audio_device_name)
        
        # Restart client if running
        if sendspin_client.is_running:
            await sendspin_client.restart()
        
        # Reload config in client
        sendspin_client.config = config_manager.get_config()
        
        config = config_manager.get_config()
        discovered_servers = _discovered_servers_cache["servers"]
        audio_devices = SendspinClient.list_audio_devices()
        return templates.TemplateResponse(
            "tab_sendspin.html",
            {
                "request": request,
                "config": config,
                "discovered_servers": discovered_servers,
                "audio_devices": audio_devices,
                "message": "Configuration saved successfully!"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/config/wifi", response_class=HTMLResponse)
async def update_wifi_config(
    request: Request,
    ssid: str = Form(...),
    password: str = Form("")
):
    """Update WiFi configuration."""
    try:
        config_manager.update_config(
            wifi_ssid=ssid,
            wifi_password=password
        )
        
        config = config_manager.get_config()
        return templates.TemplateResponse(
            "tab_wifi.html",
            {
                "request": request,
                "config": config,
                "message": "WiFi configuration saved successfully!"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/client/start", response_class=HTMLResponse)
async def start_client(request: Request):
    """Start the sendspin player."""
    try:
        if await sendspin_client.start():
            message = "Client started successfully"
        else:
            message = "Client is already running"
    except Exception as e:
        message = f"Error: {str(e)}"
    
    status = sendspin_client.get_status()
    return templates.TemplateResponse(
        "tab_status.html",
        {
            "request": request,
            "status": status,
            "message": message
        }
    )


@app.post("/api/client/stop", response_class=HTMLResponse)
async def stop_client(request: Request):
    """Stop the sendspin player."""
    try:
        if await sendspin_client.stop():
            message = "Client stopped successfully"
        else:
            message = "Client is not running"
    except Exception as e:
        message = f"Error: {str(e)}"
    
    status = sendspin_client.get_status()
    return templates.TemplateResponse(
        "tab_status.html",
        {
            "request": request,
            "status": status,
            "message": message
        }
    )


@app.post("/api/client/restart", response_class=HTMLResponse)
async def restart_client(request: Request):
    """Restart the sendspin player."""
    try:
        if await sendspin_client.restart():
            message = "Client restarted successfully"
        else:
            message = "Failed to restart client"
    except Exception as e:
        message = f"Error: {str(e)}"
    
    status = sendspin_client.get_status()
    return templates.TemplateResponse(
        "tab_status.html",
        {
            "request": request,
            "status": status,
            "message": message
        }
    )


@app.get("/api/client/status")
async def get_client_status():
    """Get current client status."""
    return sendspin_client.get_status()


@app.get("/api/audio/devices")
async def list_audio_devices():
    """List available audio output devices."""
    return {"devices": SendspinClient.list_audio_devices()}


@app.post("/api/servers/discover", response_class=HTMLResponse)
async def discover_servers(request: Request):
    """Discover available sendspin servers and cache results."""
    import time
    
    logger.info("Starting server discovery...")
    servers = await SendspinClient.discover_servers()
    
    # Update cache
    _discovered_servers_cache["servers"] = servers
    _discovered_servers_cache["timestamp"] = time.time()
    
    logger.info(f"Server discovery complete. Found {len(servers)} servers.")
    
    # Return just the server list section as HTML
    return templates.TemplateResponse(
        "server_list.html",
        {
            "request": request,
            "discovered_servers": servers
        }
    )


if __name__ == "__main__":
    # Note: We're not using log_config because we've already configured
    # the loggers manually with our custom formatter above.
    # This ensures all logs use intuitive names.
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_config=None  # Use our manual configuration instead
    )

