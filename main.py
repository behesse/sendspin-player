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
from aiosendspin_sounddevice.audio_device import AudioDeviceManager
import os

# Get log level from environment variable, default to INFO
LOG_LEVEL_ENV = os.getenv("SSP_LOG_LEVEL", "INFO").upper()
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}
# Default to INFO if invalid level is provided
LOG_LEVEL = LOG_LEVEL_MAP.get(LOG_LEVEL_ENV, logging.INFO)

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

# Configure logging with custom formatter
formatter = LoggerNameFormatter(LOG_FORMAT, LOG_DATE_FORMAT)

def configure_logger(logger_name: str) -> logging.Logger:
    """Configure a logger with the custom formatter."""
    logger_obj = logging.getLogger(logger_name)
    logger_obj.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger_obj.addHandler(handler)
    logger_obj.setLevel(LOG_LEVEL)
    logger_obj.propagate = False
    return logger_obj

# Configure all loggers
configure_logger("")  # Root logger
for logger_name in ["uvicorn.access", "uvicorn.error", "uvicorn", __name__]:
    configure_logger(logger_name)

logger = logging.getLogger(__name__)

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
sendspin_client = SendspinClient(config)
_device_manager = AudioDeviceManager()  # Shared device manager instance


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


def _render_template(template_name: str, request: Request, **context) -> HTMLResponse:
    """Helper to render templates with common context."""
    return templates.TemplateResponse(template_name, {"request": request, **context})


def _render_template_to_string(template_name: str, request: Request, **context) -> str:
    """Helper to render templates to string (for HTMX partials)."""
    response = templates.TemplateResponse(template_name, {"request": request, **context})
    # Render the template to get the HTML string
    from starlette.responses import Response
    import asyncio
    # This is a simplified approach - in practice we'd need to await the response body
    # For now, we'll use a different approach in the routes
    return ""


def _get_common_context() -> dict:
    """Get common context data used across multiple endpoints."""
    return {
        "config": config_manager.get_config(),
        "discovered_servers": _discovered_servers_cache["servers"],
        "audio_devices": SendspinClient.list_audio_devices()
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main configuration page - redirects to status by default."""
    return RedirectResponse(url="/status", status_code=302)


@app.get("/status", response_class=HTMLResponse)
async def status_page(request: Request):
    """Status page."""
    context = _get_common_context()
    context["status"] = sendspin_client.get_status()
    context["active_page"] = "status"
    return _render_template("pages/status.html", request, **context)


@app.get("/sendspin", response_class=HTMLResponse)
async def sendspin_page(request: Request):
    """Sendspin config page."""
    context = _get_common_context()
    context["active_page"] = "sendspin"
    return _render_template("pages/sendspin.html", request, **context)


def _get_device_defaults(device_name: str = "") -> tuple[int, int]:
    """Get default sample rate and channels for a device."""
    if device_name and device_name.strip():
        device = _device_manager.find_by_name(device_name, exact=False)
    else:
        device = _device_manager.get_default_device()
    
    if device:
        return int(device.default_samplerate), device.max_output_channels
    return 44100, 2  # Defaults


@app.get("/api/audio-device", response_class=HTMLResponse)
async def get_audio_device_info(request: Request):
    """Get audio device information for populating format defaults via HTMX."""
    device_name = request.query_params.get("audio_device", "")
    sample_rate, channels = _get_device_defaults(device_name)
    return _render_template("components/audio_device_inputs.html", request, sample_rate=sample_rate, channels=channels)


@app.get("/api/set-server-url", response_class=HTMLResponse)
async def set_server_url(request: Request, url: str = ""):
    """Set server URL via HTMX."""
    # Get URL from query parameter
    if not url:
        url = request.query_params.get("url", "")
    return _render_template("components/server_url_input.html", request, url=url)


@app.get("/api/status/partial", response_class=HTMLResponse)
async def status_partial(request: Request):
    """Partial status update for auto-refresh."""
    return _render_template("components/status_partial.html", request, status=sendspin_client.get_status())


@app.post("/api/config/sendspin", response_class=HTMLResponse)
async def update_sendspin_config(
    request: Request,
    server_url: str = Form(...),
    client_name: str = Form("sendspin-player"),
    audio_device: str = Form(""),
    audio_codec: str = Form("PCM"),
    audio_channels: int = Form(2),
    audio_sample_rate: int = Form(44100),
    audio_bit_depth: int = Form(16)
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
            audio_device=audio_device_name,
            audio_codec=audio_codec,
            audio_channels=audio_channels,
            audio_sample_rate=audio_sample_rate,
            audio_bit_depth=audio_bit_depth
        )
        
        # Resolve device name to AudioDevice and update client
        resolved_device = None
        if audio_device_name:
            resolved_device = _device_manager.find_by_name(audio_device_name, exact=False)
        sendspin_client._audio_device = resolved_device
        
        # Restart client if running
        if sendspin_client.is_running:
            await sendspin_client.restart()
        
        # Reload config in client
        sendspin_client.config = config_manager.get_config()
        
        # Return updated sendspin page with success message
        context = _get_common_context()
        context["active_page"] = "sendspin"
        context["message"] = "Configuration saved successfully!"
        return _render_template("pages/sendspin.html", request, **context)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def _handle_client_action(action_name: str, action_func, success_msg: str, failure_msg: str) -> tuple[str, dict]:
    """Handle client action (start/stop/restart) and return message and status."""
    try:
        result = await action_func()
        message = success_msg if result else failure_msg
    except Exception as e:
        message = f"Error: {str(e)}"
    return message, sendspin_client.get_status()


@app.post("/api/client/start", response_class=HTMLResponse)
async def start_client(request: Request):
    """Start the sendspin player."""
    message, status = await _handle_client_action(
        "start",
        sendspin_client.start,
        "Client started successfully",
        "Client is already running"
    )
    # Return updated status page content
    context = _get_common_context()
    context["status"] = status
    context["active_page"] = "status"
    context["message"] = message
    return _render_template("pages/status.html", request, **context)


@app.post("/api/client/stop", response_class=HTMLResponse)
async def stop_client(request: Request):
    """Stop the sendspin player."""
    message, status = await _handle_client_action(
        "stop",
        sendspin_client.stop,
        "Client stopped successfully",
        "Client is not running"
    )
    # Return updated status page content
    context = _get_common_context()
    context["status"] = status
    context["active_page"] = "status"
    context["message"] = message
    return _render_template("pages/status.html", request, **context)


@app.post("/api/client/restart", response_class=HTMLResponse)
async def restart_client(request: Request):
    """Restart the sendspin player."""
    message, status = await _handle_client_action(
        "restart",
        sendspin_client.restart,
        "Client restarted successfully",
        "Failed to restart client"
    )
    # Return updated status page content
    context = _get_common_context()
    context["status"] = status
    context["active_page"] = "status"
    context["message"] = message
    return _render_template("pages/status.html", request, **context)


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
        "components/server_list.html",
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

