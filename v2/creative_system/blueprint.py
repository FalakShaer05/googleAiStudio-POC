import os

from flask import Blueprint

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))

bp = Blueprint(
    "creative_system",
    __name__,
    url_prefix="/creative-system",
    template_folder="templates",
    static_folder="static",
    static_url_path="/static",
)

api_bp = Blueprint(
    "creative_system_api",
    __name__,
    url_prefix="/api",
)
