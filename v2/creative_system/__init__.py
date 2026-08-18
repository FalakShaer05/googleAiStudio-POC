"""Isolated Creative System Flask module."""


def create_blueprint():
    """Return (page blueprint, API blueprint). Importing routes registers handlers."""
    from . import routes  # noqa: F401
    from .blueprint import api_bp, bp

    return bp, api_bp


def register_creative_system(app):
    """Single-call registration used by app.py."""
    for blueprint in create_blueprint():
        app.register_blueprint(blueprint)
