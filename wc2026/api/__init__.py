"""Flask REST API and the prediction service backing it.

Re-exports keep the documented entry points stable:
    gunicorn "wc2026.api:create_app()"
    flask --app wc2026.api run
"""

from .service import (
    PredictionService,
    UnknownTeamError,
    get_service,
    predict_group_fixtures,
)
from .app import create_app, main

__all__ = [
    "create_app",
    "main",
    "PredictionService",
    "UnknownTeamError",
    "get_service",
    "predict_group_fixtures",
]
