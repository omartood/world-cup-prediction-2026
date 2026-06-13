"""Flask REST API exposing the WC-2026 prediction engine.

Run:
    python -m wc2026.api                 # dev server on http://127.0.0.1:5000
    flask --app wc2026.api run           # alternative
    gunicorn "wc2026.api:create_app()"   # production (Linux)

Endpoints (all JSON, prefixed with /api):
    GET  /api/health
    GET  /api/teams
    GET  /api/groups
    GET  /api/group-predictions
    GET  /api/knockout
    GET  /api/predict?home=Spain&away=Brazil[&neutral=true][&top_n=5]
    POST /api/predict                    {"home": "...", "away": "...", "neutral": true}
    GET  /api/champion-odds[?iterations=20000][&seed=42][&limit=N]
"""

from __future__ import annotations

from flask import Flask, jsonify, request
from flask_cors import CORS

from .. import config
from .service import UnknownTeamError, get_service


def _as_bool(value, default=True) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def create_app(*, eager: bool = False) -> Flask:
    app = Flask(__name__)
    CORS(app)

    if eager:  # fit models at startup instead of on first request
        get_service()

    # ----------------------------------------------------------------------------------
    # Routes
    # ----------------------------------------------------------------------------------
    @app.get("/api/health")
    def health():
        svc = get_service()
        return jsonify({"status": "ok", **svc.info()})

    @app.get("/api/teams")
    def teams():
        return jsonify(get_service().teams())

    @app.get("/api/groups")
    def groups():
        return jsonify(get_service().groups())

    @app.get("/api/group-predictions")
    def group_predictions():
        return jsonify(get_service().group_predictions())

    @app.route("/api/predict", methods=["GET", "POST"])
    def predict():
        if request.method == "POST":
            payload = request.get_json(silent=True) or {}
            home, away = payload.get("home"), payload.get("away")
            neutral = _as_bool(payload.get("neutral"), default=True)
            top_n = int(payload.get("top_n", 5))
        else:
            home, away = request.args.get("home"), request.args.get("away")
            neutral = _as_bool(request.args.get("neutral"), default=True)
            top_n = int(request.args.get("top_n", 5))

        if not home or not away:
            return jsonify({"error": "both 'home' and 'away' are required"}), 400
        return jsonify(get_service().predict_match(home, away, neutral=neutral, top_n=top_n))

    @app.get("/api/knockout")
    def knockout():
        return jsonify(get_service().knockout())

    @app.get("/api/champion-odds")
    def champion_odds():
        iterations = int(request.args.get("iterations", config.DEFAULT_ITERATIONS))
        seed = int(request.args.get("seed", config.RANDOM_SEED))
        iterations = max(100, min(iterations, 100_000))  # guard against abuse
        results = get_service().simulate(iterations=iterations, seed=seed)
        limit = request.args.get("limit")
        if limit is not None:
            results = results[: int(limit)]
        return jsonify({"iterations": iterations, "seed": seed, "results": results})

    # ----------------------------------------------------------------------------------
    # Error handlers (always JSON)
    # ----------------------------------------------------------------------------------
    @app.errorhandler(UnknownTeamError)
    def _unknown_team(err: UnknownTeamError):
        return jsonify({"error": str(err), "team": err.team}), 400

    @app.errorhandler(404)
    def _not_found(_err):
        return jsonify({"error": "not found"}), 404

    @app.errorhandler(Exception)
    def _server_error(err):
        app.logger.exception("unhandled error")
        return jsonify({"error": "internal server error", "detail": str(err)}), 500

    return app


def main() -> None:
    app = create_app(eager=True)
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
