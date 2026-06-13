"""Prediction service: fits the models once and serves cached, JSON-ready results.

The Flask layer (`api.py`) stays thin and delegates everything here. Fitting Elo + the
Dixon-Coles model takes a few seconds, so a process holds a single `PredictionService`
instance; the expensive tournament simulation is memoized per (iterations, seed).
"""

from __future__ import annotations

import threading

import numpy as np

from .. import config
from ..data import derive_groups, load_results, played_matches
from ..models import EloModel, PoissonModel, Simulator


def predict_group_fixtures(struct, model: PoissonModel) -> list[dict]:
    """W/D/L probabilities, expected goals, and a likely scoreline for every group fixture
    not yet played. (Shared by the CLI and the API.)"""
    rows = []
    for letter, g in struct.groups.items():
        for (home, away, neutral, played, hs, as_) in g.fixtures:
            if played:
                continue
            p_h, p_d, p_a = model.outcome_probabilities(home, away, neutral)
            sh, sa = model.most_likely_score(home, away, neutral)
            lh, la = model.expected_goals(home, away, neutral)
            rows.append({
                "group": letter, "home": home, "away": away,
                "p_home": round(p_h, 4), "p_draw": round(p_d, 4), "p_away": round(p_a, 4),
                "xg_home": round(lh, 2), "xg_away": round(la, 2),
                "likely_score": f"{sh}-{sa}",
                "prediction": home if p_h >= max(p_d, p_a) else (away if p_a >= p_d else "Draw"),
            })
    return rows


class PredictionService:
    def __init__(self) -> None:
        self.df = load_results()
        self.struct = derive_groups(self.df)
        played = played_matches(self.df)
        self.elo = EloModel().fit(played)
        self.model = PoissonModel(self.elo).fit(played)
        self._sim_cache: dict[tuple[int, int], list[dict]] = {}
        self._knockout_cache: dict | None = None
        self._sim_lock = threading.Lock()
        self.known_teams = set(self.elo.ratings) | set(self.model.idx)

    # ----------------------------------------------------------------------------------
    # Metadata
    # ----------------------------------------------------------------------------------
    def info(self) -> dict:
        wc = self.struct.all_fixtures
        n_played = int(wc["played"].sum())
        return {
            "teams": len(self.struct.teams),
            "groups": len(self.struct.groups),
            "fixtures_played": n_played,
            "fixtures_remaining": int(len(wc) - n_played),
            "model": {
                "home_advantage": round(self.model.h, 4),
                "elo_coefficient": round(self.model.w, 4),
                "dixon_coles_rho": round(self.model.rho, 4),
                "teams_fitted": len(self.model.teams),
            },
        }

    def teams(self) -> list[dict]:
        rows = [
            {"team": t, "group": self.struct.team_group[t], "elo": round(self.elo.rating(t), 1)}
            for t in self.struct.teams
        ]
        return sorted(rows, key=lambda r: r["elo"], reverse=True)

    def groups(self) -> dict:
        out = {}
        for letter, g in self.struct.groups.items():
            fixtures = [
                {
                    "home": home, "away": away, "neutral": neutral, "played": played,
                    "score": (f"{hs}-{as_}" if played else None),
                }
                for (home, away, neutral, played, hs, as_) in g.fixtures
            ]
            out[letter] = {
                "teams": [
                    {"team": t, "elo": round(self.elo.rating(t), 1)} for t in g.teams
                ],
                "fixtures": fixtures,
            }
        return out

    # ----------------------------------------------------------------------------------
    # Single-match prediction
    # ----------------------------------------------------------------------------------
    def predict_match(self, home: str, away: str, neutral: bool = True, top_n: int = 5) -> dict:
        home = config.normalize_team(home)
        away = config.normalize_team(away)
        for t in (home, away):
            if t not in self.known_teams:
                raise UnknownTeamError(t)

        p_h, p_d, p_a = self.model.outcome_probabilities(home, away, neutral)
        lh, la = self.model.expected_goals(home, away, neutral)
        mat = self.model.score_matrix(home, away, neutral)

        flat = [
            {"score": f"{i}-{j}", "prob": round(float(mat[i, j]), 4)}
            for i in range(mat.shape[0]) for j in range(mat.shape[1])
        ]
        flat.sort(key=lambda r: r["prob"], reverse=True)
        sh, sa = (int(x) for x in np.unravel_index(int(np.argmax(mat)), mat.shape))

        return {
            "home": home, "away": away, "neutral": neutral,
            "probabilities": {"home_win": round(p_h, 4), "draw": round(p_d, 4),
                              "away_win": round(p_a, 4)},
            "expected_goals": {"home": round(lh, 2), "away": round(la, 2)},
            "elo": {"home": round(self.elo.rating(home), 1),
                    "away": round(self.elo.rating(away), 1)},
            "most_likely_score": f"{sh}-{sa}",
            "top_scorelines": flat[:top_n],
            "prediction": home if p_h >= max(p_d, p_a) else (away if p_a >= p_d else "Draw"),
        }

    def group_predictions(self) -> list[dict]:
        return predict_group_fixtures(self.struct, self.model)

    # ----------------------------------------------------------------------------------
    # Projected knockout bracket (deterministic; cached)
    # ----------------------------------------------------------------------------------
    def knockout(self) -> dict:
        if self._knockout_cache is None:
            with self._sim_lock:
                if self._knockout_cache is None:
                    sim = Simulator(self.struct, self.model, self.elo)
                    self._knockout_cache = sim.project_bracket()
        return self._knockout_cache

    # ----------------------------------------------------------------------------------
    # Tournament simulation (memoized)
    # ----------------------------------------------------------------------------------
    def simulate(self, iterations: int = config.DEFAULT_ITERATIONS,
                 seed: int = config.RANDOM_SEED) -> list[dict]:
        key = (int(iterations), int(seed))
        if key in self._sim_cache:
            return self._sim_cache[key]
        with self._sim_lock:
            if key in self._sim_cache:  # double-checked: another thread may have filled it
                return self._sim_cache[key]
            sim = Simulator(self.struct, self.model, self.elo, seed=seed)
            odds = sim.run(iterations)
            records = odds.round(2).to_dict(orient="records")
            self._sim_cache[key] = records
            return records


class UnknownTeamError(ValueError):
    def __init__(self, team: str) -> None:
        super().__init__(f"Unknown team: {team!r}")
        self.team = team


# Process-wide lazy singleton -------------------------------------------------------------
_service: PredictionService | None = None
_service_lock = threading.Lock()


def get_service() -> PredictionService:
    """Return the shared service, fitting the models on first call."""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = PredictionService()
    return _service
