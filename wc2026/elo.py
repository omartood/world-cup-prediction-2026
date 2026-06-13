"""Football Elo rating engine.

Standard "World Football Elo" variant: expected score from a logistic of the rating
difference (plus home advantage), updated by a K-factor that scales with tournament
importance and goal margin. We replay the entire match history once and record each
team's rating both at the end (current strength) and as-of each date (for backtesting).
"""

from __future__ import annotations

import pandas as pd

from . import config


def _k_factor(tournament: str) -> float:
    if tournament in config.ELO_K_BY_TOURNAMENT:
        return config.ELO_K_BY_TOURNAMENT[tournament]
    if isinstance(tournament, str) and tournament.endswith("qualification"):
        return config.ELO_K_QUALIFIER_DEFAULT
    if tournament == "Friendly":
        return config.ELO_K_FRIENDLY
    return config.ELO_K_DEFAULT


def _mov_multiplier(goal_diff: int) -> float:
    """Margin-of-victory multiplier (FiveThirtyEight-style, dampened for blowouts)."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that A beats B (draw counts as half), given ratings already adjusted
    for home advantage."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


class EloModel:
    """Replays history to produce current ratings and a per-match snapshot of ratings.

    `match_ratings` holds, for every played match (indexed like the input frame), the
    pre-match home/away ratings — i.e. ratings that did NOT peek at that match's result.
    This is what backtests and the Poisson covariate consume.
    """

    def __init__(self) -> None:
        self.ratings: dict[str, float] = {}
        self.match_ratings: dict[int, tuple[float, float]] = {}

    def _rating(self, team: str) -> float:
        return self.ratings.get(team, config.ELO_BASE_RATING)

    def fit(self, played: pd.DataFrame) -> "EloModel":
        played = played.sort_values("date")
        for idx, r in played.iterrows():
            home, away = r["home_team"], r["away_team"]
            ra, rb = self._rating(home), self._rating(away)

            home_adv = 0.0 if r["neutral"] else config.ELO_HOME_ADVANTAGE
            exp_home = expected_score(ra + home_adv, rb)

            hs, as_ = int(r["home_score"]), int(r["away_score"])
            if hs > as_:
                actual_home = 1.0
            elif hs < as_:
                actual_home = 0.0
            else:
                actual_home = 0.5

            k = _k_factor(r["tournament"]) * _mov_multiplier(hs - as_)
            delta = k * (actual_home - exp_home)

            self.match_ratings[idx] = (ra, rb)
            self.ratings[home] = ra + delta
            self.ratings[away] = rb - delta
        return self

    def rating(self, team: str) -> float:
        return self._rating(team)

    def win_probabilities(self, home: str, away: str, neutral: bool) -> tuple[float, float]:
        """Return (P home win incl. half-draw, P away win incl. half-draw) — a baseline
        used by the Elo-only comparison model. Draws are split implicitly."""
        home_adv = 0.0 if neutral else config.ELO_HOME_ADVANTAGE
        p_home = expected_score(self._rating(home) + home_adv, self._rating(away))
        return p_home, 1.0 - p_home
