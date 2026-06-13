"""Backtest the Dixon-Coles+Elo model against past World Cups.

For each target tournament we train only on matches played *before* it started, then
score every match of that World Cup with three-way (W/D/L) metrics: accuracy, multiclass
log-loss, and the ranked-probability/Brier score. A calibrated Elo-only model is included
as a baseline so the goal model's added value is visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .. import config
from ..data import load_results, played_matches
from ..models import EloModel, expected_score, PoissonModel

# Tournament -> first match date (train strictly before this).
WC_BACKTESTS = {
    2014: "2014-06-12",
    2018: "2018-06-14",
    2022: "2022-11-20",
}


class EloBaseline:
    """Calibrated Elo W/D/L model: p_draw = sigmoid(a + b*|Δrating|); the remaining mass
    is split by the Elo expected score. Parameters a, b are fit on training data."""

    def __init__(self, elo: EloModel) -> None:
        self.elo = elo
        self.a = 0.0
        self.b = 0.0

    def fit(self, train: pd.DataFrame) -> "EloBaseline":
        diffs, is_draw = [], []
        for idx, r in train.iterrows():
            ra, rb = self.elo.match_ratings.get(
                idx, (config.ELO_BASE_RATING, config.ELO_BASE_RATING)
            )
            adv = 0.0 if r["neutral"] else config.ELO_HOME_ADVANTAGE
            diffs.append(abs((ra + adv) - rb))
            is_draw.append(1.0 if r["home_score"] == r["away_score"] else 0.0)
        d = np.array(diffs) / 100.0
        y = np.array(is_draw)

        def nll(p):
            z = p[0] + p[1] * d
            pr = 1.0 / (1.0 + np.exp(-z))
            pr = np.clip(pr, 1e-9, 1 - 1e-9)
            return -(y * np.log(pr) + (1 - y) * np.log(1 - pr)).sum()

        res = minimize(nll, np.array([-1.0, -0.1]), method="Nelder-Mead")
        self.a, self.b = res.x
        return self

    def probabilities(self, home, away, neutral) -> tuple[float, float, float]:
        adv = 0.0 if neutral else config.ELO_HOME_ADVANTAGE
        x = expected_score(self.elo.rating(home) + adv, self.elo.rating(away))
        diff = abs((self.elo.rating(home) + adv) - self.elo.rating(away)) / 100.0
        p_draw = 1.0 / (1.0 + np.exp(-(self.a + self.b * diff)))
        return (1 - p_draw) * x, p_draw, (1 - p_draw) * (1 - x)


def _outcome(hs, as_) -> int:
    return 0 if hs > as_ else (1 if hs == as_ else 2)  # 0 home, 1 draw, 2 away


def _metrics(probs: list[tuple], actuals: list[int]) -> dict:
    P = np.clip(np.array(probs), 1e-12, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    y = np.array(actuals)
    acc = float((P.argmax(axis=1) == y).mean())
    logloss = float(-np.log(P[np.arange(len(y)), y]).mean())
    onehot = np.eye(3)[y]
    brier = float(((P - onehot) ** 2).sum(axis=1).mean())
    return {"n": len(y), "accuracy": acc, "log_loss": logloss, "brier": brier}


def backtest() -> pd.DataFrame:
    df = load_results()
    played = played_matches(df)
    rows = []
    for year, start in WC_BACKTESTS.items():
        cutoff = pd.Timestamp(start)
        train = played[played["date"] < cutoff]
        test = played[
            (played["tournament"] == config.WC_TOURNAMENT)
            & (played["date"].dt.year == year)
            & (played["date"] >= cutoff)
        ]
        if test.empty:
            continue

        elo = EloModel().fit(train)
        model = PoissonModel(elo).fit(train)
        baseline = EloBaseline(elo).fit(train)

        m_probs, b_probs, actuals = [], [], []
        for _, r in test.iterrows():
            neutral = bool(r["neutral"])
            m_probs.append(model.outcome_probabilities(r["home_team"], r["away_team"], neutral))
            b_probs.append(baseline.probabilities(r["home_team"], r["away_team"], neutral))
            actuals.append(_outcome(r["home_score"], r["away_score"]))

        dc = _metrics(m_probs, actuals)
        bl = _metrics(b_probs, actuals)
        rows.append({
            "world_cup": year, "matches": dc["n"],
            "DC_acc": dc["accuracy"], "Elo_acc": bl["accuracy"],
            "DC_logloss": dc["log_loss"], "Elo_logloss": bl["log_loss"],
            "DC_brier": dc["brier"], "Elo_brier": bl["brier"],
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        avg = out.drop(columns=["world_cup", "matches"]).mean()
        avg_row = {"world_cup": "AVG", "matches": int(out["matches"].sum()), **avg.to_dict()}
        out = pd.concat([out, pd.DataFrame([avg_row])], ignore_index=True)
    return out


def _main() -> None:
    out = backtest()
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print("\n=== World Cup backtest: Dixon-Coles+Elo vs calibrated Elo baseline ===")
    print("(lower log-loss / Brier is better; higher accuracy is better)\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    _main()
