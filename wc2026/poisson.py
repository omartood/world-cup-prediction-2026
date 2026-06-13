"""Dixon-Coles bivariate-Poisson goal model with an Elo strength covariate.

Model (log-link):
    log(lambda_home) = c + atk[home] - def[away] + h + w * elo_diff
    log(lambda_away) = c + atk[away] - def[home]     - w * elo_diff
where elo_diff = (R_home - R_away) / 100 from the pre-match Elo ratings, `h` is the
home-advantage term (zeroed on neutral ground), and `c` is a global intercept.

Fitting is done in two fast stages:
  1. attack/defense/home/elo-coef via weighted Poisson MLE with **analytic gradients**
     (vectorized; L-BFGS-B). A small L2 penalty on attack/defense pins the otherwise
     location-confounded team effects and pushes the strength signal onto Elo.
  2. the single Dixon-Coles low-score correlation rho via a bounded 1-D search, holding
     stage-1 lambdas fixed.

Matches are exponentially time-decayed so recent form dominates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

from . import config
from .elo import EloModel

_L2 = 1.0  # ridge penalty on attack/defense effects


def _time_weights(dates: pd.Series, ref_date: pd.Timestamp) -> np.ndarray:
    age_days = (ref_date - dates).dt.days.to_numpy(dtype=float)
    age_days = np.clip(age_days, 0.0, None)
    decay = np.log(2.0) / config.DC_TIME_DECAY_HALFLIFE_DAYS
    return np.exp(-decay * age_days)


def _tau(x, y, lh, la, rho):
    """Dixon-Coles low-score adjustment (vectorized over arrays of x, y)."""
    t = np.ones_like(lh, dtype=float)
    t = np.where((x == 0) & (y == 0), 1.0 - lh * la * rho, t)
    t = np.where((x == 0) & (y == 1), 1.0 + lh * rho, t)
    t = np.where((x == 1) & (y == 0), 1.0 + la * rho, t)
    t = np.where((x == 1) & (y == 1), 1.0 - rho, t)
    return t


class PoissonModel:
    def __init__(self, elo: EloModel) -> None:
        self.elo = elo
        self.teams: list[str] = []
        self.idx: dict[str, int] = {}
        self.atk: np.ndarray | None = None
        self.dfn: np.ndarray | None = None
        self.c = 0.0
        self.h = 0.0
        self.w = 0.0
        self.rho = 0.0

    # ----------------------------------------------------------------------------------
    # Fitting
    # ----------------------------------------------------------------------------------
    def fit(self, played: pd.DataFrame) -> "PoissonModel":
        train = played[played["date"] >= pd.Timestamp(config.DC_TRAIN_FROM)].copy()
        ref_date = train["date"].max()

        self.teams = sorted(set(train["home_team"]) | set(train["away_team"]))
        self.idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        hi = train["home_team"].map(self.idx).to_numpy()
        ai = train["away_team"].map(self.idx).to_numpy()
        hs = train["home_score"].to_numpy(dtype=float)
        as_ = train["away_score"].to_numpy(dtype=float)
        neutral = train["neutral"].to_numpy(dtype=bool)
        w = _time_weights(train["date"], ref_date)

        # Pre-match Elo difference for each training match (no result leakage).
        elo_diff = np.empty(len(train))
        for k, (idx_row, _) in enumerate(train.iterrows()):
            ra, rb = self.elo.match_ratings.get(
                idx_row, (config.ELO_BASE_RATING, config.ELO_BASE_RATING)
            )
            elo_diff[k] = (ra - rb) / 100.0

        home_on = (~neutral).astype(float)

        # Parameter packing: [atk(n), dfn(n), c, h, w]
        def unpack(theta):
            atk = theta[:n]
            dfn = theta[n : 2 * n]
            c, h, wco = theta[2 * n], theta[2 * n + 1], theta[2 * n + 2]
            return atk, dfn, c, h, wco

        def lambdas(theta):
            atk, dfn, c, h, wco = unpack(theta)
            eta_h = c + atk[hi] - dfn[ai] + h * home_on + wco * elo_diff
            eta_a = c + atk[ai] - dfn[hi] - wco * elo_diff
            return np.exp(np.clip(eta_h, -4, 4)), np.exp(np.clip(eta_a, -4, 4))

        def nll_and_grad(theta):
            atk, dfn, c, h, wco = unpack(theta)
            lh, la = lambdas(theta)
            # Negative weighted Poisson log-likelihood (dropping constant log factorials).
            ll = w * (hs * np.log(lh) - lh + as_ * np.log(la) - la)
            nll = -ll.sum() + _L2 * (atk @ atk + dfn @ dfn)

            rh = w * (hs - lh)  # weighted residuals
            ra = w * (as_ - la)

            g = np.zeros_like(theta)
            # attack: +rh where team is home, +ra where team is away
            np.add.at(g[:n], hi, -rh)
            np.add.at(g[:n], ai, -ra)
            # defense: enters with -1 in opponent's eta -> +rh(away side), +ra(home side)
            np.add.at(g[n : 2 * n], ai, rh)
            np.add.at(g[n : 2 * n], hi, ra)
            g[:n] += 2 * _L2 * atk
            g[n : 2 * n] += 2 * _L2 * dfn
            g[2 * n] = -(rh + ra).sum()                 # intercept c
            g[2 * n + 1] = -(rh * home_on).sum()        # home adv h
            g[2 * n + 2] = -(elo_diff * (rh - ra)).sum()  # elo coef w
            return nll, g

        theta0 = np.zeros(2 * n + 3)
        theta0[2 * n] = np.log(max(hs.mean(), 0.3))  # intercept ~ log(mean goals)
        res = minimize(
            nll_and_grad, theta0, jac=True, method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-9},
        )
        atk, dfn, c, h, wco = unpack(res.x)
        # Center team effects for interpretability (absorb into intercept/home is unneeded
        # since we re-add means symmetrically — predictions are invariant).
        self.atk = atk - atk.mean()
        self.dfn = dfn - dfn.mean()
        self.c = c + atk.mean() - dfn.mean()  # keep lambda_home unchanged
        self.h = h
        self.w = wco

        self._fit_rho(hi, ai, hs, as_, w, lambdas(res.x))
        return self

    def _fit_rho(self, hi, ai, hs, as_, w, lam):
        lh, la = lam

        def neg_ll(rho):
            t = _tau(hs, as_, lh, la, rho)
            t = np.clip(t, 1e-9, None)
            return -(w * np.log(t)).sum()

        r = minimize_scalar(neg_ll, bounds=(-0.2, 0.2), method="bounded")
        self.rho = float(r.x)

    # ----------------------------------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------------------------------
    def _team_effects(self, team: str) -> tuple[float, float]:
        i = self.idx.get(team)
        if i is None:
            return 0.0, 0.0  # unseen team -> league-average attack/defense
        return float(self.atk[i]), float(self.dfn[i])

    def expected_goals(self, home: str, away: str, neutral: bool) -> tuple[float, float]:
        atk_h, dfn_h = self._team_effects(home)
        atk_a, dfn_a = self._team_effects(away)
        elo_diff = (self.elo.rating(home) - self.elo.rating(away)) / 100.0
        h = 0.0 if neutral else self.h
        eta_h = self.c + atk_h - dfn_a + h + self.w * elo_diff
        eta_a = self.c + atk_a - dfn_h - self.w * elo_diff
        return float(np.exp(np.clip(eta_h, -4, 4))), float(np.exp(np.clip(eta_a, -4, 4)))

    def score_matrix(self, home: str, away: str, neutral: bool) -> np.ndarray:
        """(max_goals+1) x (max_goals+1) matrix of P(home=i, away=j)."""
        lh, la = self.expected_goals(home, away, neutral)
        m = config.DC_MAX_GOALS + 1
        gh = poisson.pmf(np.arange(m), lh)
        ga = poisson.pmf(np.arange(m), la)
        mat = np.outer(gh, ga)
        # Apply Dixon-Coles correction to the four low-score cells.
        mat[0, 0] *= 1.0 - lh * la * self.rho
        mat[0, 1] *= 1.0 + lh * self.rho
        mat[1, 0] *= 1.0 + la * self.rho
        mat[1, 1] *= 1.0 - self.rho
        mat = np.clip(mat, 0.0, None)
        return mat / mat.sum()

    def outcome_probabilities(self, home: str, away: str, neutral: bool) -> tuple[float, float, float]:
        """Return (P home win, P draw, P away win)."""
        mat = self.score_matrix(home, away, neutral)
        p_home = float(np.tril(mat, -1).sum())
        p_draw = float(np.trace(mat))
        p_away = float(np.triu(mat, 1).sum())
        return p_home, p_draw, p_away

    def most_likely_score(self, home: str, away: str, neutral: bool) -> tuple[int, int]:
        mat = self.score_matrix(home, away, neutral)
        i, j = np.unravel_index(int(np.argmax(mat)), mat.shape)
        return int(i), int(j)
