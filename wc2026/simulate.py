"""Monte-Carlo simulation of the WC-2026 group stage and knockout bracket.

Group fixtures already played are treated as fixed; the remaining group games are
sampled from the Dixon-Coles score matrices. Standings use the FIFA tiebreaker order
(points -> goal difference -> goals for -> head-to-head). The 12 winners, 12 runners-up
and 8 best third-placed teams advance to a 32-team single-elimination bracket.

The bracket is a balanced 2026-format tree built by group letter (group-mates 1X/2X are
placed in opposite halves; group-stage results never produce an immediate rematch). The
exact official third-place lookup table is not in the dataset, so qualifying thirds are
slotted with a same-group-avoidance rule — structurally faithful and rematch-free.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import config
from .data import TournamentStructure
from .elo import EloModel, expected_score
from .poisson import PoissonModel

ROUND_NAMES = ["Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "Final", "Champion"]


class Simulator:
    def __init__(self, struct: TournamentStructure, model: PoissonModel, elo: EloModel,
                 seed: int = config.RANDOM_SEED) -> None:
        self.struct = struct
        self.model = model
        self.elo = elo
        self.rng = np.random.default_rng(seed)
        self._knockout_cache: dict[tuple[str, str], float] = {}
        self._prep_group_fixtures()

    # ----------------------------------------------------------------------------------
    # Group stage
    # ----------------------------------------------------------------------------------
    def _prep_group_fixtures(self) -> None:
        """Cache, per group, the played results and the sampling tables for unplayed games."""
        self.group_played: dict[str, list[tuple]] = defaultdict(list)
        self.group_unplayed: dict[str, list[dict]] = defaultdict(list)
        m = config.DC_MAX_GOALS + 1
        for letter, g in self.struct.groups.items():
            for (home, away, neutral, played, hs, as_) in g.fixtures:
                if played:
                    self.group_played[letter].append((home, away, hs, as_))
                else:
                    mat = self.model.score_matrix(home, away, neutral).ravel()
                    mat = mat / mat.sum()
                    self.group_unplayed[letter].append(
                        {"home": home, "away": away, "flat": mat, "cols": m}
                    )

    def _sample_group(self, letter: str) -> list[tuple]:
        """Simulate one group; return standings sorted best-first as
        (team, points, gd, gf, rank_in_group)."""
        teams = self.struct.groups[letter].teams
        pts = {t: 0 for t in teams}
        gf = {t: 0 for t in teams}
        ga = {t: 0 for t in teams}
        results: list[tuple] = []  # (home, away, hs, as_) for head-to-head

        def record(home, away, hs, as_):
            gf[home] += hs; ga[home] += as_
            gf[away] += as_; ga[away] += hs
            if hs > as_:
                pts[home] += 3
            elif hs < as_:
                pts[away] += 3
            else:
                pts[home] += 1; pts[away] += 1
            results.append((home, away, hs, as_))

        for (home, away, hs, as_) in self.group_played[letter]:
            record(home, away, hs, as_)
        for fx in self.group_unplayed[letter]:
            k = self.rng.choice(len(fx["flat"]), p=fx["flat"])
            hs, as_ = divmod(int(k), fx["cols"])
            record(fx["home"], fx["away"], hs, as_)

        standings = self._rank(teams, pts, gf, ga, results)
        return [(t, pts[t], gf[t] - ga[t], gf[t], i) for i, t in enumerate(standings)]

    def _rank(self, teams, pts, gf, ga, results) -> list[str]:
        def head_to_head(a, b):
            pa = pb = 0
            for (h, aw, hs, as_) in results:
                if {h, aw} == {a, b}:
                    if hs == as_:
                        pa += 1; pb += 1
                    elif (hs > as_) == (h == a):
                        pa += 3
                    else:
                        pb += 3
            return pa - pb

        import functools

        def cmp(a, b):
            if pts[a] != pts[b]:
                return pts[b] - pts[a]
            gda, gdb = gf[a] - ga[a], gf[b] - ga[b]
            if gda != gdb:
                return gdb - gda
            if gf[a] != gf[b]:
                return gf[b] - gf[a]
            h2h = head_to_head(a, b)
            if h2h != 0:
                return -h2h
            return -1 if a < b else 1  # deterministic final fallback

        return sorted(teams, key=functools.cmp_to_key(cmp))

    # ----------------------------------------------------------------------------------
    # Knockout
    # ----------------------------------------------------------------------------------
    def _penalty_prob(self, home: str, away: str) -> float:
        key = (home, away)
        if key not in self._knockout_cache:
            self._knockout_cache[key] = expected_score(
                self.elo.rating(home), self.elo.rating(away)
            )
        return self._knockout_cache[key]

    def _knockout_winner(self, a: str, b: str) -> str:
        """Single match on neutral ground; draws resolved by an Elo-tilted shootout."""
        p_home, p_draw, p_away = self.model.outcome_probabilities(a, b, neutral=True)
        r = self.rng.random()
        if r < p_home:
            return a
        if r < p_home + p_away:
            return b
        # Drawn after normal/extra time -> penalties.
        return a if self.rng.random() < self._penalty_prob(a, b) else b

    def _build_bracket(self, winners, runners, thirds) -> list[str]:
        """Return 32 teams in bracket-leaf order (adjacent pairs are R32 matches).

        Half 1 holds winners A-F, runners G-L and 4 thirds; half 2 mirrors it. This keeps
        every group's winner and runner-up in opposite halves.
        """
        letters = sorted(self.struct.groups)  # A..L
        h1_w = [winners[l] for l in letters[:6]]          # winners A-F
        h2_w = [winners[l] for l in letters[6:]]          # winners G-L
        h1_r = [runners[l] for l in letters[6:]]          # runners G-L into half 1
        h2_r = [runners[l] for l in letters[:6]]          # runners A-F into half 2
        t_h1, t_h2 = thirds[:4], thirds[4:]

        half1 = self._lay_half(h1_w, h1_r, t_h1, letters[:6])
        half2 = self._lay_half(h2_w, h2_r, t_h2, letters[6:])
        return half1 + half2

    def _lay_half(self, winners6, runners6, thirds4, winner_letters) -> list[str]:
        """Build 16 leaves (8 R32 matches) for one half: 6 winners each face a third/runner
        (avoiding a same-group third), the leftover opponents play each other."""
        wgroup = {team: let for team, let in zip(winners6, winner_letters)}
        opponents = list(thirds4) + list(runners6)  # 10 opponents for 6 winners
        leaves: list[str] = []
        used = [False] * len(opponents)

        for w in winners6:
            pick = None
            for i, opp in enumerate(opponents):
                if used[i]:
                    continue
                # avoid pairing a winner with the third-placed team from its own group
                if self.struct.team_group.get(opp) == wgroup[w]:
                    continue
                pick = i
                break
            if pick is None:  # fallback if avoidance impossible
                pick = next(i for i in range(len(opponents)) if not used[i])
            used[pick] = True
            leaves.extend([w, opponents[pick]])

        leftovers = [opponents[i] for i in range(len(opponents)) if not used[i]]
        for i in range(0, len(leftovers), 2):
            leaves.extend(leftovers[i : i + 2])
        return leaves

    def _play_bracket(self, bracket: list[str], reach: dict[str, np.ndarray]) -> str:
        """Fold the bracket to a champion, recording the deepest round each team reaches."""
        teams = list(bracket)
        for rnd in range(len(ROUND_NAMES) - 1):  # R32 .. Final
            for t in teams:
                reach[t][rnd] += 1  # reached this round
            nxt = []
            for i in range(0, len(teams), 2):
                nxt.append(self._knockout_winner(teams[i], teams[i + 1]))
            teams = nxt
        champion = teams[0]
        reach[champion][len(ROUND_NAMES) - 1] += 1  # Champion
        return champion

    # ----------------------------------------------------------------------------------
    # Monte-Carlo driver
    # ----------------------------------------------------------------------------------
    def run(self, iterations: int = config.DEFAULT_ITERATIONS):
        all_teams = self.struct.teams
        reach = {t: np.zeros(len(ROUND_NAMES)) for t in all_teams}
        group_pos = {t: np.zeros(4) for t in all_teams}  # finishing 1st..4th counts
        advance = {t: 0 for t in all_teams}              # reached Round of 32

        for _ in range(iterations):
            winners, runners, thirds_pool = {}, {}, []
            for letter in self.struct.groups:
                standings = self._sample_group(letter)
                for (team, _pts, _gd, _gf, pos) in standings:
                    group_pos[team][pos] += 1
                winners[letter] = standings[0][0]
                runners[letter] = standings[1][0]
                thirds_pool.append(standings[2])  # (team, pts, gd, gf, pos)

            # 8 best third-placed teams by (points, gd, gf)
            thirds_pool.sort(key=lambda s: (s[1], s[2], s[3]), reverse=True)
            thirds = [s[0] for s in thirds_pool[: config.N_BEST_THIRDS]]

            for t in list(winners.values()) + list(runners.values()) + thirds:
                advance[t] += 1

            bracket = self._build_bracket(winners, runners, thirds)
            self._play_bracket(bracket, reach)

        return self._summarize(iterations, reach, group_pos, advance)

    def _summarize(self, iterations, reach, group_pos, advance):
        import pandas as pd

        rows = []
        for t in self.struct.teams:
            row = {
                "team": t,
                "group": self.struct.team_group[t],
                "elo": round(self.elo.rating(t), 1),
                "win_group_%": 100 * group_pos[t][0] / iterations,
                "advance_R32_%": 100 * advance[t] / iterations,
                "reach_R16_%": 100 * reach[t][1] / iterations,
                "reach_QF_%": 100 * reach[t][2] / iterations,
                "reach_SF_%": 100 * reach[t][3] / iterations,
                "reach_final_%": 100 * reach[t][4] / iterations,
                "champion_%": 100 * reach[t][5] / iterations,
            }
            rows.append(row)
        df = pd.DataFrame(rows).sort_values("champion_%", ascending=False).reset_index(drop=True)
        return df

    # ----------------------------------------------------------------------------------
    # Deterministic single-projection bracket ("most likely" path)
    # ----------------------------------------------------------------------------------
    def _expected_group_table(self, letter: str) -> list[dict]:
        """Expected final standings for a group from model probabilities (played games
        fixed, unplayed games contribute expected points / goal difference)."""
        teams = self.struct.groups[letter].teams
        stats = {t: {"pts": 0.0, "gd": 0.0, "gf": 0.0} for t in teams}
        for (home, away, neutral, played, hs, as_) in self.struct.groups[letter].fixtures:
            if played:
                stats[home]["gf"] += hs
                stats[away]["gf"] += as_
                stats[home]["gd"] += hs - as_
                stats[away]["gd"] += as_ - hs
                if hs > as_:
                    stats[home]["pts"] += 3
                elif hs < as_:
                    stats[away]["pts"] += 3
                else:
                    stats[home]["pts"] += 1
                    stats[away]["pts"] += 1
            else:
                p_h, p_d, p_a = self.model.outcome_probabilities(home, away, neutral)
                lh, la = self.model.expected_goals(home, away, neutral)
                stats[home]["pts"] += 3 * p_h + p_d
                stats[away]["pts"] += 3 * p_a + p_d
                stats[home]["gd"] += lh - la
                stats[away]["gd"] += la - lh
                stats[home]["gf"] += lh
                stats[away]["gf"] += la

        ranked = sorted(
            teams, key=lambda t: (stats[t]["pts"], stats[t]["gd"], stats[t]["gf"]), reverse=True
        )
        return [
            {
                "team": t,
                "position": i + 1,
                "exp_points": round(stats[t]["pts"], 2),
                "exp_gd": round(stats[t]["gd"], 2),
                "exp_gf": round(stats[t]["gf"], 2),
            }
            for i, t in enumerate(ranked)
        ]

    def _project_winner(self, a: str, b: str) -> tuple[str, dict]:
        """Most-likely advancer of a single knockout tie (draw resolved by Elo penalties)."""
        p_h, p_d, p_a = self.model.outcome_probabilities(a, b, neutral=True)
        pen = expected_score(self.elo.rating(a), self.elo.rating(b))
        a_adv = p_h + p_d * pen
        b_adv = p_a + p_d * (1 - pen)
        winner = a if a_adv >= b_adv else b
        return winner, {
            "p_home": round(p_h, 4),
            "p_draw": round(p_d, 4),
            "p_away": round(p_a, 4),
            "p_advance": round(max(a_adv, b_adv), 4),
        }

    def project_bracket(self) -> dict:
        """Deterministic projection: expected group tables -> qualifiers -> a single
        most-likely knockout bracket through to the champion."""
        group_tables = {l: self._expected_group_table(l) for l in self.struct.groups}

        winners, runners = {}, {}
        thirds_pool = []  # (letter, team, stats-tuple)
        for letter, table in group_tables.items():
            winners[letter] = table[0]["team"]
            runners[letter] = table[1]["team"]
            third = table[2]
            thirds_pool.append(
                (letter, third["team"], (third["exp_points"], third["exp_gd"], third["exp_gf"]))
            )

        thirds_pool.sort(key=lambda x: x[2], reverse=True)
        qualifying = thirds_pool[: config.N_BEST_THIRDS]
        thirds = [x[1] for x in qualifying]
        third_letters = {x[0] for x in qualifying}

        # Flat list of the 32 qualifiers with their route.
        qualified = []
        for letter in sorted(group_tables):
            for row in group_tables[letter]:
                pos = row["position"]
                if pos == 1:
                    route = "Winner"
                elif pos == 2:
                    route = "Runner-up"
                elif pos == 3 and letter in third_letters:
                    route = "Best third"
                else:
                    continue
                qualified.append({
                    "team": row["team"], "group": letter, "position": pos, "route": route,
                    "exp_points": row["exp_points"], "exp_gd": row["exp_gd"],
                    "elo": round(self.elo.rating(row["team"]), 1),
                })

        # Build and play the bracket deterministically, carrying winners forward.
        teams = self._build_bracket(winners, runners, thirds)
        rounds = []
        for name, _n in config.KNOCKOUT_ROUNDS:
            matches, nxt = [], []
            for i in range(0, len(teams), 2):
                a, b = teams[i], teams[i + 1]
                winner, probs = self._project_winner(a, b)
                matches.append({
                    "home": a, "away": b, "winner": winner,
                    "home_group": self.struct.team_group.get(a),
                    "away_group": self.struct.team_group.get(b),
                    **probs,
                })
                nxt.append(winner)
            rounds.append({"round": name, "matches": matches})
            teams = nxt

        return {
            "qualified": qualified,
            "group_tables": group_tables,
            "rounds": rounds,
            "champion": teams[0],
        }
