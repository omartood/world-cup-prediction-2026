"""Data loading, cleaning, and WC-2026 structure recovery."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from . import config


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_results() -> pd.DataFrame:
    """Load results.csv, normalize names/dates, and coerce scores to nullable ints.

    Returns a frame with an extra boolean column `played` (True where both scores are
    present). Unplayed rows keep NaN scores.
    """
    df = pd.read_csv(config.RESULTS_CSV, na_values=["NA"], keep_default_na=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_team"] = df["home_team"].map(config.normalize_team)
    df["away_team"] = df["away_team"].map(config.normalize_team)
    df["country"] = df["country"].map(config.normalize_team)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    df["played"] = df["home_score"].notna() & df["away_score"].notna()
    return df.sort_values("date").reset_index(drop=True)


def played_matches(df: pd.DataFrame) -> pd.DataFrame:
    """All historical matches with a final score (used for model training)."""
    return df[df["played"]].copy()


# --------------------------------------------------------------------------------------
# WC-2026 fixtures + group recovery
# --------------------------------------------------------------------------------------
def wc2026_matches(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["tournament"] == config.WC_TOURNAMENT) & (
        df["date"].dt.year == int(config.WC_YEAR)
    )
    return df[mask].copy()


@dataclass
class Group:
    letter: str
    teams: list[str]
    # All group fixtures (both played and unplayed) as (home, away, neutral, played, hs, as_)
    fixtures: list[tuple] = field(default_factory=list)


@dataclass
class TournamentStructure:
    groups: dict[str, Group]          # letter -> Group
    team_group: dict[str, str]        # team -> group letter
    all_fixtures: pd.DataFrame        # the WC2026 group-stage matches

    @property
    def teams(self) -> list[str]:
        return [t for g in self.groups.values() for t in g.teams]


def derive_groups(df: pd.DataFrame) -> TournamentStructure:
    """Recover the 12 groups of 4 from 2026 WC fixture pairings via connected components.

    Group letters A-L are assigned deterministically (groups sorted by their member
    tuple) so results are reproducible across runs.
    """
    wc = wc2026_matches(df)

    adj: dict[str, set] = defaultdict(set)
    for _, r in wc.iterrows():
        adj[r["home_team"]].add(r["away_team"])
        adj[r["away_team"]].add(r["home_team"])

    seen: set[str] = set()
    components: list[list[str]] = []
    for team in adj:
        if team in seen:
            continue
        stack, comp = [team], set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            comp.add(node)
            stack.extend(adj[node] - seen)
        components.append(sorted(comp))

    if len(components) != config.N_GROUPS or any(
        len(c) != config.GROUP_SIZE for c in components
    ):
        raise ValueError(
            f"Expected {config.N_GROUPS} groups of {config.GROUP_SIZE}; "
            f"got sizes {sorted(len(c) for c in components)}"
        )

    components.sort()  # deterministic ordering by member tuple
    letters = [chr(ord("A") + i) for i in range(config.N_GROUPS)]

    groups: dict[str, Group] = {}
    team_group: dict[str, str] = {}
    for letter, teams in zip(letters, components):
        g = Group(letter=letter, teams=teams)
        for _, r in wc.iterrows():
            if r["home_team"] in teams and r["away_team"] in teams:
                g.fixtures.append(
                    (
                        r["home_team"],
                        r["away_team"],
                        bool(r["neutral"]),
                        bool(r["played"]),
                        None if pd.isna(r["home_score"]) else int(r["home_score"]),
                        None if pd.isna(r["away_score"]) else int(r["away_score"]),
                    )
                )
        groups[letter] = g
        for t in teams:
            team_group[t] = letter

    return TournamentStructure(groups=groups, team_group=team_group, all_fixtures=wc)


# --------------------------------------------------------------------------------------
# Manual sanity entry point: `python -m wc2026.data`
# --------------------------------------------------------------------------------------
def _main() -> None:
    df = load_results()
    struct = derive_groups(df)
    wc = struct.all_fixtures
    n_played = int(wc["played"].sum())
    print(f"WC2026 fixtures: {len(wc)}  played={n_played}  unplayed={len(wc) - n_played}")
    print(f"Historical played matches available for training: {len(played_matches(df)):,}")
    print()
    for letter, g in struct.groups.items():
        print(f"Group {letter}: {', '.join(g.teams)}")


if __name__ == "__main__":
    _main()
