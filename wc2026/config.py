"""Static configuration: paths, team-name normalization, and tournament structure.

The dataset (results.csv) contains the 2026 group fixtures but does NOT label the
official group letters (A-L) or the knockout bracket. We recover the 12 groups from
fixture pairings in `data.py`, assign them letters A-L in a deterministic order, and
slot qualifiers into the standard 48-team knockout tree defined below.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_CSV = os.path.join(DATA_DIR, "results.csv")
GOALSCORERS_CSV = os.path.join(DATA_DIR, "goalscorers.csv")
SHOOTOUTS_CSV = os.path.join(DATA_DIR, "shootouts.csv")
OUTPUT_DIR = os.path.join(ROOT, "outputs")

# --------------------------------------------------------------------------------------
# Tournament constants
# --------------------------------------------------------------------------------------
WC_TOURNAMENT = "FIFA World Cup"
WC_YEAR = "2026"
N_GROUPS = 12
GROUP_SIZE = 4
N_BEST_THIRDS = 8  # 2026 format: 12 winners + 12 runners-up + 8 best thirds -> Round of 32

# Host nations get genuine home advantage even though most WC matches are flagged neutral.
HOST_NATIONS = {"United States", "Canada", "Mexico"}

# --------------------------------------------------------------------------------------
# Team-name normalization
# --------------------------------------------------------------------------------------
# Map variant / historical spellings onto the canonical name used across the project.
# (The dataset is largely consistent; this guards against encoding drift and a few
# historical name changes that would otherwise fragment a team's match history.)
NAME_NORMALIZATION = {
    "Curacao": "Curaçao",
    "Cura�ao": "Curaçao",
    "Turkiye": "Turkey",
    "Türkiye": "Turkey",
    "USA": "United States",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "China PR": "China",
    "Czechia": "Czech Republic",
    "Cabo Verde": "Cape Verde",
    "Ivory Coast": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
}


def normalize_team(name: str) -> str:
    if name is None:
        return name
    name = name.strip()
    return NAME_NORMALIZATION.get(name, name)


# --------------------------------------------------------------------------------------
# Elo configuration
# --------------------------------------------------------------------------------------
ELO_BASE_RATING = 1500.0
ELO_HOME_ADVANTAGE = 65.0  # rating points added to the non-neutral home side
# K-factor weights by tournament importance (World Bank "World Football Elo" style).
ELO_K_BY_TOURNAMENT = {
    "FIFA World Cup": 60.0,
    "FIFA World Cup qualification": 40.0,
    "UEFA Euro": 50.0,
    "Copa América": 50.0,
    "African Cup of Nations": 50.0,
    "AFC Asian Cup": 50.0,
    "Gold Cup": 50.0,
    "UEFA Nations League": 40.0,
    "CONCACAF Nations League": 40.0,
    "Confederations Cup": 45.0,
}
ELO_K_QUALIFIER_DEFAULT = 35.0  # any "* qualification" tournament not listed above
ELO_K_FRIENDLY = 20.0
ELO_K_DEFAULT = 30.0  # other competitive tournaments

# --------------------------------------------------------------------------------------
# Dixon-Coles configuration
# --------------------------------------------------------------------------------------
# Exponential time-decay half-life (in days) for match weights when fitting the model.
DC_TIME_DECAY_HALFLIFE_DAYS = 365.0 * 3.0  # ~3 years
DC_MAX_GOALS = 10  # score-matrix is computed for 0..DC_MAX_GOALS goals per side
# Only fit on matches from this date onward (keeps the model on the modern game).
DC_TRAIN_FROM = "2006-01-01"

# --------------------------------------------------------------------------------------
# Simulation configuration
# --------------------------------------------------------------------------------------
DEFAULT_ITERATIONS = 20000
RANDOM_SEED = 42

# Knockout rounds, in order, with the number of ties contested in each.
KNOCKOUT_ROUNDS = [
    ("Round of 32", 16),
    ("Round of 16", 8),
    ("Quarter-finals", 4),
    ("Semi-finals", 2),
    ("Final", 1),
]
