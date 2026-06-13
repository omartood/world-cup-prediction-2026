"""End-to-end WC-2026 prediction pipeline.

Usage:
    python -m wc2026.main [--iterations N] [--seed S] [--no-charts]

Steps: load + clean data, recover groups, fit Elo and Dixon-Coles models, predict the
remaining group fixtures, then Monte-Carlo simulate the whole tournament. Writes results
to outputs/ and prints summary tables.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from . import config
from .data import derive_groups, load_results, played_matches
from .elo import EloModel
from .poisson import PoissonModel
from .service import predict_group_fixtures
from .simulate import Simulator


def group_predictions_frame(struct, model: PoissonModel) -> pd.DataFrame:
    """The shared group-fixture predictions, formatted as a percentage table for the CLI."""
    df = pd.DataFrame(predict_group_fixtures(struct, model))
    if df.empty:
        return df
    for col, src in [("p_home_%", "p_home"), ("p_draw_%", "p_draw"), ("p_away_%", "p_away")]:
        df[col] = (100 * df[src]).round(1)
    return df[["group", "home", "away", "p_home_%", "p_draw_%", "p_away_%",
               "xg_home", "xg_away", "likely_score", "prediction"]]


def make_charts(odds: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[charts] matplotlib unavailable ({exc}); skipping charts")
        return

    top = odds.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["team"], top["champion_%"], color="#1f77b4")
    ax.set_xlabel("Probability of winning the World Cup (%)")
    ax.set_title("WC 2026 — Champion odds (top 15)")
    for y, v in zip(range(len(top)), top["champion_%"]):
        ax.text(v + 0.1, y, f"{v:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "champion_odds.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[charts] wrote {path}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="FIFA World Cup 2026 prediction engine")
    ap.add_argument("--iterations", type=int, default=config.DEFAULT_ITERATIONS)
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("Loading data ...")
    df = load_results()
    struct = derive_groups(df)
    wc = struct.all_fixtures
    n_played = int(wc["played"].sum())
    print(f"  {len(struct.teams)} teams, 12 groups | WC2026 fixtures: "
          f"{n_played} played, {len(wc) - n_played} to simulate")

    print("Fitting Elo ratings ...")
    elo = EloModel().fit(played_matches(df))
    print("Fitting Dixon-Coles goal model ...")
    model = PoissonModel(elo).fit(played_matches(df))
    print(f"  home advantage={model.h:+.3f}, elo coef={model.w:+.3f}, "
          f"DC rho={model.rho:+.3f}, teams fitted={len(model.teams)}")

    print("Predicting remaining group fixtures ...")
    group_preds = group_predictions_frame(struct, model)
    group_preds.to_csv(os.path.join(config.OUTPUT_DIR, "group_predictions.csv"), index=False)

    print(f"Simulating tournament ({args.iterations:,} iterations) ...")
    sim = Simulator(struct, model, elo, seed=args.seed)
    odds = sim.run(args.iterations)

    advance_cols = ["team", "group", "elo", "win_group_%", "advance_R32_%"]
    odds.to_csv(os.path.join(config.OUTPUT_DIR, "champion_odds.csv"), index=False)
    odds[advance_cols + ["reach_R16_%", "reach_QF_%", "reach_SF_%"]].to_csv(
        os.path.join(config.OUTPUT_DIR, "advancement.csv"), index=False
    )

    pd.set_option("display.float_format", lambda v: f"{v:.1f}")
    print("\n=== Champion odds (top 15) ===")
    print(odds[["team", "group", "elo", "advance_R32_%", "reach_SF_%",
                "reach_final_%", "champion_%"]].head(15).to_string(index=False))

    print("\n=== Sample group-fixture predictions ===")
    print(group_preds.head(12).to_string(index=False))

    if not args.no_charts:
        make_charts(odds)

    print(f"\nDone. Outputs written to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
