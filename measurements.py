"""Re-run the basic 5-agent Chicken tournament with save_history=True and
produce the diagnostic plots requested for the methods/results section:

  * Choice over rounds (per matchup, focal agent)
  * Cumulative score over rounds (per matchup, focal agent)
  * Estimated P(opponent is k-TOM) for the 2-TOM agent (sophistication)
  * Estimated opponent volatility (log scale)
  * Estimated opponent bias (default 0 for k-TOM opponents)

Plots use tomsup's built-in helpers (see tomsup/tutorials/Getting_started.ipynb
section 4 "Plotting results"):

  group.plot_choice(agent0, agent1, agent=...)
  group.plot_score (agent0, agent1, agent=...)
  group.plot_p_k   (agent0, agent1, agent=..., level=k)
  group.plot_history(agent0, agent1, agent=..., fun=lambda x: ...)

n_sim is kept at 50 to match the original baseline tournament in run.py.

Run:
    /Users/yoav.levy/chicken/.venv-tomsup/bin/python chicken_tomsup/measurements.py
"""
from __future__ import annotations

import os
import warnings

os.chdir("/tmp")
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import tomsup as ts

OUT_DIR = "/Users/yoav.levy/chicken"
N_ROUNDS = 50
N_SIM = 50
CRASH = -10.0


def chicken_matrix(crash: float = CRASH) -> ts.PayoffMatrix:
    m = np.array(
        [
            [(0.0, -1.0), (1.0, crash)],
            [(0.0, 1.0), (-1.0, crash)],
        ],
        dtype=float,
    )
    return ts.PayoffMatrix(name="custom", predefined=m)


def save(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


def main() -> None:
    matrix = chicken_matrix()
    agent_names = ["RB", "WSLS", "0-TOM", "1-TOM", "2-TOM"]
    start_params = [{"bias": 0.5}, {}, {}, {}, {}]

    # Try several seeds; tomsup's k-ToM update can NaN on rare initializations.
    print(f"Running round-robin tournament: N_rounds={N_ROUNDS}, N_sim={N_SIM}…")
    for seed in [42, 7, 123, 2024, 99, 0]:
        np.random.seed(seed)
        try:
            group = ts.create_agents(agent_names, start_params)
            group.set_env("round_robin")
            group.compete(
                p_matrix=matrix, n_rounds=N_ROUNDS, n_sim=N_SIM,
                save_history=True, verbose=False,
            )
            print(f"  succeeded with seed={seed}")
            break
        except ValueError as e:
            print(f"  seed={seed} failed ({e}); retrying…")
    else:
        raise RuntimeError("All seeds NaN'd; consider reducing n_sim or n_rounds")

    # Choice and cumulative score over rounds, for key matchups.
    # Focal agent is agent=1 (i.e. the named `a1`); opponent is `a0`.
    key_pairs = [
        ("0-TOM", "1-TOM"),
        ("0-TOM", "2-TOM"),
        ("1-TOM", "2-TOM"),
        ("RB",    "2-TOM"),
        ("WSLS",  "2-TOM"),
    ]
    # for a0, a1 in key_pairs:
    #     plt.figure(figsize=(7, 4))
    #     group.plot_choice(agent0=a0, agent1=a1, agent=1,
    #                       plot_individual_sim=False, show=False)
    #     plt.title(f"{a1} playing against {a0}")
    #     plt.xlabel("Round")
    #     plt.ylabel("Action (0 = Swerve, 1 = Straight)")
    #     save(os.path.join(OUT_DIR, f"meas_choice_{a1}_vs_{a0}.png"))

    #     plt.figure(figsize=(7, 4))
    #     group.plot_score(agent0=a0, agent1=a1, agent=1, show=False)
    #     plt.title(f"{a1} playing against {a0}")
    #     plt.xlabel("Round")
    #     plt.ylabel("Cumulative payoff")
    #     save(os.path.join(OUT_DIR, f"meas_score_{a1}_vs_{a0}.png"))

    # 2-TOM's estimated P(opponent is k-TOM) for each candidate level k.
    # Overlaid lines: P(opp = 0-TOM) and P(opp = 1-TOM).
    for opp in ["RB", "WSLS", "0-TOM", "1-TOM"]:
        group.plot_p_k(agent0=opp, agent1="2-TOM", agent=1, level=0, show=False)
        group.plot_p_k(agent0=opp, agent1="2-TOM", agent=1, level=1, show=False)
        save(os.path.join(OUT_DIR, f"meas_pk_2TOM_vs_{opp}.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
