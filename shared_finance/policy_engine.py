from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_policy_paths(
    *,
    horizon_months: int,
    n_sims: int,
    initial_inflation: float,
    target_inflation: float,
    initial_policy_rate: float,
    neutral_rate: float,
    initial_output_gap: float,
    total_hike_bps: int,
    hike_months: int,
    expectations_persistence: float,
    inflation_persistence: float,
    rate_sensitivity: float,
    phillips_slope: float,
    supply_shock: float,
    supply_decay: float,
    credibility: float,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    policy_panel = []
    terminal_rows = []
    hike_step = (total_hike_bps / 10000.0) / max(hike_months, 1)

    for sim_id in range(n_sims):
        inflation = initial_inflation
        expected_inflation = initial_inflation
        output_gap = initial_output_gap
        policy_rate = initial_policy_rate
        current_supply = supply_shock

        for month in range(horizon_months + 1):
            if month > 0 and month <= hike_months:
                policy_rate += hike_step

            expected_inflation = (
                expectations_persistence * expected_inflation
                + (1.0 - expectations_persistence) * inflation
            )
            policy_stance = policy_rate - neutral_rate
            inflation_gap = inflation - target_inflation
            expected_gap = expected_inflation - target_inflation

            if month > 0:
                output_gap = (
                    0.72 * output_gap
                    - 0.10 * rate_sensitivity * policy_stance
                    - 0.28 * current_supply
                    + rng.normal(0.0, 0.003)
                )
                output_gap = float(np.clip(output_gap, -0.08, 0.06))
                inflation = (
                    target_inflation
                    + inflation_persistence * inflation_gap
                    + 0.22 * expected_gap
                    + 0.18 * phillips_slope * output_gap
                    + 0.28 * current_supply
                    - 0.10 * credibility * max(policy_stance, 0.0)
                    + rng.normal(0.0, 0.0022)
                )
                inflation = float(np.clip(inflation, -0.01, 0.12))
                current_supply *= supply_decay

            policy_panel.append(
                {
                    "simulation": sim_id,
                    "month": month,
                    "policy_rate": policy_rate,
                    "inflation": inflation,
                    "expected_inflation": expected_inflation,
                    "output_gap": output_gap,
                    "supply_shock": current_supply,
                }
            )

        terminal_rows.append(
            {
                "simulation": sim_id,
                "terminal_inflation": inflation,
                "terminal_output_gap": output_gap,
                "terminal_policy_rate": policy_rate,
                "hit_target": inflation <= target_inflation,
                "soft_landing": inflation <= target_inflation and output_gap > -0.015,
            }
        )

    panel = pd.DataFrame(policy_panel)
    quantile_rows = []
    for month, group in panel.groupby("month"):
        quantile_rows.append(
            {
                "month": month,
                "inflation_p10": group["inflation"].quantile(0.1),
                "inflation_p50": group["inflation"].quantile(0.5),
                "inflation_p90": group["inflation"].quantile(0.9),
                "policy_rate_p10": group["policy_rate"].quantile(0.1),
                "policy_rate_p50": group["policy_rate"].quantile(0.5),
                "policy_rate_p90": group["policy_rate"].quantile(0.9),
                "output_gap_p10": group["output_gap"].quantile(0.1),
                "output_gap_p50": group["output_gap"].quantile(0.5),
                "output_gap_p90": group["output_gap"].quantile(0.9),
            }
        )
    quantiles = pd.DataFrame(quantile_rows)
    terminals = pd.DataFrame(terminal_rows)

    summary = {
        "target_hit_probability": float(terminals["hit_target"].mean()),
        "soft_landing_probability": float(terminals["soft_landing"].mean()),
        "median_terminal_inflation": float(terminals["terminal_inflation"].median()),
        "median_terminal_policy_rate": float(terminals["terminal_policy_rate"].median()),
        "median_terminal_output_gap": float(terminals["terminal_output_gap"].median()),
    }

    return {
        "panel": panel,
        "quantiles": quantiles,
        "terminal": terminals,
        "summary": summary,
    }
