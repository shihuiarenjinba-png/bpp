from __future__ import annotations

import numpy as np
import pandas as pd


def _max_drawdown_from_prices(prices: np.ndarray) -> np.ndarray:
    running_max = np.maximum.accumulate(prices, axis=1)
    drawdowns = prices / running_max - 1.0
    return drawdowns.min(axis=1)


def _path_summary(prices: np.ndarray, annualization: float) -> dict[str, float]:
    safe_prices = np.clip(prices, 1e-6, None)
    monthly_returns = safe_prices[:, 1:] / safe_prices[:, :-1] - 1.0
    terminal = prices[:, -1] / prices[:, 0] - 1.0
    ann_return = np.power(np.clip(1.0 + terminal, 1e-9, None), annualization / monthly_returns.shape[1]) - 1.0
    ann_vol = monthly_returns.std(axis=1) * np.sqrt(annualization)
    sharpe = np.divide(
        ann_return,
        ann_vol,
        out=np.zeros_like(ann_return),
        where=ann_vol > 1e-9,
    )
    summary = {
        "expected_return": float(np.mean(ann_return)),
        "volatility": float(np.mean(ann_vol)),
        "median_terminal": float(np.median(terminal)),
        "crash_probability": float(np.mean(terminal <= -0.2)),
        "avg_max_drawdown": float(np.mean(_max_drawdown_from_prices(prices))),
        "avg_sharpe": float(np.mean(sharpe)),
    }
    return summary


def simulate_behavioral_gap(
    *,
    initial_price: float,
    expected_return: float,
    volatility: float,
    years: int,
    n_sims: int,
    overreaction: float,
    loss_aversion: float,
    herding: float,
    anchoring: float,
    sentiment_beta: float,
    seed: int,
) -> dict[str, object]:
    steps = years * 12
    dt = 1.0 / 12.0
    rng = np.random.default_rng(seed)
    shocks = rng.normal(size=(n_sims, steps))
    diffusion = volatility * np.sqrt(dt) * shocks
    drift = (expected_return - 0.5 * volatility**2) * dt
    rational_log_returns = drift + diffusion

    rational_prices = np.full((n_sims, steps + 1), initial_price, dtype=float)
    behavioral_prices = np.full((n_sims, steps + 1), initial_price, dtype=float)
    sentiment = np.zeros(n_sims, dtype=float)
    anchor_price = np.full(n_sims, initial_price, dtype=float)
    previous_behavioral_return = np.zeros(n_sims, dtype=float)
    bias_terms = np.zeros((n_sims, steps), dtype=float)

    for step in range(1, steps + 1):
        rational_step_return = rational_log_returns[:, step - 1]
        rational_prices[:, step] = rational_prices[:, step - 1] * np.exp(rational_step_return)

        sentiment = 0.78 * sentiment + rng.normal(scale=0.35 * np.sqrt(dt), size=n_sims)
        crowd_signal = np.tanh(previous_behavioral_return / max(volatility * np.sqrt(dt), 1e-6))
        anchoring_gap = (behavioral_prices[:, step - 1] - anchor_price) / np.clip(anchor_price, 1e-6, None)

        raw_bias = (
            overreaction * np.sign(previous_behavioral_return) * np.abs(previous_behavioral_return)
            + herding * crowd_signal * np.abs(crowd_signal)
            + sentiment_beta * sentiment * np.sqrt(dt)
            - loss_aversion * np.clip(-previous_behavioral_return, 0.0, None)
            - anchoring * anchoring_gap * dt
        )
        bias = np.tanh(raw_bias) * (0.55 * volatility * np.sqrt(dt))

        behavioral_step_return = np.clip(rational_step_return + bias, -0.45, 0.45)
        behavioral_prices[:, step] = behavioral_prices[:, step - 1] * np.exp(behavioral_step_return)
        bias_terms[:, step - 1] = bias
        anchor_price = 0.92 * anchor_price + 0.08 * behavioral_prices[:, step]
        previous_behavioral_return = behavioral_step_return

    dates = pd.date_range("2026-01-01", periods=steps + 1, freq="MS")
    avg_rational = rational_prices.mean(axis=0)
    avg_behavioral = behavioral_prices.mean(axis=0)
    lower_rational = np.quantile(rational_prices, 0.1, axis=0)
    upper_rational = np.quantile(rational_prices, 0.9, axis=0)
    lower_behavioral = np.quantile(behavioral_prices, 0.1, axis=0)
    upper_behavioral = np.quantile(behavioral_prices, 0.9, axis=0)

    fan_chart = pd.DataFrame(
        {
            "date": dates,
            "rational_mean": avg_rational,
            "rational_p10": lower_rational,
            "rational_p90": upper_rational,
            "behavioral_mean": avg_behavioral,
            "behavioral_p10": lower_behavioral,
            "behavioral_p90": upper_behavioral,
        }
    )

    comparison = pd.DataFrame(
        [
            {"model": "Traditional Finance", **_path_summary(rational_prices, 12.0)},
            {"model": "Behavioral Finance", **_path_summary(behavioral_prices, 12.0)},
        ]
    )
    comparison["terminal_price_mean"] = [float(rational_prices[:, -1].mean()), float(behavioral_prices[:, -1].mean())]
    comparison["gap_vs_traditional"] = comparison["terminal_price_mean"] / comparison["terminal_price_mean"].iloc[0] - 1.0

    divergence = pd.DataFrame(
        {
            "date": dates,
            "relative_gap": avg_behavioral / np.clip(avg_rational, 1e-6, None) - 1.0,
            "avg_bias_term": np.concatenate([[0.0], bias_terms.mean(axis=0)]),
        }
    )

    terminal_gap = behavioral_prices[:, -1] / np.clip(rational_prices[:, -1], 1e-6, None) - 1.0
    diagnostics = {
        "gap_probability_gt_10pct": float(np.mean(terminal_gap >= 0.1)),
        "gap_probability_lt_minus_10pct": float(np.mean(terminal_gap <= -0.1)),
        "mean_bias": float(bias_terms.mean()),
        "vol_of_bias": float(bias_terms.std()),
    }

    return {
        "fan_chart": fan_chart,
        "comparison": comparison,
        "divergence": divergence,
        "diagnostics": diagnostics,
    }
