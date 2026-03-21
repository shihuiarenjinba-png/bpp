from __future__ import annotations

import io

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
except Exception:
    GaussianHMM = None


KNOWN_FACTOR_NAMES = [
    "Mkt-RF",
    "SMB",
    "HML",
    "RMW",
    "CMA",
    "MOM",
]

FACTOR_ALIAS_MAP = {
    "mkt_rf": "Mkt-RF",
    "mktrf": "Mkt-RF",
    "market": "Mkt-RF",
    "market_excess": "Mkt-RF",
    "smb": "SMB",
    "hml": "HML",
    "rmw": "RMW",
    "cma": "CMA",
    "mom": "MOM",
    "momentum": "MOM",
    "rf": "RF",
}

REGIME_NAME_MAP = {
    0: "supportive_uptrend",
    1: "transition",
    2: "fragile_downturn",
}


def generate_sample_factor_dataset(months: int = 180, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2011-01-31", periods=months, freq="ME")

    regimes = ["supportive_uptrend", "inflation_rotation", "fragile_downturn"]
    transition = np.array(
        [
            [0.82, 0.13, 0.05],
            [0.18, 0.58, 0.24],
            [0.12, 0.28, 0.60],
        ]
    )
    current = 0
    state_ids = []
    for _ in range(months):
        state_ids.append(current)
        current = int(rng.choice([0, 1, 2], p=transition[current]))

    params = {
        "supportive_uptrend": {
            "mean": np.array([0.010, 0.003, -0.001, 0.002, 0.001, 0.004]),
            "vol": np.array([0.040, 0.028, 0.022, 0.020, 0.018, 0.035]),
        },
        "inflation_rotation": {
            "mean": np.array([0.004, -0.001, 0.006, -0.0005, 0.003, -0.001]),
            "vol": np.array([0.048, 0.032, 0.026, 0.023, 0.022, 0.036]),
        },
        "fragile_downturn": {
            "mean": np.array([-0.009, -0.003, 0.003, 0.005, 0.004, -0.002]),
            "vol": np.array([0.062, 0.042, 0.032, 0.026, 0.023, 0.045]),
        },
    }

    rows = []
    for state_id in state_ids:
        regime = regimes[state_id]
        mean = params[regime]["mean"]
        vol = params[regime]["vol"]
        corr = np.full((len(mean), len(mean)), 0.18)
        np.fill_diagonal(corr, 1.0)
        cov = np.outer(vol, vol) * corr
        draw = rng.multivariate_normal(mean, cov)
        rows.append(draw)

    factor_df = pd.DataFrame(rows, columns=KNOWN_FACTOR_NAMES, index=dates)
    factor_df["RF"] = 0.0012
    factor_df["true_regime"] = [regimes[idx] for idx in state_ids]
    return factor_df


def load_uploaded_factor_table(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    suffix = file_name.lower().split(".")[-1]
    if suffix == "csv":
        return pd.read_csv(io.BytesIO(file_bytes))
    if suffix in {"xlsx", "xlsm", "xls"}:
        return pd.read_excel(io.BytesIO(file_bytes))
    return pd.DataFrame()


def prepare_factor_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    df = frame.copy()
    renamed = {}
    date_col = None
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in {"date", "month", "年月", "年月日"}:
            date_col = col
        if key in FACTOR_ALIAS_MAP:
            renamed[col] = FACTOR_ALIAS_MAP[key]

    df.rename(columns=renamed, inplace=True)
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df[df[date_col].notna()].copy()
        df.set_index(date_col, inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.date_range("2010-01-31", periods=len(df), freq="M")

    cols = [col for col in KNOWN_FACTOR_NAMES + ["RF"] if col in df.columns]
    if len(cols) < 3:
        return pd.DataFrame()

    df = df[cols].apply(pd.to_numeric, errors="coerce")
    if df.abs().max().max() > 1.5:
        df = df / 100.0
    df.index = pd.DatetimeIndex(df.index).to_period("M").to_timestamp(how="end")
    return df.dropna(how="all")


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = values / max(temperature, 1e-6)
    shifted = scaled - np.nanmax(scaled)
    exp_values = np.exp(shifted)
    return exp_values / np.clip(exp_values.sum(), 1e-9, None)


def _spectral_signal(series: pd.Series, window: int = 48) -> float:
    clean = series.dropna().tail(window)
    if len(clean) < max(18, window // 2):
        return 0.0

    values = clean.to_numpy(dtype=float)
    values = values - values.mean()
    if np.allclose(values.std(), 0.0):
        return 0.0

    fft = np.fft.rfft(values)
    amplitudes = np.abs(fft)
    if len(amplitudes) <= 2:
        return 0.0

    dominant = 1 + int(np.argmax(amplitudes[1:]))
    freqs = np.fft.rfftfreq(len(values), d=1.0)
    phase = np.angle(fft[dominant])
    prediction = amplitudes[dominant] / len(values) * np.cos(2.0 * np.pi * freqs[dominant] * len(values) + phase)
    return float(prediction / (clean.std() + 1e-6))


def _regime_features(market_series: pd.Series, all_factors: pd.DataFrame) -> pd.DataFrame:
    cumulative = (1.0 + market_series.fillna(0.0)).cumprod()
    available_factors = [col for col in KNOWN_FACTOR_NAMES if col in all_factors.columns]
    features = pd.DataFrame(index=market_series.index)
    features["mean_6m"] = market_series.rolling(6).mean()
    features["vol_6m"] = market_series.rolling(6).std()
    features["drawdown_12m"] = cumulative / cumulative.rolling(12).max() - 1.0
    features["breadth_3m"] = (all_factors[available_factors].rolling(3).mean() > 0).mean(axis=1)
    return features.dropna()


def estimate_regimes(factor_df: pd.DataFrame) -> pd.DataFrame:
    market = factor_df["Mkt-RF"] if "Mkt-RF" in factor_df.columns else factor_df.iloc[:, 0]
    features = _regime_features(market, factor_df)
    if features.empty:
        return pd.DataFrame()

    result = features.copy()
    if GaussianHMM is not None and len(features) >= 48:
        scaled = (features - features.mean()) / features.std(ddof=0).replace(0.0, 1.0)
        model = GaussianHMM(n_components=3, covariance_type="full", n_iter=400, random_state=42)
        model.fit(scaled)
        states = model.predict(scaled)
        probabilities = model.predict_proba(scaled)
        result["state"] = states
        result["supportive_prob"] = probabilities[:, 0]
        result["transition_prob"] = probabilities[:, 1]
        result["fragile_prob"] = probabilities[:, 2]

        summary = result.groupby("state").agg(
            mean_6m=("mean_6m", "mean"),
            vol_6m=("vol_6m", "mean"),
            drawdown_12m=("drawdown_12m", "mean"),
        )
        summary["risk"] = summary["vol_6m"].rank(pct=True) + (-summary["mean_6m"]).rank(pct=True) + summary["drawdown_12m"].rank(pct=True)
        ordered = summary["risk"].sort_values().index.tolist()
        mapping = {
            ordered[0]: "supportive_uptrend",
            ordered[1]: "transition",
            ordered[2]: "fragile_downturn",
        }
        result["regime_label"] = result["state"].map(mapping)

        prob_cols = ["supportive_prob", "transition_prob", "fragile_prob"]
        label_map = {
            "supportive_uptrend": "supportive_prob",
            "transition": "transition_prob",
            "fragile_downturn": "fragile_prob",
        }
        for label, col in label_map.items():
            if col not in result.columns:
                result[col] = 0.0
    else:
        mean_std = result["mean_6m"].std(ddof=0)
        vol_std = result["vol_6m"].std(ddof=0)
        mean_scale = mean_std if mean_std and not np.isnan(mean_std) else 1.0
        vol_scale = vol_std if vol_std and not np.isnan(vol_std) else 1.0
        mean_z = (result["mean_6m"] - result["mean_6m"].mean()) / mean_scale
        vol_z = (result["vol_6m"] - result["vol_6m"].mean()) / vol_scale
        dd = result["drawdown_12m"].fillna(0.0)

        supportive_score = 1.35 * mean_z - 0.85 * vol_z + 1.2 * dd
        fragile_score = -1.15 * mean_z + 1.2 * vol_z - 1.5 * dd
        transition_score = 0.35 - 0.45 * np.abs(mean_z) + 0.2 * result["breadth_3m"]

        scores = np.column_stack([supportive_score, transition_score, fragile_score])
        probabilities = np.apply_along_axis(_softmax, 1, scores, 0.65)
        result["supportive_prob"] = probabilities[:, 0]
        result["transition_prob"] = probabilities[:, 1]
        result["fragile_prob"] = probabilities[:, 2]
        labels = np.argmax(probabilities, axis=1)
        result["regime_label"] = [REGIME_NAME_MAP[int(label)] for label in labels]

    return result


def compute_factor_forecast(
    factor_df: pd.DataFrame,
    *,
    horizon_months: int,
    temperature: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    regimes = estimate_regimes(factor_df)
    if regimes.empty:
        return pd.DataFrame(), pd.DataFrame()

    aligned = factor_df.join(regimes[["regime_label", "supportive_prob", "transition_prob", "fragile_prob"]], how="inner")
    current_regime = aligned["regime_label"].iloc[-1]
    factor_cols = [col for col in KNOWN_FACTOR_NAMES if col in aligned.columns]
    rows = []
    for factor in factor_cols:
        series = aligned[factor].dropna()
        recent = series.tail(12)
        rolling_ir = recent.mean() / (recent.std(ddof=0) + 1e-6)
        regime_mean = aligned.loc[aligned["regime_label"].eq(current_regime), factor].tail(36).mean()
        long_mean = series.tail(60).mean()
        consistency = float((recent > 0).mean())
        spectral = _spectral_signal(series, window=min(48, len(series)))
        volatility = series.tail(36).std(ddof=0)

        rows.append(
            {
                "factor": factor,
                "rolling_ir": rolling_ir,
                "regime_mean": regime_mean,
                "long_mean": long_mean,
                "consistency": consistency,
                "spectral_signal": spectral,
                "volatility": volatility,
            }
        )

    forecast = pd.DataFrame(rows)
    for col in ["rolling_ir", "regime_mean", "consistency", "spectral_signal"]:
        std = forecast[col].std(ddof=0)
        forecast[f"{col}_z"] = 0.0 if std == 0 or np.isnan(std) else (forecast[col] - forecast[col].mean()) / std

    forecast["score"] = (
        0.34 * forecast["rolling_ir_z"]
        + 0.31 * forecast["regime_mean_z"]
        + 0.19 * forecast["consistency_z"]
        + 0.16 * forecast["spectral_signal_z"]
    )
    forecast["expected_monthly_return"] = (
        0.45 * forecast["regime_mean"]
        + 0.30 * forecast["long_mean"]
        + 0.25 * forecast["rolling_ir"] * forecast["volatility"]
    )
    forecast["expected_horizon_return"] = np.power(
        1.0 + forecast["expected_monthly_return"].clip(lower=-0.9),
        horizon_months,
    ) - 1.0
    probabilities = _softmax(forecast["score"].to_numpy(dtype=float), temperature)
    forecast["lead_probability"] = probabilities
    forecast["confidence"] = np.clip(
        0.45
        + 0.25 * forecast["lead_probability"]
        + 0.15 * np.abs(forecast["score"]),
        0.0,
        0.99,
    )
    forecast.sort_values(by="lead_probability", ascending=False, inplace=True)
    forecast.reset_index(drop=True, inplace=True)
    forecast["rank"] = np.arange(1, len(forecast) + 1)
    forecast["current_regime"] = current_regime
    return forecast, regimes


def simulate_factor_outlook(
    factor_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    *,
    horizon_months: int,
    n_sims: int,
    seed: int,
) -> dict[str, object]:
    factor_cols = forecast_df["factor"].tolist()
    returns = factor_df[factor_cols].dropna()
    if returns.empty:
        return {"fan_chart": pd.DataFrame(), "winner_probabilities": pd.DataFrame(), "summary": {}}

    recent = returns.tail(max(36, horizon_months * 3))
    cov = recent.cov().to_numpy()
    mean = forecast_df.set_index("factor").loc[factor_cols, "expected_monthly_return"].to_numpy()
    weights = forecast_df.set_index("factor").loc[factor_cols, "lead_probability"].to_numpy()
    rng = np.random.default_rng(seed)

    simulations = rng.multivariate_normal(mean, cov, size=(n_sims, horizon_months))
    weighted_returns = np.tensordot(simulations, weights, axes=([2], [0]))
    cumulative_paths = np.cumprod(1.0 + weighted_returns, axis=1) - 1.0

    fan_rows = []
    for month in range(horizon_months):
        path_values = cumulative_paths[:, month]
        fan_rows.append(
            {
                "month": month + 1,
                "p10": np.quantile(path_values, 0.1),
                "p50": np.quantile(path_values, 0.5),
                "p90": np.quantile(path_values, 0.9),
                "mean": np.mean(path_values),
            }
        )

    factor_terminals = np.prod(1.0 + simulations, axis=1) - 1.0
    winners = np.argmax(factor_terminals, axis=1)
    winner_probabilities = (
        pd.Series(winners)
        .value_counts(normalize=True)
        .rename(index={idx: factor_cols[idx] for idx in range(len(factor_cols))})
        .reset_index()
    )
    winner_probabilities.columns = ["factor", "dominance_probability"]
    winner_probabilities.sort_values(by="dominance_probability", ascending=False, inplace=True)

    summary = {
        "expected_portfolio_return": float(np.mean(cumulative_paths[:, -1])),
        "median_portfolio_return": float(np.median(cumulative_paths[:, -1])),
        "loss_probability": float(np.mean(cumulative_paths[:, -1] < 0.0)),
        "best_factor": str(forecast_df.iloc[0]["factor"]),
    }

    return {
        "fan_chart": pd.DataFrame(fan_rows),
        "winner_probabilities": winner_probabilities,
        "summary": summary,
    }
