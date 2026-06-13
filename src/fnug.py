
from __future__ import annotations

from tqdm.auto import tqdm
import argparse
import csv
import copy
import hashlib
import json
import logging
import math
import os
import pickle
import random
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.ensemble import HistGradientBoostingRegressor
from torch.utils.data import Dataset, DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    data_dir: str = field(default_factory=lambda: os.environ.get("FNUG_DATA_DIR", "data/raw"))
    cache_dir: str = "cache"           # preprocessed daily quote sets
    results_dir: str = field(default_factory=lambda: os.environ.get("FNUG_RESULTS_DIR", "results"))

    years: List[int] = field(default_factory=lambda: list(range(2018, 2026)))
    option_file_fmt: str = "data_{year}.parquet"
    price_file: str = "security_prices.parquet"
    zcy_file: str = "zero_coupon_yield.parquet"
    div_file: str = "index_dividend_yield.parquet"
    vix_file: str = "VIXCLS.parquet"
    vxv_file: str = "VXVCLS.parquet"

    secid: int = 108105
    tau_min: float = 7 / 365
    tau_max: float = 270 / 365
    k_min: float = -0.30
    k_max: float = 0.30
    atm_eps: float = 0.005

    min_bid: float = 0.05
    max_rel_spread: float = 0.50
    min_open_interest: int = 0
    settlement_policy: str = "pm_only"
    atm_summary_k: float = 0.02
    short_target_days: int = 30
    long_target_days: int = 90

    train_end: int = 2021
    val_end: int = 2022

    # These are now used directly in the no-arbitrage grid construction.
    moneyness_nodes: List[float] = field(default_factory=lambda: [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20])
    maturity_nodes_days: List[int] = field(default_factory=lambda: [7, 14, 21, 30, 45, 60, 90, 120, 180, 270])
    noarb_ref_days: int = 30

    rv_horizons: List[int] = field(default_factory=lambda: [5, 21])
    stress_quantile: float = 0.90

    chunk_size: int = 400_000

@dataclass
class ModelConfig:
    input_dim: int = 8
    embed_dim: int = 64
    embed_layers: int = 3
    num_inducing: int = 32
    num_heads: int = 4
    num_isab: int = 2
    pma_seeds: int = 4
    latent_dim: int = 16
    decoder_query_dim: int = 2
    decoder_hidden: int = 128
    decoder_layers: int = 3
    local_input_dim: int = 7
    local_hidden: int = 64
    local_layers: int = 2
    local_residual_scale: float = 0.25
    head_hidden: int = 64
    head_layers: int = 2


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda"

    batch_size_s1: int = 4
    epochs_s1: int = 100
    lr_s1: float = 1e-3
    weight_decay: float = 1e-4

    batch_size_s2: int = 32
    epochs_s2: int = 60
    lr_s2: float = 1e-3

    epochs_s3: int = 30
    lr_s3_encoder: float = 1e-4
    lr_s3_heads: float = 5e-4
    warmup_epochs_s3: int = 5

    grad_clip: float = 1.0
    scheduler: str = "cosine"

    lambda_rec: float = 1.0
    lambda_ba: float = 0.6
    lambda_arb: float = 0.3
    lambda_mask: float = 0.5
    lambda_pred: float = 1.0
    lambda_struct: float = 0.25
    lambda_contrast: float = 0.10
    lambda_ba_cover: float = 0.20

    ema_decay: float = 0.99
    ema_eps: float = 1e-6

    mask_ratio_min: float = 0.15
    mask_ratio_max: float = 0.30

    ckpt_every: int = 10
    best_metric: str = "val_total_loss"
    stage3_eval_preference: str = "forecast"


DATA_CFG = DataConfig()
MODEL_CFG = ModelConfig()
TRAIN_CFG = TrainConfig()


def make_dirs(cfg: DataConfig) -> None:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.results_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(cfg.results_dir, "eval"), exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _looks_like_data_dir(path: Path, cfg: DataConfig) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    aux_ok = all((path / fname).exists() for fname in [
        cfg.price_file, cfg.zcy_file, cfg.div_file, cfg.vix_file, cfg.vxv_file
    ])
    if not aux_ok:
        return False
    option_years = [y for y in cfg.years if (path / cfg.option_file_fmt.format(year=y)).exists()]
    return len(option_years) > 0


def resolve_data_dir(cfg: DataConfig) -> Path:
    candidates = []
    if cfg.data_dir:
        candidates.append(Path(cfg.data_dir).expanduser())

    cwd = Path.cwd()
    candidates.extend([
        cwd,
        cwd.parent,
        Path(__file__).resolve().parent if "__file__" in globals() else cwd,
        (Path(__file__).resolve().parent.parent if "__file__" in globals() else cwd.parent),
    ])

    seen = set()
    ordered = []
    for c in candidates:
        try:
            rc = c.resolve()
        except Exception:
            rc = c
        key = str(rc)
        if key not in seen:
            seen.add(key)
            ordered.append(rc)

    for c in ordered:
        if _looks_like_data_dir(c, cfg):
            return c

    raise FileNotFoundError(
        "Could not resolve data_dir. Set DataConfig.data_dir or environment variable "
        "'FNUG_DATA_DIR' to the folder containing option parquet files and auxiliary parquet files."
    )


def discover_available_years(cfg: DataConfig, data_dir: Path) -> List[int]:
    return sorted([y for y in cfg.years if (data_dir / cfg.option_file_fmt.format(year=y)).exists()])


def _bs_price_np(F: float, K: float, tau: float, sigma: float, D: float, is_call: bool) -> float:
    if tau <= 1e-8 or sigma <= 1e-8:
        iv = max(F - K, 0.0) if is_call else max(K - F, 0.0)
        return D * iv
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    if is_call:
        return D * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def _bs_iv_np(mid: float, F: float, K: float, tau: float, D: float, is_call: bool,
              lo: float = 1e-4, hi: float = 10.0) -> Optional[float]:
    intrinsic = D * max((F - K if is_call else K - F), 0.0)
    if mid <= intrinsic + 1e-6:
        return None
    try:
        iv = brentq(lambda s: _bs_price_np(F, K, tau, s, D, is_call) - mid, lo, hi, xtol=1e-6, maxiter=100)
        return iv if 0 < iv < hi else None
    except (ValueError, RuntimeError):
        return None


def _build_discount_curve(days_arr: np.ndarray, rates_pct: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    T = days_arr / 365.0
    D = np.exp(-rates_pct / 100.0 * T)
    idx = np.argsort(T)
    return T[idx], D[idx]


def _interp_discount(tau: float, T_arr: np.ndarray, D_arr: np.ndarray) -> float:
    if tau <= 0:
        return 1.0
    if tau <= T_arr[0]:
        if T_arr[0] > 0:
            return np.exp(np.log(D_arr[0]) * tau / T_arr[0])
        return D_arr[0]
    if tau >= T_arr[-1]:
        r_last = -np.log(D_arr[-1]) / T_arr[-1]
        return np.exp(-r_last * tau)
    i = np.searchsorted(T_arr, tau) - 1
    i = max(0, min(i, len(T_arr) - 2))
    t0, t1 = T_arr[i], T_arr[i + 1]
    D0, D1 = D_arr[i], D_arr[i + 1]
    alpha = (tau - t0) / (t1 - t0)
    return np.exp((1 - alpha) * np.log(D0) + alpha * np.log(D1))


def _interp_log_curve_np(tau_obs: np.ndarray, y_obs: np.ndarray, tau_grid: np.ndarray) -> np.ndarray:
    tau_obs = np.asarray(tau_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    tau_grid = np.asarray(tau_grid, dtype=float)

    valid = np.isfinite(tau_obs) & np.isfinite(y_obs) & (tau_obs > 0) & (y_obs > 0)
    tau_obs, y_obs = tau_obs[valid], y_obs[valid]
    if len(tau_obs) == 0:
        return np.ones_like(tau_grid, dtype=float)

    order = np.argsort(tau_obs)
    tau_obs, y_obs = tau_obs[order], y_obs[order]
    uniq = np.unique(tau_obs)
    if len(uniq) != len(tau_obs):
        y_avg = [np.mean(y_obs[tau_obs == u]) for u in uniq]
        tau_obs = uniq
        y_obs = np.asarray(y_avg, dtype=float)

    logy = np.log(y_obs)
    out = np.empty_like(tau_grid, dtype=float)
    for i, t in enumerate(tau_grid):
        if t <= tau_obs[0]:
            out[i] = np.exp(logy[0] * t / tau_obs[0]) if tau_obs[0] > 0 else y_obs[0]
        elif t >= tau_obs[-1]:
            slope = (logy[-1] - logy[-2]) / max(tau_obs[-1] - tau_obs[-2], 1e-8) if len(tau_obs) >= 2 else 0.0
            out[i] = np.exp(logy[-1] + slope * (t - tau_obs[-1]))
        else:
            j = np.searchsorted(tau_obs, t) - 1
            j = max(0, min(j, len(tau_obs) - 2))
            a = (t - tau_obs[j]) / max(tau_obs[j + 1] - tau_obs[j], 1e-8)
            out[i] = np.exp((1 - a) * logy[j] + a * logy[j + 1])
    return out


def load_aux_data(cfg: DataConfig) -> Dict:
    ddir = resolve_data_dir(cfg)
    logger.info("Loading auxiliary data …")

    prices = pd.read_parquet(os.path.join(ddir, cfg.price_file))
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[prices["secid"] == cfg.secid][["date", "close", "return"]].copy()
    prices.sort_values("date", inplace=True)
    prices.reset_index(drop=True, inplace=True)

    zcy = pd.read_parquet(os.path.join(ddir, cfg.zcy_file))
    zcy["date"] = pd.to_datetime(zcy["date"])
    zcy.sort_values(["date", "days"], inplace=True)

    div = pd.read_parquet(os.path.join(ddir, cfg.div_file))
    div["date"] = pd.to_datetime(div["date"])
    div["expiration"] = pd.to_datetime(div["expiration"])
    div.sort_values(["date", "expiration"], inplace=True)

    vix = pd.read_parquet(os.path.join(ddir, cfg.vix_file))
    vix["observation_date"] = pd.to_datetime(vix["observation_date"])
    vix.rename(columns={"observation_date": "date", "VIXCLS": "VIX"}, inplace=True)

    vxv = pd.read_parquet(os.path.join(ddir, cfg.vxv_file))
    vxv["observation_date"] = pd.to_datetime(vxv["observation_date"])
    vxv.rename(columns={"observation_date": "date", "VXVCLS": "VXV"}, inplace=True)

    prices = prices.merge(vix[["date", "VIX"]], on="date", how="left")
    prices = prices.merge(vxv[["date", "VXV"]], on="date", how="left")

    logger.info(f"  Prices: {len(prices)} days")
    logger.info(f"  ZCY:    {zcy['date'].nunique()} dates")
    logger.info(f"  DIV:    {div['date'].nunique()} dates")
    return {"prices": prices, "zcy": zcy, "div": div}


def build_forward_for_date(trade_date: pd.Timestamp, S: float, tau_arr: np.ndarray,
                           zcy_day: pd.DataFrame, div_day: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    if len(zcy_day) > 0:
        T_r, D_r_curve = _build_discount_curve(zcy_day["days"].values.astype(float), zcy_day["rate"].values.astype(float))
    else:
        T_r, D_r_curve = np.array([1.0]), np.array([1.0])

    if len(div_day) > 0:
        exp_taus = ((div_day["expiration"] - trade_date).dt.days.values / 365.0).astype(float)
        div_rates = div_day["rate"].values.astype(float)
        mask = exp_taus > 0
        exp_taus, div_rates = exp_taus[mask], div_rates[mask]
        if len(exp_taus) > 0:
            T_q, D_q_curve = _build_discount_curve(exp_taus * 365, div_rates)
        else:
            T_q, D_q_curve = np.array([1.0]), np.array([1.0])
    else:
        T_q, D_q_curve = np.array([1.0]), np.array([1.0])

    F_arr = np.empty(len(tau_arr))
    Dr_arr = np.empty(len(tau_arr))
    for i, tau in enumerate(tau_arr):
        Dr = _interp_discount(tau, T_r, D_r_curve)
        Dq = _interp_discount(tau, T_q, D_q_curve)
        F_arr[i] = S * Dq / Dr
        Dr_arr[i] = Dr
    return F_arr, Dr_arr


def _compute_iv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["iv"] = pd.to_numeric(df["impl_volatility"], errors="coerce")
    nan_mask = df["iv"].isna() | (df["iv"] <= 0)
    if nan_mask.any():
        mids = 0.5 * (df.loc[nan_mask, "best_bid"] + df.loc[nan_mask, "best_offer"])
        Fs = df.loc[nan_mask, "F"].values
        Ks = df.loc[nan_mask, "K"].values
        taus = df.loc[nan_mask, "tau"].values
        Drs = df.loc[nan_mask, "D_r"].values
        cps = df.loc[nan_mask, "cp_flag"].values
        ivs = np.empty(nan_mask.sum())
        for i, (mid, F0, K0, tau0, D0, cp) in enumerate(zip(mids, Fs, Ks, taus, Drs, cps)):
            iv = _bs_iv_np(mid, F0, K0, tau0, D0, cp == "C")
            ivs[i] = iv if iv is not None else np.nan
        df.loc[nan_mask, "iv"] = ivs
    return df


def _compute_mid_iv_target(df: pd.DataFrame) -> np.ndarray:
    mids = 0.5 * (df["best_bid"].values + df["best_offer"].values)
    Fs = df["F"].values
    Ks = df["K"].values
    taus = df["tau"].values
    Drs = df["D_r"].values
    cps = df["cp_flag"].values
    iv_fallback = df["iv"].values
    out = np.empty(len(df), dtype=float)
    for i, (mid, F0, K0, tau0, D0, cp, iv0) in enumerate(zip(mids, Fs, Ks, taus, Drs, cps, iv_fallback)):
        iv_mid = _bs_iv_np(mid, F0, K0, tau0, D0, cp == "C")
        if iv_mid is not None and np.isfinite(iv_mid) and (0.02 < iv_mid < 5.0):
            out[i] = iv_mid
        else:
            out[i] = iv0
    return out


def _compute_vega(df: pd.DataFrame, iv_col: str = "iv") -> np.ndarray:
    F0 = df["F"].values
    K0 = df["K"].values
    tau = df["tau"].values
    iv = pd.to_numeric(df[iv_col], errors="coerce").fillna(0.20).values
    Dr = df["D_r"].values
    with np.errstate(divide="ignore", invalid="ignore"):
        sqrt_tau = np.sqrt(np.maximum(tau, 1e-8))
        sigma = np.maximum(iv, 1e-4)
        d1 = (np.log(F0 / np.maximum(K0, 1e-4)) + 0.5 * sigma**2 * tau) / (sigma * sqrt_tau)
        vega = F0 * Dr * norm.pdf(d1) * sqrt_tau
    return np.maximum(vega, 0.0)


def _apply_settlement_policy(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    if policy == "all":
        return df
    if "am_flag" not in df.columns:
        return df
    if policy == "pm_only":
        return df[df["am_flag"] == 0].copy()
    if policy == "am_only":
        return df[df["am_flag"] == 1].copy()
    raise ValueError(f"Unknown settlement_policy={policy}")


def _apply_otm_rule(df: pd.DataFrame, atm_eps: float) -> pd.DataFrame:
    put_mask = (df["cp_flag"] == "P") & (df["k"] < -atm_eps)
    call_mask = (df["cp_flag"] == "C") & (df["k"] > atm_eps)
    atm_mask = df["k"].abs() <= atm_eps

    atm_df = df[atm_mask].copy()
    if len(atm_df) > 0:
        if "vega_bs" in atm_df.columns:
            atm_df["vega_val"] = atm_df["vega_bs"].fillna(0.0)
        elif "vega" in atm_df.columns:
            atm_df["vega_val"] = pd.to_numeric(atm_df["vega"], errors="coerce").fillna(0)
        else:
            atm_df["vega_val"] = 0.0
        atm_df = (
            atm_df.sort_values("vega_val", ascending=False)
            .drop_duplicates(subset=["exdate", "K"])
            .drop(columns=["vega_val"], errors="ignore")
        )

    return pd.concat([df[put_mask], df[call_mask], atm_df], ignore_index=True)


def _liquidity_weight(df: pd.DataFrame) -> np.ndarray:
    raw = df["vega_bs"].values / (df["rel_spread"].values + 1e-4)
    denom = raw.max() + 1e-8
    return raw / denom


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    valid = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not valid.any():
        return np.nan
    return float(np.average(x[valid], weights=w[valid]))


def _canonical_total_variance(df: pd.DataFrame) -> np.ndarray:
    """
    Canonical surface definition used consistently across:
    - quote feature w
    - summary statistics
    - reconstruction target

    Priority:
    1) w_target_label  (mid-IV implied total variance)
    2) w
    3) tau * iv^2
    """
    if "w_target_label" in df.columns:
        w = df["w_target_label"].values.astype(float)
    elif "w" in df.columns:
        w = df["w"].values.astype(float)
    elif "iv" in df.columns and "tau" in df.columns:
        w = df["tau"].values.astype(float) * np.square(df["iv"].values.astype(float))
    else:
        raise KeyError("Cannot construct canonical total variance from dataframe columns.")
    return np.asarray(w, dtype=float)


def _build_noarb_reference_grids_np(
    tau_obs: np.ndarray,
    F_obs: np.ndarray,
    D_obs: np.ndarray,
    S_ref: float,
    moneyness_nodes: np.ndarray,
    tau_grid: np.ndarray,
    ref_tau_days: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build no-arbitrage evaluation grids using:
      fixed strike line K_j = m_j * F(ref_tau)
    where m_j are the configured moneyness nodes.

    This keeps strike fixed across maturities for each day, so the calendar
    monotonicity check remains well-defined.
    """
    tau_obs = np.asarray(tau_obs, dtype=float)
    F_obs = np.asarray(F_obs, dtype=float)
    D_obs = np.asarray(D_obs, dtype=float)
    tau_grid = np.asarray(tau_grid, dtype=float)

    m_nodes = np.asarray(sorted([
        float(x) for x in np.asarray(moneyness_nodes, dtype=float)
        if np.isfinite(x) and x > 0
    ]), dtype=float)
    if m_nodes.size == 0:
        raise ValueError("moneyness_nodes must contain positive finite values.")

    valid = (
        np.isfinite(tau_obs) & np.isfinite(F_obs) & np.isfinite(D_obs) &
        (tau_obs > 0) & (F_obs > 0) & (D_obs > 0)
    )

    if valid.sum() == 0:
        F_tau = np.full(len(tau_grid), max(float(S_ref), 1e-6), dtype=np.float32)
        D_tau = np.ones(len(tau_grid), dtype=np.float32)
        F_ref = max(float(S_ref), 1e-6)
    else:
        F_tau = _interp_log_curve_np(tau_obs[valid], F_obs[valid], tau_grid)
        D_tau = _interp_log_curve_np(tau_obs[valid], D_obs[valid], tau_grid)

        ref_tau = np.asarray([ref_tau_days / 365.0], dtype=float)
        F_ref = float(_interp_log_curve_np(tau_obs[valid], F_obs[valid], ref_tau)[0])

        if (not np.isfinite(F_ref)) or (F_ref <= 0):
            F_ref = float(np.nanmedian(F_obs[valid]))
        if (not np.isfinite(F_ref)) or (F_ref <= 0):
            F_ref = max(float(S_ref), 1e-6)

    K_fixed = np.maximum(F_ref * m_nodes, 1e-4)

    for j in range(1, len(K_fixed)):
        if K_fixed[j] <= K_fixed[j - 1]:
            K_fixed[j] = K_fixed[j - 1] + max(1e-4, 1e-4 * K_fixed[j - 1])

    return (
        np.asarray(F_tau, dtype=np.float32),
        np.asarray(D_tau, dtype=np.float32),
        np.asarray(K_fixed, dtype=np.float32),
    )


def _load_full_price_history(cfg: DataConfig) -> pd.DataFrame:
    """
    Full daily price/return history from the raw auxiliary file.
    This is used for stateful GARCH pseudo-OOS updates so that missing option
    dates do not collapse multi-day returns into one step.
    """
    aux = load_aux_data(cfg)
    px = aux["prices"][["date", "close", "return"]].copy()
    px = px.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return px


def garch_forecast_on_anchor_dates(
    prices_df: pd.DataFrame,
    fit_end_date: pd.Timestamp | str,
    anchor_dates,
    horizon: int = 1,
) -> np.ndarray:
    """
    Pseudo-OOS GARCH forecasts aligned to arbitrary anchor dates.

    Protocol:
    - GARCH parameters are estimated exactly once using returns up to fit_end_date.
    - After that, the conditional variance state is updated sequentially using
      every realized daily return after fit_end_date.
    - No parameter re-fitting is performed on validation data before test-time
      evaluation.

    Forecast at date t uses return_t (close-of-t information), matching the
    pipeline's forward target convention: rv_h[t] depends on t+1, ..., t+h.
    """
    try:
        from arch import arch_model
    except ImportError:
        logger.warning("arch package not found. GARCH baseline unavailable. Install with: pip install arch")
        return np.full(len(anchor_dates), np.nan)

    fit_end_date = pd.Timestamp(fit_end_date)
    anchor_index = pd.to_datetime(pd.Index(anchor_dates))
    anchor_set = set(anchor_index)

    px = prices_df[["date", "return"]].copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values("date").reset_index(drop=True)

    train_mask = px["date"] <= fit_end_date
    train_returns = px.loc[train_mask, "return"].values.astype(float)
    train_returns = train_returns[np.isfinite(train_returns)]

    if len(train_returns) < 50:
        return np.full(len(anchor_index), np.nan)

    am = arch_model(train_returns * 100.0, mean="Zero", vol="Garch", p=1, q=1, dist="normal")
    res = am.fit(disp="off", show_warning=False)

    p = res.params
    omega = float(p.get("omega", 0.0))
    alpha = float(p.get("alpha[1]", 0.0))
    beta = float(p.get("beta[1]", 0.0))
    persistence = alpha + beta

    sigma2_state = float(res.conditional_volatility[-1] ** 2)

    post_df = px.loc[px["date"] > fit_end_date, ["date", "return"]].copy()
    forecasts = {}

    for _, row in post_df.iterrows():
        d = pd.Timestamp(row["date"])
        r = float(row["return"])

        if np.isfinite(r):
            eps_t_pct = r * 100.0
            sigma2_state = max(omega + alpha * eps_t_pct**2 + beta * sigma2_state, 1e-12)

        path_h = np.empty(horizon, dtype=float)
        path_h[0] = sigma2_state
        for h in range(1, horizon):
            path_h[h] = max(omega + persistence * path_h[h - 1], 1e-12)

        avg_var_raw = np.mean(path_h) / 1e4
        vol_forecast = np.sqrt(avg_var_raw * 252.0)

        if d in anchor_set:
            forecasts[d] = float(vol_forecast)

    return np.asarray([forecasts.get(pd.Timestamp(d), np.nan) for d in anchor_index], dtype=float)


def downside_semivar_forecast_on_anchor_dates(
    prices_df: pd.DataFrame,
    fit_end_date: pd.Timestamp | str,
    anchor_dates,
    horizon: int = 5,
) -> np.ndarray:
    vol_forecast = garch_forecast_on_anchor_dates(prices_df, fit_end_date, anchor_dates, horizon=horizon)
    return np.sqrt(0.5 * np.square(vol_forecast))


def _nearest_tau_band(df: pd.DataFrame, target_days: int, tol_days: int = 20) -> pd.DataFrame:
    target_tau = target_days / 365.0
    dist = (df["tau"] - target_tau).abs()
    band = df[dist <= tol_days / 365.0].copy()
    if len(band) == 0:
        if len(df) == 0:
            return df.copy()
        nearest = dist.min()
        band = df[dist <= nearest + 1e-12].copy()
    return band


def _daily_option_summary(df: pd.DataFrame, cfg: DataConfig) -> Dict[str, float]:
    w_canon = _canonical_total_variance(df)
    iv = np.sqrt(np.maximum(w_canon, 0.0) / np.maximum(df["tau"].values.astype(float), 1e-8))
    liq = np.maximum(df["liq_w"].values.astype(float), 1e-8)
    k = df["k"].values.astype(float)

    atm_mask = np.abs(k) <= cfg.atm_summary_k
    atm_iv = _weighted_mean(iv[atm_mask], liq[atm_mask]) if atm_mask.any() else float(iv[np.argmin(np.abs(k))])

    short_df = _nearest_tau_band(df, cfg.short_target_days)
    short_iv = _weighted_mean(
        np.sqrt(np.maximum(_canonical_total_variance(short_df), 0.0) / np.maximum(short_df["tau"].values, 1e-8)),
        short_df["liq_w"].values
    )

    long_df = _nearest_tau_band(df, cfg.long_target_days)
    long_iv = _weighted_mean(
        np.sqrt(np.maximum(_canonical_total_variance(long_df), 0.0) / np.maximum(long_df["tau"].values, 1e-8)),
        long_df["liq_w"].values
    )

    put_band = df[(df["k"] >= -0.18) & (df["k"] <= -0.12)]
    call_band = df[(df["k"] >= 0.12) & (df["k"] <= 0.18)]
    put_iv = _weighted_mean(
        np.sqrt(np.maximum(_canonical_total_variance(put_band), 0.0) / np.maximum(put_band["tau"].values, 1e-8)),
        put_band["liq_w"].values
    ) if len(put_band) > 0 else np.nan
    call_iv = _weighted_mean(
        np.sqrt(np.maximum(_canonical_total_variance(call_band), 0.0) / np.maximum(call_band["tau"].values, 1e-8)),
        call_band["liq_w"].values
    ) if len(call_band) > 0 else np.nan

    skew = put_iv - call_iv if np.isfinite(put_iv) and np.isfinite(call_iv) else np.nan
    term_slope = long_iv - short_iv if np.isfinite(long_iv) and np.isfinite(short_iv) else np.nan

    return {
        "atm_iv": float(atm_iv) if np.isfinite(atm_iv) else np.nan,
        "short_iv": float(short_iv) if np.isfinite(short_iv) else np.nan,
        "long_iv": float(long_iv) if np.isfinite(long_iv) else np.nan,
        "term_slope": float(term_slope) if np.isfinite(term_slope) else np.nan,
        "skew_proxy": float(skew) if np.isfinite(skew) else np.nan,
        "quote_density": int(len(df)),
        "am_share": float(df["am_flag"].mean()) if "am_flag" in df.columns and len(df) > 0 else np.nan,
    }


def _putcall_parity_diagnostics(df: pd.DataFrame, atm_eps: float = 0.03, tol: float = 0.05) -> Dict[str, float]:
    atm = df[df["k"].abs() < atm_eps].copy()
    residuals: List[float] = []
    n_pairs = 0
    if len(atm) == 0:
        return {"n_pairs": 0, "median_residual": np.nan, "max_residual": np.nan, "frac_bad": np.nan, "ok": True}

    for (_, K0), grp in atm.groupby(["exdate", "K"]):
        if "C" in grp["cp_flag"].values and "P" in grp["cp_flag"].values:
            n_pairs += 1
            c_row = grp[grp["cp_flag"] == "C"].iloc[0]
            p_row = grp[grp["cp_flag"] == "P"].iloc[0]
            F0 = float(c_row["F"])
            D0 = float(c_row["D_r"])
            C = float(c_row["mid"])
            P = float(p_row["mid"])
            residuals.append(abs((C - P) - D0 * (F0 - K0)) / (C + P + 1e-4))

    if n_pairs == 0:
        return {"n_pairs": 0, "median_residual": np.nan, "max_residual": np.nan, "frac_bad": np.nan, "ok": True}

    arr = np.asarray(residuals, dtype=float)
    frac_bad = float(np.mean(arr > tol))
    return {
        "n_pairs": int(n_pairs),
        "median_residual": float(np.nanmedian(arr)),
        "max_residual": float(np.nanmax(arr)),
        "frac_bad": frac_bad,
        "ok": bool(frac_bad == 0.0),
    }


def _pack_quote_dict(df: pd.DataFrame, prices_df: pd.DataFrame, dstr: str, cfg: DataConfig) -> dict:
    surface_w = _canonical_total_variance(df).astype(np.float32)

    features = np.column_stack([
        df["k"].values,
        df["tau"].values,
        surface_w,
        df["best_bid"].values,
        df["best_offer"].values,
        df["rel_spread"].values,
        np.log1p(df.get("open_interest", pd.Series(0, index=df.index)).fillna(0).values
                 + df.get("volume", pd.Series(0, index=df.index)).fillna(0).values),
        (df["cp_flag"] == "C").astype(float).values,
    ]).astype(np.float32)

    summary = _daily_option_summary(df, cfg)
    return {
        "date": dstr,
        "features": features,
        "liq_w": df["liq_w"].values.astype(np.float32),
        "w_target": df["w_target_label"].values.astype(np.float32),
        "bid_px": df["best_bid"].values.astype(np.float32),
        "ask_px": df["best_offer"].values.astype(np.float32),
        "F": df["F"].values.astype(np.float32),
        "K": df["K"].values.astype(np.float32),
        "D_r": df["D_r"].values.astype(np.float32),
        "tau": df["tau"].values.astype(np.float32),
        "k": df["k"].values.astype(np.float32),
        "am_flag": df["am_flag"].values.astype(np.int8),
        "S": float(prices_df[prices_df["date"] == pd.Timestamp(dstr)]["close"].iloc[0]),
        "N": len(df),
        **summary,
    }


class RVTargetBuilder:
    def __init__(self, prices_df: pd.DataFrame, cfg: DataConfig):
        self.prices = prices_df.sort_values("date").reset_index(drop=True)
        self.cfg = cfg
        self._targets: Optional[pd.DataFrame] = None
        self._stress_thr: Optional[float] = None

    def build(self) -> pd.DataFrame:
        if self._targets is not None:
            return self._targets

        ret = self.prices["return"].values.astype(float)
        dates = self.prices["date"].values
        n = len(ret)

        rv5 = np.full(n, np.nan)
        rv21 = np.full(n, np.nan)
        dsv5 = np.full(n, np.nan)
        rv5_back = np.full(n, np.nan)
        rv21_back = np.full(n, np.nan)
        dsv5_back = np.full(n, np.nan)

        for t in range(n):
            if t + 1 + 5 <= n:
                r5_fwd = ret[t + 1:t + 1 + 5]
                if np.all(np.isfinite(r5_fwd)):
                    rv5[t] = np.sqrt(252 / 5 * np.sum(r5_fwd**2))
                    dsv5[t] = np.sqrt(252 / 5 * np.sum(r5_fwd**2 * (r5_fwd < 0)))

            if t + 1 + 21 <= n:
                r21_fwd = ret[t + 1:t + 1 + 21]
                if np.all(np.isfinite(r21_fwd)):
                    rv21[t] = np.sqrt(252 / 21 * np.sum(r21_fwd**2))

            if t - 5 + 1 >= 0:
                r5_back = ret[t - 5 + 1:t + 1]
                if np.all(np.isfinite(r5_back)):
                    rv5_back[t] = np.sqrt(252 / 5 * np.sum(r5_back**2))
                    dsv5_back[t] = np.sqrt(252 / 5 * np.sum(r5_back**2 * (r5_back < 0)))

            if t - 21 + 1 >= 0:
                r21_back = ret[t - 21 + 1:t + 1]
                if np.all(np.isfinite(r21_back)):
                    rv21_back[t] = np.sqrt(252 / 21 * np.sum(r21_back**2))

        tdf = pd.DataFrame({
            "date": dates,
            "rv5": rv5,
            "rv21": rv21,
            "dsv5": dsv5,
            "rv5_back": rv5_back,
            "rv21_back": rv21_back,
            "dsv5_back": dsv5_back,
        })

        train_mask = tdf["date"] <= pd.Timestamp(f"{self.cfg.train_end}-12-31")
        train_rv21 = tdf.loc[train_mask, "rv21"].dropna()
        if len(train_rv21) == 0:
            raise ValueError("No finite training-set RV21 values found for stress threshold construction.")

        self._stress_thr = float(np.nanpercentile(train_rv21, self.cfg.stress_quantile * 100))
        logger.info(f"Stress threshold (q{self.cfg.stress_quantile * 100:.0f} train RV21) = {self._stress_thr:.4f}")

        tdf["stress21"] = np.where(np.isfinite(tdf["rv21"]), (tdf["rv21"] > self._stress_thr).astype(np.float32), np.nan)
        self._targets = tdf
        return tdf

    @property
    def stress_threshold(self) -> float:
        if self._stress_thr is None:
            self.build()
        return self._stress_thr


def _process_year(year: int, aux: Dict, cfg: DataConfig) -> Dict[str, dict]:
    data_dir = resolve_data_dir(cfg)
    filepath = os.path.join(str(data_dir), cfg.option_file_fmt.format(year=year))
    if not os.path.exists(filepath):
        logger.warning(f"  {filepath} not found – skipping year {year}")
        return {}

    prices_df = aux["prices"]
    zcy_df = aux["zcy"]
    div_df = aux["div"]

    daily_quotes: Dict[str, list] = {}
    logger.info(f"  Reading {year} …")

    df_year = pd.read_parquet(filepath)
    df_year["date"] = pd.to_datetime(df_year["date"])
    df_year["exdate"] = pd.to_datetime(df_year["exdate"])

    df_year = df_year[df_year["secid"] == cfg.secid]
    df_year = df_year[df_year["exercise_style"].fillna("E") == "E"]
    df_year = df_year[df_year["best_bid"] >= cfg.min_bid]
    df_year = df_year[df_year["best_offer"] > df_year["best_bid"]]
    if cfg.min_open_interest > 0 and "open_interest" in df_year.columns:
        df_year = df_year[df_year["open_interest"].fillna(0) >= cfg.min_open_interest]
    if len(df_year) == 0:
        return {}

    df_year["K"] = df_year["strike_price"] / 1000.0
    df_year["tau"] = (df_year["exdate"] - df_year["date"]).dt.days / 365.0
    df_year = df_year[(df_year["tau"] >= cfg.tau_min) & (df_year["tau"] <= cfg.tau_max)]
    if len(df_year) == 0:
        return {}

    for tdate, grp in df_year.groupby("date"):
        tdate_str = str(tdate.date())

        price_row = prices_df[prices_df["date"] == tdate]
        if len(price_row) == 0:
            continue
        S = float(price_row["close"].iloc[0])

        zcy_day = zcy_df[zcy_df["date"] == tdate]
        div_day = div_df[div_df["date"] == tdate]

        tau_arr = grp["tau"].values.astype(float)
        F_arr, Dr_arr = build_forward_for_date(tdate, S, tau_arr, zcy_day, div_day)

        grp = grp.copy()
        grp["F"] = F_arr
        grp["D_r"] = Dr_arr
        grp["k"] = np.log(grp["K"].values / F_arr)

        grp = grp[(grp["k"] >= cfg.k_min) & (grp["k"] <= cfg.k_max)]
        if len(grp) == 0:
            continue

        grp = _compute_iv(grp)
        grp = grp[grp["iv"].notna() & (grp["iv"] > 0.02) & (grp["iv"] < 5.0)]
        if len(grp) == 0:
            continue

        # Preserve the original vendor/fallback IV only for diagnostics.
        grp["iv_raw_vendor"] = grp["iv"].astype(float)
        grp["w_raw_iv"] = grp["tau"] * grp["iv_raw_vendor"]**2

        grp["mid_iv_target"] = _compute_mid_iv_target(grp)
        grp = grp[np.isfinite(grp["mid_iv_target"]) & (grp["mid_iv_target"] > 0.02) & (grp["mid_iv_target"] < 5.0)]
        if len(grp) == 0:
            continue

        grp["w_target_label"] = grp["tau"] * grp["mid_iv_target"]**2

        # Canonicalize the in-pipeline surface definition completely:
        # from this point onward, summary stats, vega/liquidity weights,
        # and reconstruction targets all use the same mid-IV surface.
        grp["iv"] = grp["mid_iv_target"]
        grp["w"] = grp["w_target_label"]

        grp["mid"] = 0.5 * (grp["best_bid"] + grp["best_offer"])
        grp["rel_spread"] = (grp["best_offer"] - grp["best_bid"]) / grp["mid"].clip(lower=1e-4)
        grp = grp[grp["rel_spread"] <= cfg.max_rel_spread]
        if len(grp) == 0:
            continue

        grp["vega_bs"] = _compute_vega(grp, iv_col="iv")

        if "am_set_flag" in grp.columns:
            grp["am_flag"] = grp["am_set_flag"].fillna(0).astype(int)
        elif "am_settlement" in grp.columns:
            grp["am_flag"] = grp["am_settlement"].fillna(0).astype(int)
        else:
            grp["am_flag"] = 0

        grp = _apply_settlement_policy(grp, cfg.settlement_policy)
        if len(grp) == 0:
            continue

        parity_diag = _putcall_parity_diagnostics(grp)
        grp = _apply_otm_rule(grp, cfg.atm_eps)
        if len(grp) == 0:
            continue

        grp["liq_w"] = _liquidity_weight(grp)
        grp.attrs["parity_diag"] = parity_diag
        daily_quotes.setdefault(tdate_str, []).append(grp)

    result: Dict[str, dict] = {}
    for dstr, dfs in daily_quotes.items():
        df = pd.concat(dfs, ignore_index=True)
        parity_diag = dfs[0].attrs.get("parity_diag", {})
        if parity_diag.get("n_pairs", 0) > 0 and not parity_diag.get("ok", True):
            logger.warning(
                f"  {dstr}: put-call parity issue before OTM collapsing | "
                f"pairs={parity_diag['n_pairs']} median_resid={parity_diag['median_residual']:.4f} "
                f"max_resid={parity_diag['max_residual']:.4f} frac_bad={parity_diag['frac_bad']:.2%}"
            )
        result[dstr] = _pack_quote_dict(df, prices_df, dstr, cfg)

    logger.info(f"  Year {year}: {len(result)} trading days processed")
    return result


def preprocess_all(cfg: DataConfig = DATA_CFG, force_rebuild: bool = False) -> str:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    cache_path = os.path.join(cfg.cache_dir, "daily_quotes.pkl")
    if os.path.exists(cache_path) and not force_rebuild:
        logger.info(f"Cache found at {cache_path} – skipping rebuild. Pass force_rebuild=True to regenerate.")
        return cache_path

    data_dir = resolve_data_dir(cfg)
    available_years = discover_available_years(cfg, data_dir)
    if len(available_years) == 0:
        raise FileNotFoundError(f"No option parquet files found in {data_dir}")

    missing_years = sorted(set(cfg.years) - set(available_years))
    if missing_years:
        logger.warning(f"Missing option files for years: {missing_years}")

    aux = load_aux_data(cfg)
    rv_b = RVTargetBuilder(aux["prices"], cfg)
    targets_df = rv_b.build()

    all_daily: Dict[str, dict] = {}
    for year in available_years:
        all_daily.update(_process_year(year, aux, cfg))

    if len(all_daily) == 0:
        raise ValueError("Preprocessing produced zero daily quote sets. Check filters, paths, and secid.")

    targets_map = targets_df.set_index("date").to_dict("index")
    for dstr, record in all_daily.items():
        tdate = pd.Timestamp(dstr)
        if tdate in targets_map:
            trow = targets_map[tdate]
            for k in ["rv5", "rv21", "dsv5", "stress21", "rv5_back", "rv21_back", "dsv5_back"]:
                record[k] = float(trow.get(k, np.nan))
        else:
            for k in ["rv5", "rv21", "dsv5", "stress21", "rv5_back", "rv21_back", "dsv5_back"]:
                record[k] = np.nan

    vix_map = aux["prices"].set_index("date")[["VIX", "VXV"]].to_dict("index")
    for dstr, record in all_daily.items():
        tdate = pd.Timestamp(dstr)
        if tdate in vix_map:
            record["VIX"] = float(vix_map[tdate].get("VIX", np.nan))
            record["VXV"] = float(vix_map[tdate].get("VXV", np.nan))
        else:
            record["VIX"] = np.nan
            record["VXV"] = np.nan

    split_counts = {
        "train": sum(1 for d in all_daily if int(str(d)[:4]) <= cfg.train_end),
        "val": sum(1 for d in all_daily if cfg.train_end < int(str(d)[:4]) <= cfg.val_end),
        "test": sum(1 for d in all_daily if int(str(d)[:4]) > cfg.val_end),
    }
    if min(split_counts.values()) == 0:
        raise ValueError(f"Empty split detected after preprocessing: {split_counts}.")

    all_daily["__metadata__"] = {
        "stress_threshold": float(rv_b.stress_threshold),
        "available_years": available_years,
        "split_counts": split_counts,
    }

    with open(cache_path, "wb") as f:
        pickle.dump(all_daily, f, protocol=4)

    logger.info(f"Saved {len(all_daily)} daily records to {cache_path}")
    return cache_path


class OptionDataset(Dataset):
    def __init__(self, cache_path: str, split: str, cfg: DataConfig = DATA_CFG):
        with open(cache_path, "rb") as f:
            all_data = pickle.load(f)

        self.records = []
        self.metadata = all_data.get("__metadata__", {})
        for dstr, rec in sorted(all_data.items()):
            if isinstance(dstr, str) and dstr.startswith("__"):
                continue
            yr = int(str(dstr)[:4])
            if split == "train" and yr <= cfg.train_end:
                self.records.append(rec)
            elif split == "val" and cfg.train_end < yr <= cfg.val_end:
                self.records.append(rec)
            elif split == "test" and yr > cfg.val_end:
                self.records.append(rec)

        logger.info(f"OptionDataset [{split}]: {len(self.records)} days")
        if len(self.records) == 0:
            raise ValueError(f"OptionDataset [{split}] is empty. Metadata={self.metadata}.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return dict(self.records[idx])


def collate_fn(batch: List[dict]) -> dict:
    max_N = max(b["N"] for b in batch)
    B = len(batch)
    feat_dim = batch[0]["features"].shape[1]

    features = torch.zeros(B, max_N, feat_dim)
    liq_w = torch.zeros(B, max_N)
    w_target = torch.zeros(B, max_N)
    bid_px = torch.zeros(B, max_N)
    ask_px = torch.zeros(B, max_N)
    F_t = torch.zeros(B, max_N)
    K_t = torch.zeros(B, max_N)
    D_r_t = torch.zeros(B, max_N)
    tau_t = torch.zeros(B, max_N)
    k_t = torch.zeros(B, max_N)
    mask = torch.zeros(B, max_N, dtype=torch.bool)

    rv5 = torch.full((B,), float("nan"))
    rv21 = torch.full((B,), float("nan"))
    dsv5 = torch.full((B,), float("nan"))
    stress21 = torch.full((B,), float("nan"))

    rv5_back = torch.full((B,), float("nan"))
    rv21_back = torch.full((B,), float("nan"))
    dsv5_back = torch.full((B,), float("nan"))

    valid_v = torch.zeros(B, dtype=torch.bool)

    VIX_v = torch.full((B,), float("nan"))
    VXV_v = torch.full((B,), float("nan"))
    S_v = torch.full((B,), float("nan"))
    atm_iv_v = torch.full((B,), float("nan"))
    short_iv_v = torch.full((B,), float("nan"))
    term_slope_v = torch.full((B,), float("nan"))
    skew_v = torch.full((B,), float("nan"))
    quote_density_v = torch.full((B,), float("nan"))
    am_share_v = torch.full((B,), float("nan"))

    dates = []
    N_each = []

    for i, b in enumerate(batch):
        N = b["N"]
        N_each.append(N)

        features[i, :N] = torch.from_numpy(b["features"])
        liq_w[i, :N] = torch.from_numpy(b["liq_w"])
        w_target[i, :N] = torch.from_numpy(b["w_target"])
        bid_px[i, :N] = torch.from_numpy(b["bid_px"])
        ask_px[i, :N] = torch.from_numpy(b["ask_px"])
        F_t[i, :N] = torch.from_numpy(b["F"])
        K_t[i, :N] = torch.from_numpy(b["K"])
        D_r_t[i, :N] = torch.from_numpy(b["D_r"])
        tau_t[i, :N] = torch.from_numpy(b["tau"])
        k_t[i, :N] = torch.from_numpy(b["k"])
        mask[i, :N] = True

        for name, tgt in [("rv5", rv5), ("rv21", rv21), ("dsv5", dsv5), ("stress21", stress21),
                          ("rv5_back", rv5_back), ("rv21_back", rv21_back), ("dsv5_back", dsv5_back),
                          ("VIX", VIX_v), ("VXV", VXV_v), ("S", S_v), ("atm_iv", atm_iv_v),
                          ("short_iv", short_iv_v), ("term_slope", term_slope_v), ("skew_proxy", skew_v),
                          ("quote_density", quote_density_v), ("am_share", am_share_v)]:
            val = b.get(name, np.nan)
            tgt[i] = float(val) if np.isfinite(val) else float("nan")

        valid_v[i] = bool(
            np.isfinite(b.get("rv5", np.nan)) and
            np.isfinite(b.get("rv21", np.nan)) and
            np.isfinite(b.get("dsv5", np.nan)) and
            np.isfinite(b.get("stress21", np.nan))
        )
        dates.append(b["date"])

    return {
        "features": features,
        "liq_w": liq_w,
        "w_target": w_target,
        "bid_px": bid_px,
        "ask_px": ask_px,
        "F": F_t,
        "K": K_t,
        "D_r": D_r_t,
        "tau": tau_t,
        "k": k_t,
        "mask": mask,
        "rv5": rv5,
        "rv21": rv21,
        "dsv5": dsv5,
        "stress21": stress21,
        "rv5_back": rv5_back,
        "rv21_back": rv21_back,
        "dsv5_back": dsv5_back,
        "valid": valid_v,
        "VIX": VIX_v,
        "VXV": VXV_v,
        "S": S_v,
        "atm_iv": atm_iv_v,
        "short_iv": short_iv_v,
        "term_slope": term_slope_v,
        "skew_proxy": skew_v,
        "quote_density": quote_density_v,
        "am_share": am_share_v,
        "dates": dates,
        "N_each": N_each,
    }


def build_loaders(cache_path: str, cfg_data: DataConfig, cfg_train: TrainConfig):
    train_ds = OptionDataset(cache_path, "train", cfg_data)
    val_ds = OptionDataset(cache_path, "val", cfg_data)

    train_loader = DataLoader(train_ds, batch_size=cfg_train.batch_size_s1, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg_train.batch_size_s1, shuffle=False,
                            collate_fn=collate_fn, num_workers=0, pin_memory=True)
    return train_loader, val_loader


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def bs_price_from_w(F0: torch.Tensor, K0: torch.Tensor, w: torch.Tensor, D0: torch.Tensor, is_call: bool = True) -> torch.Tensor:
    eps = 1e-8
    w = w.clamp(min=eps)
    sqrt_w = w.sqrt()
    k_log = torch.log(F0.clamp(min=eps) / K0.clamp(min=eps))
    d1 = (k_log + 0.5 * w) / sqrt_w
    d2 = (k_log - 0.5 * w) / sqrt_w
    if is_call:
        price = D0 * (F0 * _normal_cdf(d1) - K0 * _normal_cdf(d2))
    else:
        price = D0 * (K0 * _normal_cdf(-d2) - F0 * _normal_cdf(-d1))
    return price.clamp(min=0.0)

class MAB(nn.Module):
    def __init__(self, d: int, h: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, X: torch.Tensor, Y: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        H, _ = self.attn(
            X, Y, Y,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        X = self.norm1(X + H)
        X = self.norm2(X + self.ff(X))
        return X


class ISAB(nn.Module):
    def __init__(self, d: int, h: int, m: int, dropout: float = 0.0):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, m, d) * 0.02)
        self.mab1 = MAB(d, h, dropout)
        self.mab2 = MAB(d, h, dropout)

    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = X.shape[0]
        I = self.I.expand(B, -1, -1)
        kpm = (~mask) if mask is not None else None
        H = self.mab1(I, X, key_padding_mask=kpm)
        Z = self.mab2(X, H)
        return Z


class PMA(nn.Module):
    def __init__(self, d: int, h: int, k_seeds: int = 1, dropout: float = 0.0):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, k_seeds, d) * 0.02)
        self.mab = MAB(d, h, dropout)
        self.rff = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = X.shape[0]
        S = self.S.expand(B, -1, -1)
        kpm = (~mask) if mask is not None else None
        return self.mab(S, self.rff(X), key_padding_mask=kpm)


class QuoteEmbedding(nn.Module):
    def __init__(self, in_dim: int, d: int, n_layers: int):
        super().__init__()
        layers = []
        cur = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(cur, d), nn.GELU()]
            cur = d
        layers.append(nn.Linear(cur, d))
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))


class SetEncoder(nn.Module):
    SUMMARY_NAMES = ["atm_iv", "skew_proxy", "term_slope", "short_iv"]

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.embed_dim
        self.embed = QuoteEmbedding(cfg.input_dim, d, cfg.embed_layers)
        self.isabs = nn.ModuleList([ISAB(d, cfg.num_heads, cfg.num_inducing) for _ in range(cfg.num_isab)])
        self.pma = PMA(d, cfg.num_heads, k_seeds=cfg.pma_seeds)
        self.proj = nn.Linear(d * cfg.pma_seeds, cfg.latent_dim)
        self.norm_z = nn.LayerNorm(cfg.latent_dim)
        self.summary_heads = nn.ModuleDict({
            name: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
            for name in self.SUMMARY_NAMES
        })

    def encode_full(self, features: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        X = self.embed(features)
        for isab in self.isabs:
            X = isab(X, mask)
        seed_tokens = self.pma(X, mask)
        B = seed_tokens.shape[0]
        z = self.norm_z(self.proj(seed_tokens.reshape(B, -1)))
        summary_preds = {}
        for j, name in enumerate(self.SUMMARY_NAMES):
            idx = min(j, seed_tokens.shape[1] - 1)
            summary_preds[name] = self.summary_heads[name](seed_tokens[:, idx, :]).squeeze(-1)
        return {"z": z, "quote_memory": X, "seed_tokens": seed_tokens, "summary_preds": summary_preds}

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.encode_full(features, mask)["z"]


class SurfaceDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig, eps: float = 1e-4):
        super().__init__()
        cur = cfg.latent_dim + cfg.decoder_query_dim
        h = cfg.decoder_hidden
        layers = []
        for _ in range(cfg.decoder_layers - 1):
            layers += [nn.Linear(cur, h), nn.GELU()]
            cur = h
        layers.append(nn.Linear(cur, 1))
        self.net = nn.Sequential(*layers)
        self.eps = eps

    def forward(self, z: torch.Tensor, k: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        if k.dim() == 1:
            k = k.unsqueeze(0).expand(z.shape[0], -1)
            tau = tau.unsqueeze(0).expand(z.shape[0], -1)
        B, n_q = k.shape
        z_exp = z.unsqueeze(1).expand(-1, n_q, -1)
        query = torch.stack([k, tau], dim=-1)
        inp = torch.cat([z_exp, query], dim=-1)
        out = self.net(inp).squeeze(-1)
        return F.softplus(out) + self.eps


class QueryConditionedResidual(nn.Module):
    LOCAL_IDX = [0, 1, 3, 4, 5, 6, 7]

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.embed_dim
        self.quote_embed = QuoteEmbedding(cfg.local_input_dim, d, max(2, cfg.embed_layers - 1))
        self.query_proj = nn.Sequential(nn.Linear(cfg.latent_dim + 2, d), nn.GELU(), nn.Linear(d, d))
        self.attn = nn.MultiheadAttention(d, cfg.num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, cfg.local_hidden), nn.GELU(), nn.Linear(cfg.local_hidden, d))
        self.norm2 = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)
        self.scale = cfg.local_residual_scale

    def _build_memory(self, features: torch.Tensor) -> torch.Tensor:
        return self.quote_embed(features[:, :, self.LOCAL_IDX])

    def forward(self, z: torch.Tensor, k: torch.Tensor, tau: torch.Tensor,
                features: torch.Tensor, quote_mask: torch.Tensor) -> torch.Tensor:
        if k.dim() == 1:
            k = k.unsqueeze(0).expand(z.shape[0], -1)
            tau = tau.unsqueeze(0).expand(z.shape[0], -1)
        B, n_q = k.shape
        z_exp = z.unsqueeze(1).expand(-1, n_q, -1)
        q_inp = torch.cat([z_exp, torch.stack([k, tau], dim=-1)], dim=-1)
        q = self.query_proj(q_inp)
        memory = self._build_memory(features)
        ctx, _ = self.attn(
            q, memory, memory,
            key_padding_mask=~quote_mask,
            need_weights=False,
        )
        h = self.norm1(q + ctx)
        h = self.norm2(h + self.ff(h))
        return self.scale * torch.tanh(self.out(h).squeeze(-1))


class FullSurfaceModel(nn.Module):
    def __init__(self, model_cfg: ModelConfig = MODEL_CFG):
        super().__init__()
        self.encoder = SetEncoder(model_cfg)
        self.global_decoder = SurfaceDecoder(model_cfg)
        self.local_decoder = QueryConditionedResidual(model_cfg)
        self.cfg = model_cfg

    def encode(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(features, mask)

    def encode_full(self, features: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.encoder.encode_full(features, mask)

    def decode(self, z: torch.Tensor, k: torch.Tensor, tau: torch.Tensor,
               features: Optional[torch.Tensor] = None, quote_mask: Optional[torch.Tensor] = None,
               use_local: bool = False) -> torch.Tensor:
        w_global = self.global_decoder(z, k, tau)
        if (not use_local) or (features is None) or (quote_mask is None):
            return w_global
        delta_log = self.local_decoder(z, k, tau, features, quote_mask)
        return w_global * torch.exp(delta_log)

    def decode_global(self, z: torch.Tensor, k: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return self.global_decoder(z, k, tau)

    def reconstruct_observed(self, features: torch.Tensor, encoder_mask: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        enc = self.encode_full(features, encoder_mask)
        z = enc["z"]
        k_q = features[:, :, 0]
        tau_q = features[:, :, 1]
        w_hat = self.decode(z, k_q, tau_q, features=features, quote_mask=encoder_mask, use_local=True)
        return enc, w_hat

    def forward(self, features: torch.Tensor, mask: torch.Tensor, k_query: torch.Tensor, tau_query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.encode_full(features, mask)
        z = enc["z"]
        w_hat = self.decode(z, k_query, tau_query, use_local=False)
        return z, w_hat


class EMAScaler:
    def __init__(self, names, decay: float = 0.99, eps: float = 1e-6):
        self.decay = decay
        self.eps = eps
        self._ema = {n: 1.0 for n in names}
        self._init = {n: False for n in names}

    def scale(self, losses: Dict[str, torch.Tensor], update: bool = True) -> Dict[str, torch.Tensor]:
        scaled = {}
        for name, val in losses.items():
            v = val.item() if isinstance(val, torch.Tensor) else float(val)
            if update:
                if not self._init[name]:
                    self._ema[name] = v + self.eps
                    self._init[name] = True
                else:
                    self._ema[name] = self.decay * self._ema[name] + (1 - self.decay) * v
                denom = self._ema[name] + self.eps
            else:
                denom = (self._ema[name] + self.eps) if self._init[name] else (max(v, self.eps) + self.eps)
            scaled[name] = val / denom
        return scaled

    def state_dict(self) -> dict:
        return {"ema": self._ema, "init": self._init}

    def load_state_dict(self, d: dict):
        self._ema = d["ema"]
        self._init = d["init"]


def reconstruction_loss(w_hat: torch.Tensor, w_true: torch.Tensor, liq_w: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    sq_err = (w_hat - w_true) ** 2
    weighted = sq_err * liq_w * mask.float()
    denom = liq_w * mask.float()
    return weighted.sum() / (denom.sum() + 1e-8)


def mask_recovery_loss(w_hat: torch.Tensor, w_true: torch.Tensor, liq_w: torch.Tensor, mask_quotes: torch.Tensor) -> torch.Tensor:
    sq_err = (w_hat - w_true) ** 2
    weights = liq_w * mask_quotes.float()
    return (sq_err * weights).sum() / (weights.sum() + 1e-8)


def bidask_loss(w_hat: torch.Tensor, F0: torch.Tensor, K0: torch.Tensor, D_r: torch.Tensor,
                tau: torch.Tensor, k: torch.Tensor, bid_px: torch.Tensor, ask_px: torch.Tensor,
                mask: torch.Tensor, cp_flag: torch.Tensor, lambda_cover: float = 0.20) -> torch.Tensor:
    is_call = cp_flag > 0.5
    w_hat_c = w_hat.clamp(min=1e-8)
    price_call = bs_price_from_w(F0, K0, w_hat_c, D_r, is_call=True)
    price_put = bs_price_from_w(F0, K0, w_hat_c, D_r, is_call=False)
    price_hat = torch.where(is_call, price_call, price_put)
    spread = (ask_px - bid_px).clamp(min=1e-4)
    mid = 0.5 * (bid_px + ask_px)
    sigma = 0.5 * spread
    nll = 0.5 * ((price_hat - mid) / sigma) ** 2 + torch.log(sigma)
    above_ask = torch.relu(price_hat - ask_px) / spread
    below_bid = torch.relu(bid_px - price_hat) / spread
    cover = above_ask ** 2 + below_bid ** 2
    loss = (nll + lambda_cover * cover) * mask.float()
    return loss.sum() / (mask.float().sum() + 1e-8)


def structured_summary_loss(summary_preds: Dict[str, torch.Tensor], batch: dict, device: torch.device) -> torch.Tensor:
    losses = []
    for name in ["atm_iv", "skew_proxy", "term_slope", "short_iv"]:
        target = batch[name].to(device)
        pred = summary_preds[name]
        valid = torch.isfinite(target)
        if valid.any():
            losses.append(F.mse_loss(pred[valid], target[valid]))
    if len(losses) == 0:
        return torch.tensor(0.0, device=device)
    return sum(losses) / len(losses)


def seed_orthogonality_loss(seed_tokens: torch.Tensor) -> torch.Tensor:
    S = F.normalize(seed_tokens, dim=-1)
    G = torch.bmm(S, S.transpose(1, 2))
    I = torch.eye(S.shape[1], device=S.device).unsqueeze(0)
    return ((G - I) ** 2).mean()


def bootstrap_consistency_loss(model: FullSurfaceModel, features: torch.Tensor, full_mask: torch.Tensor) -> torch.Tensor:
    rand = torch.rand_like(full_mask.float())
    mask2 = full_mask & (rand > 0.25)
    no_vis = ~mask2.any(dim=1)
    mask2[no_vis] = full_mask[no_vis]
    z1 = model.encode(features, full_mask)
    z2 = model.encode(features, mask2)
    invariance = (1.0 - F.cosine_similarity(z1, z2, dim=1)).mean()
    std = torch.sqrt(z1.var(dim=0, unbiased=False) + 1e-4)
    variance_reg = F.relu(1.0 - std).mean()
    return invariance + 0.10 * variance_reg


def noarb_penalty(model: FullSurfaceModel, z: torch.Tensor,
                  tau_obs: torch.Tensor, F_obs: torch.Tensor, D_obs: torch.Tensor,
                  S_day: torch.Tensor, moneyness_nodes: torch.Tensor, tau_grid: torch.Tensor,
                  ref_tau_days: int = 30) -> torch.Tensor:
    device = z.device
    B = z.shape[0]
    m_nodes_np = moneyness_nodes.detach().cpu().numpy().astype(float)
    tau_grid_np = tau_grid.detach().cpu().numpy().astype(float)

    n_K = len(m_nodes_np)
    n_T = len(tau_grid_np)

    F_grid_np = np.zeros((B, n_T), dtype=np.float32)
    D_grid_np = np.zeros((B, n_T), dtype=np.float32)
    K_fixed_np = np.zeros((B, n_K), dtype=np.float32)

    tau_obs_np = tau_obs.detach().cpu().numpy()
    F_obs_np = F_obs.detach().cpu().numpy()
    D_obs_np = D_obs.detach().cpu().numpy()
    S_day_np = S_day.detach().cpu().numpy()

    for b in range(B):
        F_tau_b, D_tau_b, K_fix_b = _build_noarb_reference_grids_np(
            tau_obs=tau_obs_np[b],
            F_obs=F_obs_np[b],
            D_obs=D_obs_np[b],
            S_ref=float(S_day_np[b]),
            moneyness_nodes=m_nodes_np,
            tau_grid=tau_grid_np,
            ref_tau_days=ref_tau_days,
        )
        F_grid_np[b] = F_tau_b
        D_grid_np[b] = D_tau_b
        K_fixed_np[b] = K_fix_b

    F_grid = torch.tensor(F_grid_np, device=device)
    D_grid = torch.tensor(D_grid_np, device=device)
    K_fixed = torch.tensor(K_fixed_np, device=device)

    K_grid = K_fixed.unsqueeze(-1).expand(B, n_K, n_T)
    F_exp = F_grid.unsqueeze(1).expand(B, n_K, n_T)
    D_exp = D_grid.unsqueeze(1).expand(B, n_K, n_T)

    k_query = torch.log(K_grid / F_exp.clamp(min=1e-8))
    tau_query = tau_grid.view(1, 1, n_T).expand(B, n_K, n_T)

    w_hat = model.decode(z, k_query.reshape(B, -1), tau_query.reshape(B, -1)).reshape(B, n_K, n_T)
    C = bs_price_from_w(F_exp, K_grid, w_hat, D_exp, is_call=True)

    # Calendar monotonicity: fixed strike, later maturity call price should not be lower.
    cal_viol = torch.relu(C[:, :, :-1] - C[:, :, 1:])
    L_cal = (cal_viol ** 2).mean()

    # Butterfly convexity in strike.
    if n_K >= 3:
        delta1 = K_grid[:, 1:-1, :] - K_grid[:, :-2, :]
        delta2 = K_grid[:, 2:, :] - K_grid[:, 1:-1, :]
        term_left = (C[:, 1:-1, :] - C[:, :-2, :]) / delta1.clamp(min=1e-8)
        term_right = (C[:, 2:, :] - C[:, 1:-1, :]) / delta2.clamp(min=1e-8)
        bfly = term_right - term_left
        L_bfly = (torch.relu(-bfly) ** 2).mean()
    else:
        L_bfly = torch.tensor(0.0, device=device)

    return L_cal + L_bfly


class SurfaceLoss(nn.Module):
    def __init__(self, train_cfg: TrainConfig = TRAIN_CFG, data_cfg: DataConfig = DATA_CFG):
        super().__init__()
        self.cfg = train_cfg
        self.scaler = EMAScaler(
            names=["rec", "ba", "arb", "mask", "struct", "contrast"],
            decay=train_cfg.ema_decay,
            eps=train_cfg.ema_eps
        )

        m_nodes = np.asarray(sorted(data_cfg.moneyness_nodes), dtype=np.float32)
        tau_nodes = (np.array(data_cfg.maturity_nodes_days) / 365.0).astype(np.float32)

        self.register_buffer("moneyness_nodes", torch.from_numpy(m_nodes))
        self.register_buffer("tau_grid", torch.from_numpy(tau_nodes))
        self.noarb_ref_days = int(data_cfg.noarb_ref_days)

        # --- lazy no-arb computation (train-only) ---
        # Val and deterministic passes always recompute. Training steps recompute
        # every `noarb_every` steps and reuse the previous detached value in between,
        # yielding ~25% wall-clock reduction per epoch.
        self.noarb_every = 4
        self._arb_step = 0
        self._last_arb: Optional[torch.Tensor] = None

    def forward(self, model: FullSurfaceModel, batch: dict, device: torch.device,
                deterministic: bool = False, update_scaler: bool = True) -> Tuple[torch.Tensor, dict]:
        feat = batch["features"].to(device)
        mask = batch["mask"].to(device)
        liq_w = batch["liq_w"].to(device)
        w_true = batch["w_target"].to(device)
        bid_px = batch["bid_px"].to(device)
        ask_px = batch["ask_px"].to(device)
        F_q = batch["F"].to(device)
        K_q = batch["K"].to(device)
        D_q = batch["D_r"].to(device)
        tau_q = batch["tau"].to(device)
        k_q = batch["k"].to(device)
        cp_flag = feat[:, :, 7]

        if deterministic:
            ratio = 0.5 * (self.cfg.mask_ratio_min + self.cfg.mask_ratio_max)
            B, N = mask.shape
            row_idx = torch.arange(B, device=device, dtype=torch.float32).unsqueeze(1)
            col_idx = torch.arange(N, device=device, dtype=torch.float32).unsqueeze(0)
            pseudo = torch.frac(torch.sin((row_idx + 1.0) * 12.9898 + (col_idx + 1.0) * 78.233) * 43758.5453)
            rand = pseudo.expand_as(mask.float())
        else:
            ratio = np.random.uniform(self.cfg.mask_ratio_min, self.cfg.mask_ratio_max)
            rand = torch.rand_like(mask.float())

        quote_mask = mask & (rand > ratio)
        masked_q = mask & (rand <= ratio)
        no_vis = ~quote_mask.any(dim=1)
        quote_mask[no_vis] = mask[no_vis]
        masked_q[no_vis] = False

        enc, w_hat = model.reconstruct_observed(feat, quote_mask)
        z = enc["z"]

        L_rec = reconstruction_loss(w_hat, w_true, liq_w, quote_mask)
        L_mask = mask_recovery_loss(w_hat, w_true, liq_w, masked_q)
        L_ba = bidask_loss(
            w_hat, F_q, K_q, D_q, tau_q, k_q, bid_px, ask_px, mask, cp_flag,
            lambda_cover=self.cfg.lambda_ba_cover
        )
        # No-arb penalty is expensive (O(B) Python loop with np.quantile and
        # log-linear curve interpolation). In deterministic/val mode we always
        # compute it. In training mode we recompute every `noarb_every` steps
        # and reuse the previous detached value in between. Reusing a detached
        # tensor means no gradient flows through L_arb on skipped steps, but
        # the model still receives a fresh finance-consistency signal every
        # `noarb_every` steps, which is enough to keep calendar/butterfly
        # violations at near-zero rates in practice.
        need_arb = deterministic or (self._last_arb is None) or (self._arb_step % self.noarb_every == 0)
        if need_arb:
            L_arb = noarb_penalty(
                model=model,
                z=z,
                tau_obs=tau_q,
                F_obs=F_q,
                D_obs=D_q,
                S_day=batch["S"].to(device),
                moneyness_nodes=self.moneyness_nodes,
                tau_grid=self.tau_grid,
                ref_tau_days=self.noarb_ref_days,
            )
            if not deterministic:
                self._last_arb = L_arb.detach()
        else:
            L_arb = self._last_arb
        if not deterministic:
            self._arb_step += 1

        L_struct = structured_summary_loss(enc["summary_preds"], batch, device) + 0.05 * seed_orthogonality_loss(
            enc["seed_tokens"])
        L_contrast = torch.tensor(0.0, device=device) if deterministic else bootstrap_consistency_loss(model, feat,
                                                                                                       quote_mask)
        raw = {
            "rec": L_rec,
            "ba": L_ba,
            "arb": L_arb,
            "mask": L_mask,
            "struct": L_struct,
            "contrast": L_contrast
        }
        scaled = self.scaler.scale(raw, update=update_scaler)

        total = (
            self.cfg.lambda_rec * scaled["rec"] +
            self.cfg.lambda_ba * scaled["ba"] +
            self.cfg.lambda_arb * scaled["arb"] +
            self.cfg.lambda_mask * scaled["mask"] +
            self.cfg.lambda_struct * scaled["struct"] +
            self.cfg.lambda_contrast * scaled["contrast"]
        )

        info = {k: float(v.item()) for k, v in raw.items()}
        info["total"] = float(total.item())
        return total, info



class RegressionHead(nn.Module):
    def __init__(self, d_in: int, hidden: int, n_layers: int, eps: float = 1e-8):
        super().__init__()
        layers = []
        cur = d_in
        for _ in range(n_layers - 1):
            layers += [nn.Linear(cur, hidden), nn.GELU()]
            cur = hidden
        layers.append(nn.Linear(cur, 1))
        self.net = nn.Sequential(*layers)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x).squeeze(-1)
        return F.softplus(raw) + self.eps


class ClassificationHead(nn.Module):
    def __init__(self, d_in: int, hidden: int, n_layers: int):
        super().__init__()
        layers = []
        cur = d_in
        for _ in range(n_layers - 1):
            layers += [nn.Linear(cur, hidden), nn.GELU()]
            cur = hidden
        layers.append(nn.Linear(cur, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DownstreamHeads(nn.Module):
    def __init__(self, model_cfg: ModelConfig = MODEL_CFG):
        super().__init__()
        d = model_cfg.latent_dim
        xdim = d + 2 + d + d
        h = model_cfg.head_hidden
        nl = model_cfg.head_layers
        self.rv5 = RegressionHead(xdim, h, nl)
        self.rv21 = RegressionHead(xdim, h, nl)
        self.dsv5 = RegressionHead(xdim, h, nl)
        self.stress21 = ClassificationHead(xdim, h, nl)

    def _make_input(self, z: torch.Tensor, side: Optional[torch.Tensor]) -> torch.Tensor:
        B = z.shape[0]
        if side is None:
            side = torch.full((B, 2), 20.0, device=z.device, dtype=z.dtype)
        # Fill NaN / inf with training-set mean (~20) rather than 0.
        # Zero would misrepresent "missing" as "ultra-low volatility",
        # landing the regime gate in a region the head has never seen.
        side = torch.where(
            torch.isnan(side) | torch.isinf(side),
            torch.full_like(side, 20.0),
            side,
        )
        # Center + rescale VIX/VXV to O(1) so that the interaction term
        # z * vix has the same magnitude as z itself, instead of being
        # dominated by the raw VIX level (typically 15-80).
        vix = (side[:, 0:1] - 20.0) / 15.0
        vxv = (side[:, 1:2] - 20.0) / 15.0
        return torch.cat([z, vix, vxv, z * vix, z * vxv], dim=1)

    def forward(self, z: torch.Tensor, side: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x = self._make_input(z, side)
        return {
            "rv5": self.rv5(x),
            "rv21": self.rv21(x),
            "dsv5": self.dsv5(x),
            "stress21": self.stress21(x),
        }


def downstream_loss(preds: Dict[str, torch.Tensor], batch: dict, device: torch.device) -> Tuple[torch.Tensor, dict]:
    if "valid" in batch:
        valid = batch["valid"].to(device=device, dtype=torch.bool)
    else:
        valid = (~torch.isnan(batch["rv5"]) & ~torch.isnan(batch["rv21"]) &
                 ~torch.isnan(batch["dsv5"]) & ~torch.isnan(batch["stress21"])).to(device)
    if not valid.any():
        zero = torch.tensor(0.0, device=device)
        return zero, {"rv5": 0.0, "rv21": 0.0, "dsv5": 0.0, "stress": 0.0}

    def reg_loss(key: str):
        y_hat = preds[key][valid]
        y = batch[key].to(device)[valid]
        return F.huber_loss(y_hat, y, delta=0.5)

    L_rv5 = reg_loss("rv5")
    L_rv21 = reg_loss("rv21")
    L_dsv5 = reg_loss("dsv5")
    logit = preds["stress21"][valid]
    y_cls = batch["stress21"].to(device)[valid]
    L_cls = F.binary_cross_entropy_with_logits(logit, y_cls)

    total = L_rv5 + L_rv21 + L_dsv5 + L_cls
    return total, {"rv5": L_rv5.item(), "rv21": L_rv21.item(), "dsv5": L_dsv5.item(), "stress": L_cls.item()}


def train_epoch(model: FullSurfaceModel, loss_fn: SurfaceLoss, loader: DataLoader,
                optimizer: torch.optim.Optimizer, device: torch.device, grad_clip: float) -> dict:
    model.train()
    accum = {}
    n_batches = 0
    n_oom = 0

    for batch in tqdm(loader, desc="Stage1 train", leave=False):
        try:
            optimizer.zero_grad()
            total, info = loss_fn(model, batch, device)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            for k, v in info.items():
                accum[k] = accum.get(k, 0.0) + v
            n_batches += 1
        except torch.cuda.OutOfMemoryError:
            n_oom += 1
            optimizer.zero_grad()
            # Free any partially-computed tensors from the failed forward/backward.
            for _name in ("total", "info"):
                if _name in locals():
                    del locals()[_name]
            torch.cuda.empty_cache()
            logger.warning(
                f"OOM skipped, max_N={batch['features'].shape[1]}, "
                f"cumulative OOMs this epoch: {n_oom}"
            )
            continue

    if n_oom > 0:
        logger.info(f"Epoch finished with {n_oom} OOM batch(es) skipped out of {n_batches + n_oom}.")

    return {k: v / max(n_batches, 1) for k, v in accum.items()}

@torch.no_grad()
def val_epoch(model: FullSurfaceModel, loss_fn: SurfaceLoss, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    accum = {}
    n_batches = 0
    for batch in loader:
        _, info = loss_fn(model, batch, device, deterministic=True, update_scaler=False)
        for k, v in info.items():
            accum[k] = accum.get(k, 0.0) + v
        n_batches += 1
    return {k: v / n_batches for k, v in accum.items()}


def save_checkpoint(state: dict, path: str):
    torch.save(state, path)
    logger.info(f"  Checkpoint saved → {path}")


def load_checkpoint(path: str, model: FullSurfaceModel, optimizer=None, device: str = "cpu") -> int:
    ckpt = torch.load(path, map_location=device)
    if "model" not in ckpt:
        raise ValueError(f"{path} is not a valid stage-1 checkpoint. Keys: {list(ckpt.keys())}")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    epoch = ckpt.get("epoch", 0)
    logger.info(f"Loaded checkpoint from epoch {epoch}: {path}")
    return epoch


def train_stage1(cfg_data: DataConfig = DATA_CFG, cfg_model: ModelConfig = MODEL_CFG,
                 cfg_train: TrainConfig = TRAIN_CFG, resume: bool = False):
    make_dirs(cfg_data)
    set_seed(cfg_train.seed)
    device = torch.device(cfg_train.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    cache_path = os.path.join(cfg_data.cache_dir, "daily_quotes.pkl")
    train_loader, val_loader = build_loaders(cache_path, cfg_data, cfg_train)

    model = FullSurfaceModel(cfg_model).to(device)
    loss_fn = SurfaceLoss(cfg_train, cfg_data).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(loss_fn.parameters()),
                                  lr=cfg_train.lr_s1, weight_decay=cfg_train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg_train.epochs_s1,
                                                           eta_min=cfg_train.lr_s1 * 0.01)

    start_epoch = 0
    best_val = float("inf")
    ckpt_dir = os.path.join(cfg_data.results_dir, "checkpoints")

    if resume:
        last = os.path.join(ckpt_dir, "stage1_last.pt")
        if os.path.exists(last):
            start_epoch = load_checkpoint(last, model, optimizer, str(device))
            ckpt_last = torch.load(last, map_location=str(device))
            if "loss_scaler" in ckpt_last:
                loss_fn.scaler.load_state_dict(ckpt_last["loss_scaler"])
                logger.info("EMAScaler state restored from checkpoint.")
            best_path = os.path.join(ckpt_dir, "stage1_best.pt")
            if os.path.exists(best_path):
                best_val = torch.load(best_path, map_location=str(device)).get("best_val", float("inf"))

    history = []
    for epoch in range(start_epoch, cfg_train.epochs_s1):
        t0 = time.time()
        train_info = train_epoch(model, loss_fn, train_loader, optimizer, device, cfg_train.grad_clip)
        val_info = val_epoch(model, loss_fn, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        val_loss = val_info.get("total", val_info.get("rec", 0.0))
        lr_now = optimizer.param_groups[0]["lr"]

        train_str = "  ".join(f"tr_{k}={v:.4f}" for k, v in train_info.items())
        val_str = "  ".join(f"va_{k}={v:.4f}" for k, v in val_info.items())
        logger.info(f"[S1 epoch {epoch+1:3d}/{cfg_train.epochs_s1}] lr={lr_now:.5f}  {train_str}  {val_str}  ({elapsed:.1f}s)")

        row = {"epoch": epoch + 1, "lr": lr_now}
        row.update({f"train_{k}": v for k, v in train_info.items()})
        row.update({f"val_{k}": v for k, v in val_info.items()})
        history.append(row)

        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "loss_scaler": loss_fn.scaler.state_dict(),
            "best_val": best_val,
            "history": history,
        }

        if (epoch + 1) % cfg_train.ckpt_every == 0:
            save_checkpoint(state, os.path.join(ckpt_dir, f"stage1_ep{epoch+1:04d}.pt"))
        save_checkpoint(state, os.path.join(ckpt_dir, "stage1_last.pt"))

        if val_loss < best_val:
            best_val = val_loss
            state["best_val"] = best_val
            save_checkpoint(state, os.path.join(ckpt_dir, "stage1_best.pt"))
            logger.info(f"  ↑ New best val loss: {best_val:.6f}")

    logger.info(f"Stage 1 complete.  Best val total loss = {best_val:.6f}")

    hist_path = os.path.join(cfg_data.results_dir, "stage1_history.csv")
    if history:
        with open(hist_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=history[0].keys())
            w.writeheader()
            w.writerows(history)
        logger.info(f"History saved to {hist_path}")

    return model, history


@torch.no_grad()
def _surface_noarb_counts_one_day(model: FullSurfaceModel, z_i: torch.Tensor,
                                  tau_obs_i: torch.Tensor, F_obs_i: torch.Tensor, D_obs_i: torch.Tensor,
                                  S_day_i,
                                  device: torch.device, cfg_data: DataConfig, tol: float = 1e-8) -> tuple[int, int, int, int]:
    tau_obs = tau_obs_i.detach().cpu().numpy().astype(float)
    F_obs = F_obs_i.detach().cpu().numpy().astype(float)
    D_obs = D_obs_i.detach().cpu().numpy().astype(float)
    S_day = float(S_day_i.detach().cpu().item()) if torch.is_tensor(S_day_i) else float(S_day_i)

    valid = (
        np.isfinite(tau_obs) & np.isfinite(F_obs) &
        np.isfinite(D_obs) & (tau_obs > 0.0) & (F_obs > 0.0) & (D_obs > 0.0)
    )
    if valid.sum() < 2:
        return 0, 0, 0, 0

    tau_grid = np.asarray(cfg_data.maturity_nodes_days, dtype=float) / 365.0
    F_tau, D_tau, K_fixed = _build_noarb_reference_grids_np(
        tau_obs=tau_obs[valid],
        F_obs=F_obs[valid],
        D_obs=D_obs[valid],
        S_ref=S_day,
        moneyness_nodes=np.asarray(cfg_data.moneyness_nodes, dtype=float),
        tau_grid=tau_grid,
        ref_tau_days=cfg_data.noarb_ref_days,
    )

    if len(K_fixed) < 3 or len(tau_grid) < 2:
        return 0, 0, 0, 0

    k_query = np.log(K_fixed[None, :] / np.maximum(F_tau[:, None], 1e-8))
    tau_query = np.broadcast_to(tau_grid[:, None], k_query.shape)

    k_t = torch.tensor(k_query, dtype=torch.float32, device=device).unsqueeze(0)
    tau_t = torch.tensor(tau_query, dtype=torch.float32, device=device).unsqueeze(0)
    z_t = z_i.unsqueeze(0)

    w_hat = model.decode(z_t, k_t.reshape(1, -1), tau_t.reshape(1, -1)).reshape(1, len(tau_grid), len(K_fixed)).squeeze(0)

    F_t = torch.tensor(F_tau[:, None], dtype=torch.float32, device=device)
    K_t = torch.tensor(K_fixed[None, :], dtype=torch.float32, device=device)
    D_t = torch.tensor(D_tau[:, None], dtype=torch.float32, device=device)

    C = bs_price_from_w(F_t, K_t, w_hat, D_t, is_call=True).detach().cpu().numpy()  # [n_T, n_K]

    cal_bad_mask = C[1:, :] + tol < C[:-1, :]
    cal_bad = int(cal_bad_mask.sum())
    cal_total = int(cal_bad_mask.size)

    delta1 = K_fixed[1:-1] - K_fixed[:-2]
    delta2 = K_fixed[2:] - K_fixed[1:-1]
    term_left = (C[:, 1:-1] - C[:, :-2]) / np.maximum(delta1[None, :], 1e-8)
    term_right = (C[:, 2:] - C[:, 1:-1]) / np.maximum(delta2[None, :], 1e-8)
    bfly_measure = term_right - term_left
    bfly_bad_mask = bfly_measure < -tol
    bfly_bad = int(bfly_bad_mask.sum())
    bfly_total = int(bfly_bad_mask.size)

    return cal_bad, cal_total, bfly_bad, bfly_total


@torch.no_grad()
def evaluate_reconstruction(model: FullSurfaceModel, loader: DataLoader, device: torch.device,
                            cfg_data: DataConfig = DATA_CFG) -> dict:
    model.eval()

    price_sq_err_sum = 0.0
    iv_sq_err_sum = 0.0
    ba_viol_sum = 0.0
    n_quotes_total = 0.0

    cal_bad_total = 0
    cal_total = 0
    bfly_bad_total = 0
    bfly_total = 0

    within_half_spread_sum = 0.0
    within_one_spread_sum = 0.0
    norm_exceed_all = []

    for batch in loader:
        feat = batch["features"].to(device)
        mask = batch["mask"].to(device)
        w_true = batch["w_target"].to(device)
        tau_q = batch["tau"].to(device)
        F_q = batch["F"].to(device)
        K_q = batch["K"].to(device)
        D_q = batch["D_r"].to(device)
        bid_px = batch["bid_px"].to(device)
        ask_px = batch["ask_px"].to(device)

        enc, w_hat = model.reconstruct_observed(features=feat, encoder_mask=mask)
        z = enc["z"]

        cp_flag = feat[:, :, 7]
        is_call = cp_flag > 0.5
        price_call = bs_price_from_w(F_q, K_q, w_hat, D_q, is_call=True)
        price_put = bs_price_from_w(F_q, K_q, w_hat, D_q, is_call=False)
        price_hat = torch.where(is_call, price_call, price_put)

        mid_true = 0.5 * (bid_px + ask_px)
        m = mask.float()
        price_sq_err_sum += (((price_hat - mid_true) ** 2) * m).sum().item()

        iv_hat = (w_hat / tau_q.clamp(min=1e-6)).sqrt()
        iv_true = (w_true / tau_q.clamp(min=1e-6)).sqrt()
        iv_sq_err_sum += (((iv_hat - iv_true) ** 2) * m).sum().item()

        ba_viol = ((price_hat < bid_px) | (price_hat > ask_px)).float() * m
        ba_viol_sum += ba_viol.sum().item()

        spread = (ask_px - bid_px).clamp(min=1e-4)
        abs_dev = (price_hat - mid_true).abs()

        within_half = (abs_dev <= 0.5 * spread).float() * m
        within_one = (abs_dev <= 1.0 * spread).float() * m
        within_half_spread_sum += within_half.sum().item()
        within_one_spread_sum += within_one.sum().item()

        above_ask = torch.relu(price_hat - ask_px)
        below_bid = torch.relu(bid_px - price_hat)
        norm_exceed = torch.maximum(above_ask, below_bid) / spread
        norm_exceed_valid = norm_exceed[mask].detach().cpu().numpy()
        if norm_exceed_valid.size > 0:
            norm_exceed_all.append(norm_exceed_valid)

        n_quotes_total += m.sum().item()

        B = feat.shape[0]
        for b in range(B):
            mb = mask[b]
            if int(mb.sum().item()) < 8:
                continue
            cb, ct, bb, bt = _surface_noarb_counts_one_day(
                model=model,
                z_i=z[b],
                tau_obs_i=tau_q[b, mb],
                F_obs_i=F_q[b, mb],
                D_obs_i=D_q[b, mb],
                S_day_i=batch["S"][b],
                device=device,
                cfg_data=cfg_data,
            )
            cal_bad_total += cb
            cal_total += ct
            bfly_bad_total += bb
            bfly_total += bt

    if n_quotes_total <= 0:
        return {k: np.nan for k in [
            "price_rmse", "iv_rmse", "ba_violation_rate",
            "ba_within_half_spread_rate", "ba_within_one_spread_rate",
            "ba_norm_exceed_mean", "ba_norm_exceed_q50", "ba_norm_exceed_q90",
            "calendar_violation_rate", "butterfly_violation_rate",
            "n_quotes", "n_days",
        ]}

    norm_exceed_all = np.concatenate(norm_exceed_all) if len(norm_exceed_all) > 0 else np.array([], dtype=float)

    return {
        "price_rmse": float(np.sqrt(price_sq_err_sum / n_quotes_total)),
        "iv_rmse": float(np.sqrt(iv_sq_err_sum / n_quotes_total)),
        "ba_violation_rate": float(ba_viol_sum / n_quotes_total),
        "ba_within_half_spread_rate": float(within_half_spread_sum / n_quotes_total),
        "ba_within_one_spread_rate": float(within_one_spread_sum / n_quotes_total),
        "ba_norm_exceed_mean": float(np.mean(norm_exceed_all)) if norm_exceed_all.size > 0 else np.nan,
        "ba_norm_exceed_q50": float(np.quantile(norm_exceed_all, 0.50)) if norm_exceed_all.size > 0 else np.nan,
        "ba_norm_exceed_q90": float(np.quantile(norm_exceed_all, 0.90)) if norm_exceed_all.size > 0 else np.nan,
        "calendar_violation_rate": float(cal_bad_total / cal_total) if cal_total > 0 else np.nan,
        "butterfly_violation_rate": float(bfly_bad_total / bfly_total) if bfly_total > 0 else np.nan,
        "n_quotes": int(n_quotes_total),
        "n_days": int(len(loader.dataset)) if hasattr(loader, "dataset") else np.nan,
    }


def evaluate_surface_checkpoint(checkpoint_name: str, cache_path: str,
                                data_cfg: DataConfig = DATA_CFG, model_cfg: ModelConfig = MODEL_CFG,
                                device: Optional[str] = None) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    ckpt_path = os.path.join(data_cfg.results_dir, "checkpoints", checkpoint_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    surface_model = FullSurfaceModel(model_cfg).to(device_t)
    ck = torch.load(ckpt_path, map_location=device_t)

    if checkpoint_name.startswith("stage3"):
        if "surface" not in ck:
            raise ValueError(f"{checkpoint_name} expected key 'surface', got {list(ck.keys())}")
        surface_model.load_state_dict(ck["surface"])
        logger.info(f"[surface-only eval] Loaded Stage 3 checkpoint: {checkpoint_name}")
    elif checkpoint_name.startswith("stage1"):
        if "model" not in ck:
            raise ValueError(f"{checkpoint_name} expected key 'model', got {list(ck.keys())}")
        surface_model.load_state_dict(ck["model"])
        logger.info(f"[surface-only eval] Loaded Stage 1 checkpoint: {checkpoint_name}")
    else:
        if "surface" in ck:
            surface_model.load_state_dict(ck["surface"])
        elif "model" in ck:
            surface_model.load_state_dict(ck["model"])
        else:
            raise ValueError(f"Unknown checkpoint format in {checkpoint_name}. Keys: {list(ck.keys())}")

    test_ds = OptionDataset(cache_path, split="test", cfg=data_cfg)
    # Test set (2023+) contains days with N ~ 2000 quotes; attention is O(N^2),
    # so keep eval batch size small even under torch.no_grad().
    test_ld = DataLoader(test_ds, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)
    metrics = evaluate_reconstruction(surface_model, test_ld, device_t, cfg_data=data_cfg)

    out_path = os.path.join(data_cfg.results_dir, "eval", f"surface_eval_{checkpoint_name.replace('.pt', '')}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame([metrics]).to_csv(out_path, index=False)

    logger.info(f"[surface-only eval] Saved -> {out_path}")
    logger.info(f"[surface-only eval] Metrics: {metrics}")
    return metrics


@torch.no_grad()
def extract_latents(model: FullSurfaceModel, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    records = []
    for batch in loader:
        feat = batch["features"].to(device)
        mask = batch["mask"].to(device)
        z = model.encode(feat, mask).cpu().numpy()

        B = z.shape[0]
        records.append(pd.DataFrame([
            {
                "date": batch["dates"][i],
                **{f"z_{j}": float(z[i, j]) for j in range(z.shape[1])},
                "rv5": float(batch["rv5"][i]),
                "rv21": float(batch["rv21"][i]),
                "dsv5": float(batch["dsv5"][i]),
                "stress21": float(batch["stress21"][i]),
                "rv5_back": float(batch["rv5_back"][i]),
                "rv21_back": float(batch["rv21_back"][i]),
                "dsv5_back": float(batch["dsv5_back"][i]),
                "_valid": bool(batch["valid"][i]),
                "VIX": float(batch["VIX"][i]),
                "VXV": float(batch["VXV"][i]),
                "S": float(batch["S"][i]),
                "atm_iv": float(batch["atm_iv"][i]),
                "short_iv": float(batch["short_iv"][i]),
                "term_slope": float(batch["term_slope"][i]),
                "skew_proxy": float(batch["skew_proxy"][i]),
                "N_quotes": int(batch["N_each"][i]),
                "am_share": float(batch["am_share"][i]),
            }
            for i in range(B)
        ]))

    df = pd.concat(records, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


class LatentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, d_latent: int):
        z_cols = [f"z_{j}" for j in range(d_latent)]
        self.Z = torch.tensor(df[z_cols].values, dtype=torch.float32)
        # Preserve NaN here; DownstreamHeads._make_input fills it with the
        # training-set VIX mean (~20). Filling with 0 would conflate
        # "missing" with "VIX == 0".
        side_df = df[["VIX", "VXV"]].replace([np.inf, -np.inf], np.nan)
        self.side = torch.tensor(side_df.values, dtype=torch.float32)
        self.rv5 = torch.tensor(df["rv5"].values, dtype=torch.float32)
        self.rv21 = torch.tensor(df["rv21"].values, dtype=torch.float32)
        self.dsv5 = torch.tensor(df["dsv5"].values, dtype=torch.float32)
        self.stress21 = torch.tensor(df["stress21"].values, dtype=torch.float32)
        if "_valid" in df.columns:
            self.valid = df["_valid"].values.astype(bool)
        else:
            self.valid = (~df[["rv5", "rv21", "dsv5", "stress21"]].isna().any(axis=1)).values

    def __len__(self):
        return len(self.Z)

    def __getitem__(self, idx):
        return {
            "z": self.Z[idx],
            "side": self.side[idx],
            "rv5": self.rv5[idx],
            "rv21": self.rv21[idx],
            "dsv5": self.dsv5[idx],
            "stress21": self.stress21[idx],
            "valid": bool(self.valid[idx]),
        }


def train_stage2(surface_model: FullSurfaceModel, cache_path: str,
                 cfg_data: DataConfig = DATA_CFG, cfg_model: ModelConfig = MODEL_CFG,
                 cfg_train: TrainConfig = TRAIN_CFG) -> DownstreamHeads:
    set_seed(cfg_train.seed + 1000)
    device = torch.device(cfg_train.device if torch.cuda.is_available() else "cpu")
    logger.info("=== Stage 2: freeze encoder, train heads ===")

    train_loader, val_loader = build_loaders(cache_path, cfg_data, cfg_train)

    logger.info("Extracting train latents …")
    train_df = extract_latents(surface_model, train_loader, device)
    logger.info("Extracting val latents …")
    val_df = extract_latents(surface_model, val_loader, device)

    train_df = train_df[train_df["_valid"]].reset_index(drop=True)
    val_df = val_df[val_df["_valid"]].reset_index(drop=True)

    train_ds = LatentDataset(train_df, cfg_model.latent_dim)
    val_ds = LatentDataset(val_df, cfg_model.latent_dim)

    train_ld = DataLoader(train_ds, batch_size=cfg_train.batch_size_s2, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=cfg_train.batch_size_s2, shuffle=False)

    heads = DownstreamHeads(cfg_model).to(device)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=cfg_train.lr_s2, weight_decay=cfg_train.weight_decay)

    best_val = float("inf")
    ckpt_dir = os.path.join(cfg_data.results_dir, "checkpoints")

    for epoch in range(cfg_train.epochs_s2):
        heads.train()
        tr_loss = 0.0
        for batch in train_ld:
            optimizer.zero_grad()
            z = batch["z"].to(device)
            side = batch["side"].to(device)
            preds = heads(z, side)
            loss, _ = downstream_loss(preds, batch, device)
            loss.backward()
            nn.utils.clip_grad_norm_(heads.parameters(), cfg_train.grad_clip)
            optimizer.step()
            tr_loss += loss.item()

        heads.eval()
        va_loss = 0.0
        with torch.no_grad():
            for batch in val_ld:
                z = batch["z"].to(device)
                side = batch["side"].to(device)
                preds = heads(z, side)
                loss, _ = downstream_loss(preds, batch, device)
                va_loss += loss.item()

        tr_loss /= max(len(train_ld), 1)
        va_loss /= max(len(val_ld), 1)

        if (epoch + 1) % 10 == 0:
            logger.info(f"[S2 epoch {epoch+1:3d}/{cfg_train.epochs_s2}] tr_loss={tr_loss:.4f}  va_loss={va_loss:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save({"heads": heads.state_dict(), "epoch": epoch + 1},
                       os.path.join(ckpt_dir, "stage2_best.pt"))

    logger.info(f"Stage 2 complete.  Best val loss = {best_val:.6f}")
    ckpt = torch.load(os.path.join(ckpt_dir, "stage2_best.pt"), map_location=device)
    heads.load_state_dict(ckpt["heads"])
    return heads


class ScalarEMANormalizer:
    def __init__(self, names, decay: float = 0.9, eps: float = 1e-6):
        self.decay = decay
        self.eps = eps
        self.state = {n: None for n in names}

    def update(self, name: str, value: float):
        v = float(value)
        if self.state[name] is None:
            self.state[name] = max(v, self.eps)
        else:
            self.state[name] = self.decay * self.state[name] + (1 - self.decay) * v

    def normalize(self, name: str, value: float) -> float:
        denom = self.state[name] if self.state[name] is not None else max(float(value), self.eps)
        return float(value) / (denom + self.eps)


def train_stage3(surface_model: FullSurfaceModel, heads: DownstreamHeads, cache_path: str,
                 cfg_data: DataConfig = DATA_CFG, cfg_model: ModelConfig = MODEL_CFG,
                 cfg_train: TrainConfig = TRAIN_CFG):
    set_seed(cfg_train.seed + 2000)
    device = torch.device(cfg_train.device if torch.cuda.is_available() else "cpu")
    logger.info("=== Stage 3: joint fine-tune ===")

    train_loader, val_loader = build_loaders(cache_path, cfg_data, cfg_train)
    loss_surf = SurfaceLoss(cfg_train, cfg_data).to(device)

    optimizer = torch.optim.AdamW([
        {"params": surface_model.parameters(), "lr": cfg_train.lr_s3_encoder},
        {"params": heads.parameters(), "lr": cfg_train.lr_s3_heads},
    ], weight_decay=cfg_train.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg_train.epochs_s3)
    ckpt_dir = os.path.join(cfg_data.results_dir, "checkpoints")

    best_surface = float("inf")
    best_forecast = float("inf")
    best_joint = float("inf")

    normer = ScalarEMANormalizer(["surface", "forecast"], decay=0.9, eps=1e-6)

    def _save(name: str, epoch: int, va_surface: float, va_forecast: float, va_joint: float):
        torch.save({
            "surface": surface_model.state_dict(),
            "heads": heads.state_dict(),
            "epoch": epoch,
            "val_surface": va_surface,
            "val_forecast": va_forecast,
            "val_joint": va_joint,
        }, os.path.join(ckpt_dir, name))

    for epoch in range(cfg_train.epochs_s3):
        warmup_ratio = min(1.0, (epoch + 1) / cfg_train.warmup_epochs_s3)
        lambda_pred_train = cfg_train.lambda_pred * warmup_ratio
        lambda_pred_eval = cfg_train.lambda_pred

        surface_model.train()
        heads.train()
        tr_surface = tr_forecast = tr_joint = 0.0

        def _prep_side(batch):
            # Consistent with DownstreamHeads._make_input: fill missing with 20,
            # not 0, so that the regime gate is never driven to an unseen region.
            side = torch.stack([batch["VIX"], batch["VXV"]], dim=1).to(device)
            return torch.where(
                torch.isnan(side) | torch.isinf(side),
                torch.full_like(side, 20.0),
                side,
            )

        for batch in train_loader:
            optimizer.zero_grad()
            L_surf, _ = loss_surf(surface_model, batch, device)

            feat = batch["features"].to(device)
            mask = batch["mask"].to(device)
            z = surface_model.encode(feat, mask)

            side = _prep_side(batch)
            preds = heads(z, side)
            L_pred, _ = downstream_loss(preds, batch, device)
            total = L_surf + lambda_pred_train * L_pred

            total.backward()
            nn.utils.clip_grad_norm_(list(surface_model.parameters()) + list(heads.parameters()), cfg_train.grad_clip)
            optimizer.step()

            tr_surface += L_surf.item()
            tr_forecast += L_pred.item()
            tr_joint += total.item()

        scheduler.step()

        surface_model.eval()
        heads.eval()
        va_surface = 0.0
        va_forecast = 0.0
        n_val_surface = 0
        n_val_forecast = 0

        with torch.no_grad():
            for batch in val_loader:
                # Surface loss is well-defined on every batch, regardless of
                # whether the forward RV targets are available.
                L_s, _ = loss_surf(surface_model, batch, device, deterministic=True, update_scaler=False)
                va_surface += L_s.item()
                n_val_surface += 1

                # Forecast loss is only meaningful on batches that contain any
                # valid forward target. Tail val batches near year-end can have
                # valid.sum() == 0 (RV21 needs 21 future trading days). Without
                # this guard, downstream_loss returns 0, which would incorrectly
                # pull va_forecast toward zero and corrupt model selection.
                if bool(batch["valid"].any()):
                    feat = batch["features"].to(device)
                    mask = batch["mask"].to(device)
                    z = surface_model.encode(feat, mask)

                    side = _prep_side(batch)
                    preds = heads(z, side)
                    L_p, _ = downstream_loss(preds, batch, device)
                    va_forecast += L_p.item()
                    n_val_forecast += 1

        tr_surface /= max(len(train_loader), 1)
        tr_forecast /= max(len(train_loader), 1)
        tr_joint /= max(len(train_loader), 1)

        va_surface /= max(n_val_surface, 1)
        va_forecast /= max(n_val_forecast, 1)

        norm_s = normer.normalize("surface", va_surface)
        norm_p = normer.normalize("forecast", va_forecast)
        va_joint = norm_s + lambda_pred_eval * norm_p

        normer.update("surface", va_surface)
        normer.update("forecast", va_forecast)

        logger.info(
            f"[S3 epoch {epoch+1:3d}/{cfg_train.epochs_s3}] "
            f"λ_pred_train={lambda_pred_train:.3f}  "
            f"tr_joint={tr_joint:.4f} tr_surf={tr_surface:.4f} tr_pred={tr_forecast:.4f}  "
            f"va_joint={va_joint:.4f} va_surf={va_surface:.4f} va_pred={va_forecast:.4f}"
        )

        ep = epoch + 1
        if va_surface < best_surface:
            best_surface = va_surface
            _save("stage3_best_surface.pt", ep, va_surface, va_forecast, va_joint)

        if va_forecast < best_forecast:
            best_forecast = va_forecast
            _save("stage3_best_forecast.pt", ep, va_surface, va_forecast, va_joint)
            _save("stage3_best.pt", ep, va_surface, va_forecast, va_joint)

        if va_joint < best_joint:
            best_joint = va_joint
            _save("stage3_best_joint.pt", ep, va_surface, va_forecast, va_joint)

        _save("stage3_last.pt", ep, va_surface, va_forecast, va_joint)

    logger.info(f"Stage 3 complete. best_surface={best_surface:.6f}  best_forecast={best_forecast:.6f}  best_joint={best_joint:.6f}")


class SetForecastOnlyModel(nn.Module):
    """
    Clean ablation: permutation-invariant quote-set encoder + downstream heads
    trained directly on forecasting loss, with no surface decoder and no surface
    reconstruction losses. This tests whether surface anchoring adds value beyond
    quote-set aggregation.
    """
    def __init__(self, model_cfg: ModelConfig = MODEL_CFG):
        super().__init__()
        self.encoder = SetEncoder(model_cfg)
        self.heads = DownstreamHeads(model_cfg)

    def encode(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(features, mask)

    def forward(self, features: torch.Tensor, mask: torch.Tensor,
                side: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        z = self.encode(features, mask)
        return self.heads(z, side)


def _prep_side_from_batch(batch: dict, device: torch.device, use_side_info: bool = True) -> Optional[torch.Tensor]:
    if not use_side_info:
        return None
    side = torch.stack([batch["VIX"], batch["VXV"]], dim=1).to(device)
    return torch.where(
        torch.isnan(side) | torch.isinf(side),
        torch.full_like(side, 20.0),
        side,
    )


def train_set_forecast_only(
    cache_path: str,
    cfg_data: DataConfig = DATA_CFG,
    cfg_model: ModelConfig = MODEL_CFG,
    cfg_train: TrainConfig = TRAIN_CFG,
    use_side_info: bool = True,
    run_name: str = "set_forecast_only",
) -> tuple[SetForecastOnlyModel, DownstreamHeads]:
    """
    Train the set-encoder-only forecasting ablation from scratch.

    Saved checkpoint:
        results_dir/checkpoints/{run_name}_best.pt
    """
    make_dirs(cfg_data)
    set_seed(cfg_train.seed)
    device = torch.device(cfg_train.device if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_loaders(cache_path, cfg_data, cfg_train)
    model = SetForecastOnlyModel(cfg_model).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_train.lr_s3_heads, weight_decay=cfg_train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg_train.epochs_s2, 1))

    ckpt_dir = os.path.join(cfg_data.results_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val = float("inf")
    history = []

    for epoch in range(cfg_train.epochs_s2):
        model.train()
        tr_loss = 0.0
        tr_batches = 0
        tr_parts = {"rv5": 0.0, "rv21": 0.0, "dsv5": 0.0, "stress": 0.0}

        for batch in tqdm(train_loader, desc=f"{run_name} train", leave=False):
            if not bool(batch["valid"].any()):
                continue
            feat = batch["features"].to(device)
            mask = batch["mask"].to(device)
            side = _prep_side_from_batch(batch, device, use_side_info=use_side_info)

            optimizer.zero_grad(set_to_none=True)
            preds = model(feat, mask, side)
            loss, info = downstream_loss(preds, batch, device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg_train.grad_clip)
            optimizer.step()

            tr_loss += float(loss.item())
            tr_batches += 1
            for k in tr_parts:
                tr_parts[k] += float(info.get(k, 0.0))

        scheduler.step()

        model.eval()
        va_loss = 0.0
        va_batches = 0
        va_parts = {"rv5": 0.0, "rv21": 0.0, "dsv5": 0.0, "stress": 0.0}
        with torch.no_grad():
            for batch in val_loader:
                if not bool(batch["valid"].any()):
                    continue
                feat = batch["features"].to(device)
                mask = batch["mask"].to(device)
                side = _prep_side_from_batch(batch, device, use_side_info=use_side_info)
                preds = model(feat, mask, side)
                loss, info = downstream_loss(preds, batch, device)
                va_loss += float(loss.item())
                va_batches += 1
                for k in va_parts:
                    va_parts[k] += float(info.get(k, 0.0))

        tr_loss /= max(tr_batches, 1)
        va_loss /= max(va_batches, 1)
        tr_parts = {k: v / max(tr_batches, 1) for k, v in tr_parts.items()}
        va_parts = {k: v / max(va_batches, 1) for k, v in va_parts.items()}

        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": tr_loss,
            "val_loss": va_loss,
            **{f"train_{k}": v for k, v in tr_parts.items()},
            **{f"val_{k}": v for k, v in va_parts.items()},
            "use_side_info": bool(use_side_info),
        }
        history.append(row)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"[{run_name} epoch {epoch+1:3d}/{cfg_train.epochs_s2}] "
                f"tr_loss={tr_loss:.4f} va_loss={va_loss:.4f} "
                f"va_rv5={va_parts['rv5']:.4f} va_rv21={va_parts['rv21']:.4f} "
                f"va_dsv5={va_parts['dsv5']:.4f} va_cls={va_parts['stress']:.4f}"
            )

        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                "encoder": model.encoder.state_dict(),
                "heads": model.heads.state_dict(),
                "model": model.state_dict(),
                "epoch": epoch + 1,
                "best_val": best_val,
                "use_side_info": bool(use_side_info),
                "run_name": run_name,
                "history": history,
            }, os.path.join(ckpt_dir, f"{run_name}_best.pt"))

    hist_path = os.path.join(cfg_data.results_dir, f"{run_name}_history.csv")
    if history:
        pd.DataFrame(history).to_csv(hist_path, index=False)

    ckpt = torch.load(os.path.join(ckpt_dir, f"{run_name}_best.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    logger.info(f"{run_name} complete. best val loss = {best_val:.6f}")
    return model, model.heads




class SingleFactorCalibrator:
    """
    Target-specific single-factor calibration baseline.

    Supported forms:
      - linear:     y = a + b x
      - loglinear:  log(y) = a + b log(x)

    The fit method selects the form with validation_regression_score():
    RV5/RV21 use validation QLIKE; DSV5 uses validation MAE.
    """
    def __init__(self, forms: tuple[str, ...] = ("linear", "loglinear"), eps: float = 1e-8):
        self.forms = forms
        self.eps = eps
        self.form_: Optional[str] = None
        self.beta_: Optional[np.ndarray] = None

    def _design(self, x: np.ndarray, form: str) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if form == "linear":
            return np.column_stack([np.ones_like(x), x])
        if form == "loglinear":
            return np.column_stack([np.ones_like(x), np.log(np.maximum(x, self.eps))])
        raise ValueError(f"Unknown calibration form: {form}")

    def _fit_beta(self, x: np.ndarray, y: np.ndarray, form: str) -> np.ndarray:
        X = self._design(x, form)
        y = np.asarray(y, dtype=float)
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        if form == "loglinear":
            valid &= y > self.eps
            y_fit = np.log(np.maximum(y[valid], self.eps))
        else:
            y_fit = y[valid]
        if valid.sum() == 0:
            raise ValueError(f"SingleFactorCalibrator.fit found no finite rows for form={form}.")
        return np.linalg.lstsq(X[valid], y_fit, rcond=None)[0]

    def _predict_with(self, beta: np.ndarray, x: np.ndarray, form: str) -> np.ndarray:
        X = self._design(x, form)
        pred = X @ beta
        if form == "loglinear":
            pred = np.exp(pred)
        pred = np.where(np.isfinite(pred), pred, np.nan)
        return np.maximum(pred, self.eps)

    def fit(self, x_train, y_train, x_val=None, y_val=None, target: str = "rv5"):
        x_train = np.asarray(x_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)
        x_val = None if x_val is None else np.asarray(x_val, dtype=float)
        y_val = None if y_val is None else np.asarray(y_val, dtype=float)

        best = None
        for form in self.forms:
            beta = self._fit_beta(x_train, y_train, form)
            if x_val is not None and y_val is not None:
                pred_val = self._predict_with(beta, x_val, form)
                score = validation_regression_score(
                    y_true=y_val,
                    y_pred=pred_val,
                    target=target,
                    train_y=y_train,
                    eps=self.eps,
                )
            else:
                score = 0.0
            if best is None or score < best[0]:
                best = (score, form, beta)

        if best is None:
            raise ValueError("SingleFactorCalibrator.fit failed to fit any candidate form.")
        _, self.form_, self.beta_ = best
        return self

    def predict(self, x) -> np.ndarray:
        if self.beta_ is None or self.form_ is None:
            raise ValueError("SingleFactorCalibrator.predict called before fit().")
        x = np.asarray(x, dtype=float)
        return self._predict_with(self.beta_, x, self.form_)


def fit_scalar_calibrated_baselines(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """
    Fit calibrated single-factor baselines for VIX, ATM-IV, and ShortEnd-IV.

    Each target gets its own calibrator, selected between linear and log-linear
    forms using validation_regression_score(): RV5/RV21 use QLIKE, while DSV5 uses MAE.
    """
    feature_map = {
        "VIX-alone": ("VIX", lambda s: np.asarray(s, dtype=float) / 100.0),
        "ATM-IV": ("atm_iv", lambda s: np.asarray(s, dtype=float)),
        "ShortEnd-IV": ("short_iv", lambda s: np.asarray(s, dtype=float)),
    }
    out: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    for label, (col, transform) in feature_map.items():
        x_train_full = transform(train_df[col].values)
        x_val_full = transform(val_df[col].values)
        x_test_full = transform(test_df[col].values)

        target_out: Dict[str, Dict[str, np.ndarray]] = {}
        for target in ["rv5", "rv21", "dsv5"]:
            y_train_full = train_df[target].values.astype(float)
            y_val_full = val_df[target].values.astype(float)

            train_valid = np.isfinite(x_train_full) & np.isfinite(y_train_full) & (x_train_full > 0)
            val_valid = np.isfinite(x_val_full) & np.isfinite(y_val_full) & (x_val_full > 0)
            if train_valid.sum() < 50:
                raise ValueError(f"{label} calibration has too few valid training rows for {target}.")
            if val_valid.sum() < 20:
                raise ValueError(f"{label} calibration has too few valid validation rows for {target}.")

            cal = SingleFactorCalibrator(forms=("linear", "loglinear")).fit(
                x_train=x_train_full[train_valid],
                y_train=y_train_full[train_valid],
                x_val=x_val_full[val_valid],
                y_val=y_val_full[val_valid],
                target=target,
            )

            val_pred_full = np.full(len(val_df), np.nan, dtype=float)
            val_pred_full[val_valid] = cal.predict(x_val_full[val_valid])

            test_pred_full = np.full(len(test_df), np.nan, dtype=float)
            test_valid = np.isfinite(x_test_full) & (x_test_full > 0)
            test_pred_full[test_valid] = cal.predict(x_test_full[test_valid])

            target_out[target] = {
                "val": val_pred_full,
                "test": test_pred_full,
                "form": cal.form_,
            }

        out[label] = target_out

    return out


class HARModel:
    def __init__(self, use_options: bool = False, serving_time: str = "close_t"):
        """
        serving_time='close_t' means:
          at anchor date t, we use close-of-t information
          to forecast forward targets defined on t+1, ..., t+h.

        This matches:
          rv_h[t], dsv_h[t], stress21[t].
        """
        if serving_time != "close_t":
            raise ValueError("This pipeline only supports serving_time='close_t'.")
        self.use_options = use_options
        self.serving_time = serving_time
        self.coef_ = None
        self._n_lag = 22

    def _build_X(self, info_series: np.ndarray, vix: Optional[np.ndarray] = None,
                 vxv: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        info_series = np.asarray(info_series, dtype=np.float64)
        vix_arr = None if vix is None else np.asarray(vix, dtype=np.float64)
        vxv_arr = None if vxv is None else np.asarray(vxv, dtype=np.float64)

        rows, idx = [], []
        n = len(info_series)
        for t in range(self._n_lag - 1, n):
            win = info_series[t - self._n_lag + 1:t + 1]
            if np.isnan(win).any():
                continue

            row = [1.0, win[-1], win[-5:].mean(), win.mean()]

            if self.use_options:
                if vix_arr is None or not np.isfinite(vix_arr[t]):
                    continue
                row.append(vix_arr[t] / 100.0)
                row.append(vxv_arr[t] / (vix_arr[t] + 1e-6) if vxv_arr is not None and np.isfinite(vxv_arr[t]) else 0.0)

            rows.append(row)
            idx.append(t)

        if len(rows) == 0:
            return np.empty((0, 4 + (2 if self.use_options else 0))), np.empty((0,), dtype=int)
        return np.asarray(rows, dtype=np.float64), np.asarray(idx, dtype=int)

    def _feature_row_from_history(self, info_hist: np.ndarray,
                                  vix_hist: Optional[np.ndarray] = None,
                                  vxv_hist: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if len(info_hist) < self._n_lag:
            return None

        win = np.asarray(info_hist[-self._n_lag:], dtype=np.float64)
        if np.isnan(win).any():
            return None

        row = [1.0, win[-1], win[-5:].mean(), win.mean()]

        if self.use_options:
            if vix_hist is None or len(vix_hist) < 1:
                return None
            vix_last = float(vix_hist[-1])
            if not np.isfinite(vix_last):
                return None
            row.append(vix_last / 100.0)
            row.append(
                float(vxv_hist[-1]) / (vix_last + 1e-6)
                if vxv_hist is not None and len(vxv_hist) >= 1 and np.isfinite(vxv_hist[-1]) else 0.0
            )

        return np.asarray(row, dtype=np.float64)

    def fit(self, info_train: np.ndarray, y_train: np.ndarray,
            vix: Optional[np.ndarray] = None, vxv: Optional[np.ndarray] = None):
        X, idx = self._build_X(info_train, vix, vxv)
        y = np.asarray(y_train, dtype=np.float64)[idx]
        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        if valid.sum() == 0:
            raise ValueError("HARModel.fit found no finite training rows.")
        self.coef_ = np.linalg.lstsq(X[valid], y[valid], rcond=None)[0]

    def predict_pseudo_oos(self, info_train: np.ndarray, info_test: np.ndarray,
                           vix_train: Optional[np.ndarray] = None, vix_test: Optional[np.ndarray] = None,
                           vxv_train: Optional[np.ndarray] = None, vxv_test: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Each prediction at evaluation index i uses close-of-t information for that same
        anchor date t, consistent with forward labels built from future returns only.
        """
        if self.coef_ is None:
            raise ValueError("HARModel must be fit before predict_pseudo_oos.")

        info_hist = list(np.asarray(info_train, dtype=np.float64))
        vix_hist = list(np.asarray(vix_train, dtype=np.float64)) if vix_train is not None else None
        vxv_hist = list(np.asarray(vxv_train, dtype=np.float64)) if vxv_train is not None else None

        info_test = np.asarray(info_test, dtype=np.float64)
        preds = np.full(len(info_test), np.nan, dtype=np.float64)

        for i in range(len(info_test)):
            info_hist.append(float(info_test[i]))
            if vix_hist is not None and vix_test is not None:
                vix_hist.append(float(vix_test[i]))
            if vxv_hist is not None and vxv_test is not None:
                vxv_hist.append(float(vxv_test[i]))

            row = self._feature_row_from_history(
                np.asarray(info_hist, dtype=np.float64),
                None if vix_hist is None else np.asarray(vix_hist, dtype=np.float64),
                None if vxv_hist is None else np.asarray(vxv_hist, dtype=np.float64),
            )
            if row is not None and np.isfinite(row).all():
                preds[i] = float(row @ self.coef_)

        return preds


class HARSummaryModel:
    def __init__(self):
        self.beta = None

    def _design(self, rv_back, atm_iv, short_iv, term_slope, skew_proxy, VIX, VXV):
        return np.column_stack([
            np.ones_like(rv_back, dtype=float),
            rv_back.astype(float),
            atm_iv.astype(float),
            short_iv.astype(float),
            term_slope.astype(float),
            skew_proxy.astype(float),
            VIX.astype(float),
            VXV.astype(float),
        ])

    def fit(self, rv_back, y, atm_iv, short_iv, term_slope, skew_proxy, VIX, VXV):
        X = self._design(rv_back, atm_iv, short_iv, term_slope, skew_proxy, VIX, VXV)
        y = np.asarray(y, dtype=float)
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        if valid.sum() == 0:
            raise ValueError("HARSummaryModel.fit found no finite training rows.")
        self.beta = np.linalg.lstsq(X[valid], y[valid], rcond=None)[0]
        return self

    def predict(self, rv_back, atm_iv, short_iv, term_slope, skew_proxy, VIX, VXV):
        if self.beta is None:
            raise ValueError("HARSummaryModel.predict called before fit().")
        X = self._design(rv_back, atm_iv, short_iv, term_slope, skew_proxy, VIX, VXV)
        pred = X @ self.beta
        pred = np.where(np.isfinite(pred), pred, np.nan)
        return np.maximum(pred, 1e-8)




# ============================================================
# Additional evaluation-only baselines
# ============================================================

def _choose_boosting_backend(require_xgboost: Optional[bool] = None) -> tuple[str, object]:
    """
    Select the nonlinear summary learner used in the paper tables.

    Paper-facing runs should not silently replace XGBoost-Summary with a
    different backend, because that changes the baseline identity in the result
    table. By default we require xgboost. For exploratory/local debugging only,
    set environment variable CIKM_REQUIRE_XGBOOST=0 to allow LightGBM/HGBR
    fallbacks with explicitly different labels.
    """
    if require_xgboost is None:
        require_xgboost = os.environ.get("CIKM_REQUIRE_XGBOOST", "1") != "0"

    try:
        import xgboost as xgb  # type: ignore
        return "XGBoost-Summary", xgb
    except Exception as e:
        if require_xgboost:
            raise ImportError(
                "Paper reproduction requires xgboost so that the baseline label "
                "XGBoost-Summary matches the reported table. Install xgboost or "
                "set CIKM_REQUIRE_XGBOOST=0 for exploratory fallback runs."
            ) from e

    try:
        import lightgbm as lgb  # type: ignore
        logger.warning("xgboost unavailable; using LightGBM-Summary exploratory fallback.")
        return "LightGBM-Summary", lgb
    except Exception:
        pass

    logger.warning("xgboost/lightgbm unavailable; using HGBR-Summary exploratory fallback.")
    return "HGBR-Summary", HistGradientBoostingRegressor

def _build_summary_matrix(df: pd.DataFrame, rv_back_col: str) -> pd.DataFrame:
    cols = [rv_back_col, "atm_iv", "short_iv", "term_slope", "skew_proxy", "VIX", "VXV"]
    X = df[cols].copy()
    X.columns = ["rv_back", "atm_iv", "short_iv", "term_slope", "skew_proxy", "VIX", "VXV"]
    return X


def _median_impute_from_train(X_train: pd.DataFrame, *others: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    med = X_train.replace([np.inf, -np.inf], np.nan).median(numeric_only=True)

    def _apply(X: pd.DataFrame) -> pd.DataFrame:
        return X.replace([np.inf, -np.inf], np.nan).fillna(med)

    out = [_apply(X_train)]
    out.extend(_apply(X) for X in others)
    return tuple(out)


def fit_boosted_summary_baselines(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                                  seed: int = 42) -> tuple[str, Dict[str, Dict[str, np.ndarray]]]:
    label, backend = _choose_boosting_backend()
    task_map = {"rv5": "rv5_back", "rv21": "rv21_back", "dsv5": "dsv5_back"}
    out: Dict[str, Dict[str, np.ndarray]] = {}

    for target, rv_back_col in task_map.items():
        X_train_raw = _build_summary_matrix(train_df, rv_back_col)
        X_val_raw = _build_summary_matrix(val_df, rv_back_col)
        X_test_raw = _build_summary_matrix(test_df, rv_back_col)
        X_train, X_val, X_test = _median_impute_from_train(X_train_raw, X_val_raw, X_test_raw)

        y_train = train_df[target].values.astype(float)
        y_val = val_df[target].values.astype(float)

        train_valid = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1).values
        val_valid = np.isfinite(y_val) & np.isfinite(X_val).all(axis=1).values
        if train_valid.sum() < 50:
            raise ValueError(f"Boosted summary baseline has too few valid training rows for {target}.")
        if val_valid.sum() < 20:
            raise ValueError(f"Boosted summary baseline has too few valid validation rows for {target}.")

        candidates = []
        if label == "XGBoost-Summary":
            for max_depth in [2, 3]:
                for n_estimators in [200, 400]:
                    for lr in [0.03, 0.05]:
                        candidates.append(backend.XGBRegressor(
                            objective="reg:squarederror",
                            tree_method="hist",
                            max_depth=max_depth,
                            n_estimators=n_estimators,
                            learning_rate=lr,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            reg_lambda=1.0,
                            random_state=seed,
                        ))
        elif label == "LightGBM-Summary":
            for max_depth in [2, 3, -1]:
                for n_estimators in [200, 400]:
                    for lr in [0.03, 0.05]:
                        candidates.append(backend.LGBMRegressor(
                            objective="regression",
                            max_depth=max_depth,
                            n_estimators=n_estimators,
                            learning_rate=lr,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            reg_lambda=1.0,
                            random_state=seed,
                            verbosity=-1,
                        ))
        else:
            for max_depth in [3, None]:
                for max_iter in [300, 500]:
                    for lr in [0.03, 0.05]:
                        candidates.append(HistGradientBoostingRegressor(
                            learning_rate=lr,
                            max_iter=max_iter,
                            max_depth=max_depth,
                            min_samples_leaf=20,
                            random_state=seed,
                        ))

        best_model = None
        best_score = np.inf
        for model in candidates:
            model.fit(X_train.loc[train_valid].values, y_train[train_valid])
            pred_val = np.asarray(model.predict(X_val.values), dtype=float)
            pred_val = np.maximum(pred_val, 1e-8)
            score = validation_regression_score(
                y_true=y_val[val_valid],
                y_pred=pred_val[val_valid],
                target=target,
                train_y=y_train[train_valid],
            )
            if score < best_score:
                best_score = score
                best_model = model

        if best_model is None:
            raise RuntimeError(f"Failed to fit boosted summary baseline for {target}.")

        out[target] = {
            "val": np.maximum(np.asarray(best_model.predict(X_val.values), dtype=float), 1e-8),
            "test": np.maximum(np.asarray(best_model.predict(X_test.values), dtype=float), 1e-8),
        }

    return label, out


def record_to_grid_surface(record: dict, cfg_data: DataConfig, n_k: int = 21, n_tau: int = 16) -> np.ndarray:
    k = np.asarray(record["k"], dtype=float)
    tau = np.asarray(record["tau"], dtype=float)
    w_target = np.asarray(record["w_target"], dtype=float)
    liq_w = np.asarray(record.get("liq_w", np.ones_like(k)), dtype=float)

    valid = np.isfinite(k) & np.isfinite(tau) & np.isfinite(w_target) & np.isfinite(liq_w) & (tau > 0)
    k = k[valid]
    tau = tau[valid]
    w_target = w_target[valid]
    liq_w = liq_w[valid]

    grid_sum = np.zeros((n_tau, n_k), dtype=np.float32)
    grid_w = np.zeros((n_tau, n_k), dtype=np.float32)

    if len(k) > 0:
        iv = np.sqrt(np.maximum(w_target, 0.0) / np.maximum(tau, 1e-8))
        k_idx = np.floor((k - cfg_data.k_min) / max(cfg_data.k_max - cfg_data.k_min, 1e-8) * n_k).astype(int)
        t_idx = np.floor((tau - cfg_data.tau_min) / max(cfg_data.tau_max - cfg_data.tau_min, 1e-8) * n_tau).astype(int)
        k_idx = np.clip(k_idx, 0, n_k - 1)
        t_idx = np.clip(t_idx, 0, n_tau - 1)
        for kk, tt, vv, ww in zip(k_idx, t_idx, iv, liq_w):
            grid_sum[tt, kk] += float(vv * max(ww, 1e-6))
            grid_w[tt, kk] += float(max(ww, 1e-6))

    grid_iv = np.divide(grid_sum, grid_w, out=np.zeros_like(grid_sum), where=grid_w > 0)
    obs_mask = (grid_w > 0).astype(np.float32)
    return np.stack([grid_iv.astype(np.float32), obs_mask], axis=0)


def build_grid_map(cache_path: str, split: str, cfg_data: DataConfig, n_k: int = 21, n_tau: int = 16) -> Dict[pd.Timestamp, np.ndarray]:
    ds = OptionDataset(cache_path, split, cfg_data)
    out: Dict[pd.Timestamp, np.ndarray] = {}
    for rec in ds.records:
        out[pd.Timestamp(rec["date"])] = record_to_grid_surface(rec, cfg_data, n_k=n_k, n_tau=n_tau)
    return out


def attach_grid_tensor(df: pd.DataFrame, grid_map: Dict[pd.Timestamp, np.ndarray], n_k: int = 21, n_tau: int = 16) -> np.ndarray:
    fallback = np.zeros((2, n_tau, n_k), dtype=np.float32)
    mats = [grid_map.get(pd.Timestamp(d), fallback) for d in pd.to_datetime(df["date"]).values]
    return np.stack(mats, axis=0).astype(np.float32)


class SmallGridCNNRegressor(nn.Module):
    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        raw = self.head(h).squeeze(-1)
        return F.softplus(raw) + 1e-6


class SmallGridTransformerRegressor(nn.Module):
    def __init__(self, n_tau: int = 16, n_k: int = 21, d_model: int = 32,
                 nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.n_tau = n_tau
        self.n_k = n_k
        n_tokens = n_tau * n_k

        tau_coords = torch.linspace(0.0, 1.0, steps=n_tau).view(n_tau, 1).expand(n_tau, n_k)
        k_coords = torch.linspace(0.0, 1.0, steps=n_k).view(1, n_k).expand(n_tau, n_k)
        self.register_buffer("tau_coords", tau_coords.reshape(1, n_tokens, 1), persistent=False)
        self.register_buffer("k_coords", k_coords.reshape(1, n_tokens, 1), persistent=False)

        self.token_proj = nn.Linear(4, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens + 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c != 2 or h != self.n_tau or w != self.n_k:
            raise ValueError(f"Expected input of shape (B, 2, {self.n_tau}, {self.n_k}), got {tuple(x.shape)}")

        iv = x[:, 0].reshape(b, -1, 1)
        obs = x[:, 1].reshape(b, -1, 1)
        tau = self.tau_coords.expand(b, -1, -1)
        kk = self.k_coords.expand(b, -1, -1)
        tokens = torch.cat([iv, obs, tau, kk], dim=-1)
        tokens = self.token_proj(tokens)

        cls = self.cls_token.expand(b, -1, -1)
        hseq = torch.cat([cls, tokens], dim=1) + self.pos_embed
        hseq = self.encoder(hseq)
        cls_out = self.norm(hseq[:, 0])
        raw = self.head(cls_out).squeeze(-1)
        return F.softplus(raw) + 1e-6


def _normalize_grid_inputs(train_x: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    train_x = train_x.copy()
    other_list = [arr.copy() for arr in others]
    observed = train_x[:, 0][train_x[:, 1] > 0.5]
    mean = float(np.mean(observed)) if observed.size > 0 else float(np.mean(train_x[:, 0]))
    std = float(np.std(observed)) if observed.size > 1 else 1.0
    std = max(std, 1e-6)

    def _apply(arr: np.ndarray) -> np.ndarray:
        arr[:, 0] = ((arr[:, 0] - mean) / std) * arr[:, 1]
        return arr

    out = [_apply(train_x)]
    out.extend(_apply(arr) for arr in other_list)
    return tuple(out)


def _fit_single_grid_cnn(train_x: np.ndarray, train_y: np.ndarray,
                         val_x: np.ndarray, val_y: np.ndarray,
                         test_x: np.ndarray, device: torch.device,
                         seed: int = 42, max_epochs: int = 150,
                         batch_size: int = 64, lr: float = 3e-4,
                         patience: int = 20, target: str = "rv5",
                         weight_decay: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_x, val_x, test_x = _normalize_grid_inputs(train_x, val_x, test_x)

    train_ds = TensorDataset(torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32))
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = SmallGridCNNRegressor().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=lr * 0.05)

    val_x_t = torch.tensor(val_x, dtype=torch.float32, device=device)
    test_x_t = torch.tensor(test_x, dtype=torch.float32, device=device)

    best_state = None
    best_score = np.inf
    patience_left = patience

    for _epoch in range(max_epochs):
        model.train()
        for xb, yb in train_ld:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = F.huber_loss(pred, yb, delta=0.5)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x_t).detach().cpu().numpy()
        val_score = validation_regression_score(
            y_true=val_y,
            y_pred=val_pred,
            target=target,
            train_y=train_y,
        )
        if val_score < best_score:
            best_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
        scheduler.step()
        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_pred = np.maximum(model(val_x_t).detach().cpu().numpy(), 1e-8)
        test_pred = np.maximum(model(test_x_t).detach().cpu().numpy(), 1e-8)
    return val_pred, test_pred


def fit_grid_cnn_baselines(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                           cache_path: str, cfg_data: DataConfig,
                           device: torch.device, seed: int = 42) -> Dict[str, Dict[str, np.ndarray]]:
    train_grid_map = build_grid_map(cache_path, "train", cfg_data)
    val_grid_map = build_grid_map(cache_path, "val", cfg_data)
    test_grid_map = build_grid_map(cache_path, "test", cfg_data)

    train_x_all = attach_grid_tensor(train_df, train_grid_map)
    val_x_all = attach_grid_tensor(val_df, val_grid_map)
    test_x_all = attach_grid_tensor(test_df, test_grid_map)

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for target in ["rv5", "rv21", "dsv5"]:
        y_train = train_df[target].values.astype(float)
        y_val = val_df[target].values.astype(float)
        y_test = test_df[target].values.astype(float)

        train_valid = np.isfinite(y_train)
        val_valid = np.isfinite(y_val)
        if train_valid.sum() < 50 or val_valid.sum() < 20:
            raise ValueError(f"Grid-CNN baseline has too few valid rows for {target}.")

        val_pred, test_pred = _fit_single_grid_cnn(
            train_x=train_x_all[train_valid],
            train_y=y_train[train_valid],
            val_x=val_x_all[val_valid],
            val_y=y_val[val_valid],
            test_x=test_x_all,
            device=device,
            seed=seed,
            target=target,
        )

        pred_val_full = np.full(len(val_df), np.nan, dtype=float)
        pred_val_full[val_valid] = val_pred
        out[target] = {"val": pred_val_full, "test": test_pred}

    return out


def _fit_single_grid_transformer(train_x: np.ndarray, train_y: np.ndarray,
                                 val_x: np.ndarray, val_y: np.ndarray,
                                 test_x: np.ndarray, device: torch.device,
                                 seed: int = 42, max_epochs: int = 150,
                                 batch_size: int = 64, lr: float = 3e-4,
                                 patience: int = 20, target: str = "rv5",
                                 weight_decay: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_x, val_x, test_x = _normalize_grid_inputs(train_x, val_x, test_x)
    n_tau = int(train_x.shape[2])
    n_k = int(train_x.shape[3])

    train_ds = TensorDataset(torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32))
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = SmallGridTransformerRegressor(n_tau=n_tau, n_k=n_k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=lr * 0.05)

    val_x_t = torch.tensor(val_x, dtype=torch.float32, device=device)
    test_x_t = torch.tensor(test_x, dtype=torch.float32, device=device)

    best_state = None
    best_score = np.inf
    patience_left = patience

    for _epoch in range(max_epochs):
        model.train()
        for xb, yb in train_ld:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = F.huber_loss(pred, yb, delta=0.5)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x_t).detach().cpu().numpy()
        val_score = validation_regression_score(
            y_true=val_y,
            y_pred=val_pred,
            target=target,
            train_y=train_y,
        )
        if val_score < best_score:
            best_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
        scheduler.step()
        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_pred = np.maximum(model(val_x_t).detach().cpu().numpy(), 1e-8)
        test_pred = np.maximum(model(test_x_t).detach().cpu().numpy(), 1e-8)
    return val_pred, test_pred


def fit_grid_transformer_baselines(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                                   cache_path: str, cfg_data: DataConfig,
                                   device: torch.device, seed: int = 42) -> Dict[str, Dict[str, np.ndarray]]:
    train_grid_map = build_grid_map(cache_path, "train", cfg_data)
    val_grid_map = build_grid_map(cache_path, "val", cfg_data)
    test_grid_map = build_grid_map(cache_path, "test", cfg_data)

    train_x_all = attach_grid_tensor(train_df, train_grid_map)
    val_x_all = attach_grid_tensor(val_df, val_grid_map)
    test_x_all = attach_grid_tensor(test_df, test_grid_map)

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for target in ["rv5", "rv21", "dsv5"]:
        y_train = train_df[target].values.astype(float)
        y_val = val_df[target].values.astype(float)
        train_valid = np.isfinite(y_train)
        val_valid = np.isfinite(y_val)
        if train_valid.sum() < 50 or val_valid.sum() < 20:
            raise ValueError(f"Grid-Transformer baseline has too few valid rows for {target}.")

        val_pred, test_pred = _fit_single_grid_transformer(
            train_x=train_x_all[train_valid],
            train_y=y_train[train_valid],
            val_x=val_x_all[val_valid],
            val_y=y_val[val_valid],
            test_x=test_x_all,
            device=device,
            seed=seed,
            target=target,
        )

        pred_val_full = np.full(len(val_df), np.nan, dtype=float)
        pred_val_full[val_valid] = val_pred
        out[target] = {"val": pred_val_full, "test": test_pred}

    return out



def _extract_side_matrix(df: pd.DataFrame) -> np.ndarray:
    side = df[["VIX", "VXV"]].replace([np.inf, -np.inf], np.nan).fillna(20.0).values.astype(np.float32)
    return side


def _make_side_augmented_features(h: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
    side = torch.where(
        torch.isnan(side) | torch.isinf(side),
        torch.full_like(side, 20.0),
        side,
    )
    vix = (side[:, 0:1] - 20.0) / 15.0
    vxv = (side[:, 1:2] - 20.0) / 15.0
    return torch.cat([h, vix, vxv, h * vix, h * vxv], dim=1)


class SmallGridCNNSideRegressor(nn.Module):
    """Grid-CNN baseline with the same VIX/VXV side-channel style as OurModel."""
    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(32 * 3 + 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        h = _make_side_augmented_features(h, side)
        raw = self.head(h).squeeze(-1)
        return F.softplus(raw) + 1e-6


class SmallGridTransformerSideRegressor(nn.Module):
    """Grid-Transformer baseline with VIX/VXV side-channel interactions."""
    def __init__(self, n_tau: int = 16, n_k: int = 21, d_model: int = 32,
                 nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.n_tau = n_tau
        self.n_k = n_k
        n_tokens = n_tau * n_k
        tau_coords = torch.linspace(0.0, 1.0, steps=n_tau).view(n_tau, 1).expand(n_tau, n_k)
        k_coords = torch.linspace(0.0, 1.0, steps=n_k).view(1, n_k).expand(n_tau, n_k)
        self.register_buffer("tau_coords", tau_coords.reshape(1, n_tokens, 1), persistent=False)
        self.register_buffer("k_coords", k_coords.reshape(1, n_tokens, 1), persistent=False)
        self.token_proj = nn.Linear(4, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens + 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 3 + 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c != 2 or h != self.n_tau or w != self.n_k:
            raise ValueError(f"Expected input of shape (B, 2, {self.n_tau}, {self.n_k}), got {tuple(x.shape)}")
        iv = x[:, 0].reshape(b, -1, 1)
        obs = x[:, 1].reshape(b, -1, 1)
        tau = self.tau_coords.expand(b, -1, -1)
        kk = self.k_coords.expand(b, -1, -1)
        tokens = torch.cat([iv, obs, tau, kk], dim=-1)
        tokens = self.token_proj(tokens)
        cls = self.cls_token.expand(b, -1, -1)
        hseq = torch.cat([cls, tokens], dim=1) + self.pos_embed
        hseq = self.encoder(hseq)
        cls_out = self.norm(hseq[:, 0])
        h_aug = _make_side_augmented_features(cls_out, side)
        raw = self.head(h_aug).squeeze(-1)
        return F.softplus(raw) + 1e-6


def _fit_single_grid_side_model(train_x: np.ndarray, train_side: np.ndarray, train_y: np.ndarray,
                                val_x: np.ndarray, val_side: np.ndarray, val_y: np.ndarray,
                                test_x: np.ndarray, test_side: np.ndarray,
                                device: torch.device, model_kind: str,
                                seed: int = 42, max_epochs: int = 150,
                                batch_size: int = 64, lr: float = 3e-4,
                                patience: int = 20, target: str = "rv5",
                                weight_decay: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_x, val_x, test_x = _normalize_grid_inputs(train_x, val_x, test_x)
    train_side = np.asarray(train_side, dtype=np.float32)
    val_side = np.asarray(val_side, dtype=np.float32)
    test_side = np.asarray(test_side, dtype=np.float32)

    train_ds = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_side, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32),
    )
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    if model_kind == "cnn":
        model = SmallGridCNNSideRegressor().to(device)
    elif model_kind == "transformer":
        model = SmallGridTransformerSideRegressor(n_tau=int(train_x.shape[2]), n_k=int(train_x.shape[3])).to(device)
    else:
        raise ValueError(f"Unknown model_kind={model_kind}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=lr * 0.05)
    val_x_t = torch.tensor(val_x, dtype=torch.float32, device=device)
    val_side_t = torch.tensor(val_side, dtype=torch.float32, device=device)
    test_x_t = torch.tensor(test_x, dtype=torch.float32, device=device)
    test_side_t = torch.tensor(test_side, dtype=torch.float32, device=device)

    best_state = None
    best_score = np.inf
    patience_left = patience
    for _epoch in range(max_epochs):
        model.train()
        for xb, sb, yb in train_ld:
            xb = xb.to(device)
            sb = sb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb, sb)
            loss = F.huber_loss(pred, yb, delta=0.5)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(val_x_t, val_side_t).detach().cpu().numpy()
        val_score = validation_regression_score(y_true=val_y, y_pred=val_pred, target=target, train_y=train_y)
        if val_score < best_score:
            best_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
        scheduler.step()
        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = np.maximum(model(val_x_t, val_side_t).detach().cpu().numpy(), 1e-8)
        test_pred = np.maximum(model(test_x_t, test_side_t).detach().cpu().numpy(), 1e-8)
    return val_pred, test_pred


def fit_grid_side_baselines(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                            cache_path: str, cfg_data: DataConfig, device: torch.device,
                            seed: int = 42, model_kind: str = "cnn") -> Dict[str, Dict[str, np.ndarray]]:
    train_grid_map = build_grid_map(cache_path, "train", cfg_data)
    val_grid_map = build_grid_map(cache_path, "val", cfg_data)
    test_grid_map = build_grid_map(cache_path, "test", cfg_data)
    train_x_all = attach_grid_tensor(train_df, train_grid_map)
    val_x_all = attach_grid_tensor(val_df, val_grid_map)
    test_x_all = attach_grid_tensor(test_df, test_grid_map)
    train_side_all = _extract_side_matrix(train_df)
    val_side_all = _extract_side_matrix(val_df)
    test_side_all = _extract_side_matrix(test_df)

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for target in ["rv5", "rv21", "dsv5"]:
        y_train = train_df[target].values.astype(float)
        y_val = val_df[target].values.astype(float)
        train_valid = np.isfinite(y_train)
        val_valid = np.isfinite(y_val)
        if train_valid.sum() < 50 or val_valid.sum() < 20:
            raise ValueError(f"Grid-{model_kind}+Side baseline has too few valid rows for {target}.")
        val_pred, test_pred = _fit_single_grid_side_model(
            train_x=train_x_all[train_valid],
            train_side=train_side_all[train_valid],
            train_y=y_train[train_valid],
            val_x=val_x_all[val_valid],
            val_side=val_side_all[val_valid],
            val_y=y_val[val_valid],
            test_x=test_x_all,
            test_side=test_side_all,
            device=device,
            model_kind=model_kind,
            seed=seed,
            target=target,
        )
        pred_val_full = np.full(len(val_df), np.nan, dtype=float)
        pred_val_full[val_valid] = val_pred
        out[target] = {"val": pred_val_full, "test": test_pred}
    return out

def fit_garch_oos(train_returns: np.ndarray, test_returns: np.ndarray, horizon: int = 1, model: str = "GARCH") -> np.ndarray:
    # Retained for backward compatibility, but full evaluation now uses
    # garch_forecast_on_anchor_dates() with full daily price history.
    try:
        from arch import arch_model
    except ImportError:
        logger.warning("arch package not found. GARCH baseline unavailable. Install with: pip install arch")
        return np.full(len(test_returns), np.nan)

    train_returns = np.asarray(train_returns, dtype=float)
    test_returns = np.asarray(test_returns, dtype=float)
    train_returns = train_returns[np.isfinite(train_returns)]
    if len(train_returns) < 50:
        return np.full(len(test_returns), np.nan)

    am = arch_model(train_returns * 100.0, mean="Zero", vol="Garch", p=1, q=1, dist="normal")
    res = am.fit(disp="off", show_warning=False)

    p = res.params
    omega = float(p.get("omega", 0.0))
    alpha = float(p.get("alpha[1]", 0.0))
    beta = float(p.get("beta[1]", 0.0))
    persistence = alpha + beta

    sigma2_state = float(res.conditional_volatility[-1] ** 2)
    out = np.full(len(test_returns), np.nan, dtype=float)

    for i, r in enumerate(test_returns):
        if not np.isfinite(r):
            continue
        eps_t_pct = float(r) * 100.0
        sigma2_state = max(omega + alpha * eps_t_pct**2 + beta * sigma2_state, 1e-12)
        path_h = np.empty(horizon, dtype=float)
        path_h[0] = sigma2_state
        for h in range(1, horizon):
            path_h[h] = max(omega + persistence * path_h[h - 1], 1e-12)
        avg_var_raw = np.mean(path_h) / 1e4
        out[i] = np.sqrt(avg_var_raw * 252.0)

    return out


def fit_downside_semivar_oos(train_returns, test_returns, horizon=5):
    vol_forecast = fit_garch_oos(train_returns, test_returns, horizon=horizon)
    var_ann = np.square(vol_forecast)
    dsv_ann = 0.5 * var_ann
    return np.sqrt(dsv_ann)


def positive_floor_from_train(y_train: np.ndarray, q: float = 0.05, eps: float = 1e-8) -> float:
    """
    Pre-test floor for semivolatility-style targets with structural zero windows.

    The floor is estimated only from positive training-period observations, so it
    is fixed before validation/test evaluation and does not leak future data.
    """
    y_train = np.asarray(y_train, dtype=float)
    pos = y_train[np.isfinite(y_train) & (y_train > eps)]
    if len(pos) == 0:
        return float(eps)
    return float(max(np.quantile(pos, q), eps))


def qlike_loss_series(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-8,
    mode: str = "standard",
    floor_value: Optional[float] = None,
) -> np.ndarray:
    """
    Volatility-scale QLIKE series.

    mode='standard':
        Evaluate only finite rows with y_true>eps and y_pred>eps. Non-positive
        rows remain NaN rather than being assigned an arbitrary fixed penalty.
    mode='floor':
        Replace both y_true and y_pred by max(value, floor_value). This is the
        recommended robustness diagnostic for DSV5, whose realized target has
        structural zero windows.
    mode='positive_subset':
        Alias of standard, kept for explicit diagnostics.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = np.full_like(y_true, np.nan, dtype=float)

    finite = np.isfinite(y_true) & np.isfinite(y_pred)

    if mode in ("standard", "positive_subset"):
        valid = finite & (y_true > eps) & (y_pred > eps)
        if np.any(valid):
            yt = y_true[valid]
            yp = y_pred[valid]
        else:
            return out
    elif mode == "floor":
        if floor_value is None or not np.isfinite(floor_value) or floor_value <= 0:
            raise ValueError("mode='floor' requires a positive finite floor_value.")
        valid = finite
        if np.any(valid):
            yt = np.maximum(y_true[valid], floor_value)
            yp = np.maximum(y_pred[valid], floor_value)
        else:
            return out
    else:
        raise ValueError(f"Unknown QLIKE mode: {mode}")

    ratio = (yt ** 2) / np.maximum(yp ** 2, eps)
    out[valid] = ratio - np.log(np.maximum(ratio, eps)) - 1.0
    return out


def legacy_penalized_qlike_loss_series(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-8,
    invalid_penalty: float = 100.0,
) -> np.ndarray:
    """
    Backward-compatible diagnostic for DSV5. Structural-zero windows make the
    fixed penalty dominate the average.
    """
    out = qlike_loss_series(y_true, y_pred, eps=eps, mode="standard")
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    positive = finite & (y_true > eps) & (y_pred > eps)
    out[finite & ~positive] = invalid_penalty
    return out


def qlike_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-8,
    mode: str = "standard",
    floor_value: Optional[float] = None,
) -> float:
    loss = qlike_loss_series(y_true, y_pred, eps=eps, mode=mode, floor_value=floor_value)
    return float(np.nanmean(loss)) if np.isfinite(loss).any() else np.nan


def validation_regression_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target: str,
    train_y: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> float:
    """
    Unified validation-selection score for regression baselines.

    RV5/RV21 use standard positive-domain QLIKE, matching the main risk-sensitive
    comparison. DSV5 has structural zero windows, so candidate models are selected
    by MAE rather than by standard/positive-subset QLIKE. This keeps DSV5 model
    selection aligned with the revised main-reporting rule: DSV5 primary metrics
    are RMSE/MAE, while floored QLIKE is only a robustness diagnostic.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() == 0:
        return np.inf

    yt = y_true[valid]
    yp = np.maximum(y_pred[valid], eps)

    if str(target).lower() == "dsv5":
        return float(np.mean(np.abs(yt - yp)))

    score = qlike_loss(yt, yp, eps=eps, mode="standard")
    return float(score) if np.isfinite(score) else np.inf


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-8,
    task: Optional[str] = None,
    dsv5_floor: Optional[float] = None,
) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[valid], y_pred[valid]
    if len(yt) == 0:
        return {
            "rmse": np.nan, "mae": np.nan, "qlike": np.nan,
            "qlike_floored": np.nan, "qlike_positive_subset": np.nan,
            "invalid_pred_rate": np.nan, "zero_y_rate": np.nan, "n": 0,
        }

    out = {
        "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
        "mae": float(np.mean(np.abs(yt - yp))),
        "invalid_pred_rate": float(np.mean(yp <= eps)),
        "zero_y_rate": float(np.mean(yt <= eps)),
        "n": int(valid.sum()),
    }

    if task == "dsv5":
        # DSV5 has structural zero windows. Main tables should use RMSE/MAE.
        # Floored and positive-subset QLIKE are kept as robustness diagnostics.
        out["qlike"] = np.nan
        out["qlike_positive_subset"] = qlike_loss(yt, yp, eps=eps, mode="positive_subset")
        out["qlike_floored"] = (
            qlike_loss(yt, yp, eps=eps, mode="floor", floor_value=dsv5_floor)
            if dsv5_floor is not None else np.nan
        )
    else:
        out["qlike"] = qlike_loss(yt, yp, eps=eps, mode="standard")
        out["qlike_positive_subset"] = np.nan
        out["qlike_floored"] = np.nan

    return out


def dsv5_zero_diagnostics(
    y_true: np.ndarray,
    pred_map: Dict[str, np.ndarray],
    train_y: np.ndarray,
    eps: float = 1e-8,
    floor_q: float = 0.05,
) -> pd.DataFrame:
    """Create Appendix diagnostics for the structural-zero DSV5 target."""
    y_true = np.asarray(y_true, dtype=float)
    floor = positive_floor_from_train(train_y, q=floor_q, eps=eps)
    rows = []

    for name, y_pred in pred_map.items():
        y_pred = np.asarray(y_pred, dtype=float)
        finite = np.isfinite(y_true) & np.isfinite(y_pred)
        if finite.sum() == 0:
            rows.append({
                "model": name, "n": 0, "zero_y_rate": np.nan,
                "invalid_old_rate": np.nan, "old_penalty_contribution": np.nan,
                "dsv5_floor": floor, "qlike_positive_subset": np.nan,
                "qlike_floored": np.nan, "rmse": np.nan, "mae": np.nan,
            })
            continue

        invalid_old = finite & ~((y_true > eps) & (y_pred > eps))
        loss_pos = qlike_loss_series(y_true, y_pred, eps=eps, mode="positive_subset")
        loss_floor = qlike_loss_series(y_true, y_pred, eps=eps, mode="floor", floor_value=floor)

        yt = y_true[finite]
        yp = y_pred[finite]
        rows.append({
            "model": name,
            "n": int(finite.sum()),
            "zero_y_rate": float(np.mean(yt <= eps)),
            "invalid_old_rate": float(np.mean(invalid_old[finite])),
            "old_penalty_contribution": float(np.mean(invalid_old[finite]) * 100.0),
            "dsv5_floor": floor,
            "qlike_positive_subset": float(np.nanmean(loss_pos)) if np.isfinite(loss_pos).any() else np.nan,
            "qlike_floored": float(np.nanmean(loss_floor)) if np.isfinite(loss_floor).any() else np.nan,
            "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
            "mae": float(np.mean(np.abs(yt - yp))),
        })

    return pd.DataFrame(rows)




def save_forecast_prediction_panel(
    out_path: str,
    test_df: pd.DataFrame,
    y_true_map: Dict[str, np.ndarray],
    pred_map: Dict[str, Dict[str, np.ndarray]],
) -> str:
    """
    Save per-date targets and predictions for appendix diagnostics.

    This is a diagnostic artifact only. It does not change the main forecasting,
    DM, or surface results. It makes DSV5 zero-window diagnostics, floor
    sensitivity, and future metric audits reproducible from a single panel.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if len(y_true_map) == 0:
        raise ValueError("y_true_map is empty; cannot build prediction panel.")
    first_y = np.asarray(next(iter(y_true_map.values())), dtype=float)
    n = len(first_y)

    panel = pd.DataFrame()
    if "date" in test_df.columns:
        if len(test_df) != n:
            raise ValueError(f"test_df has length {len(test_df)}, but targets have length {n}.")
        panel["date"] = pd.to_datetime(test_df["date"]).dt.strftime("%Y-%m-%d")
    else:
        panel["date"] = np.arange(n)

    for target, y in y_true_map.items():
        arr = np.asarray(y, dtype=float)
        if len(arr) != n:
            raise ValueError(f"Target {target} has length {len(arr)}, expected {n}.")
        panel[f"y_{target}"] = arr

    for model_name, task_preds in pred_map.items():
        safe_model = str(model_name).replace("/", "_").replace(" ", "_")
        for target, pred in task_preds.items():
            arr = np.asarray(pred, dtype=float)
            if len(arr) != n:
                raise ValueError(
                    f"Prediction length mismatch for {model_name}/{target}: "
                    f"{len(arr)} vs expected {n}."
                )
            panel[f"{safe_model}__{target}__pred"] = arr

    panel.to_csv(out_path, index=False)
    logger.info(f"Forecast prediction panel saved -> {out_path}")
    return out_path

def build_dsv5_floor_sensitivity_from_prediction_panel(
    panel_path: str,
    train_dsv5: np.ndarray,
    out_path: str,
    floor_qs: Tuple[float, ...] = (0.01, 0.05, 0.10),
    eps: float = 1e-8,
) -> pd.DataFrame:
    """
    Recompute DSV5 floored QLIKE under several training-only floor quantiles.

    This is a robustness table for the nonstandard DSV5 F-QLIKE diagnostic.
    It uses only test-panel predictions and training-period DSV5 values.
    """
    if not os.path.exists(panel_path):
        raise FileNotFoundError(f"Prediction panel not found: {panel_path}")
    panel = pd.read_csv(panel_path)
    if "y_dsv5" not in panel.columns:
        raise ValueError(f"Prediction panel {panel_path} does not contain y_dsv5.")

    y = panel["y_dsv5"].to_numpy(dtype=float)
    pred_cols = [c for c in panel.columns if c.endswith("__dsv5__pred")]
    rows = []
    for q in floor_qs:
        floor = positive_floor_from_train(train_dsv5, q=q, eps=eps)
        for col in pred_cols:
            model = col.replace("__dsv5__pred", "")
            pred = panel[col].to_numpy(dtype=float)
            valid = np.isfinite(y) & np.isfinite(pred)
            m = regression_metrics(y, pred, task="dsv5", dsv5_floor=floor)
            rows.append({
                "floor_q": q,
                "dsv5_floor": floor,
                "model": model,
                "n": int(valid.sum()),
                "zero_y_rate": float(np.mean(y[valid] <= eps)) if valid.any() else np.nan,
                "invalid_pred_rate": float(np.mean(pred[valid] <= eps)) if valid.any() else np.nan,
                "qlike_floored": qlike_loss(y, pred, eps=eps, mode="floor", floor_value=floor),
                "qlike_positive_subset": qlike_loss(y, pred, eps=eps, mode="positive_subset"),
                "rmse": m.get("rmse", np.nan),
                "mae": m.get("mae", np.nan),
            })
    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    logger.info(f"DSV5 floor-sensitivity table saved -> {out_path}")
    return out_df


def build_dsv5_zero_diagnostics_from_prediction_panel(
    panel_path: str,
    train_dsv5: np.ndarray,
    out_path: str,
    floor_q: float = 0.05,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """
    Rebuild the DSV5 zero-window diagnostics from the saved prediction panel.

    This lets --paper-diagnostics --diag-dsv5 regenerate both the zero-window
    table and the floor-sensitivity table without rerunning model evaluation.
    """
    if not os.path.exists(panel_path):
        raise FileNotFoundError(f"Prediction panel not found: {panel_path}")
    panel = pd.read_csv(panel_path)
    if "y_dsv5" not in panel.columns:
        raise ValueError(f"Prediction panel {panel_path} does not contain y_dsv5.")

    y = panel["y_dsv5"].to_numpy(dtype=float)
    pred_cols = [c for c in panel.columns if c.endswith("__dsv5__pred")]
    pred_map = {
        c.replace("__dsv5__pred", ""): panel[c].to_numpy(dtype=float)
        for c in pred_cols
    }
    out_df = dsv5_zero_diagnostics(
        y_true=y,
        pred_map=pred_map,
        train_y=train_dsv5,
        eps=eps,
        floor_q=floor_q,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    logger.info(f"DSV5 zero-window diagnostics saved -> {out_path}")
    return out_df

def _collect_target_from_cache(cache_path: str, split: str, target: str, cfg_data: DataConfig = DATA_CFG) -> np.ndarray:
    ds = OptionDataset(cache_path, split, cfg_data)
    vals = []
    for rec in ds.records:
        vals.append(rec.get(target, np.nan))
    return np.asarray(vals, dtype=float)


def audit_csv_parquet_equivalence(
    data_dir: str,
    stems: List[str],
    out_path: str,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> pd.DataFrame:
    """Full-value CSV/Parquet audit; stronger than shape/head checks.

    Date-like columns are normalized before comparison to avoid false positives
    such as ``2018-01-02`` versus ``2018-01-02 00:00:00``.
    """
    rows = []

    def normalize_obj(s0: pd.Series) -> pd.Series:
        return s0.astype("string").str.strip().fillna("<NA>")

    def datetime_mismatch(a0: pd.Series, b0: pd.Series) -> Optional[np.ndarray]:
        a_dt = pd.to_datetime(a0, errors="coerce")
        b_dt = pd.to_datetime(b0, errors="coerce")
        parse_share = min(a_dt.notna().mean(), b_dt.notna().mean())
        if parse_share < 0.95:
            return None
        a_key = a_dt.dt.normalize().astype("string").fillna("<NA>")
        b_key = b_dt.dt.normalize().astype("string").fillna("<NA>")
        return a_key.ne(b_key).to_numpy()

    for stem in stems:
        csv_path = Path(data_dir) / f"{stem}.csv"
        pq_path = Path(data_dir) / f"{stem}.parquet"
        if not csv_path.exists() or not pq_path.exists():
            rows.append({
                "file": stem, "status": "missing",
                "csv_exists": bool(csv_path.exists()), "parquet_exists": bool(pq_path.exists()),
                "csv_shape": "", "parquet_shape": "", "shape_ok": False, "cols_ok": False,
                "column": "", "n_mismatch": np.nan, "max_abs_diff": np.nan,
            })
            continue

        df_csv = pd.read_csv(csv_path, low_memory=False)
        df_pq = pd.read_parquet(pq_path)
        shape_ok = df_csv.shape == df_pq.shape
        cols_ok = list(df_csv.columns) == list(df_pq.columns)
        if not shape_ok or not cols_ok:
            rows.append({
                "file": stem, "status": "shape_or_column_mismatch",
                "csv_exists": True, "parquet_exists": True,
                "csv_shape": str(df_csv.shape), "parquet_shape": str(df_pq.shape),
                "shape_ok": shape_ok, "cols_ok": cols_ok,
                "column": "", "n_mismatch": np.nan, "max_abs_diff": np.nan,
            })
            continue

        any_bad = False
        for col in df_csv.columns:
            a = df_csv[col]
            b = df_pq[col]

            date_neq = None
            if "date" in str(col).lower() or str(col).lower() in {"expiration", "exdate"}:
                date_neq = datetime_mismatch(a, b)
            if date_neq is not None:
                if bool(date_neq.any()):
                    any_bad = True
                    rows.append({
                        "file": stem, "status": "value_mismatch",
                        "csv_exists": True, "parquet_exists": True,
                        "csv_shape": str(df_csv.shape), "parquet_shape": str(df_pq.shape),
                        "shape_ok": True, "cols_ok": True,
                        "column": col, "n_mismatch": int(date_neq.sum()),
                        "max_abs_diff": np.nan,
                    })
                continue

            a_num = pd.to_numeric(a, errors="coerce")
            b_num = pd.to_numeric(b, errors="coerce")
            numeric_share = min(a_num.notna().mean(), b_num.notna().mean())
            if numeric_share > 0.95:
                av = a_num.to_numpy(dtype=float)
                bv = b_num.to_numpy(dtype=float)
                ok_mask = np.isclose(av, bv, rtol=rtol, atol=atol, equal_nan=True)
                if not bool(ok_mask.all()):
                    any_bad = True
                    diff = np.abs(av - bv)
                    rows.append({
                        "file": stem, "status": "value_mismatch",
                        "csv_exists": True, "parquet_exists": True,
                        "csv_shape": str(df_csv.shape), "parquet_shape": str(df_pq.shape),
                        "shape_ok": True, "cols_ok": True,
                        "column": col, "n_mismatch": int((~ok_mask).sum()),
                        "max_abs_diff": float(np.nanmax(diff)) if np.isfinite(diff).any() else np.nan,
                    })
            else:
                aa = normalize_obj(a)
                bb = normalize_obj(b)
                neq = aa.ne(bb)
                if bool(neq.any()):
                    any_bad = True
                    rows.append({
                        "file": stem, "status": "value_mismatch",
                        "csv_exists": True, "parquet_exists": True,
                        "csv_shape": str(df_csv.shape), "parquet_shape": str(df_pq.shape),
                        "shape_ok": True, "cols_ok": True,
                        "column": col, "n_mismatch": int(neq.sum()),
                        "max_abs_diff": np.nan,
                    })
        if not any_bad:
            rows.append({
                "file": stem, "status": "ok",
                "csv_exists": True, "parquet_exists": True,
                "csv_shape": str(df_csv.shape), "parquet_shape": str(df_pq.shape),
                "shape_ok": True, "cols_ok": True,
                "column": "", "n_mismatch": 0, "max_abs_diff": 0.0,
            })

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    logger.info(f"CSV/Parquet audit saved -> {out_path}")
    return out_df

def audit_iv_inversion_failures(
    cfg: DataConfig = DATA_CFG,
    out_path: Optional[str] = None,
    max_rows_per_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Diagnostic-only scan of midpoint-IV inversion edge cases.

    The function does not change preprocessing or cached quote sets. It reports
    midpoint-at/below-intrinsic and direct midpoint-IV inversion failures both
    after the k-filter stage and after the retained quote filters used by the
    production preprocessing path (mid-IV target, relative spread, settlement,
    and OTM/ATM representative selection).
    """
    if out_path is None:
        out_path = os.path.join(cfg.results_dir, "diagnostics", "iv_inversion_diagnostics.csv")

    aux = load_aux_data(cfg)
    prices_df = aux["prices"]
    zcy_df = aux["zcy"]
    div_df = aux["div"]
    data_dir = resolve_data_dir(cfg)
    rows = []

    def _stage_iv_stats(grp_stage: pd.DataFrame, prefix: str) -> Dict[str, float]:
        if len(grp_stage) == 0:
            return {
                f"n_{prefix}": 0,
                f"n_{prefix}_mid_le_intrinsic": 0,
                f"{prefix}_mid_le_intrinsic_rate": np.nan,
                f"n_{prefix}_near_atm": 0,
                f"n_{prefix}_near_atm_mid_le_intrinsic": 0,
                f"{prefix}_near_atm_mid_le_intrinsic_rate": np.nan,
                f"n_{prefix}_mid_iv_finite": 0,
                f"{prefix}_mid_iv_failure_rate": np.nan,
            }
        mid = 0.5 * (grp_stage["best_bid"].values.astype(float) + grp_stage["best_offer"].values.astype(float))
        F0 = grp_stage["F"].values.astype(float)
        K0 = grp_stage["K"].values.astype(float)
        tau = grp_stage["tau"].values.astype(float)
        Dr = grp_stage["D_r"].values.astype(float)
        is_call = grp_stage["cp_flag"].astype(str).str.upper().eq("C").values
        intrinsic = Dr * np.maximum(np.where(is_call, F0 - K0, K0 - F0), 0.0)
        mid_le_intrinsic = mid <= intrinsic + 1e-6
        near_atm = np.abs(grp_stage["k"].values.astype(float)) <= cfg.atm_eps

        iv_mid = []
        for m, f, k0, t0, d0, c in zip(mid, F0, K0, tau, Dr, is_call):
            val = _bs_iv_np(float(m), float(f), float(k0), float(t0), float(d0), bool(c))
            iv_mid.append(np.nan if val is None else float(val))
        iv_mid = np.asarray(iv_mid, dtype=float)
        finite_iv = np.isfinite(iv_mid) & (iv_mid > 0.02) & (iv_mid < 5.0)
        n = len(grp_stage)
        n_near = int(near_atm.sum())
        return {
            f"n_{prefix}": int(n),
            f"n_{prefix}_mid_le_intrinsic": int(mid_le_intrinsic.sum()),
            f"{prefix}_mid_le_intrinsic_rate": float(mid_le_intrinsic.mean()) if n else np.nan,
            f"n_{prefix}_near_atm": n_near,
            f"n_{prefix}_near_atm_mid_le_intrinsic": int((near_atm & mid_le_intrinsic).sum()),
            f"{prefix}_near_atm_mid_le_intrinsic_rate": float((near_atm & mid_le_intrinsic).sum() / max(n_near, 1)),
            f"n_{prefix}_mid_iv_finite": int(finite_iv.sum()),
            f"{prefix}_mid_iv_failure_rate": float(1.0 - finite_iv.sum() / max(n, 1)),
        }

    for year in cfg.years:
        filepath = Path(data_dir) / cfg.option_file_fmt.format(year=year)
        if not filepath.exists():
            rows.append({"year": year, "date": "", "status": "missing_file"})
            continue
        df_year = pd.read_parquet(filepath)
        df_year["date"] = pd.to_datetime(df_year["date"])
        df_year["exdate"] = pd.to_datetime(df_year["exdate"])
        df_year = df_year[df_year["secid"] == cfg.secid]
        if max_rows_per_year is not None and len(df_year) > max_rows_per_year:
            df_year = df_year.sample(n=max_rows_per_year, random_state=cfg.secid + int(year)).sort_index()

        df_year = df_year[df_year["exercise_style"].fillna("E") == "E"]
        df_year = df_year[df_year["best_bid"] >= cfg.min_bid]
        df_year = df_year[df_year["best_offer"] > df_year["best_bid"]]
        if cfg.min_open_interest > 0 and "open_interest" in df_year.columns:
            df_year = df_year[df_year["open_interest"].fillna(0) >= cfg.min_open_interest]
        if len(df_year) == 0:
            rows.append({"year": year, "date": "", "status": "empty_after_basic_filters"})
            continue

        df_year["K"] = df_year["strike_price"] / 1000.0
        df_year["tau"] = (df_year["exdate"] - df_year["date"]).dt.days / 365.0
        df_year = df_year[(df_year["tau"] >= cfg.tau_min) & (df_year["tau"] <= cfg.tau_max)]
        if len(df_year) == 0:
            rows.append({"year": year, "date": "", "status": "empty_after_tau_filter"})
            continue

        for tdate, grp0 in df_year.groupby("date"):
            price_row = prices_df[prices_df["date"] == tdate]
            if len(price_row) == 0:
                continue
            S = float(price_row["close"].iloc[0])
            zcy_day = zcy_df[zcy_df["date"] == tdate]
            div_day = div_df[div_df["date"] == tdate]

            grp = grp0.copy()
            tau_arr = grp["tau"].values.astype(float)
            F_arr, Dr_arr = build_forward_for_date(tdate, S, tau_arr, zcy_day, div_day)
            grp["F"] = F_arr
            grp["D_r"] = Dr_arr
            grp["k"] = np.log(grp["K"].values / F_arr)
            grp_k = grp[(grp["k"] >= cfg.k_min) & (grp["k"] <= cfg.k_max)].copy()
            if len(grp_k) == 0:
                continue

            row = {"year": year, "date": str(pd.Timestamp(tdate).date()), "status": "ok"}
            row.update(_stage_iv_stats(grp_k, "after_k"))

            grp_ret = _compute_iv(grp_k)
            grp_ret = grp_ret[grp_ret["iv"].notna() & (grp_ret["iv"] > 0.02) & (grp_ret["iv"] < 5.0)].copy()
            row["n_after_vendor_or_fallback_iv"] = int(len(grp_ret))
            if len(grp_ret) > 0:
                grp_ret["mid_iv_target"] = _compute_mid_iv_target(grp_ret)
                grp_ret = grp_ret[
                    np.isfinite(grp_ret["mid_iv_target"]) &
                    (grp_ret["mid_iv_target"] > 0.02) &
                    (grp_ret["mid_iv_target"] < 5.0)
                ].copy()
            row["n_after_mid_iv_target"] = int(len(grp_ret))

            if len(grp_ret) > 0:
                grp_ret["iv"] = grp_ret["mid_iv_target"].astype(float)
                grp_ret["w"] = grp_ret["tau"] * grp_ret["iv"] ** 2
                grp_ret["mid"] = 0.5 * (grp_ret["best_bid"] + grp_ret["best_offer"])
                grp_ret["rel_spread"] = (grp_ret["best_offer"] - grp_ret["best_bid"]) / grp_ret["mid"].clip(lower=1e-4)
                grp_ret = grp_ret[grp_ret["rel_spread"] <= cfg.max_rel_spread].copy()
            row["n_after_rel_spread"] = int(len(grp_ret))

            if len(grp_ret) > 0:
                grp_ret["vega_bs"] = _compute_vega(grp_ret, iv_col="iv")
                if "am_set_flag" in grp_ret.columns:
                    grp_ret["am_flag"] = grp_ret["am_set_flag"].fillna(0).astype(int)
                elif "am_settlement" in grp_ret.columns:
                    grp_ret["am_flag"] = grp_ret["am_settlement"].fillna(0).astype(int)
                else:
                    grp_ret["am_flag"] = 0
                grp_ret = _apply_settlement_policy(grp_ret, cfg.settlement_policy)
            row["n_after_settlement"] = int(len(grp_ret))

            if len(grp_ret) > 0:
                grp_final = _apply_otm_rule(grp_ret, cfg.atm_eps).copy()
            else:
                grp_final = grp_ret
            row["n_after_otm_atm_selection"] = int(len(grp_final))
            row.update(_stage_iv_stats(grp_final, "final_retained"))

            row["n_after_basic_filters"] = row.get("n_after_k", 0)
            row["n_mid_le_intrinsic"] = row.get("n_after_k_mid_le_intrinsic", 0)
            row["mid_le_intrinsic_rate"] = row.get("after_k_mid_le_intrinsic_rate", np.nan)
            row["n_near_atm"] = row.get("n_after_k_near_atm", 0)
            row["n_near_atm_mid_le_intrinsic"] = row.get("n_after_k_near_atm_mid_le_intrinsic", 0)
            row["near_atm_mid_le_intrinsic_rate"] = row.get("after_k_near_atm_mid_le_intrinsic_rate", np.nan)
            row["n_mid_iv_finite"] = row.get("n_after_k_mid_iv_finite", 0)
            row["mid_iv_failure_rate"] = row.get("after_k_mid_iv_failure_rate", np.nan)
            rows.append(row)

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    summary_path = out_path.replace(".csv", "_summary.csv")
    if not out_df.empty and "n_after_k" in out_df.columns:
        ok = out_df[out_df.get("status", "") == "ok"].copy()
        if not ok.empty:
            def _rate(num_col: str, den_col: str) -> float:
                num = float(ok.get(num_col, pd.Series(dtype=float)).sum())
                den = float(ok.get(den_col, pd.Series(dtype=float)).sum())
                return num / max(den, 1.0)

            summary = pd.DataFrame([{
                "n_dates": int(len(ok)),
                "n_after_k": int(ok["n_after_k"].sum()),
                "after_k_mid_le_intrinsic_rate": _rate("n_after_k_mid_le_intrinsic", "n_after_k"),
                "after_k_mid_iv_failure_rate": float(1.0 - ok["n_after_k_mid_iv_finite"].sum() / max(ok["n_after_k"].sum(), 1)),
                "n_after_otm_atm_selection": int(ok["n_after_otm_atm_selection"].sum()),
                "final_retained_mid_le_intrinsic_rate": _rate("n_final_retained_mid_le_intrinsic", "n_final_retained"),
                "final_retained_mid_iv_failure_rate": float(1.0 - ok["n_final_retained_mid_iv_finite"].sum() / max(ok["n_final_retained"].sum(), 1)),
                "final_retained_near_atm_mid_le_intrinsic_rate": _rate("n_final_retained_near_atm_mid_le_intrinsic", "n_final_retained_near_atm"),
            }])
            summary.to_csv(summary_path, index=False)
            logger.info(f"IV inversion diagnostic summary saved -> {summary_path}")
    logger.info(f"IV inversion diagnostics saved -> {out_path}")
    return out_df

def run_paper_diagnostics(
    diag_csv_parquet: bool = False,
    diag_dsv5: bool = False,
    diag_iv_failures: bool = False,
    max_rows_per_year: Optional[int] = None,
) -> None:
    """Run diagnostic-only audits without changing trained models or main results."""
    diag_dir = os.path.join(DATA_CFG.results_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    if diag_csv_parquet:
        stems = [
            *(f"data_{y}" for y in DATA_CFG.years),
            "security_prices", "index_dividend_yield", "zero_coupon_yield",
            "VIXCLS", "VXVCLS",
        ]
        audit_csv_parquet_equivalence(
            data_dir=str(resolve_data_dir(DATA_CFG)),
            stems=list(stems),
            out_path=os.path.join(diag_dir, "csv_parquet_audit.csv"),
        )

    if diag_dsv5:
        cache_path = os.path.join(DATA_CFG.cache_dir, "daily_quotes.pkl")
        panel_path = os.path.join(diag_dir, "test_prediction_panel.csv")
        if not os.path.exists(panel_path):
            logger.warning(
                f"Prediction panel not found at {panel_path}. Run step 4 evaluation first; "
                "the patched full_evaluation writes this panel automatically."
            )
        else:
            train_dsv5 = _collect_target_from_cache(cache_path, "train", "dsv5", DATA_CFG)
            build_dsv5_zero_diagnostics_from_prediction_panel(
                panel_path=panel_path,
                train_dsv5=train_dsv5,
                out_path=os.path.join(diag_dir, "dsv5_zero_diagnostics.csv"),
                floor_q=0.05,
            )
            build_dsv5_floor_sensitivity_from_prediction_panel(
                panel_path=panel_path,
                train_dsv5=train_dsv5,
                out_path=os.path.join(diag_dir, "dsv5_floor_sensitivity.csv"),
                floor_qs=(0.01, 0.05, 0.10),
            )

    if diag_iv_failures:
        audit_iv_inversion_failures(
            cfg=DATA_CFG,
            out_path=os.path.join(diag_dir, "iv_inversion_diagnostics.csv"),
            max_rows_per_year=max_rows_per_year,
        )

def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: Optional[float] = None) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    valid = ~(np.isnan(y_true) | np.isnan(y_score))
    yt = y_true[valid].astype(int)
    ys = y_score[valid]
    if len(yt) == 0 or len(np.unique(yt)) < 2:
        return {"auc": np.nan, "f1": np.nan, "bal_acc": np.nan, "n": int(valid.sum())}

    out = {"auc": float(roc_auc_score(yt, ys)), "f1": np.nan, "bal_acc": np.nan, "n": int(valid.sum())}
    if threshold is not None:
        yp = (ys >= float(threshold)).astype(int)
        out["f1"] = float(f1_score(yt, yp, zero_division=0))
        out["bal_acc"] = float(balanced_accuracy_score(yt, yp))
    return out


def choose_threshold_on_validation(y_true: np.ndarray, y_score: np.ndarray,
                                   objective: str = "bal_acc", grid: Optional[np.ndarray] = None) -> float:
    valid = ~(np.isnan(y_true) | np.isnan(y_score))
    yt, ys = y_true[valid].astype(int), y_score[valid]
    if grid is None:
        grid = np.linspace(0.05, 0.95, 181)

    best_thr = 0.5
    best_val = -np.inf
    for thr in grid:
        yp = (ys >= thr).astype(int)
        if objective == "bal_acc":
            val = balanced_accuracy_score(yt, yp)
        elif objective == "f1":
            val = f1_score(yt, yp, zero_division=0)
        else:
            raise ValueError(f"Unknown objective={objective}")
        if val > best_val:
            best_val = val
            best_thr = float(thr)
    return best_thr


def block_bootstrap_dm(loss_a: np.ndarray, loss_b: np.ndarray, block_len: int = 22,
                       n_boot: int = 2000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    loss_a = np.asarray(loss_a, dtype=float)
    loss_b = np.asarray(loss_b, dtype=float)
    valid = ~(np.isnan(loss_a) | np.isnan(loss_b))
    d = loss_a[valid] - loss_b[valid]
    n = len(d)
    if n < max(30, block_len):
        return {"p_value": np.nan, "obs_diff": np.nan, "n": int(n)}

    obs_diff = float(np.mean(d))
    d0 = d - obs_diff

    ge = 0
    n_blocks = int(np.ceil(n / block_len))
    for _ in range(n_boot):
        starts = rng.integers(0, n - block_len + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block_len, n)) for s in starts])[:n]
        boot_diff = float(np.mean(d0[idx]))
        if abs(boot_diff) >= abs(obs_diff):
            ge += 1
    return {"p_value": float((ge + 1) / (n_boot + 1)), "obs_diff": obs_diff, "n": int(n)}


def full_evaluation(surface_model, heads: DownstreamHeads, cache_path: str,
                    cfg_data: DataConfig = DATA_CFG, cfg_model: ModelConfig = MODEL_CFG,
                    cfg_train: TrainConfig = TRAIN_CFG,
                    primary_model_name: str = "OurModel",
                    use_side_info: bool = True,
                    include_reconstruction: bool = True,
                    include_grid_appendix: bool = True,
                    include_grid_side_appendix: bool = False) -> Dict[str, dict]:
    device = torch.device(cfg_train.device if torch.cuda.is_available() else "cpu")
    surface_model.eval()
    heads.eval()

    train_ds = OptionDataset(cache_path, "train", cfg_data)
    val_ds = OptionDataset(cache_path, "val", cfg_data)
    test_ds = OptionDataset(cache_path, "test", cfg_data)

    train_ld = DataLoader(train_ds, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_ld = DataLoader(test_ds, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)

    with torch.no_grad():
        train_df = extract_latents(surface_model, train_ld, device)
        val_df_full = extract_latents(surface_model, val_ld, device)
        test_df_full = extract_latents(surface_model, test_ld, device)

    _vv = val_df_full["_valid"].values.astype(bool) if "_valid" in val_df_full.columns else val_df_full[["rv5", "rv21", "dsv5", "stress21"]].notna().all(axis=1).values
    _tv = test_df_full["_valid"].values.astype(bool) if "_valid" in test_df_full.columns else test_df_full[["rv5", "rv21", "dsv5", "stress21"]].notna().all(axis=1).values

    n_val_total = len(val_df_full)
    n_test_total = len(test_df_full)

    val_df = val_df_full[_vv].reset_index(drop=True)
    test_df = test_df_full[_tv].reset_index(drop=True)

    logger.info(f"Val rows after valid-target filter: {len(val_df)} / {n_val_total} (dropped {n_val_total - len(val_df)} tail-NaN rows)")
    logger.info(f"Test rows after valid-target filter: {len(test_df)} / {n_test_total} (dropped {n_test_total - len(test_df)} tail-NaN rows)")

    d = cfg_model.latent_dim
    z_cols = [f"z_{j}" for j in range(d)]
    Z_val = torch.tensor(val_df[z_cols].values, dtype=torch.float32, device=device)
    Z_test = torch.tensor(test_df[z_cols].values, dtype=torch.float32, device=device)
    if use_side_info:
        side_val = torch.tensor(val_df[["VIX", "VXV"]].replace([np.inf, -np.inf], np.nan).fillna(20.0).values, dtype=torch.float32, device=device)
        side_test = torch.tensor(test_df[["VIX", "VXV"]].replace([np.inf, -np.inf], np.nan).fillna(20.0).values, dtype=torch.float32, device=device)
    else:
        side_val = None
        side_test = None

    with torch.no_grad():
        out_val = heads(Z_val, side_val)
        out_test = heads(Z_test, side_test)

    preds_val = {
        "rv5": out_val["rv5"].detach().cpu().numpy(),
        "rv21": out_val["rv21"].detach().cpu().numpy(),
        "dsv5": out_val["dsv5"].detach().cpu().numpy(),
        "stress21": torch.sigmoid(out_val["stress21"]).detach().cpu().numpy(),
    }
    preds_test = {
        "rv5": out_test["rv5"].detach().cpu().numpy(),
        "rv21": out_test["rv21"].detach().cpu().numpy(),
        "dsv5": out_test["dsv5"].detach().cpu().numpy(),
        "stress21": torch.sigmoid(out_test["stress21"]).detach().cpu().numpy(),
    }

    y_val_rv5 = val_df["rv5"].values.astype(float)
    y_val_rv21 = val_df["rv21"].values.astype(float)
    y_val_dsv5 = val_df["dsv5"].values.astype(float)
    y_val_stress = val_df["stress21"].values.astype(float)

    y_rv5 = test_df["rv5"].values.astype(float)
    y_rv21 = test_df["rv21"].values.astype(float)
    y_dsv5 = test_df["dsv5"].values.astype(float)
    y_stress = test_df["stress21"].values.astype(float)

    # DSV5 can be exactly zero when no negative return occurs over the forward
    # five-day window. Use a training-only positive floor for robustness QLIKE.
    dsv5_floor = positive_floor_from_train(train_df["dsv5"].values, q=0.05, eps=1e-8)
    logger.info(f"DSV5 robustness floor = {dsv5_floor:.8g} "
                f"(5th percentile of positive train DSV5)")

    # -----------------------------
    # Baselines: HAR / HAR+Opt / HAR-Summary
    # -----------------------------
    har_rv5 = HARModel(use_options=False, serving_time="close_t")
    har_rv21 = HARModel(use_options=False, serving_time="close_t")
    har_dsv5 = HARModel(use_options=False, serving_time="close_t")

    har_opt_rv5 = HARModel(use_options=True, serving_time="close_t")
    har_opt_rv21 = HARModel(use_options=True, serving_time="close_t")
    har_opt_dsv5 = HARModel(use_options=True, serving_time="close_t")

    har_rv5.fit(train_df["rv5_back"].values, train_df["rv5"].values)
    har_rv21.fit(train_df["rv21_back"].values, train_df["rv21"].values)
    har_dsv5.fit(train_df["dsv5_back"].values, train_df["dsv5"].values)

    har_opt_rv5.fit(train_df["rv5_back"].values, train_df["rv5"].values, train_df["VIX"].values, train_df["VXV"].values)
    har_opt_rv21.fit(train_df["rv21_back"].values, train_df["rv21"].values, train_df["VIX"].values, train_df["VXV"].values)
    har_opt_dsv5.fit(train_df["dsv5_back"].values, train_df["dsv5"].values, train_df["VIX"].values, train_df["VXV"].values)

    har_sum_rv5 = HARSummaryModel().fit(
        train_df["rv5_back"].values, train_df["rv5"].values,
        train_df["atm_iv"].values, train_df["short_iv"].values,
        train_df["term_slope"].values, train_df["skew_proxy"].values,
        train_df["VIX"].values, train_df["VXV"].values
    )
    har_sum_rv21 = HARSummaryModel().fit(
        train_df["rv21_back"].values, train_df["rv21"].values,
        train_df["atm_iv"].values, train_df["short_iv"].values,
        train_df["term_slope"].values, train_df["skew_proxy"].values,
        train_df["VIX"].values, train_df["VXV"].values
    )
    har_sum_dsv5 = HARSummaryModel().fit(
        train_df["dsv5_back"].values, train_df["dsv5"].values,
        train_df["atm_iv"].values, train_df["short_iv"].values,
        train_df["term_slope"].values, train_df["skew_proxy"].values,
        train_df["VIX"].values, train_df["VXV"].values
    )

    val_har_rv5_pred = har_rv5.predict_pseudo_oos(train_df["rv5_back"].values, val_df["rv5_back"].values)
    val_har_rv21_pred = har_rv21.predict_pseudo_oos(train_df["rv21_back"].values, val_df["rv21_back"].values)
    val_har_dsv5_pred = har_dsv5.predict_pseudo_oos(train_df["dsv5_back"].values, val_df["dsv5_back"].values)

    val_har_opt_rv5_pred = har_opt_rv5.predict_pseudo_oos(
        train_df["rv5_back"].values, val_df["rv5_back"].values,
        train_df["VIX"].values, val_df["VIX"].values,
        train_df["VXV"].values, val_df["VXV"].values
    )
    val_har_opt_rv21_pred = har_opt_rv21.predict_pseudo_oos(
        train_df["rv21_back"].values, val_df["rv21_back"].values,
        train_df["VIX"].values, val_df["VIX"].values,
        train_df["VXV"].values, val_df["VXV"].values
    )
    val_har_opt_dsv5_pred = har_opt_dsv5.predict_pseudo_oos(
        train_df["dsv5_back"].values, val_df["dsv5_back"].values,
        train_df["VIX"].values, val_df["VIX"].values,
        train_df["VXV"].values, val_df["VXV"].values
    )

    val_har_sum_rv5_pred = har_sum_rv5.predict(
        val_df["rv5_back"].values, val_df["atm_iv"].values, val_df["short_iv"].values,
        val_df["term_slope"].values, val_df["skew_proxy"].values,
        val_df["VIX"].values, val_df["VXV"].values
    )
    val_har_sum_rv21_pred = har_sum_rv21.predict(
        val_df["rv21_back"].values, val_df["atm_iv"].values, val_df["short_iv"].values,
        val_df["term_slope"].values, val_df["skew_proxy"].values,
        val_df["VIX"].values, val_df["VXV"].values
    )
    val_har_sum_dsv5_pred = har_sum_dsv5.predict(
        val_df["dsv5_back"].values, val_df["atm_iv"].values, val_df["short_iv"].values,
        val_df["term_slope"].values, val_df["skew_proxy"].values,
        val_df["VIX"].values, val_df["VXV"].values
    )

    hist_rv5_back = np.concatenate([train_df["rv5_back"].values, val_df_full["rv5_back"].values])
    hist_rv21_back = np.concatenate([train_df["rv21_back"].values, val_df_full["rv21_back"].values])
    hist_dsv5_back = np.concatenate([train_df["dsv5_back"].values, val_df_full["dsv5_back"].values])
    hist_vix = np.concatenate([train_df["VIX"].values, val_df_full["VIX"].values])
    hist_vxv = np.concatenate([train_df["VXV"].values, val_df_full["VXV"].values])

    har_rv5_pred = har_rv5.predict_pseudo_oos(hist_rv5_back, test_df["rv5_back"].values)
    har_rv21_pred = har_rv21.predict_pseudo_oos(hist_rv21_back, test_df["rv21_back"].values)
    har_dsv5_pred = har_dsv5.predict_pseudo_oos(hist_dsv5_back, test_df["dsv5_back"].values)

    har_opt_rv5_pred = har_opt_rv5.predict_pseudo_oos(
        hist_rv5_back, test_df["rv5_back"].values,
        hist_vix, test_df["VIX"].values,
        hist_vxv, test_df["VXV"].values
    )
    har_opt_rv21_pred = har_opt_rv21.predict_pseudo_oos(
        hist_rv21_back, test_df["rv21_back"].values,
        hist_vix, test_df["VIX"].values,
        hist_vxv, test_df["VXV"].values
    )
    har_opt_dsv5_pred = har_opt_dsv5.predict_pseudo_oos(
        hist_dsv5_back, test_df["dsv5_back"].values,
        hist_vix, test_df["VIX"].values,
        hist_vxv, test_df["VXV"].values
    )

    har_sum_rv5_pred = har_sum_rv5.predict(
        test_df["rv5_back"].values, test_df["atm_iv"].values, test_df["short_iv"].values,
        test_df["term_slope"].values, test_df["skew_proxy"].values,
        test_df["VIX"].values, test_df["VXV"].values
    )
    har_sum_rv21_pred = har_sum_rv21.predict(
        test_df["rv21_back"].values, test_df["atm_iv"].values, test_df["short_iv"].values,
        test_df["term_slope"].values, test_df["skew_proxy"].values,
        test_df["VIX"].values, test_df["VXV"].values
    )
    har_sum_dsv5_pred = har_sum_dsv5.predict(
        test_df["dsv5_back"].values, test_df["atm_iv"].values, test_df["short_iv"].values,
        test_df["term_slope"].values, test_df["skew_proxy"].values,
        test_df["VIX"].values, test_df["VXV"].values
    )

    # -----------------------------
    # New evaluation-only baselines
    # -----------------------------
    boosted_label, boosted_preds = fit_boosted_summary_baselines(train_df, val_df, test_df, seed=cfg_train.seed)

    # Fixed-grid neural baselines are exported to a separate appendix CSV.
    grid_label = "Grid-CNN"
    grid_tr_label = "Grid-Transformer"
    grid_preds = None
    grid_tr_preds = None
    if include_grid_appendix:
        grid_preds = fit_grid_cnn_baselines(train_df, val_df, test_df, cache_path, cfg_data, device=device, seed=cfg_train.seed)
        grid_tr_preds = fit_grid_transformer_baselines(train_df, val_df, test_df, cache_path, cfg_data, device=device, seed=cfg_train.seed)

    # Optional side-information-matched fixed-grid baselines.
    grid_side_label = "Grid-CNN+Side"
    grid_side_tr_label = "Grid-Transformer+Side"
    grid_side_preds = None
    grid_side_tr_preds = None
    if include_grid_side_appendix:
        grid_side_preds = fit_grid_side_baselines(train_df, val_df, test_df, cache_path, cfg_data, device=device, seed=cfg_train.seed, model_kind="cnn")
        grid_side_tr_preds = fit_grid_side_baselines(train_df, val_df, test_df, cache_path, cfg_data, device=device, seed=cfg_train.seed, model_kind="transformer")

    # -----------------------------
    # Calibrated single-factor baselines
    # -----------------------------
    scalar_preds = fit_scalar_calibrated_baselines(train_df, val_df, test_df)
    for _label in ["VIX-alone", "ATM-IV", "ShortEnd-IV"]:
        logger.info(
            f"{_label} calibrated forms | "
            f"rv5={scalar_preds[_label]['rv5']['form']} "
            f"rv21={scalar_preds[_label]['rv21']['form']} "
            f"dsv5={scalar_preds[_label]['dsv5']['form']}"
        )

    # -----------------------------
    # GARCH baselines with full daily history alignment
    # -----------------------------
    prices_full = _load_full_price_history(cfg_data)

    val_dates = pd.to_datetime(val_df["date"].values)
    test_dates = pd.to_datetime(test_df["date"].values)

    val_garch_pred_rv5 = garch_forecast_on_anchor_dates(
        prices_df=prices_full,
        fit_end_date=pd.Timestamp(f"{cfg_data.train_end}-12-31"),
        anchor_dates=val_dates,
        horizon=5,
    )
    val_garch_pred_rv21 = garch_forecast_on_anchor_dates(
        prices_df=prices_full,
        fit_end_date=pd.Timestamp(f"{cfg_data.train_end}-12-31"),
        anchor_dates=val_dates,
        horizon=21,
    )
    val_garch_pred_dsv5 = downside_semivar_forecast_on_anchor_dates(
        prices_df=prices_full,
        fit_end_date=pd.Timestamp(f"{cfg_data.train_end}-12-31"),
        anchor_dates=val_dates,
        horizon=5,
    )

    garch_pred_rv5 = garch_forecast_on_anchor_dates(
        prices_df=prices_full,
        fit_end_date=pd.Timestamp(f"{cfg_data.train_end}-12-31"),
        anchor_dates=test_dates,
        horizon=5,
    )
    garch_pred_rv21 = garch_forecast_on_anchor_dates(
        prices_df=prices_full,
        fit_end_date=pd.Timestamp(f"{cfg_data.train_end}-12-31"),
        anchor_dates=test_dates,
        horizon=21,
    )
    garch_pred_dsv5 = downside_semivar_forecast_on_anchor_dates(
        prices_df=prices_full,
        fit_end_date=pd.Timestamp(f"{cfg_data.train_end}-12-31"),
        anchor_dates=test_dates,
        horizon=5,
    )

    stress_ref = np.nanpercentile(train_df["rv21"].values, cfg_data.stress_quantile * 100)
    stress_scale = np.nanstd(train_df["rv21"].values) + 1e-6

    def stress_score_from_vol(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return 1.0 / (1.0 + np.exp(-(x - stress_ref) / stress_scale))

    val_scores = {
        primary_model_name: preds_val["stress21"],
        "VIX-alone": stress_score_from_vol(scalar_preds["VIX-alone"]["rv21"]["val"]),
        "ATM-IV": stress_score_from_vol(scalar_preds["ATM-IV"]["rv21"]["val"]),
        "ShortEnd-IV": stress_score_from_vol(scalar_preds["ShortEnd-IV"]["rv21"]["val"]),
        "HAR-RV": stress_score_from_vol(val_har_rv21_pred),
        "HAR-RV+Opt": stress_score_from_vol(val_har_opt_rv21_pred),
        "HAR-Summary": stress_score_from_vol(val_har_sum_rv21_pred),
        boosted_label: stress_score_from_vol(boosted_preds["rv21"]["val"]),
        "GARCH": stress_score_from_vol(val_garch_pred_rv21),
    }

    test_scores = {
        primary_model_name: preds_test["stress21"],
        "VIX-alone": stress_score_from_vol(scalar_preds["VIX-alone"]["rv21"]["test"]),
        "ATM-IV": stress_score_from_vol(scalar_preds["ATM-IV"]["rv21"]["test"]),
        "ShortEnd-IV": stress_score_from_vol(scalar_preds["ShortEnd-IV"]["rv21"]["test"]),
        "HAR-RV": stress_score_from_vol(har_rv21_pred),
        "HAR-RV+Opt": stress_score_from_vol(har_opt_rv21_pred),
        "HAR-Summary": stress_score_from_vol(har_sum_rv21_pred),
        boosted_label: stress_score_from_vol(boosted_preds["rv21"]["test"]),
        "GARCH": stress_score_from_vol(garch_pred_rv21),
    }

    def safe_choose_threshold(y_true: np.ndarray, y_score: np.ndarray, objective: str = "bal_acc") -> Optional[float]:
        valid = ~(np.isnan(y_true) | np.isnan(y_score))
        yt = y_true[valid]
        ys = y_score[valid]
        if len(yt) == 0 or len(np.unique(yt.astype(int))) < 2:
            return None
        return choose_threshold_on_validation(yt, ys, objective=objective)

    cls_thresholds = {name: safe_choose_threshold(y_val_stress, score, objective="bal_acc") for name, score in val_scores.items()}

    results: Dict[str, dict] = {}

    def add_model(name: str, y_rv5_sub: np.ndarray, y_rv21_sub: np.ndarray, y_dsv5_sub: np.ndarray, y_stress_sub: np.ndarray,
                  rv5_hat: np.ndarray, rv21_hat: np.ndarray, dsv5_hat: np.ndarray, stress_hat: np.ndarray,
                  cls_threshold: Optional[float]) -> None:
        rv5_m = regression_metrics(y_rv5_sub, rv5_hat, task="rv5")
        rv21_m = regression_metrics(y_rv21_sub, rv21_hat, task="rv21")
        dsv5_m = regression_metrics(y_dsv5_sub, dsv5_hat, task="dsv5", dsv5_floor=dsv5_floor)
        stress_m = classification_metrics(y_stress_sub, stress_hat, threshold=cls_threshold)
        stress_m["cls_threshold"] = float(cls_threshold) if cls_threshold is not None else np.nan
        stress_m["threshold_source"] = "validation_bal_acc_tuned" if cls_threshold is not None else "none"
        results[name] = {"rv5": rv5_m, "rv21": rv21_m, "dsv5": dsv5_m, "stress21": stress_m}

    add_model(primary_model_name, y_rv5, y_rv21, y_dsv5, y_stress,
              preds_test["rv5"], preds_test["rv21"], preds_test["dsv5"], test_scores[primary_model_name], cls_thresholds[primary_model_name])
    add_model("VIX-alone", y_rv5, y_rv21, y_dsv5, y_stress,
              scalar_preds["VIX-alone"]["rv5"]["test"], scalar_preds["VIX-alone"]["rv21"]["test"], scalar_preds["VIX-alone"]["dsv5"]["test"],
              test_scores["VIX-alone"], cls_thresholds["VIX-alone"])
    add_model("ATM-IV", y_rv5, y_rv21, y_dsv5, y_stress,
              scalar_preds["ATM-IV"]["rv5"]["test"], scalar_preds["ATM-IV"]["rv21"]["test"], scalar_preds["ATM-IV"]["dsv5"]["test"],
              test_scores["ATM-IV"], cls_thresholds["ATM-IV"])
    add_model("ShortEnd-IV", y_rv5, y_rv21, y_dsv5, y_stress,
              scalar_preds["ShortEnd-IV"]["rv5"]["test"], scalar_preds["ShortEnd-IV"]["rv21"]["test"], scalar_preds["ShortEnd-IV"]["dsv5"]["test"],
              test_scores["ShortEnd-IV"], cls_thresholds["ShortEnd-IV"])
    add_model("HAR-RV", y_rv5, y_rv21, y_dsv5, y_stress,
              har_rv5_pred, har_rv21_pred, har_dsv5_pred, test_scores["HAR-RV"], cls_thresholds["HAR-RV"])
    add_model("HAR-RV+Opt", y_rv5, y_rv21, y_dsv5, y_stress,
              har_opt_rv5_pred, har_opt_rv21_pred, har_opt_dsv5_pred, test_scores["HAR-RV+Opt"], cls_thresholds["HAR-RV+Opt"])
    add_model("HAR-Summary", y_rv5, y_rv21, y_dsv5, y_stress,
              har_sum_rv5_pred, har_sum_rv21_pred, har_sum_dsv5_pred, test_scores["HAR-Summary"], cls_thresholds["HAR-Summary"])
    add_model(boosted_label, y_rv5, y_rv21, y_dsv5, y_stress,
              boosted_preds["rv5"]["test"], boosted_preds["rv21"]["test"], boosted_preds["dsv5"]["test"],
              test_scores[boosted_label], cls_thresholds[boosted_label])
    # Grid-CNN / Grid-Transformer are exported as appendix reference baselines.
    add_model("GARCH", y_rv5, y_rv21, y_dsv5, y_stress,
              garch_pred_rv5, garch_pred_rv21, garch_pred_dsv5, test_scores["GARCH"], cls_thresholds["GARCH"])

    calm_idx = test_df["VIX"].values <= 20
    stress_idx = test_df["VIX"].values > 20
    for regime, idx in [("calm", calm_idx), ("stress", stress_idx)]:
        if idx.any():
            add_model(f"{primary_model_name}_{regime}", y_rv5[idx], y_rv21[idx], y_dsv5[idx], y_stress[idx],
                      preds_test["rv5"][idx], preds_test["rv21"][idx], preds_test["dsv5"][idx], test_scores[primary_model_name][idx],
                      cls_thresholds[primary_model_name])
            add_model(f"HAR-RV_{regime}", y_rv5[idx], y_rv21[idx], y_dsv5[idx], y_stress[idx],
                      har_rv5_pred[idx], har_rv21_pred[idx], har_dsv5_pred[idx], test_scores["HAR-RV"][idx],
                      cls_thresholds["HAR-RV"])

    def _slug(x: str) -> str:
        return x.lower().replace("+", "plus").replace("-", "_").replace(" ", "_").replace("/", "_")

    block_sweep_rows = []

    def add_dm_result(tag: str, target_name: str, baseline_name: str,
                      y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> None:
        if target_name == "dsv5":
            loss_a = qlike_loss_series(y_true, pred_a, mode="floor", floor_value=dsv5_floor)
            loss_b = qlike_loss_series(y_true, pred_b, mode="floor", floor_value=dsv5_floor)
            loss_label = "floored_qlike"
        else:
            loss_a = qlike_loss_series(y_true, pred_a, mode="standard")
            loss_b = qlike_loss_series(y_true, pred_b, mode="standard")
            loss_label = "standard_qlike"

        dm_out = block_bootstrap_dm(loss_a, loss_b, block_len=22, n_boot=2000, seed=42)

        valid = np.isfinite(y_true) & np.isfinite(pred_a) & np.isfinite(pred_b)
        if valid.sum() < 30:
            results[tag] = {
                "comparison": tag, "target": target_name, "baseline": baseline_name,
                "loss_label": loss_label,
                "p_value": np.nan, "obs_diff_qlike": np.nan,
                "our_rmse": np.nan, "baseline_rmse": np.nan, "n": int(valid.sum()),
            }
            return

        e_a = y_true[valid] - pred_a[valid]
        e_b = y_true[valid] - pred_b[valid]
        results[tag] = {
            "comparison": tag,
            "target": target_name,
            "baseline": baseline_name,
            "loss_label": loss_label,
            "p_value": dm_out["p_value"],
            "obs_diff_qlike": dm_out["obs_diff"],
            "our_rmse": float(np.sqrt(np.mean(e_a ** 2))),
            "baseline_rmse": float(np.sqrt(np.mean(e_b ** 2))),
            "n": int(valid.sum()),
        }

        for L in [5, 10, 22]:
            sw = block_bootstrap_dm(loss_a, loss_b, block_len=L, n_boot=2000, seed=42)
            block_sweep_rows.append({
                "target": target_name,
                "baseline": baseline_name,
                "loss_label": loss_label,
                "block_len": L,
                "p_value": sw.get("p_value", np.nan),
                "obs_diff_qlike": sw.get("obs_diff", np.nan),
                "n": sw.get("n", np.nan),
            })


    baseline_reg_preds = {
        "VIX-alone": {"rv5": scalar_preds["VIX-alone"]["rv5"]["test"], "rv21": scalar_preds["VIX-alone"]["rv21"]["test"], "dsv5": scalar_preds["VIX-alone"]["dsv5"]["test"]},
        "ATM-IV": {"rv5": scalar_preds["ATM-IV"]["rv5"]["test"], "rv21": scalar_preds["ATM-IV"]["rv21"]["test"], "dsv5": scalar_preds["ATM-IV"]["dsv5"]["test"]},
        "ShortEnd-IV": {"rv5": scalar_preds["ShortEnd-IV"]["rv5"]["test"], "rv21": scalar_preds["ShortEnd-IV"]["rv21"]["test"], "dsv5": scalar_preds["ShortEnd-IV"]["dsv5"]["test"]},
        "HAR-RV": {"rv5": har_rv5_pred, "rv21": har_rv21_pred, "dsv5": har_dsv5_pred},
        "HAR-RV+Opt": {"rv5": har_opt_rv5_pred, "rv21": har_opt_rv21_pred, "dsv5": har_opt_dsv5_pred},
        "HAR-Summary": {"rv5": har_sum_rv5_pred, "rv21": har_sum_rv21_pred, "dsv5": har_sum_dsv5_pred},
        boosted_label: {"rv5": boosted_preds["rv5"]["test"], "rv21": boosted_preds["rv21"]["test"], "dsv5": boosted_preds["dsv5"]["test"]},
        "GARCH": {"rv5": garch_pred_rv5, "rv21": garch_pred_rv21, "dsv5": garch_pred_dsv5},
    }
    target_pack = {
        "rv5": (y_rv5, preds_test["rv5"]),
        "rv21": (y_rv21, preds_test["rv21"]),
        "dsv5": (y_dsv5, preds_test["dsv5"])
    }
    for target_name, (y_true, our_pred) in target_pack.items():
        for baseline_name, pred_dict in baseline_reg_preds.items():
            tag = f"dm_test_{target_name}_vs_{_slug(baseline_name)}"
            add_dm_result(tag, target_name, baseline_name, y_true, our_pred, pred_dict[target_name])

    # Fixed-grid reference baselines.
    appendix_grid_rows = []
    if include_grid_appendix and grid_preds is not None and grid_tr_preds is not None:
        for _glabel, _gpreds in [(grid_label, grid_preds), (grid_tr_label, grid_tr_preds)]:
            for _target, _y in [("rv5", y_rv5), ("rv21", y_rv21), ("dsv5", y_dsv5)]:
                _m = regression_metrics(
                    _y,
                    _gpreds[_target]["test"],
                    task=_target,
                    dsv5_floor=dsv5_floor if _target == "dsv5" else None,
                )
                appendix_grid_rows.append({
                    "model": _glabel,
                    "target": _target,
                    **_m,
                })

    appendix_grid_side_rows = []
    if include_grid_side_appendix and grid_side_preds is not None and grid_side_tr_preds is not None:
        for _glabel, _gpreds in [(grid_side_label, grid_side_preds), (grid_side_tr_label, grid_side_tr_preds)]:
            for _target, _y in [("rv5", y_rv5), ("rv21", y_rv21), ("dsv5", y_dsv5)]:
                _m = regression_metrics(
                    _y,
                    _gpreds[_target]["test"],
                    task=_target,
                    dsv5_floor=dsv5_floor if _target == "dsv5" else None,
                )
                appendix_grid_side_rows.append({
                    "model": _glabel,
                    "target": _target,
                    **_m,
                })

    dsv5_pred_map = {
        primary_model_name: preds_test["dsv5"],
        "VIX-alone": scalar_preds["VIX-alone"]["dsv5"]["test"],
        "ATM-IV": scalar_preds["ATM-IV"]["dsv5"]["test"],
        "ShortEnd-IV": scalar_preds["ShortEnd-IV"]["dsv5"]["test"],
        "HAR-RV": har_dsv5_pred,
        "HAR-RV+Opt": har_opt_dsv5_pred,
        "HAR-Summary": har_sum_dsv5_pred,
        boosted_label: boosted_preds["dsv5"]["test"],
        "GARCH": garch_pred_dsv5,
    }
    if include_grid_appendix and grid_preds is not None and grid_tr_preds is not None:
        dsv5_pred_map[grid_label] = grid_preds["dsv5"]["test"]
        dsv5_pred_map[grid_tr_label] = grid_tr_preds["dsv5"]["test"]
    if include_grid_side_appendix and grid_side_preds is not None and grid_side_tr_preds is not None:
        dsv5_pred_map[grid_side_label] = grid_side_preds["dsv5"]["test"]
        dsv5_pred_map[grid_side_tr_label] = grid_side_tr_preds["dsv5"]["test"]

    # Diagnostic prediction panel and DSV5 floor-sensitivity table.
    diag_pred_map = {
        primary_model_name: {"rv5": preds_test["rv5"], "rv21": preds_test["rv21"], "dsv5": preds_test["dsv5"]},
        "VIX-alone": {"rv5": scalar_preds["VIX-alone"]["rv5"]["test"], "rv21": scalar_preds["VIX-alone"]["rv21"]["test"], "dsv5": scalar_preds["VIX-alone"]["dsv5"]["test"]},
        "ATM-IV": {"rv5": scalar_preds["ATM-IV"]["rv5"]["test"], "rv21": scalar_preds["ATM-IV"]["rv21"]["test"], "dsv5": scalar_preds["ATM-IV"]["dsv5"]["test"]},
        "ShortEnd-IV": {"rv5": scalar_preds["ShortEnd-IV"]["rv5"]["test"], "rv21": scalar_preds["ShortEnd-IV"]["rv21"]["test"], "dsv5": scalar_preds["ShortEnd-IV"]["dsv5"]["test"]},
        "HAR-RV": {"rv5": har_rv5_pred, "rv21": har_rv21_pred, "dsv5": har_dsv5_pred},
        "HAR-RV+Opt": {"rv5": har_opt_rv5_pred, "rv21": har_opt_rv21_pred, "dsv5": har_opt_dsv5_pred},
        "HAR-Summary": {"rv5": har_sum_rv5_pred, "rv21": har_sum_rv21_pred, "dsv5": har_sum_dsv5_pred},
        boosted_label: {"rv5": boosted_preds["rv5"]["test"], "rv21": boosted_preds["rv21"]["test"], "dsv5": boosted_preds["dsv5"]["test"]},
        "GARCH": {"rv5": garch_pred_rv5, "rv21": garch_pred_rv21, "dsv5": garch_pred_dsv5},
    }
    if include_grid_appendix and grid_preds is not None and grid_tr_preds is not None:
        diag_pred_map[grid_label] = {"rv5": grid_preds["rv5"]["test"], "rv21": grid_preds["rv21"]["test"], "dsv5": grid_preds["dsv5"]["test"]}
        diag_pred_map[grid_tr_label] = {"rv5": grid_tr_preds["rv5"]["test"], "rv21": grid_tr_preds["rv21"]["test"], "dsv5": grid_tr_preds["dsv5"]["test"]}
    if include_grid_side_appendix and grid_side_preds is not None and grid_side_tr_preds is not None:
        diag_pred_map[grid_side_label] = {"rv5": grid_side_preds["rv5"]["test"], "rv21": grid_side_preds["rv21"]["test"], "dsv5": grid_side_preds["dsv5"]["test"]}
        diag_pred_map[grid_side_tr_label] = {"rv5": grid_side_tr_preds["rv5"]["test"], "rv21": grid_side_tr_preds["rv21"]["test"], "dsv5": grid_side_tr_preds["dsv5"]["test"]}

    diag_dir = os.path.join(cfg_data.results_dir, "diagnostics")
    panel_path = save_forecast_prediction_panel(
        out_path=os.path.join(diag_dir, "test_prediction_panel.csv"),
        test_df=test_df,
        y_true_map={"rv5": y_rv5, "rv21": y_rv21, "dsv5": y_dsv5},
        pred_map=diag_pred_map,
    )
    build_dsv5_floor_sensitivity_from_prediction_panel(
        panel_path=panel_path,
        train_dsv5=train_df["dsv5"].values,
        out_path=os.path.join(diag_dir, "dsv5_floor_sensitivity.csv"),
        floor_qs=(0.01, 0.05, 0.10),
    )

    results["__dsv5_zero_diagnostics__"] = {
        "rows": dsv5_zero_diagnostics(
            y_true=y_dsv5,
            pred_map=dsv5_pred_map,
            train_y=train_df["dsv5"].values,
            eps=1e-8,
            floor_q=0.05,
        ).to_dict("records")
    }
    results["__block_length_sweep__"] = {"rows": block_sweep_rows}
    results["__appendix_grid_baselines__"] = {"rows": appendix_grid_rows}
    results["__appendix_grid_side_baselines__"] = {"rows": appendix_grid_side_rows}

    if not include_reconstruction:
        return results

    recon_test_metrics = evaluate_reconstruction(surface_model, test_ld, device, cfg_data=cfg_data)
    results["Reconstruction"] = {
        "surface": {
            "rmse": float(recon_test_metrics.get("price_rmse", np.nan)),
            "iv_rmse": float(recon_test_metrics.get("iv_rmse", np.nan)),
            "ba_violation_rate": float(recon_test_metrics.get("ba_violation_rate", np.nan)),
            "ba_within_half_spread_rate": float(recon_test_metrics.get("ba_within_half_spread_rate", np.nan)),
            "ba_within_one_spread_rate": float(recon_test_metrics.get("ba_within_one_spread_rate", np.nan)),
            "ba_norm_exceed_mean": float(recon_test_metrics.get("ba_norm_exceed_mean", np.nan)),
            "ba_norm_exceed_q50": float(recon_test_metrics.get("ba_norm_exceed_q50", np.nan)),
            "ba_norm_exceed_q90": float(recon_test_metrics.get("ba_norm_exceed_q90", np.nan)),
            "mae": np.nan,
            "qlike": np.nan,
            "invalid_pred_rate": np.nan,
            "auc": np.nan,
            "f1": np.nan,
            "bal_acc": np.nan,
            "calendar_violation_rate": float(recon_test_metrics.get("calendar_violation_rate", np.nan)),
            "butterfly_violation_rate": float(recon_test_metrics.get("butterfly_violation_rate", np.nan)),
            "n": int(recon_test_metrics.get("n_quotes", np.nan)) if np.isfinite(recon_test_metrics.get("n_quotes", np.nan)) else np.nan,
            "n_days": int(recon_test_metrics.get("n_days", np.nan)) if np.isfinite(recon_test_metrics.get("n_days", np.nan)) else np.nan,
            "n_quotes": int(recon_test_metrics.get("n_quotes", np.nan)) if np.isfinite(recon_test_metrics.get("n_quotes", np.nan)) else np.nan,
        }
    }

    return results


def plot_latent_trajectory(surface_model: FullSurfaceModel, cache_path: str,
                           cfg_data: DataConfig = DATA_CFG, cfg_model: ModelConfig = MODEL_CFG,
                           cfg_train: TrainConfig = TRAIN_CFG,
                           date_start: str = "2020-01-01", date_end: str = "2020-06-30",
                           out_dir: str = "results/eval"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from sklearn.decomposition import PCA
    except ImportError:
        logger.warning("matplotlib / sklearn PCA not available – skipping trajectory plot.")
        return

    device = torch.device(cfg_train.device if torch.cuda.is_available() else "cpu")
    surface_model.eval()

    all_records = []
    for split in ["train", "val", "test"]:
        try:
            ds = OptionDataset(cache_path, split, cfg_data)
            ld = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0)
            all_records.append(extract_latents(surface_model, ld, device))
        except Exception:
            pass

    if not all_records:
        return

    full_df = pd.concat(all_records).sort_values("date")
    sub = full_df[(full_df["date"] >= date_start) & (full_df["date"] <= date_end)].copy()
    if len(sub) == 0:
        logger.warning(f"No data in window {date_start} – {date_end}")
        return

    d = cfg_model.latent_dim
    Z = sub[[f"z_{j}" for j in range(d)]].values
    dates = sub["date"].values
    vix = sub["VIX"].values
    rv21 = sub["rv21"].values

    pca = PCA(n_components=2)
    Z_2d = pca.fit_transform(Z)

    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, Z_2d[:, 0], label="Latent PC1")
    ax1.set_title(f"Latent PC1 vs. VIX  ({date_start} – {date_end})")
    ax2 = ax1.twinx()
    ax2.plot(dates, vix, label="VIX", alpha=0.6)
    ax1.set_ylabel("PC1")
    ax2.set_ylabel("VIX")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax3 = fig.add_subplot(gs[1, 0])
    sc = ax3.scatter(Z_2d[:, 0], Z_2d[:, 1], c=range(len(Z_2d)), cmap="RdYlGn_r", s=20)
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_title("Latent trajectory (colour = time)")
    plt.colorbar(sc, ax=ax3)

    ax4 = fig.add_subplot(gs[1, 1])
    valid = ~np.isnan(rv21)
    ax4.scatter(Z_2d[valid, 0], rv21[valid], s=10, alpha=0.5)
    ax4.set_xlabel("Latent PC1")
    ax4.set_ylabel("RV21")
    ax4.set_title("Latent PC1 vs. RV21")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"latent_traj_{date_start[:7]}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Latent trajectory saved → {out_path}")


def print_results_table(results: Dict[str, dict]) -> None:
    def _fmt(x, n=5):
        return f"{x:.{n}f}" if np.isfinite(x) else "nan"

    print("\n" + "=" * 275)
    print(
        f"{'Model':<24}  {'Task':<14}  "
        f"{'RMSE':>10}  {'IV_RMSE':>10}  {'BA_Vio':>10}  {'HalfSpr':>10}  {'OneSpr':>10}  {'BAq90':>10}  "
        f"{'MAE':>10}  {'QLIKE':>10}  {'QLIKEflr':>10}  {'InvRate':>10}  "
        f"{'AUC*':>8}  {'F1@ValThr':>10}  {'Bal@ValThr':>11}  "
        f"{'CalVio':>10}  {'BflyVio':>10}"
    )
    print("-" * 275)

    for model_name, tasks in results.items():
        if model_name.startswith("dm_test_") or model_name.startswith("__") or model_name == "Reconstruction":
            continue
        for task_name, m in tasks.items():
            print(
                f"{model_name:<24}  {task_name:<14}  "
                f"{_fmt(m.get('rmse', np.nan)):>10}  "
                f"{_fmt(m.get('iv_rmse', np.nan)):>10}  "
                f"{_fmt(m.get('ba_violation_rate', np.nan)):>10}  "
                f"{_fmt(m.get('ba_within_half_spread_rate', np.nan)):>10}  "
                f"{_fmt(m.get('ba_within_one_spread_rate', np.nan)):>10}  "
                f"{_fmt(m.get('ba_norm_exceed_q90', np.nan)):>10}  "
                f"{_fmt(m.get('mae', np.nan)):>10}  "
                f"{_fmt(m.get('qlike', np.nan)):>10}  "
                f"{_fmt(m.get('qlike_floored', np.nan)):>10}  "
                f"{_fmt(m.get('invalid_pred_rate', np.nan)):>10}  "
                f"{_fmt(m.get('auc', np.nan), 3):>8}  "
                f"{_fmt(m.get('f1', np.nan), 3):>10}  "
                f"{_fmt(m.get('bal_acc', np.nan), 3):>11}  "
                f"{_fmt(m.get('calendar_violation_rate', np.nan)):>10}  "
                f"{_fmt(m.get('butterfly_violation_rate', np.nan)):>10}"
            )

    if "Reconstruction" in results:
        print("\n[Surface consistency only — not downstream forecasting]")
        for task_name, m in results["Reconstruction"].items():
            print(
                f"{'Reconstruction':<24}  {task_name:<14}  "
                f"{_fmt(m.get('rmse', np.nan)):>10}  "
                f"{_fmt(m.get('iv_rmse', np.nan)):>10}  "
                f"{_fmt(m.get('ba_violation_rate', np.nan)):>10}  "
                f"{_fmt(m.get('ba_within_half_spread_rate', np.nan)):>10}  "
                f"{_fmt(m.get('ba_within_one_spread_rate', np.nan)):>10}  "
                f"{_fmt(m.get('ba_norm_exceed_q90', np.nan)):>10}  "
                f"{_fmt(m.get('mae', np.nan)):>10}  "
                f"{_fmt(m.get('qlike', np.nan)):>10}  "
                f"{_fmt(m.get('qlike_floored', np.nan)):>10}  "
                f"{_fmt(m.get('invalid_pred_rate', np.nan)):>10}  "
                f"{_fmt(m.get('auc', np.nan), 3):>8}  "
                f"{_fmt(m.get('f1', np.nan), 3):>10}  "
                f"{_fmt(m.get('bal_acc', np.nan), 3):>11}  "
                f"{_fmt(m.get('calendar_violation_rate', np.nan)):>10}  "
                f"{_fmt(m.get('butterfly_violation_rate', np.nan)):>10}"
            )


def save_results(results: Dict[str, dict], out_path: str) -> None:
    rows = []
    dm_rows = []

    for model_name, tasks in results.items():
        if model_name.startswith("dm_test_"):
            dm_rows.append({
                "comparison": tasks.get("comparison", model_name),
                "target": tasks.get("target", np.nan),
                "baseline": tasks.get("baseline", np.nan),
                "loss_label": tasks.get("loss_label", np.nan),
                "p_value": tasks.get("p_value", np.nan),
                "obs_diff_qlike": tasks.get("obs_diff_qlike", np.nan),
                "our_rmse": tasks.get("our_rmse", np.nan),
                "baseline_rmse": tasks.get("baseline_rmse", np.nan),
                "n": tasks.get("n", np.nan),
            })
            continue

        if model_name.startswith("__"):
            continue

        for task_name, m in tasks.items():
            evidence_group = "surface_consistency" if model_name == "Reconstruction" else "forecasting"
            rows.append({
                "model": model_name,
                "task": task_name,
                "evidence_group": evidence_group,
                "rmse": m.get("rmse", np.nan),
                "iv_rmse": m.get("iv_rmse", np.nan),
                "ba_violation_rate": m.get("ba_violation_rate", np.nan),
                "ba_within_half_spread_rate": m.get("ba_within_half_spread_rate", np.nan),
                "ba_within_one_spread_rate": m.get("ba_within_one_spread_rate", np.nan),
                "ba_norm_exceed_mean": m.get("ba_norm_exceed_mean", np.nan),
                "ba_norm_exceed_q50": m.get("ba_norm_exceed_q50", np.nan),
                "ba_norm_exceed_q90": m.get("ba_norm_exceed_q90", np.nan),
                "mae": m.get("mae", np.nan),
                "qlike": m.get("qlike", np.nan),
                "qlike_floored": m.get("qlike_floored", np.nan),
                "qlike_positive_subset": m.get("qlike_positive_subset", np.nan),
                "zero_y_rate": m.get("zero_y_rate", np.nan),
                "invalid_pred_rate": m.get("invalid_pred_rate", np.nan),
                "auc": m.get("auc", np.nan),
                "f1": m.get("f1", np.nan),
                "bal_acc": m.get("bal_acc", np.nan),
                "cls_threshold": m.get("cls_threshold", np.nan),
                "threshold_source": m.get("threshold_source", np.nan),
                "calendar_violation_rate": m.get("calendar_violation_rate", np.nan),
                "butterfly_violation_rate": m.get("butterfly_violation_rate", np.nan),
                "n": m.get("n", np.nan),
                "n_days": m.get("n_days", np.nan),
                "n_quotes": m.get("n_quotes", np.nan),
            })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df_main = pd.DataFrame(rows)
    main_cols = [
        "model", "task", "evidence_group",
        "rmse", "iv_rmse", "ba_violation_rate",
        "ba_within_half_spread_rate", "ba_within_one_spread_rate",
        "ba_norm_exceed_mean", "ba_norm_exceed_q50", "ba_norm_exceed_q90",
        "mae", "qlike", "qlike_floored", "qlike_positive_subset", "zero_y_rate", "invalid_pred_rate",
        "auc", "f1", "bal_acc", "cls_threshold", "threshold_source",
        "calendar_violation_rate", "butterfly_violation_rate", "n", "n_days", "n_quotes",
    ]
    df_main = df_main.reindex(columns=main_cols)
    df_main.to_csv(out_path, index=False)

    df_main[df_main["evidence_group"] == "forecasting"].to_csv(out_path.replace(".csv", "_forecasting.csv"), index=False)
    df_main[df_main["evidence_group"] == "surface_consistency"].to_csv(out_path.replace(".csv", "_surface.csv"), index=False)

    if dm_rows:
        dm_path = out_path.replace(".csv", "_dm.csv")
        df_dm = pd.DataFrame(dm_rows).reindex(columns=[
            "comparison", "target", "baseline", "loss_label", "p_value",
            "obs_diff_qlike", "our_rmse", "baseline_rmse", "n"
        ])
        df_dm.to_csv(dm_path, index=False)

    # Appendix diagnostics and robustness tables.
    if "__dsv5_zero_diagnostics__" in results:
        diag_path = out_path.replace(".csv", "_dsv5_zero_diagnostics.csv")
        pd.DataFrame(results["__dsv5_zero_diagnostics__"].get("rows", [])).to_csv(diag_path, index=False)

    if "__block_length_sweep__" in results:
        sweep_path = out_path.replace(".csv", "_block_length_sweep.csv")
        pd.DataFrame(results["__block_length_sweep__"].get("rows", [])).to_csv(sweep_path, index=False)

    if "__appendix_grid_baselines__" in results:
        grid_path = out_path.replace(".csv", "_appendix_grid_baselines.csv")
        pd.DataFrame(results["__appendix_grid_baselines__"].get("rows", [])).to_csv(grid_path, index=False)

    if "__appendix_grid_side_baselines__" in results:
        grid_side_path = out_path.replace(".csv", "_appendix_grid_side_baselines.csv")
        pd.DataFrame(results["__appendix_grid_side_baselines__"].get("rows", [])).to_csv(grid_side_path, index=False)


def load_stage2_heads_or_raise(heads: DownstreamHeads, results_dir: str, device: str) -> str:
    s2 = os.path.join(results_dir, "checkpoints", "stage2_best.pt")
    if not os.path.exists(s2):
        raise FileNotFoundError(
            "stage2_best.pt not found. Cannot proceed because downstream heads would otherwise remain randomly initialized."
        )
    ck = torch.load(s2, map_location=device)
    if "heads" not in ck:
        raise ValueError(f"stage2_best.pt is malformed. Keys found: {list(ck.keys())}")
    heads.load_state_dict(ck["heads"])
    logger.info("Loaded stage2_best.pt")
    return "stage2_best.pt"


def load_stage3_checkpoint_for_eval(surface_model: FullSurfaceModel, heads: DownstreamHeads,
                                    results_dir: str, device: str, preference: str = "forecast") -> str:
    pref_orders = {
        "forecast": ["stage3_best_forecast.pt", "stage3_best.pt", "stage3_best_joint.pt", "stage3_best_surface.pt"],
        "joint": ["stage3_best_joint.pt", "stage3_best_forecast.pt", "stage3_best.pt", "stage3_best_surface.pt"],
        "surface": ["stage3_best_surface.pt", "stage3_best_forecast.pt", "stage3_best.pt", "stage3_best_joint.pt"],
    }
    candidates = pref_orders.get(preference, ["stage3_best_forecast.pt", "stage3_best.pt", "stage3_best_joint.pt", "stage3_best_surface.pt"])

    for name in candidates:
        p = os.path.join(results_dir, "checkpoints", name)
        if os.path.exists(p):
            ck = torch.load(p, map_location=device)
            if "surface" not in ck or "heads" not in ck:
                raise ValueError(f"{name} is not a valid stage-3 checkpoint. Keys found: {list(ck.keys())}")
            surface_model.load_state_dict(ck["surface"])
            heads.load_state_dict(ck["heads"])
            logger.info(f"Loaded {name}")
            return name

    raise FileNotFoundError("No stage-3 checkpoint found for evaluation.")


def run_pipeline(steps, skip_preprocess=False, rebuild=False, resume=False, include_grid_appendix=True, include_grid_side_appendix=False):
    make_dirs(DATA_CFG)

    if 0 in steps and not skip_preprocess:
        logger.info("=" * 60)
        logger.info("STEP 0: Preprocessing raw option data")
        logger.info("=" * 60)
        cache_path = preprocess_all(DATA_CFG, force_rebuild=rebuild)
        logger.info(f"Cache ready: {cache_path}")
    else:
        cache_path = os.path.join(DATA_CFG.cache_dir, "daily_quotes.pkl")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if 1 in steps:
        logger.info("=" * 60)
        logger.info("STEP 1: Stage 1 – Representation Pretraining")
        logger.info("=" * 60)
        model, _ = train_stage1(DATA_CFG, MODEL_CFG, TRAIN_CFG, resume=resume)
    else:
        model = FullSurfaceModel(MODEL_CFG).to(device)
        s1 = os.path.join(DATA_CFG.results_dir, "checkpoints", "stage1_best.pt")
        if os.path.exists(s1):
            load_checkpoint(s1, model, device=device)
        elif any(s in steps for s in [2, 3]):
            raise FileNotFoundError("stage1_best.pt not found. Steps 2/3 require a trained stage-1 checkpoint.")
        else:
            logger.warning("stage1_best.pt not found.")

    if 2 in steps or 3 in steps:
        logger.info("=" * 60)
        logger.info("STEP 2+3: Downstream Training")
        logger.info("=" * 60)

        if 2 in steps:
            heads = train_stage2(model, cache_path, DATA_CFG, MODEL_CFG, TRAIN_CFG)
        else:
            heads = DownstreamHeads(MODEL_CFG).to(device)
            load_stage2_heads_or_raise(heads, DATA_CFG.results_dir, device)

        if 3 in steps:
            train_stage3(model, heads, cache_path, DATA_CFG, MODEL_CFG, TRAIN_CFG)

    if 4 in steps:
        logger.info("=" * 60)
        logger.info("STEP 4: Full Evaluation")
        logger.info("=" * 60)

        surface_model = FullSurfaceModel(MODEL_CFG).to(device)
        heads = DownstreamHeads(MODEL_CFG).to(device)

        s1 = os.path.join(DATA_CFG.results_dir, "checkpoints", "stage1_best.pt")
        try:
            loaded_name = load_stage3_checkpoint_for_eval(surface_model, heads, DATA_CFG.results_dir, device,
                                                          preference=TRAIN_CFG.stage3_eval_preference)
        except FileNotFoundError:
            if not os.path.exists(s1):
                raise FileNotFoundError("No stage-3 checkpoint found, and stage1_best.pt is also missing. Cannot run evaluation.")
            load_checkpoint(s1, surface_model, device=device)
            load_stage2_heads_or_raise(heads, DATA_CFG.results_dir, device)
            loaded_name = "stage1_best.pt + stage2_best.pt"
            logger.warning(f"Stage-3 checkpoint missing, fallback to {loaded_name}")

        results = full_evaluation(
            surface_model, heads, cache_path, DATA_CFG, MODEL_CFG, TRAIN_CFG,
            primary_model_name="OurModel",
            use_side_info=True,
            include_reconstruction=True,
            include_grid_appendix=include_grid_appendix,
            include_grid_side_appendix=include_grid_side_appendix,
        )
        print_results_table(results)

        out = os.path.join(DATA_CFG.results_dir, "eval", "test_results.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        save_results(results, out)

        stage1_surface = evaluate_surface_checkpoint("stage1_best.pt", cache_path, DATA_CFG, MODEL_CFG, device) \
            if os.path.exists(os.path.join(DATA_CFG.results_dir, "checkpoints", "stage1_best.pt")) else None
        stage3_forecast_surface = evaluate_surface_checkpoint("stage3_best_forecast.pt", cache_path, DATA_CFG, MODEL_CFG, device) \
            if os.path.exists(os.path.join(DATA_CFG.results_dir, "checkpoints", "stage3_best_forecast.pt")) else None
        stage3_surface_surface = evaluate_surface_checkpoint("stage3_best_surface.pt", cache_path, DATA_CFG, MODEL_CFG, device) \
            if os.path.exists(os.path.join(DATA_CFG.results_dir, "checkpoints", "stage3_best_surface.pt")) else None
        stage3_joint_surface = evaluate_surface_checkpoint("stage3_best_joint.pt", cache_path, DATA_CFG, MODEL_CFG, device) \
            if os.path.exists(os.path.join(DATA_CFG.results_dir, "checkpoints", "stage3_best_joint.pt")) else None

        print("Loaded eval checkpoint:", loaded_name)
        print("Stage1 surface-only:", stage1_surface)
        print("Stage3 forecast-best surface-only:", stage3_forecast_surface)
        print("Stage3 surface-best surface-only:", stage3_surface_surface)
        print("Stage3 joint-best surface-only:", stage3_joint_surface)

        for window in [("2020-01-15", "2020-06-30"), ("2022-01-01", "2022-12-31")]:
            plot_latent_trajectory(surface_model, cache_path, date_start=window[0], date_end=window[1],
                                   out_dir=os.path.join(DATA_CFG.results_dir, "eval"))


def run_ablation():
    variants = {
        "A0_full": {},
        "A1_no_arb": {"lambda_arb": 0.0},
        "A2_no_ba": {"lambda_ba": 0.0},
        "A3_no_mask": {"lambda_mask": 0.0},
        "A5_latent32": {"latent_dim": 32},
    }

    for name, overrides in variants.items():
        logger.info(f"\n{'='*60}\n  ABLATION: {name}\n{'='*60}")

        data_cfg = DataConfig()
        model_cfg = ModelConfig()
        train_cfg = TrainConfig()

        for k, v in overrides.items():
            if hasattr(train_cfg, k):
                setattr(train_cfg, k, v)
            elif hasattr(model_cfg, k):
                setattr(model_cfg, k, v)

        data_cfg.results_dir = os.path.join("results", "ablations", name)
        os.makedirs(os.path.join(data_cfg.results_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(data_cfg.results_dir, "eval"), exist_ok=True)

        cache_path = os.path.join(data_cfg.cache_dir, "daily_quotes.pkl")

        train_stage1(data_cfg, model_cfg, train_cfg, resume=False)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        surface_model = FullSurfaceModel(model_cfg).to(device)
        s1 = os.path.join(data_cfg.results_dir, "checkpoints", "stage1_best.pt")
        ck1 = torch.load(s1, map_location=device)
        if "model" not in ck1:
            raise ValueError(f"[{name}] stage1_best.pt malformed. Keys found: {list(ck1.keys())}")
        surface_model.load_state_dict(ck1["model"])

        heads = train_stage2(surface_model, cache_path, data_cfg, model_cfg, train_cfg)
        train_stage3(surface_model, heads, cache_path, data_cfg, model_cfg, train_cfg)

        eval_surface = FullSurfaceModel(model_cfg).to(device)
        eval_heads = DownstreamHeads(model_cfg).to(device)

        try:
            loaded_name = load_stage3_checkpoint_for_eval(eval_surface, eval_heads, data_cfg.results_dir, device,
                                                          preference=train_cfg.stage3_eval_preference)
        except FileNotFoundError:
            eval_surface.load_state_dict(ck1["model"])
            load_stage2_heads_or_raise(eval_heads, data_cfg.results_dir, device)
            loaded_name = "stage1_best.pt + stage2_best.pt"
            logger.warning(f"[{name}] Stage-3 checkpoint missing, fallback to {loaded_name}")

        results = full_evaluation(eval_surface, eval_heads, cache_path, data_cfg, model_cfg, train_cfg)
        print_results_table(results)
        out = os.path.join(data_cfg.results_dir, "eval", "test_results.csv")
        save_results(results, out)
        logger.info(f"Ablation {name} complete -> {out} | loaded: {loaded_name}")



def _parse_seed_list(seed_text: str) -> List[int]:
    return [int(s.strip()) for s in str(seed_text).split(",") if s.strip()]


NEURAL_SEED_MODELS = [
    "OurModel",
    "OurModel_calm",
    "OurModel_stress",
    "SetForecastOnly",
    "SetForecastOnly_NoSide",
]


def aggregate_seed_result_csvs(
    root_dir: str,
    out_name: str = "seed_sensitivity_summary.csv",
    model_filter: Optional[List[str]] = None,
    neural_only: bool = True,
) -> Optional[str]:
    """
    Aggregate seed-sensitivity CSVs under root_dir/**/test_results_forecasting.csv.

    By default, only neural specifications are aggregated. This prevents the
    seed appendix from being polluted by deterministic baselines such as VIX,
    ATM-IV, HAR, or GARCH, which are repeatedly written in each seed folder but
    are not neural seed-sensitivity objects.
    """
    root = Path(root_dir)
    files = sorted(root.glob("**/test_results_forecasting.csv"))
    rows = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        seed = np.nan
        for part in fp.parts:
            if part.startswith("seed_"):
                try:
                    seed = int(part.replace("seed_", ""))
                    break
                except Exception:
                    pass
        df = df.copy()
        df["seed"] = seed
        df["source_file"] = str(fp)
        rows.append(df)

    if not rows:
        return None

    all_df_raw = pd.concat(rows, ignore_index=True)
    raw_path = root / out_name.replace(".csv", "_all_models_raw.csv")
    all_df_raw.to_csv(raw_path, index=False)

    if model_filter is None and neural_only:
        model_filter = NEURAL_SEED_MODELS
    if model_filter is not None:
        all_df = all_df_raw[all_df_raw["model"].isin(model_filter)].copy()
    else:
        all_df = all_df_raw.copy()

    all_path = root / out_name.replace(".csv", "_all_rows.csv")
    all_df.to_csv(all_path, index=False)

    if all_df.empty:
        return str(all_path)

    metric_cols = [
        "rmse", "mae", "qlike", "qlike_floored", "qlike_positive_subset",
        "auc", "f1", "bal_acc", "invalid_pred_rate", "zero_y_rate",
    ]
    present = [c for c in metric_cols if c in all_df.columns]
    group_cols = ["model", "task"]
    summary_parts = []
    for metric in present:
        tmp = (
            all_df.groupby(group_cols, dropna=False)[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std", "count": f"{metric}_count"})
        )
        summary_parts.append(tmp)

    if not summary_parts:
        return str(all_path)

    summary = summary_parts[0]
    for tmp in summary_parts[1:]:
        summary = summary.merge(tmp, on=group_cols, how="outer")

    out_path = root / out_name
    summary.to_csv(out_path, index=False)
    return str(out_path)

def run_set_forecast_only_experiments(
    seeds: List[int],
    include_no_side: bool = True,
    include_grid_appendix: bool = False,
) -> None:
    """
    Run the clean set-encoder-only ablation:
      1) SetForecastOnly: quote-set encoder + heads, VIX/VXV side info retained.
      2) SetForecastOnly_NoSide: same, but no VIX/VXV side info.
    """
    base_results_dir = os.path.join("results", "set_forecast_only")
    cache_path_default = os.path.join(DATA_CFG.cache_dir, "daily_quotes.pkl")
    if not os.path.exists(cache_path_default):
        logger.info("Cache missing; preprocessing first.")
        preprocess_all(DATA_CFG, force_rebuild=False)

    for seed in seeds:
        for use_side, label, run_name in [
            (True, "SetForecastOnly", "set_forecast_only"),
            (False, "SetForecastOnly_NoSide", "set_forecast_only_no_side"),
        ]:
            if (not include_no_side) and (not use_side):
                continue

            data_cfg = DataConfig()
            model_cfg = ModelConfig()
            train_cfg = TrainConfig(seed=seed)
            data_cfg.results_dir = os.path.join(base_results_dir, f"seed_{seed}", run_name)

            make_dirs(data_cfg)
            cache_path = os.path.join(data_cfg.cache_dir, "daily_quotes.pkl")

            logger.info(f"\n{'='*70}\nSET-FORECAST-ONLY ABLATION | seed={seed} | {label}\n{'='*70}")
            model, heads = train_set_forecast_only(
                cache_path=cache_path,
                cfg_data=data_cfg,
                cfg_model=model_cfg,
                cfg_train=train_cfg,
                use_side_info=use_side,
                run_name=run_name,
            )

            results = full_evaluation(
                model, heads, cache_path, data_cfg, model_cfg, train_cfg,
                primary_model_name=label,
                use_side_info=use_side,
                include_reconstruction=False,
                include_grid_appendix=include_grid_appendix,
            )
            print_results_table(results)
            out = os.path.join(data_cfg.results_dir, "eval", "test_results.csv")
            save_results(results, out)
            logger.info(f"{label} seed={seed} results saved -> {out}")

    summary = aggregate_seed_result_csvs(base_results_dir)
    if summary:
        logger.info(f"Set-forecast-only seed summary saved -> {summary}")


def run_multi_seed(
    seeds: List[int],
    steps: List[int],
    skip_preprocess: bool = True,
    rebuild: bool = False,
    resume: bool = False,
    run_full: bool = True,
    run_setonly: bool = True,
    include_grid_appendix: bool = False,
) -> None:
    """
    Multi-seed driver for neural specifications.

    Full model uses the existing staged pipeline in separate results folders:
        results/seeds/full/seed_<s>/
    Set-only ablations use:
        results/set_forecast_only/seed_<s>/...
    """
    original_seed = TRAIN_CFG.seed
    original_results_dir = DATA_CFG.results_dir

    try:
        if run_full:
            full_root = os.path.join("results", "seeds", "full")
            for seed in seeds:
                TRAIN_CFG.seed = int(seed)
                DATA_CFG.results_dir = os.path.join(full_root, f"seed_{seed}")
                make_dirs(DATA_CFG)
                logger.info(f"\n{'='*70}\nFULL MODEL MULTI-SEED RUN | seed={seed}\n{'='*70}")
                run_pipeline(
                    steps=steps,
                    skip_preprocess=skip_preprocess,
                    rebuild=rebuild,
                    resume=resume,
                    include_grid_appendix=include_grid_appendix,
                )
            summary = aggregate_seed_result_csvs(full_root)
            if summary:
                logger.info(f"Full-model seed summary saved -> {summary}")

        if run_setonly:
            run_set_forecast_only_experiments(
                seeds=seeds,
                include_no_side=True,
                include_grid_appendix=False,
            )
    finally:
        TRAIN_CFG.seed = original_seed
        DATA_CFG.results_dir = original_results_dir




# =============================================================================
# Option-Implied Belief-State Deep Hedging Extension
# =============================================================================
# HedgeSet extension using the trained option-state encoder. The existing stages
# learn the surface-regularized latent option state. The extension below treats
# the trained latent state as a frozen option-implied belief state and tests
# whether it is useful as the Markov input for risk-sensitive hedging under
# transaction costs.
#
# Evaluation protocol:
#   * tradable PnL is marked from observed option quotes, not decoder prices;
#   * liability episodes are built from exact contract keys, not gridded marks;
#   * episode paths must remain fully inside train/validation/test splits;
#   * the policy uses a differentiable no-trade band to avoid overtrading;
#   * model selection uses validation CVaR, not test performance;
#   * delta and no-hedge baselines are evaluated under the identical cost model.
# =============================================================================


@dataclass
class HedgingConfig:
    cache_dir: str = field(default_factory=lambda: os.environ.get("FNUG_HEDGE_CACHE_DIR", "cache_hedging"))
    results_dir: str = field(default_factory=lambda: os.path.join(DATA_CFG.results_dir, "hedging"))

    horizon: int = 5
    start_stride: int = 5
    target_tau_days: int = 30
    tau_tol_days: int = 10

    # Liability selection filters.  These are stricter than the forecasting
    # filters because pathwise PnL is much more sensitive to stale/wide quotes.
    min_liability_bid: float = 0.10
    max_liability_rel_spread: float = 0.30
    max_path_rel_spread: float = 0.50
    min_path_bid: float = 0.05

    # For pathwise mark-to-market, use a wider moneyness filter than the
    # representation model.  Otherwise crash/stress episodes can be dropped
    # after the selected liability moves outside the forecasting surface range.
    path_k_min: float = -0.60
    path_k_max: float = 0.60

    # If True, each finite-horizon episode pays the transaction cost required
    # to close the remaining underlying hedge at the terminal date.  This makes
    # closed-episode tail losses comparable across policies and prevents large
    # terminal hedge positions from receiving a hidden cost advantage.
    terminal_liquidation: bool = True

    # Liability families supported by the episode builder.
    families: List[str] = field(default_factory=lambda: ["atm_call", "put25", "atm_straddle"])

    # Hedging control and transaction cost.
    cost_bps: float = 2.0
    cost_bps_list: List[float] = field(default_factory=lambda: [1.0, 2.0, 5.0])
    h_max: float = 3.0
    trade_max: float = 1.0
    band_max: float = 1.0
    smooth_temp: float = 0.02

    # Risk-sensitive training.
    cvar_alpha: float = 0.95
    lambda_mean: float = 0.15
    lambda_tc: float = 0.05
    lambda_turnover: float = 0.02
    lambda_pos: float = 0.001

    # Policy training.
    epochs: int = 250
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    hidden_layers: int = 3
    patience: int = 35
    min_delta: float = 1e-5

    # State modes.  latent is the proposed belief state; summary and grid are
    # baselines using the same no-trade-band policy architecture.
    state_modes: List[str] = field(default_factory=lambda: ["latent", "summary", "grid"])


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _bs_delta_np(F: float, K: float, tau: float, sigma: float, D: float, S: float, is_call: bool) -> float:
    """Forward-form Black--Scholes delta with respect to the underlying close S."""
    if S <= 0 or F <= 0 or K <= 0:
        return 0.0
    sigma = max(float(sigma), 1e-6)
    tau = max(float(tau), 1e-8)
    sqrt_tau = math.sqrt(tau)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * tau) / (sigma * sqrt_tau)
    # Since F = S * Dq / Dr, the price delta wrt S equals Dq*N(d1) for calls
    # and Dq*(N(d1)-1) for puts.  Dq = F*Dr/S.
    Dq = max(float(F) * float(D) / max(float(S), 1e-8), 0.0)
    nd1 = float(norm.cdf(d1))
    return Dq * nd1 if is_call else Dq * (nd1 - 1.0)


def _parse_list_arg(text: str) -> List[str]:
    raw = str(text).strip()
    if raw.lower() in {"", "none", "no", "off", "false", "0"}:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _date_split_from_timestamp(d: pd.Timestamp, cfg: DataConfig = DATA_CFG) -> str:
    y = int(pd.Timestamp(d).year)
    if y <= cfg.train_end:
        return "train"
    if y <= cfg.val_end:
        return "val"
    return "test"


def preprocess_hedging_chains(
    data_cfg: DataConfig = DATA_CFG,
    hedge_cfg: HedgingConfig = HedgingConfig(),
    force_rebuild: bool = False,
) -> str:
    """Build a quote-level contract table for hedging episodes.

    The forecasting cache intentionally collapses redundant calls/puts and does
    not store exact expiration keys.  For hedging, that is not enough: realized
    PnL must be marked on the same liability contract through time.  This
    function therefore rereads the raw option files and saves a lightweight
    contract table with exact date/exdate/strike/cp keys and observed mid marks.
    """
    os.makedirs(hedge_cfg.cache_dir, exist_ok=True)
    out_path = os.path.join(hedge_cfg.cache_dir, "hedging_chains.parquet")
    if os.path.exists(out_path) and not force_rebuild:
        logger.info(f"Hedging chain cache found: {out_path}")
        return out_path

    ddir = resolve_data_dir(data_cfg)
    years = discover_available_years(data_cfg, ddir)
    aux = load_aux_data(data_cfg)
    prices_df = aux["prices"].copy()
    zcy_df = aux["zcy"]
    div_df = aux["div"]
    price_map = prices_df.set_index("date")

    rows = []
    max_tau = max(data_cfg.tau_max, (hedge_cfg.target_tau_days + 3 * hedge_cfg.tau_tol_days + hedge_cfg.horizon + 5) / 365.0)

    for year in years:
        fp = os.path.join(str(ddir), data_cfg.option_file_fmt.format(year=year))
        if not os.path.exists(fp):
            continue
        logger.info(f"[hedging chains] reading {year} …")
        df_year = pd.read_parquet(fp)
        df_year["date"] = pd.to_datetime(df_year["date"])
        df_year["exdate"] = pd.to_datetime(df_year["exdate"])
        df_year = df_year[df_year["secid"] == data_cfg.secid].copy()
        if getattr(data_cfg, "require_european_style", True) and "exercise_style" in df_year.columns:
            df_year = df_year[df_year["exercise_style"].fillna("E") == "E"]
        df_year = df_year[df_year["best_bid"] >= hedge_cfg.min_path_bid]
        df_year = df_year[df_year["best_offer"] > df_year["best_bid"]]
        if len(df_year) == 0:
            continue

        df_year["K"] = df_year["strike_price"] / 1000.0
        df_year["strike_key"] = pd.to_numeric(df_year["strike_price"], errors="coerce").fillna(-1).astype(np.int64)
        df_year["tau"] = (df_year["exdate"] - df_year["date"]).dt.days / 365.0
        df_year = df_year[(df_year["tau"] >= data_cfg.tau_min) & (df_year["tau"] <= max_tau)]
        if len(df_year) == 0:
            continue

        year_rows = []
        for tdate, grp in tqdm(df_year.groupby("date"), desc=f"hedge-chain {year}", leave=False):
            if tdate not in price_map.index:
                continue
            S = float(price_map.loc[tdate, "close"])
            zcy_day = zcy_df[zcy_df["date"] == tdate]
            div_day = div_df[div_df["date"] == tdate]
            tau_arr = grp["tau"].values.astype(float)
            F_arr, Dr_arr = build_forward_for_date(tdate, S, tau_arr, zcy_day, div_day)
            g = grp.copy()
            g["F"] = F_arr
            g["D_r"] = Dr_arr
            g["k"] = np.log(g["K"].values / np.maximum(F_arr, 1e-8))
            # Hedging mark-to-market must keep the selected contract observable
            # even after underlying moves.  Use the wider hedging path filter,
            # not the narrower forecasting-surface filter.
            g = g[(g["k"] >= hedge_cfg.path_k_min) & (g["k"] <= hedge_cfg.path_k_max)]
            if len(g) == 0:
                continue
            g = _compute_iv(g)
            g = g[g["iv"].notna() & (g["iv"] > 0.02) & (g["iv"] < 5.0)]
            if len(g) == 0:
                continue
            # Use midpoint-implied volatility for hedging Greeks whenever it is
            # numerically invertible.  This keeps the SmiledDelta baseline and
            # liability state features aligned with the observed mid marks used
            # in pathwise PnL.
            g["mid_iv_target"] = _compute_mid_iv_target(g)
            g = g[np.isfinite(g["mid_iv_target"]) & (g["mid_iv_target"] > 0.02) & (g["mid_iv_target"] < 5.0)]
            if len(g) == 0:
                continue
            g["iv"] = g["mid_iv_target"]
            g["mid"] = 0.5 * (g["best_bid"] + g["best_offer"])
            g["rel_spread"] = (g["best_offer"] - g["best_bid"]) / g["mid"].clip(lower=1e-4)
            g = g[g["rel_spread"] <= hedge_cfg.max_path_rel_spread]
            if len(g) == 0:
                continue

            if "am_set_flag" in g.columns:
                g["am_flag"] = g["am_set_flag"].fillna(0).astype(int)
            elif "am_settlement" in g.columns:
                g["am_flag"] = g["am_settlement"].fillna(0).astype(int)
            else:
                g["am_flag"] = 0
            g = _apply_settlement_policy(g, data_cfg.settlement_policy)
            if len(g) == 0:
                continue

            g["vega_bs"] = _compute_vega(g, iv_col="iv")
            g["delta_bs"] = [
                _bs_delta_np(F0, K0, tau0, sig0, D0, S, cp == "C")
                for F0, K0, tau0, sig0, D0, cp in zip(g["F"], g["K"], g["tau"], g["iv"], g["D_r"], g["cp_flag"])
            ]
            g["liq_score"] = g["vega_bs"] / (g["rel_spread"] + 1e-4)
            g["S"] = S
            g["VIX"] = _safe_float(price_map.loc[tdate, "VIX"], 20.0) if "VIX" in price_map.columns else 20.0
            g["VXV"] = _safe_float(price_map.loc[tdate, "VXV"], 20.0) if "VXV" in price_map.columns else 20.0
            g["date"] = pd.to_datetime(g["date"])
            g["exdate"] = pd.to_datetime(g["exdate"])
            keep_cols = [
                "date", "exdate", "strike_key", "K", "cp_flag", "tau", "k", "F", "D_r",
                "best_bid", "best_offer", "mid", "rel_spread", "iv", "delta_bs", "vega_bs",
                "liq_score", "S", "VIX", "VXV", "am_flag",
            ]
            year_rows.append(g[keep_cols])

        if year_rows:
            rows.append(pd.concat(year_rows, ignore_index=True))

    if not rows:
        raise ValueError("No hedging chain rows were produced. Check filters and data paths.")

    chains = pd.concat(rows, ignore_index=True)
    chains.sort_values(["date", "exdate", "strike_key", "cp_flag"], inplace=True)
    chains.reset_index(drop=True, inplace=True)
    chains.to_parquet(out_path, index=False)
    logger.info(f"Saved hedging chain table: {out_path} | rows={len(chains):,} dates={chains['date'].nunique():,}")
    return out_path


def load_surface_model_for_hedging(
    data_cfg: DataConfig = DATA_CFG,
    model_cfg: ModelConfig = MODEL_CFG,
    preference: str = "forecast",
    device: Optional[str] = None,
) -> FullSurfaceModel:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    surface_model = FullSurfaceModel(model_cfg).to(device_t)
    heads = DownstreamHeads(model_cfg).to(device_t)
    load_stage3_checkpoint_for_eval(surface_model, heads, data_cfg.results_dir, device, preference=preference)
    surface_model.eval()
    return surface_model


@torch.no_grad()
def build_hedging_state_cache(
    cache_path: str,
    data_cfg: DataConfig = DATA_CFG,
    model_cfg: ModelConfig = MODEL_CFG,
    hedge_cfg: HedgingConfig = HedgingConfig(),
    checkpoint_preference: str = "forecast",
    force_rebuild: bool = False,
) -> str:
    """Extract daily latent, summary, and fixed-grid states for hedging."""
    os.makedirs(hedge_cfg.cache_dir, exist_ok=True)
    out_path = os.path.join(hedge_cfg.cache_dir, f"hedging_state_cache_{checkpoint_preference}.pkl")
    require_setonly = bool(getattr(hedge_cfg, "require_setonly_state", False))
    if os.path.exists(out_path) and not force_rebuild:
        if require_setonly:
            try:
                with open(out_path, "rb") as f:
                    cached_payload = pickle.load(f)
                cached_df = cached_payload.get("state_df", pd.DataFrame())
                if any(str(c).startswith("setonly_z_") for c in cached_df.columns):
                    logger.info(f"Hedging state cache found: {out_path}")
                    return out_path
                logger.info("Existing hedging state cache lacks set-only latents; rebuilding.")
            except Exception:
                logger.info("Could not inspect existing hedging state cache; rebuilding.")
        else:
            logger.info(f"Hedging state cache found: {out_path}")
            return out_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_surface_model_for_hedging(data_cfg, model_cfg, preference=checkpoint_preference, device=device)

    # Optional no-surface/set-only latent state ablation.  If the clean
    # SetForecastOnly ablation checkpoint exists, we add setonly_z_* columns to
    # the same state cache.  HedgeSet state mode "setonly" or "nosurface"
    # will then build policy states from this representation.
    setonly_model = None
    explicit_setonly = str(getattr(hedge_cfg, "setonly_checkpoint_path", "") or os.environ.get("HEDGESET_SETONLY_CKPT", "") or "").strip()
    candidate_setonly_paths = []
    if explicit_setonly:
        candidate_setonly_paths.append(explicit_setonly)
    candidate_setonly_paths.extend([
        os.path.join(data_cfg.results_dir, "checkpoints", "set_forecast_only_best.pt"),
        os.path.join("results", "set_forecast_only", f"seed_{getattr(TRAIN_CFG, 'seed', 42)}", "set_forecast_only", "checkpoints", "set_forecast_only_best.pt"),
        os.path.join("results", "set_forecast_only", "seed_42", "set_forecast_only", "checkpoints", "set_forecast_only_best.pt"),
    ])
    setonly_ckpt = next((c for c in candidate_setonly_paths if c and os.path.exists(c)), None)
    if setonly_ckpt is not None:
        try:
            setonly_model = SetForecastOnlyModel(model_cfg).to(torch.device(device))
            ck_set = torch.load(setonly_ckpt, map_location=device)
            if "model" in ck_set:
                setonly_model.load_state_dict(ck_set["model"])
            elif "encoder" in ck_set:
                setonly_model.encoder.load_state_dict(ck_set["encoder"])
            else:
                raise ValueError(f"Unsupported checkpoint keys: {list(ck_set.keys())}")
            setonly_model.eval()
            logger.info(f"Loaded set-only/no-surface encoder for hedging ablation: {setonly_ckpt}")
        except Exception as exc:
            if require_setonly:
                raise
            logger.warning(f"Could not load set-only checkpoint for hedging ablation: {exc}")
            setonly_model = None
    elif require_setonly:
        raise FileNotFoundError(
            "state_mode=setonly/nosurface was requested but no set-forecast-only checkpoint was found. "
            "Run --set-forecast-only first or pass --hedgeset-setonly-checkpoint."
        )
    else:
        logger.warning("Set-only checkpoint not found; HedgeSet state_mode=setonly/nosurface will be unavailable. Run --set-forecast-only first to enable it.")

    dfs = []
    grid_map: Dict[str, np.ndarray] = {}
    with open(cache_path, "rb") as f:
        all_data = pickle.load(f)

    for split in ["train", "val", "test"]:
        ds = OptionDataset(cache_path, split, data_cfg)
        ld = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)
        df_split = extract_latents(model, ld, torch.device(device))
        if setonly_model is not None:
            df_set = extract_latents(setonly_model, ld, torch.device(device))
            rename = {f"z_{j}": f"setonly_z_{j}" for j in range(model_cfg.latent_dim)}
            keep_cols = ["date"] + list(rename.values())
            df_set = df_set.rename(columns=rename)[keep_cols]
            df_split = df_split.merge(df_set, on="date", how="left")
        df_split["split"] = split
        dfs.append(df_split)
        for rec in ds.records:
            dstr = str(pd.Timestamp(rec["date"]).date())
            grid_map[dstr] = record_to_grid_surface(rec, data_cfg, n_k=21, n_tau=16).astype(np.float32)

    state_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    state_df["date_str"] = state_df["date"].dt.strftime("%Y-%m-%d")

    payload = {
        "state_df": state_df,
        "grid_map": grid_map,
        "latent_dim": model_cfg.latent_dim,
        "grid_shape": (2, 16, 21),
        "checkpoint_preference": checkpoint_preference,
        "source_cache_path": cache_path,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    logger.info(f"Saved hedging state cache: {out_path} | days={len(state_df):,}")
    return out_path


def _state_row_to_summary_vec(row: pd.Series) -> np.ndarray:
    return np.asarray([
        _safe_float(row.get("atm_iv", np.nan), 0.20),
        _safe_float(row.get("short_iv", np.nan), 0.20),
        _safe_float(row.get("term_slope", np.nan), 0.0),
        _safe_float(row.get("skew_proxy", np.nan), 0.0),
        _safe_float(row.get("rv5_back", np.nan), 0.20),
        _safe_float(row.get("rv21_back", np.nan), 0.20),
        _safe_float(row.get("dsv5_back", np.nan), 0.10),
        math.log1p(max(_safe_float(row.get("N_quotes", 0.0), 0.0), 0.0)) / 10.0,
    ], dtype=np.float32)


def _scaled_side_from_row(row: pd.Series) -> np.ndarray:
    vix = (_safe_float(row.get("VIX", np.nan), 20.0) - 20.0) / 15.0
    vxv = (_safe_float(row.get("VXV", np.nan), 20.0) - 20.0) / 15.0
    return np.asarray([vix, vxv], dtype=np.float32)


def _liability_feat(
    k: float,
    tau: float,
    cp_code: float,
    delta: float,
    vega: float,
    L0: float,
    L_t: Optional[float] = None,
) -> np.ndarray:
    """Complete liability-side state used by HedgeSet policies.

    The policy state exposes both liability delta and liability vega. Vega is
    divided by 100 to match the instrument feature scaling used in HedgeSet
    candidate features.
    L_t/L0 gives the controller a normalized mark-to-market state without
    leaking future information.
    """
    L0_s = max(_safe_float(L0, 1.0), 1.0)
    Lt_s = _safe_float(L_t, L0_s) if L_t is not None else L0_s
    return np.asarray([
        _safe_float(k, 0.0),
        _safe_float(tau, 30.0 / 365.0),
        _safe_float(cp_code, 0.0),
        _safe_float(delta, 0.0),
        _safe_float(vega, 0.0) / 100.0,
        math.log1p(L0_s) / 10.0,
        Lt_s / L0_s - 1.0,
    ], dtype=np.float32)


def _select_single_liability(start_df: pd.DataFrame, family: str, cfg: HedgingConfig) -> Optional[pd.Series]:
    df = start_df.copy()
    target_tau = cfg.target_tau_days / 365.0
    tau_tol = cfg.tau_tol_days / 365.0
    df = df[(df["tau"].sub(target_tau).abs() <= tau_tol)]
    df = df[(df["best_bid"] >= cfg.min_liability_bid) & (df["rel_spread"] <= cfg.max_liability_rel_spread)]
    if len(df) == 0:
        return None

    if family == "atm_call":
        sub = df[df["cp_flag"] == "C"].copy()
        if len(sub) == 0:
            return None
        sub["select_score"] = sub["k"].abs() + 0.25 * (sub["tau"] - target_tau).abs() + 0.01 * sub["rel_spread"]
        return sub.sort_values(["select_score", "rel_spread"], ascending=True).iloc[0]

    if family == "put25":
        sub = df[df["cp_flag"] == "P"].copy()
        if len(sub) == 0:
            return None
        sub["select_score"] = (sub["delta_bs"] + 0.25).abs() + 0.10 * (sub["tau"] - target_tau).abs() + 0.01 * sub["rel_spread"]
        return sub.sort_values(["select_score", "rel_spread"], ascending=True).iloc[0]

    raise ValueError(f"Unsupported single liability family: {family}")


def _select_straddle_liability(start_df: pd.DataFrame, cfg: HedgingConfig) -> Optional[dict]:
    target_tau = cfg.target_tau_days / 365.0
    tau_tol = cfg.tau_tol_days / 365.0
    df = start_df.copy()
    df = df[(df["tau"].sub(target_tau).abs() <= tau_tol)]
    df = df[(df["best_bid"] >= cfg.min_liability_bid) & (df["rel_spread"] <= cfg.max_liability_rel_spread)]
    if len(df) == 0:
        return None
    calls = df[df["cp_flag"] == "C"].copy()
    puts = df[df["cp_flag"] == "P"].copy()
    if len(calls) == 0 or len(puts) == 0:
        return None
    m = calls.merge(
        puts,
        on=["date", "exdate", "strike_key", "K"],
        suffixes=("_C", "_P"),
        how="inner",
    )
    if len(m) == 0:
        return None
    # k_C and k_P should be identical for the same strike/date; use average for robustness.
    m["k_avg"] = 0.5 * (m["k_C"] + m["k_P"])
    m["tau_avg"] = 0.5 * (m["tau_C"] + m["tau_P"])
    m["rel_avg"] = 0.5 * (m["rel_spread_C"] + m["rel_spread_P"])
    m["select_score"] = m["k_avg"].abs() + 0.25 * (m["tau_avg"] - target_tau).abs() + 0.01 * m["rel_avg"]
    row = m.sort_values(["select_score", "rel_avg"], ascending=True).iloc[0]
    return {
        "date": row["date"],
        "exdate": row["exdate"],
        "strike_key": int(row["strike_key"]),
        "K": float(row["K"]),
    }


def _get_contract_row(contract_index: pd.DataFrame, date: pd.Timestamp, exdate: pd.Timestamp, strike_key: int, cp: str) -> Optional[pd.Series]:
    try:
        row = contract_index.loc[(pd.Timestamp(date), pd.Timestamp(exdate), int(strike_key), cp)]
        if isinstance(row, pd.DataFrame):
            row = row.sort_values("rel_spread").iloc[0]
        return row
    except KeyError:
        return None


def _build_one_episode(
    family: str,
    path_dates: List[pd.Timestamp],
    chains_by_date: Dict[pd.Timestamp, pd.DataFrame],
    contract_index: pd.DataFrame,
    state_by_date: Dict[str, pd.Series],
    grid_map: Dict[str, np.ndarray],
    latent_cols: List[str],
    hedge_cfg: HedgingConfig,
    data_cfg: DataConfig = DATA_CFG,
) -> Optional[dict]:
    t0 = path_dates[0]
    start_df = chains_by_date.get(t0)
    if start_df is None or len(start_df) == 0:
        return None

    if family in {"atm_call", "put25"}:
        sel = _select_single_liability(start_df, family, hedge_cfg)
        if sel is None:
            return None
        exdate = pd.Timestamp(sel["exdate"])
        strike_key = int(sel["strike_key"])
        cp = str(sel["cp_flag"])
        cp_code = 1.0 if cp == "C" else -1.0
        legs = [(cp, 1.0)]
    elif family == "atm_straddle":
        sel2 = _select_straddle_liability(start_df, hedge_cfg)
        if sel2 is None:
            return None
        exdate = pd.Timestamp(sel2["exdate"])
        strike_key = int(sel2["strike_key"])
        cp_code = 0.0
        legs = [("C", 1.0), ("P", 1.0)]
    else:
        raise ValueError(f"Unsupported liability family: {family}")

    S_path, L_path = [], []
    delta_path, k_path, tau_path = [], [], []
    latent_states, latent_summary_states, summary_states, grid_states, side_states, liab_states = [], [], [], [], [], []

    L0 = None
    for idx, d in enumerate(path_dates):
        leg_rows = []
        for cp_leg, _qty in legs:
            r = _get_contract_row(contract_index, d, exdate, strike_key, cp_leg)
            if r is None:
                return None
            if _safe_float(r.get("best_bid"), 0.0) < hedge_cfg.min_path_bid:
                return None
            if _safe_float(r.get("rel_spread"), 999.0) > hedge_cfg.max_path_rel_spread:
                return None
            leg_rows.append(r)

        S_t = float(leg_rows[0]["S"])
        L_t = float(sum(float(r["mid"]) for r in leg_rows))
        delta_t = float(sum(float(r["delta_bs"]) for r in leg_rows))
        vega_t = float(sum(float(r.get("vega_bs", 0.0)) for r in leg_rows))
        k_t = float(np.mean([float(r["k"]) for r in leg_rows]))
        tau_t = float(np.mean([float(r["tau"]) for r in leg_rows]))
        if not np.isfinite(L_t) or L_t <= 0:
            return None
        if L0 is None:
            L0 = L_t

        S_path.append(S_t)
        L_path.append(L_t)
        if idx < len(path_dates) - 1:
            dstr = str(pd.Timestamp(d).date())
            if dstr not in state_by_date or dstr not in grid_map:
                return None
            st = state_by_date[dstr]
            z = np.asarray([_safe_float(st.get(c), 0.0) for c in latent_cols], dtype=np.float32)
            side = _scaled_side_from_row(st)
            summary = _state_row_to_summary_vec(st)
            grid = np.asarray(grid_map[dstr], dtype=np.float32).reshape(-1)
            liab = _liability_feat(k_t, tau_t, cp_code, delta_t, vega_t, L0, L_t)

            latent_state = np.concatenate([z, side, z * side[0], z * side[1], liab], axis=0).astype(np.float32)
            latent_summary_state = np.concatenate(
                [z, summary, side, z * side[0], z * side[1], summary * side[0], summary * side[1], liab],
                axis=0,
            ).astype(np.float32)
            summary_state = np.concatenate([summary, side, liab], axis=0).astype(np.float32)
            grid_state = np.concatenate([grid, side, liab], axis=0).astype(np.float32)

            latent_states.append(latent_state)
            latent_summary_states.append(latent_summary_state)
            summary_states.append(summary_state)
            grid_states.append(grid_state)
            side_states.append(side)
            liab_states.append(liab)
            delta_path.append(delta_t)
            k_path.append(k_t)
            tau_path.append(tau_t)

    H = hedge_cfg.horizon
    if len(S_path) != H + 1 or len(L_path) != H + 1 or len(delta_path) != H:
        return None

    scale = max(float(L0), 1.0)
    return {
        "family": family,
        "start_date": str(path_dates[0].date()),
        "end_date": str(path_dates[-1].date()),
        "exdate": str(pd.Timestamp(exdate).date()),
        "strike_key": int(strike_key),
        "cp_code": float(cp_code),
        "H": H,
        "S": np.asarray(S_path, dtype=np.float32),
        "L": np.asarray(L_path, dtype=np.float32),
        "delta": np.asarray(delta_path, dtype=np.float32),
        "k": np.asarray(k_path, dtype=np.float32),
        "tau": np.asarray(tau_path, dtype=np.float32),
        "scale": float(scale),
        "latent_state": np.asarray(latent_states, dtype=np.float32),
        "latent_summary_state": np.asarray(latent_summary_states, dtype=np.float32),
        "summary_state": np.asarray(summary_states, dtype=np.float32),
        "grid_state": np.asarray(grid_states, dtype=np.float32),
        "dates": [str(pd.Timestamp(d).date()) for d in path_dates],
        "split": _date_split_from_timestamp(path_dates[0], data_cfg),
    }


def build_hedging_episodes(
    chains_path: str,
    state_cache_path: str,
    family: str,
    data_cfg: DataConfig = DATA_CFG,
    hedge_cfg: HedgingConfig = HedgingConfig(),
    force_rebuild: bool = False,
) -> str:
    """Construct leakage-safe historical hedging episodes for one liability family."""
    os.makedirs(hedge_cfg.cache_dir, exist_ok=True)
    out_path = os.path.join(hedge_cfg.cache_dir, f"episodes_{family}_H{hedge_cfg.horizon}_stride{hedge_cfg.start_stride}.pkl")
    if os.path.exists(out_path) and not force_rebuild:
        logger.info(f"Hedging episode cache found: {out_path}")
        return out_path

    chains = pd.read_parquet(chains_path)
    chains["date"] = pd.to_datetime(chains["date"])
    chains["exdate"] = pd.to_datetime(chains["exdate"])
    chains.sort_values(["date", "exdate", "strike_key", "cp_flag"], inplace=True)
    chains_by_date = {pd.Timestamp(d): g.copy() for d, g in chains.groupby("date")}
    contract_index = chains.set_index(["date", "exdate", "strike_key", "cp_flag"], drop=False)

    with open(state_cache_path, "rb") as f:
        payload = pickle.load(f)
    state_df = payload["state_df"].copy()
    state_df["date"] = pd.to_datetime(state_df["date"])
    state_df["date_str"] = state_df["date"].dt.strftime("%Y-%m-%d")
    grid_map = payload["grid_map"]
    latent_cols = sorted([c for c in state_df.columns if c.startswith("z_")], key=lambda s: int(s.split("_")[1]))
    state_by_date = {row["date_str"]: row for _, row in state_df.iterrows()}

    available_dates = sorted([pd.Timestamp(d) for d in state_df["date"].unique()])
    episodes: List[dict] = []
    skipped = 0

    for pos in tqdm(range(0, len(available_dates) - hedge_cfg.horizon, hedge_cfg.start_stride), desc=f"episodes {family}"):
        path_dates = available_dates[pos:pos + hedge_cfg.horizon + 1]
        if len(path_dates) != hedge_cfg.horizon + 1:
            continue
        # Leakage guard: the whole path must remain inside a single split.
        splits = {_date_split_from_timestamp(d, data_cfg) for d in path_dates}
        if len(splits) != 1:
            skipped += 1
            continue
        ep = _build_one_episode(
            family=family,
            path_dates=path_dates,
            chains_by_date=chains_by_date,
            contract_index=contract_index,
            state_by_date=state_by_date,
            grid_map=grid_map,
            latent_cols=latent_cols,
            hedge_cfg=hedge_cfg,
            data_cfg=data_cfg,
        )
        if ep is None:
            skipped += 1
            continue
        episodes.append(ep)

    if len(episodes) == 0:
        raise ValueError(f"No hedging episodes produced for family={family}.")

    meta = {
        "family": family,
        "horizon": hedge_cfg.horizon,
        "start_stride": hedge_cfg.start_stride,
        "target_tau_days": hedge_cfg.target_tau_days,
        "tau_tol_days": hedge_cfg.tau_tol_days,
        "n_episodes": len(episodes),
        "n_skipped": skipped,
        "split_counts": pd.Series([ep["split"] for ep in episodes]).value_counts().to_dict(),
    }
    with open(out_path, "wb") as f:
        pickle.dump({"episodes": episodes, "meta": meta}, f, protocol=4)
    logger.info(f"Saved episodes: {out_path} | {meta}")
    return out_path


class HedgingEpisodeDataset(Dataset):
    def __init__(self, episodes_path: str, split: str, state_mode: str):
        with open(episodes_path, "rb") as f:
            payload = pickle.load(f)
        self.meta = payload.get("meta", {})
        self.episodes = [ep for ep in payload["episodes"] if ep.get("split") == split]
        self.split = split
        self.state_mode = state_mode
        key = f"{state_mode}_state"
        if len(self.episodes) == 0:
            raise ValueError(f"No episodes for split={split} in {episodes_path}. Meta={self.meta}")
        if key not in self.episodes[0]:
            raise ValueError(f"State mode {state_mode} unavailable. Expected key {key}.")

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict:
        ep = self.episodes[idx]
        key = f"{self.state_mode}_state"
        return {
            "base_state": torch.tensor(ep[key], dtype=torch.float32),
            "S": torch.tensor(ep["S"], dtype=torch.float32),
            "L": torch.tensor(ep["L"], dtype=torch.float32),
            "delta": torch.tensor(ep["delta"], dtype=torch.float32),
            "scale": torch.tensor(float(ep["scale"]), dtype=torch.float32),
            "start_date": ep["start_date"],
            "family": ep["family"],
        }


def hedge_collate_fn(batch: List[dict]) -> dict:
    return {
        "base_state": torch.stack([b["base_state"] for b in batch], dim=0),
        "S": torch.stack([b["S"] for b in batch], dim=0),
        "L": torch.stack([b["L"] for b in batch], dim=0),
        "delta": torch.stack([b["delta"] for b in batch], dim=0),
        "scale": torch.stack([b["scale"] for b in batch], dim=0),
        "start_date": [b["start_date"] for b in batch],
        "family": [b["family"] for b in batch],
    }


class FeatureNormalizer(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float().clamp(min=1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def fit_feature_normalizer(ds: HedgingEpisodeDataset, device: torch.device) -> FeatureNormalizer:
    mats = []
    for ep in ds.episodes:
        mats.append(np.asarray(ep[f"{ds.state_mode}_state"], dtype=np.float32))
    X = np.concatenate(mats, axis=0)
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
    return FeatureNormalizer(torch.tensor(mean), torch.tensor(std)).to(device)


class NoTradeBandPolicy(nn.Module):
    def __init__(self, input_dim: int, cfg: HedgingConfig):
        super().__init__()
        layers = []
        cur = input_dim + 1  # plus previous hedge holding
        for _ in range(max(1, cfg.hidden_layers - 1)):
            layers += [nn.Linear(cur, cfg.hidden_dim), nn.GELU(), nn.LayerNorm(cfg.hidden_dim)]
            cur = cfg.hidden_dim
        layers.append(nn.Linear(cur, 2))
        self.net = nn.Sequential(*layers)
        self.h_max = float(cfg.h_max)
        self.trade_max = float(cfg.trade_max)
        self.band_max = float(cfg.band_max)
        self.smooth_temp = float(cfg.smooth_temp)

    def forward(self, x_norm: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inp = torch.cat([x_norm, (h_prev / self.h_max).unsqueeze(1)], dim=1)
        out = self.net(inp)
        mu = self.h_max * torch.tanh(out[:, 0])
        band = self.band_max * torch.sigmoid(out[:, 1])
        gap = mu - h_prev
        abs_gap = torch.abs(gap)
        # Smooth approximation to relu(abs_gap - band).
        active_mag = self.smooth_temp * F.softplus((abs_gap - band) / max(self.smooth_temp, 1e-6))
        trade = torch.sign(gap) * active_mag
        trade = torch.clamp(trade, -self.trade_max, self.trade_max)
        h = torch.clamp(h_prev + trade, -self.h_max, self.h_max)
        return h, mu, band


class HedgingAgent(nn.Module):
    def __init__(self, input_dim: int, normalizer: FeatureNormalizer, cfg: HedgingConfig):
        super().__init__()
        self.normalizer = normalizer
        self.policy = NoTradeBandPolicy(input_dim, cfg)
        self.cvar_eta = nn.Parameter(torch.tensor(0.0))
        self.cfg = cfg

    def act(self, base_state_t: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.normalizer(base_state_t)
        return self.policy(x, h_prev)


def _rollout_policy(
    agent: Optional[HedgingAgent],
    batch: dict,
    hedge_cfg: HedgingConfig,
    cost_bps: float,
    mode: str = "policy",
) -> Dict[str, torch.Tensor]:
    base = batch["base_state"]
    S = batch["S"]
    L = batch["L"]
    delta = batch["delta"]
    scale = batch["scale"].clamp(min=1.0)
    device = S.device
    B, H = delta.shape

    h_prev = torch.zeros(B, device=device)
    pnl = torch.zeros(B, device=device)
    tc_total = torch.zeros(B, device=device)
    turnover = torch.zeros(B, device=device)
    abs_h_sum = torch.zeros(B, device=device)

    for t in range(H):
        if mode == "no_hedge":
            h = torch.zeros_like(h_prev)
        elif mode == "delta":
            h = torch.clamp(delta[:, t], -hedge_cfg.h_max, hedge_cfg.h_max)
        elif mode == "policy":
            if agent is None:
                raise ValueError("agent is required for mode='policy'")
            h, _, _ = agent.act(base[:, t, :], h_prev)
        else:
            raise ValueError(f"Unknown rollout mode: {mode}")

        dS = S[:, t + 1] - S[:, t]
        dL = L[:, t + 1] - L[:, t]
        trade = h - h_prev
        tc = (cost_bps / 10000.0) * S[:, t].abs() * trade.abs()
        pnl = pnl + h * dS - dL - tc
        tc_total = tc_total + tc
        turnover = turnover + trade.abs()
        abs_h_sum = abs_h_sum + h.abs()
        h_prev = h

    if hedge_cfg.terminal_liquidation:
        # Close any remaining hedge at the terminal date.  The mark-to-market
        # price move of the terminal position is already included by the summed
        # h_t dS_t terms; this line adds only the missing execution cost of
        # liquidating h_H.
        tc_close = (cost_bps / 10000.0) * S[:, -1].abs() * h_prev.abs()
        pnl = pnl - tc_close
        tc_total = tc_total + tc_close
        turnover = turnover + h_prev.abs()

    loss_raw = -pnl
    loss_norm = loss_raw / scale
    pnl_norm = pnl / scale
    tc_norm = tc_total / scale
    return {
        "loss_raw": loss_raw,
        "loss_norm": loss_norm,
        "pnl_norm": pnl_norm,
        "tc_norm": tc_norm,
        "turnover": turnover,
        "avg_abs_h": abs_h_sum / float(H),
    }


def _cvar_torch(losses: torch.Tensor, alpha: float, eta: Optional[torch.Tensor] = None) -> torch.Tensor:
    if eta is None:
        eta = torch.quantile(losses.detach(), alpha)
    return eta + torch.relu(losses - eta).mean() / max(1.0 - alpha, 1e-6)


def hedging_training_loss(agent: HedgingAgent, batch: dict, hedge_cfg: HedgingConfig, cost_bps: float) -> Tuple[torch.Tensor, dict]:
    out = _rollout_policy(agent, batch, hedge_cfg, cost_bps, mode="policy")
    losses = out["loss_norm"]
    cvar = _cvar_torch(losses, hedge_cfg.cvar_alpha, agent.cvar_eta)
    mean_loss = losses.mean()
    tc_pen = out["tc_norm"].mean()
    turn_pen = out["turnover"].mean()
    pos_pen = out["avg_abs_h"].pow(2).mean()
    total = (
        cvar
        + hedge_cfg.lambda_mean * mean_loss
        + hedge_cfg.lambda_tc * tc_pen
        + hedge_cfg.lambda_turnover * turn_pen
        + hedge_cfg.lambda_pos * pos_pen
    )
    info = {
        "total": float(total.detach().cpu()),
        "cvar": float(cvar.detach().cpu()),
        "mean_loss": float(mean_loss.detach().cpu()),
        "tc": float(tc_pen.detach().cpu()),
        "turnover": float(turn_pen.detach().cpu()),
    }
    return total, info


def _metrics_from_episode_outputs(df: pd.DataFrame, label: str, family: str, state_mode: str, cost_bps: float) -> dict:
    x = df["loss_norm"].values.astype(float)
    pnl = df["pnl_norm"].values.astype(float)
    tc = df["tc_norm"].values.astype(float)
    turnover = df["turnover"].values.astype(float)
    abs_h = df["avg_abs_h"].values.astype(float)
    def q(p):
        return float(np.nanquantile(x, p)) if len(x) else np.nan
    out = {
        "model": label,
        "family": family,
        "state_mode": state_mode,
        "cost_bps": float(cost_bps),
        "n": int(len(df)),
        "mean_loss_norm": float(np.nanmean(x)),
        "std_loss_norm": float(np.nanstd(x)),
        "rmse_loss_norm": float(np.sqrt(np.nanmean(np.square(x)))),
        "var_95": q(0.95),
        "var_99": q(0.99),
        "cvar_95": float(np.nanmean(x[x >= q(0.95)])) if len(x) else np.nan,
        "cvar_99": float(np.nanmean(x[x >= q(0.99)])) if len(x) else np.nan,
        "shortfall_prob": float(np.nanmean(x > 0.0)) if len(x) else np.nan,
        "mean_pnl_norm": float(np.nanmean(pnl)),
        "mean_tc_norm": float(np.nanmean(tc)),
        "mean_turnover": float(np.nanmean(turnover)),
        "mean_abs_hedge": float(np.nanmean(abs_h)),
    }
    return out


@torch.no_grad()
def evaluate_hedging_agent(
    agent: Optional[HedgingAgent],
    loader: DataLoader,
    hedge_cfg: HedgingConfig,
    cost_bps: float,
    model_label: str,
    family: str,
    state_mode: str,
    mode: str = "policy",
) -> Tuple[dict, pd.DataFrame]:
    rows = []
    device = next(agent.parameters()).device if agent is not None else torch.device("cpu")
    for batch in loader:
        batch_t = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = _rollout_policy(agent, batch_t, hedge_cfg, cost_bps, mode=mode)
        B = len(batch["start_date"])
        for i in range(B):
            rows.append({
                "model": model_label,
                "family": family,
                "state_mode": state_mode,
                "cost_bps": float(cost_bps),
                "start_date": batch["start_date"][i],
                "loss_norm": float(out["loss_norm"][i].detach().cpu()),
                "loss_raw": float(out["loss_raw"][i].detach().cpu()),
                "pnl_norm": float(out["pnl_norm"][i].detach().cpu()),
                "tc_norm": float(out["tc_norm"][i].detach().cpu()),
                "turnover": float(out["turnover"][i].detach().cpu()),
                "avg_abs_h": float(out["avg_abs_h"][i].detach().cpu()),
            })
    df = pd.DataFrame(rows)
    return _metrics_from_episode_outputs(df, model_label, family, state_mode, cost_bps), df


def train_hedging_policy_one(
    episodes_path: str,
    family: str,
    state_mode: str,
    cost_bps: float,
    hedge_cfg: HedgingConfig,
    out_dir: str,
    seed: int = 42,
) -> Tuple[dict, pd.DataFrame]:
    set_seed(seed)
    device = torch.device(TRAIN_CFG.device if torch.cuda.is_available() else "cpu")
    train_ds = HedgingEpisodeDataset(episodes_path, "train", state_mode)
    val_ds = HedgingEpisodeDataset(episodes_path, "val", state_mode)
    test_ds = HedgingEpisodeDataset(episodes_path, "test", state_mode)

    train_ld = DataLoader(train_ds, batch_size=hedge_cfg.batch_size, shuffle=True, collate_fn=hedge_collate_fn, drop_last=False)
    val_ld = DataLoader(val_ds, batch_size=hedge_cfg.batch_size, shuffle=False, collate_fn=hedge_collate_fn)
    test_ld = DataLoader(test_ds, batch_size=hedge_cfg.batch_size, shuffle=False, collate_fn=hedge_collate_fn)

    normer = fit_feature_normalizer(train_ds, device)
    input_dim = int(train_ds[0]["base_state"].shape[1])
    agent = HedgingAgent(input_dim=input_dim, normalizer=normer, cfg=hedge_cfg).to(device)
    opt = torch.optim.AdamW(agent.parameters(), lr=hedge_cfg.lr, weight_decay=hedge_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(hedge_cfg.epochs, 1), eta_min=hedge_cfg.lr * 0.05)

    os.makedirs(out_dir, exist_ok=True)
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(hedge_cfg.epochs):
        agent.train()
        acc = {"total": 0.0, "cvar": 0.0, "mean_loss": 0.0, "tc": 0.0, "turnover": 0.0}
        n_batches = 0
        for batch in train_ld:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            opt.zero_grad()
            loss, info = hedging_training_loss(agent, batch, hedge_cfg, cost_bps)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            opt.step()
            for k in acc:
                acc[k] += info.get(k, 0.0)
            n_batches += 1
        scheduler.step()
        for k in acc:
            acc[k] /= max(n_batches, 1)

        agent.eval()
        val_metrics, _ = evaluate_hedging_agent(agent, val_ld, hedge_cfg, cost_bps, "Policy", family, state_mode, mode="policy")
        val_score = val_metrics["cvar_95"]
        row = {"epoch": epoch + 1, "lr": opt.param_groups[0]["lr"], **{f"train_{k}": v for k, v in acc.items()}, **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))}}
        history.append(row)
        if (epoch + 1) % 25 == 0 or epoch == 0:
            logger.info(f"[hedge {family}/{state_mode}/cost={cost_bps:g} epoch {epoch+1:03d}] train_cvar={acc['cvar']:.4f} val_cvar95={val_score:.4f}")

        if val_score < best_val - hedge_cfg.min_delta:
            best_val = val_score
            best_state = copy.deepcopy(agent.state_dict())
            bad_epochs = 0
            torch.save({"agent": best_state, "epoch": epoch + 1, "best_val_cvar95": best_val}, os.path.join(out_dir, "best_policy.pt"))
        else:
            bad_epochs += 1
            if bad_epochs >= hedge_cfg.patience:
                logger.info(f"Early stopping hedge policy at epoch {epoch+1}; best val CVaR95={best_val:.6f}")
                break

    if best_state is not None:
        agent.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "training_history.csv"), index=False)

    test_metrics, test_panel = evaluate_hedging_agent(agent, test_ld, hedge_cfg, cost_bps, f"Policy-{state_mode}", family, state_mode, mode="policy")
    test_panel.to_csv(os.path.join(out_dir, "test_episode_panel.csv"), index=False)
    pd.DataFrame([test_metrics]).to_csv(os.path.join(out_dir, "test_metrics.csv"), index=False)
    return test_metrics, test_panel


def evaluate_static_hedging_baselines(
    episodes_path: str,
    family: str,
    state_mode_for_loader: str,
    cost_bps: float,
    hedge_cfg: HedgingConfig,
    out_dir: str,
) -> Tuple[List[dict], pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)
    test_ds = HedgingEpisodeDataset(episodes_path, "test", state_mode_for_loader)
    test_ld = DataLoader(test_ds, batch_size=hedge_cfg.batch_size, shuffle=False, collate_fn=hedge_collate_fn)
    metrics = []
    panels = []
    for label, mode in [("NoHedge", "no_hedge"), ("SmiledDelta", "delta")]:
        m, p = evaluate_hedging_agent(None, test_ld, hedge_cfg, cost_bps, label, family, "static", mode=mode)
        metrics.append(m)
        panels.append(p)
    all_panel = pd.concat(panels, ignore_index=True)
    all_panel.to_csv(os.path.join(out_dir, "static_baseline_episode_panel.csv"), index=False)
    pd.DataFrame(metrics).to_csv(os.path.join(out_dir, "static_baseline_metrics.csv"), index=False)
    return metrics, all_panel


def _bootstrap_stat_from_losses(x: np.ndarray, metric: str, alpha: float = 0.95, regimes: Optional[np.ndarray] = None) -> float:
    x = np.asarray(x, dtype=float)
    metric = str(metric).lower()
    if regimes is not None:
        regimes = np.asarray(regimes)
        valid0 = np.isfinite(x) & pd.notna(regimes)
        x_reg = x[valid0]
        r_reg = regimes[valid0]
        if x_reg.size == 0:
            return np.nan
        if metric in {"worst_regime_cvar95", "worst_regime_cvar_95"}:
            vals = []
            for rid in np.unique(r_reg):
                xr = x_reg[r_reg == rid]
                if xr.size == 0:
                    continue
                q = np.quantile(xr, 0.95)
                vals.append(float(np.mean(xr[xr >= q])))
            return float(np.nanmax(vals)) if vals else np.nan
        if metric in {"worst_regime_shortfall", "worst_regime_shortfall_prob"}:
            vals = [float(np.mean(x_reg[r_reg == rid] > 0.0)) for rid in np.unique(r_reg) if np.any(r_reg == rid)]
            return float(np.nanmax(vals)) if vals else np.nan
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if metric in {"mean", "mean_loss", "loss", "loss_norm"}:
        return float(np.mean(x))
    if metric in {"shortfall", "shortfall_prob"}:
        return float(np.mean(x > 0.0))
    if metric in {"cvar95", "cvar_95"}:
        q = np.quantile(x, 0.95)
        return float(np.mean(x[x >= q]))
    if metric in {"cvar99", "cvar_99"}:
        q = np.quantile(x, 0.99)
        return float(np.mean(x[x >= q]))
    if metric in {"var95", "var_95"}:
        return float(np.quantile(x, 0.95))
    if metric in {"turnover", "mean_turnover", "tc", "mean_tc", "mean_tc_norm"}:
        return float(np.mean(x))
    raise ValueError(f"Unknown bootstrap metric={metric}")


def paired_bootstrap_hedging(
    panel: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric_col: str = "loss_norm",
    n_boot: int = 2000,
    seed: int = 42,
    metric: str = "mean_loss",
) -> dict:
    """Paired bootstrap over test episodes.

    Negative `diff` favors model_a for loss, CVaR, VaR, shortfall, and turnover
    metrics.  For CVaR/shortfall, the statistic is recomputed on each bootstrap
    resample rather than approximated by mean loss differences.
    """
    rng = np.random.default_rng(seed)
    metric_l = str(metric).lower()
    if metric_l in {"turnover", "mean_turnover"}:
        col = "turnover"
    elif metric_l in {"tc", "mean_tc", "mean_tc_norm"}:
        col = "tc_norm"
    else:
        col = metric_col
    a_cols = ["start_date", col]
    if "regime_id" in panel.columns:
        a_cols.append("regime_id")
    a = panel[panel["model"] == model_a][a_cols].rename(columns={col: "a"})
    b = panel[panel["model"] == model_b][["start_date", col]].rename(columns={col: "b"})
    m = a.merge(b, on="start_date", how="inner")
    if len(m) == 0:
        return {"model_a": model_a, "model_b": model_b, "metric": metric, "n": 0, "diff": np.nan, "p_two_sided": np.nan}
    xa = m["a"].values.astype(float)
    xb = m["b"].values.astype(float)
    regimes = m["regime_id"].values if "regime_id" in m.columns else None
    obs = _bootstrap_stat_from_losses(xa, metric, regimes=regimes) - _bootstrap_stat_from_losses(xb, metric, regimes=regimes)
    n = len(m)
    boot = np.empty(n_boot, dtype=float)
    for r in range(n_boot):
        idx = rng.integers(0, n, size=n)
        reg_idx = regimes[idx] if regimes is not None else None
        boot[r] = _bootstrap_stat_from_losses(xa[idx], metric, regimes=reg_idx) - _bootstrap_stat_from_losses(xb[idx], metric, regimes=reg_idx)
    # Null-centered paired bootstrap.  The bootstrap distribution is centered
    # at the observed statistic before testing against zero.  Centering at the
    # bootstrap mean is close asymptotically but can be noticeably off in small
    # tail-risk samples.
    centered = boot - obs if np.isfinite(obs) else boot
    pval = float(np.mean(np.abs(centered) >= abs(obs))) if np.isfinite(obs) else np.nan
    return {
        "model_a": model_a,
        "model_b": model_b,
        "metric": metric,
        "n": int(n),
        "diff": float(obs),
        "mean_diff": float(obs),  # legacy column name
        "ci95_low": float(np.nanquantile(boot, 0.025)),
        "ci95_high": float(np.nanquantile(boot, 0.975)),
        "p_two_sided": min(pval, 1.0) if np.isfinite(pval) else np.nan,
    }


def paired_bootstrap_hedging_stat(
    panel: pd.DataFrame,
    model_a: str,
    model_b: str,
    seed: int = 42,
    metric: str = "cvar_95",
    n_boot: int = 2000,
) -> dict:
    """Alias for HedgeSet paired bootstrap statistics."""
    return paired_bootstrap_hedging(panel, model_a, model_b, seed=seed, metric=metric, n_boot=n_boot)


def run_hedging_pipeline(
    families: List[str],
    state_modes: List[str],
    cost_bps_list: List[float],
    horizon: int = 5,
    stride: int = 5,
    checkpoint_preference: str = "forecast",
    rebuild_chains: bool = False,
    rebuild_states: bool = False,
    rebuild_episodes: bool = False,
    seed: int = 42,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    patience: Optional[int] = None,
    terminal_liquidation: bool = True,
) -> None:
    """End-to-end hedging experiment driver."""
    hedge_cfg = HedgingConfig()
    hedge_cfg.horizon = int(horizon)
    hedge_cfg.start_stride = int(stride)
    hedge_cfg.families = families
    hedge_cfg.state_modes = state_modes
    hedge_cfg.cost_bps_list = cost_bps_list
    hedge_cfg.terminal_liquidation = bool(terminal_liquidation)
    if epochs is not None:
        hedge_cfg.epochs = int(epochs)
    if batch_size is not None:
        hedge_cfg.batch_size = int(batch_size)
    if lr is not None:
        hedge_cfg.lr = float(lr)
    if patience is not None:
        hedge_cfg.patience = int(patience)
    hedge_cfg.results_dir = os.path.join(DATA_CFG.results_dir, "hedging")
    os.makedirs(hedge_cfg.results_dir, exist_ok=True)
    os.makedirs(hedge_cfg.cache_dir, exist_ok=True)

    cache_path = os.path.join(DATA_CFG.cache_dir, "daily_quotes.pkl")
    if not os.path.exists(cache_path):
        logger.info("Forecasting cache missing; preprocessing first.")
        preprocess_all(DATA_CFG, force_rebuild=False)

    chains_path = preprocess_hedging_chains(DATA_CFG, hedge_cfg, force_rebuild=rebuild_chains)
    state_cache_path = build_hedging_state_cache(
        cache_path=cache_path,
        data_cfg=DATA_CFG,
        model_cfg=MODEL_CFG,
        hedge_cfg=hedge_cfg,
        checkpoint_preference=checkpoint_preference,
        force_rebuild=rebuild_states,
    )

    all_metrics = []
    all_panels = []
    all_boot = []

    for family in families:
        ep_path = build_hedging_episodes(chains_path, state_cache_path, family, DATA_CFG, hedge_cfg, force_rebuild=rebuild_episodes)
        for cost_bps in cost_bps_list:
            static_dir = os.path.join(hedge_cfg.results_dir, family, f"cost_{cost_bps:g}bps", "static")
            static_metrics, static_panel = evaluate_static_hedging_baselines(ep_path, family, state_modes[0], cost_bps, hedge_cfg, static_dir)
            all_metrics.extend(static_metrics)
            all_panels.append(static_panel)

            policy_panels = []
            for mode in state_modes:
                out_dir = os.path.join(hedge_cfg.results_dir, family, f"cost_{cost_bps:g}bps", f"policy_{mode}")
                m, p = train_hedging_policy_one(ep_path, family, mode, cost_bps, hedge_cfg, out_dir, seed=seed)
                all_metrics.append(m)
                all_panels.append(p)
                policy_panels.append(p)

            # Pairwise bootstrap for the latent policy and comparator baselines.
            panel_here = pd.concat([static_panel] + policy_panels, ignore_index=True)
            for comparator in ["SmiledDelta", "Policy-summary", "Policy-grid"]:
                if comparator in set(panel_here["model"]):
                    all_boot.append(paired_bootstrap_hedging(panel_here, "Policy-latent", comparator, seed=seed))

    metrics_df = pd.DataFrame(all_metrics)
    panels_df = pd.concat(all_panels, ignore_index=True) if all_panels else pd.DataFrame()
    boot_df = pd.DataFrame(all_boot)

    metrics_path = os.path.join(hedge_cfg.results_dir, "hedging_test_metrics_all.csv")
    panel_path = os.path.join(hedge_cfg.results_dir, "hedging_test_episode_panel_all.csv")
    boot_path = os.path.join(hedge_cfg.results_dir, "hedging_pairwise_bootstrap.csv")
    metrics_df.to_csv(metrics_path, index=False)
    panels_df.to_csv(panel_path, index=False)
    boot_df.to_csv(boot_path, index=False)
    logger.info(f"Hedging metrics saved -> {metrics_path}")
    logger.info(f"Hedging episode panel saved -> {panel_path}")
    logger.info(f"Hedging bootstrap saved -> {boot_path}")

    # Compact console view.
    keep = ["model", "family", "state_mode", "cost_bps", "n", "mean_loss_norm", "cvar_95", "cvar_99", "shortfall_prob", "mean_tc_norm", "mean_turnover"]
    present = [c for c in keep if c in metrics_df.columns]
    logger.info("\n" + metrics_df[present].sort_values(["family", "cost_bps", "cvar_95"]).to_string(index=False))


# =============================================================================
# QuoteSet-to-HedgeSet Deep Hedging Extension
# =============================================================================
# HedgeSet block with a variable-size tradable hedge universe. Each episode
# observes a surface-regularized market belief state together with a candidate
# set of liquid hedge instruments selected from the option chain. The policy is
# permutation-equivariant over candidate instruments and outputs state-dependent
# no-trade bands for each instrument.
#
# Evaluation protocol:
#   * liability and option hedge-instrument marks are observed quote mids;
#   * option transaction costs use half-spread plus a proportional bps component;
#   * the same liability contract cannot be selected as its own hedge instrument;
#   * all episode paths stay within one calendar split;
#   * policies are selected by validation CVaR and evaluated once on test data.
# =============================================================================


@dataclass
class HedgeSetConfig:
    cache_dir: str = field(default_factory=lambda: os.environ.get("FNUG_HEDGE_CACHE_DIR", "cache_hedging"))
    results_dir: str = field(default_factory=lambda: os.path.join(DATA_CFG.results_dir, "hedgeset"))

    horizon: int = 5
    start_stride: int = 5
    target_tau_days: int = 30
    tau_tol_days: int = 10
    candidate_rules: List[str] = field(default_factory=lambda: [
        "underlying",
        "atm_call30", "atm_put30", "put25_30", "put10_30", "call25_30", "call10_30", "atm_straddle30",
        "put25_60", "call25_60", "atm_straddle60",
    ])
    max_instruments: int = 12
    min_instruments: int = 3
    candidate_set_name: str = "custom"
    setonly_checkpoint_path: str = ""
    require_setonly_state: bool = False

    # Static baseline and sweep controls.
    delta_no_trade_band: float = 0.10
    delta_no_trade_band_grid: List[float] = field(default_factory=lambda: [0.00, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30])
    ls_ridge: float = 1e-3
    ls_vega_scale: float = 100.0
    ls_spread_penalty: float = 0.05
    bootstrap_stats: List[str] = field(default_factory=lambda: ["mean_loss", "cvar_95", "cvar_99", "shortfall_prob", "mean_turnover", "mean_tc_norm"])
    bootstrap_reps: int = 2000
    run_name: str = "main"
    policy_variants: List[str] = field(default_factory=lambda: ["full"])
    seeds: List[int] = field(default_factory=lambda: [42])

    # Policy ablation switches.
    use_no_trade_band: bool = True
    use_sparse_gate: bool = True

    # Liability/path filters match the checked underlying-only version.
    min_liability_bid: float = 0.10
    max_liability_rel_spread: float = 0.30
    max_path_rel_spread: float = 0.50
    min_path_bid: float = 0.05
    path_k_min: float = -0.60
    path_k_max: float = 0.60
    terminal_liquidation: bool = True

    families: List[str] = field(default_factory=lambda: ["atm_call", "put25", "atm_straddle"])
    state_modes: List[str] = field(default_factory=lambda: ["latent", "summary", "grid"])
    cost_bps_list: List[float] = field(default_factory=lambda: [1.0, 2.0, 5.0])

    # Risk-sensitive policy optimization.
    cvar_alpha: float = 0.95
    lambda_mean: float = 0.15
    lambda_tc: float = 0.05
    lambda_turnover: float = 0.02
    lambda_pos: float = 0.001
    lambda_sparse: float = 0.002

    # Residual classical-hedge prior. The policy learns a bounded residual around
    # a classical anchor. Supported anchors are:
    #   none, delta, delta_nt, greedy_deltavega, ls_deltavega.
    residual_base: str = "ls_deltavega"
    # Hard residual control uses q_policy = q_base + residual.  residual_scale
    # is an absolute position cap for the learned correction, not multiplied by
    # q_max.
    residual_scale: float = 0.15
    imitation_epochs: int = 10
    imitation_lambda: float = 1.0
    finetune_imitation_lambda: float = 0.02

    # Excess-loss risk control relative to the classical residual base.
    lambda_excess_cvar: float = 0.50
    lambda_excess_relu: float = 0.25
    lambda_residual_l2: float = 0.05
    lambda_residual_trade: float = 0.02

    # Factor-neutral residual overlay.  The LS base already hedges delta/vega;
    # the neural layer is projected toward the null space of the candidate-set
    # delta/vega loading matrix.
    factor_neutral_residual: bool = True
    factor_neutral_ridge: float = 1e-4
    lambda_factor_neutral: float = 0.10

    # Global uncertainty/trust gate.  A scalar trust gate multiplies the entire
    # residual overlay. The gate is initialized below one.
    use_global_trust_gate: bool = True
    trust_gate_init_bias: float = -2.0
    trust_gate_temp: float = 1.0
    lambda_trust_l1: float = 0.01

    # Static regime-conditioned excess risk relative to the LS base.
    # Disabled in the final specification; adaptive stress-conditioned variants
    # are left as an extension rather than part of the default model.
    lambda_regime_excess_cvar: float = 0.0
    regime_excess_eta: float = 8.0

    # Auxiliary dense supervision.  The policy predicts short-horizon base-loss
    # and transition proxies from the current belief/candidate-set state.  This
    # supplies dense gradients before the CVaR tail objective dominates.
    lambda_aux_risk: float = 0.05

    # Modern baseline suite.  These baselines use the same episodes, candidate
    # instruments, cost accounting, validation selection, and paired bootstrap as
    # the FNUG model.  They are off by default and enabled with
    # --hedgeset-modern-baselines ivgrid,ntb_ls,cvar_from_scratch,qiao_residual_ls or all.
    modern_baselines: List[str] = field(default_factory=list)
    ivgrid_baseline_state_mode: str = "grid"
    cvar_from_scratch_state_mode: str = "latent"
    ntb_ls_state_mode: str = "summary"
    ntb_ls_band_max: float = 1.0
    ntb_ls_lambda_band_l1: float = 0.001
    ntb_ls_use_smooth_gate: bool = True

    # Qiao-style plain anchored residual baseline.
    # This protocol adaptation keeps the hard LSDeltaVega anchor but removes
    # FNUG's factor-neutral projection, trust gate, excess-CVaR objective, and
    # auxiliary risk losses.  It trains by local squared hedging error.
    qiao_residual_state_mode: str = "latent"
    qiao_residual_base: str = "ls_deltavega"
    qiao_residual_scale: float = 0.15
    qiao_hidden_dim: int = 96
    qiao_hidden_layers: int = 3
    qiao_lambda_local_mse: float = 1.0
    qiao_lambda_residual_l2: float = 0.02
    qiao_lambda_residual_trade: float = 0.01
    qiao_lambda_tc: float = 0.0
    qiao_lambda_turnover: float = 0.0

    # Validation selection.  CVaR on a small validation split is noisy, so the
    # best checkpoint is selected by a stable risk score rather than pure CVaR95.
    val_score_cvar95_weight: float = 1.0
    val_score_cvar99_weight: float = 0.25
    val_score_mean_weight: float = 0.10
    val_score_turnover_weight: float = 0.02
    val_score_tc_weight: float = 0.05
    val_score_worst_regime_weight: float = 0.25
    val_score_excess_cvar95_weight: float = 1.00
    val_score_excess_cvar99_weight: float = 0.25
    val_score_excess_relu_weight: float = 0.50
    val_score_worst_regime_excess_weight: float = 0.0
    val_score_factor_weight: float = 0.10
    val_score_trust_weight: float = 0.02
    val_feasible_excess_cvar95_max: float = 0.012
    val_feasible_excess_cvar95_grid: List[float] = field(default_factory=lambda: [0.007, 0.008, 0.009, 0.012])
    val_feasible_turnover_max: Optional[float] = None
    val_feasible_tc_norm_max: Optional[float] = None
    val_feasible_delta_exposure_max: float = 1e-4
    val_feasible_vega_exposure_max: float = 1e-4

    # Ambiguity-aware regime-robust extension.  This is disabled by default and
    # activated through the "ambiguous" policy variant or the CLI switch below.
    # The regime labels are assigned from train-split option-implied regime
    # scores and then carried into validation/test without refitting.
    ambiguity_aware: bool = False
    lambda_ambiguity: float = 0.25
    ambiguity_eta: float = 8.0
    ambiguity_min_count: int = 2
    regime_bins: int = 3

    # Policy architecture/training.
    embed_dim: int = 96
    num_heads: int = 4
    hidden_dim: int = 96
    hidden_layers: int = 3
    q_max: float = 3.0
    trade_max: float = 1.0
    band_max: float = 1.0
    gate_temp: float = 1.0
    smooth_temp: float = 0.02
    epochs: int = 250
    batch_size: int = 48
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 35
    min_delta: float = 1e-5


def _hs_parse_rule_tau(rule: str, default_days: int = 30) -> int:
    txt = str(rule).lower().strip()
    digits = ""
    for ch in reversed(txt):
        if ch.isdigit():
            digits = ch + digits
        elif digits:
            break
    if digits:
        try:
            return int(digits)
        except Exception:
            return int(default_days)
    return int(default_days)


def _hs_rule_core(rule: str) -> str:
    txt = str(rule).lower().strip()
    for suffix in ["_30", "_60", "_90", "30", "60", "90"]:
        if txt.endswith(suffix):
            return txt[: -len(suffix)].rstrip("_")
    return txt


def _hs_contract_key(row: pd.Series, cp_override: Optional[str] = None) -> Tuple[str, int, str]:
    cp = str(cp_override if cp_override is not None else row.get("cp_flag"))
    return (str(pd.Timestamp(row["exdate"]).date()), int(row["strike_key"]), cp)


def _hs_liability_keys(sel: Any, family: str) -> set:
    keys = set()
    if family in {"atm_call", "put25"}:
        keys.add(_hs_contract_key(sel))
    elif family == "atm_straddle":
        exd = str(pd.Timestamp(sel["exdate"]).date())
        sk = int(sel["strike_key"])
        keys.add((exd, sk, "C"))
        keys.add((exd, sk, "P"))
    return keys


def _hs_select_single(start_df: pd.DataFrame, rule: str, liability_keys: set, cfg: HedgeSetConfig) -> Optional[dict]:
    core = _hs_rule_core(rule)
    tau_days = _hs_parse_rule_tau(rule, cfg.target_tau_days)
    target_tau = tau_days / 365.0
    tau_tol = cfg.tau_tol_days / 365.0

    if core in {"atm_call", "atm_c"}:
        cp = "C"; target_delta = None; target_k = 0.0
    elif core in {"atm_put", "atm_p"}:
        cp = "P"; target_delta = None; target_k = 0.0
    elif core == "put25":
        cp = "P"; target_delta = -0.25; target_k = None
    elif core == "put10":
        cp = "P"; target_delta = -0.10; target_k = None
    elif core == "call25":
        cp = "C"; target_delta = 0.25; target_k = None
    elif core == "call10":
        cp = "C"; target_delta = 0.10; target_k = None
    else:
        return None

    df = start_df.copy()
    df = df[(df["cp_flag"] == cp) & (df["tau"].sub(target_tau).abs() <= tau_tol)]
    df = df[(df["best_bid"] >= cfg.min_liability_bid) & (df["rel_spread"] <= cfg.max_liability_rel_spread)]
    if len(df) == 0:
        return None
    df = df[[not _hs_contract_key(r) in liability_keys for _, r in df.iterrows()]].copy()
    if len(df) == 0:
        return None

    if target_delta is not None:
        df["select_score"] = (df["delta_bs"] - target_delta).abs() + 0.15 * (df["tau"] - target_tau).abs() + 0.002 * df["rel_spread"]
    else:
        df["select_score"] = df["k"].abs() + 0.20 * (df["tau"] - target_tau).abs() + 0.002 * df["rel_spread"]
    row = df.sort_values(["select_score", "rel_spread"], ascending=True).iloc[0]
    return {
        "name": str(rule),
        "kind": "option",
        "legs": [(pd.Timestamp(row["exdate"]), int(row["strike_key"]), str(row["cp_flag"]), 1.0)],
        "type_id": 1.0 if str(row["cp_flag"]) == "C" else -1.0,
        "start_liq": float(row.get("liq_score", 0.0)),
    }


def _hs_select_straddle(start_df: pd.DataFrame, rule: str, liability_keys: set, cfg: HedgeSetConfig) -> Optional[dict]:
    tau_days = _hs_parse_rule_tau(rule, cfg.target_tau_days)
    target_tau = tau_days / 365.0
    tau_tol = cfg.tau_tol_days / 365.0
    df = start_df.copy()
    df = df[(df["tau"].sub(target_tau).abs() <= tau_tol)]
    df = df[(df["best_bid"] >= cfg.min_liability_bid) & (df["rel_spread"] <= cfg.max_liability_rel_spread)]
    if len(df) == 0:
        return None
    calls = df[df["cp_flag"] == "C"].copy()
    puts = df[df["cp_flag"] == "P"].copy()
    if len(calls) == 0 or len(puts) == 0:
        return None
    m = calls.merge(puts, on=["date", "exdate", "strike_key", "K"], suffixes=("_C", "_P"), how="inner")
    if len(m) == 0:
        return None
    # Avoid buying back the exact liability straddle.
    keep = []
    for _, row in m.iterrows():
        exd = str(pd.Timestamp(row["exdate"]).date())
        sk = int(row["strike_key"])
        keep.append(((exd, sk, "C") not in liability_keys) and ((exd, sk, "P") not in liability_keys))
    m = m[np.asarray(keep, dtype=bool)].copy()
    if len(m) == 0:
        return None
    m["k_avg"] = 0.5 * (m["k_C"] + m["k_P"])
    m["tau_avg"] = 0.5 * (m["tau_C"] + m["tau_P"])
    m["rel_avg"] = 0.5 * (m["rel_spread_C"] + m["rel_spread_P"])
    m["liq_sum"] = m.get("liq_score_C", 0.0) + m.get("liq_score_P", 0.0)
    m["select_score"] = m["k_avg"].abs() + 0.20 * (m["tau_avg"] - target_tau).abs() + 0.002 * m["rel_avg"]
    row = m.sort_values(["select_score", "rel_avg"], ascending=True).iloc[0]
    return {
        "name": str(rule),
        "kind": "straddle",
        "legs": [(pd.Timestamp(row["exdate"]), int(row["strike_key"]), "C", 1.0),
                 (pd.Timestamp(row["exdate"]), int(row["strike_key"]), "P", 1.0)],
        "type_id": 2.0,
        "start_liq": float(row.get("liq_sum", 0.0)),
    }


def _hs_select_candidate_specs(start_df: pd.DataFrame, liability_keys: set, cfg: HedgeSetConfig) -> List[dict]:
    specs = [{"name": "underlying", "kind": "underlying", "legs": [], "type_id": 0.0, "start_liq": 1e12}]
    for rule in cfg.candidate_rules:
        r = str(rule).strip().lower()
        if not r or r == "underlying":
            continue
        if _hs_rule_core(r) == "atm_straddle":
            spec = _hs_select_straddle(start_df, r, liability_keys, cfg)
        else:
            spec = _hs_select_single(start_df, r, liability_keys, cfg)
        if spec is not None:
            # avoid duplicate leg sets
            key = tuple((str(pd.Timestamp(e).date()), int(k), cp, float(q)) for e, k, cp, q in spec["legs"])
            existing = {tuple((str(pd.Timestamp(e).date()), int(k), cp, float(q)) for e, k, cp, q in s.get("legs", [])) for s in specs}
            if key not in existing:
                specs.append(spec)
    # Underlying first, then highest starting liquidity.
    head = [s for s in specs if s["kind"] == "underlying"]
    rest = sorted([s for s in specs if s["kind"] != "underlying"], key=lambda x: -float(x.get("start_liq", 0.0)))
    return (head + rest)[: max(1, int(cfg.max_instruments))]


def _hs_leg_row(contract_index: pd.DataFrame, d: pd.Timestamp, exdate: pd.Timestamp, strike_key: int, cp: str) -> Optional[pd.Series]:
    return _get_contract_row(contract_index, d, exdate, int(strike_key), str(cp))


def _hs_build_instrument_path(spec: dict, path_dates: List[pd.Timestamp], contract_index: pd.DataFrame,
                              S_path: List[float], cfg: HedgeSetConfig) -> Optional[dict]:
    H = len(path_dates) - 1
    if spec["kind"] == "underlying":
        price = np.asarray(S_path, dtype=np.float32)
        half_spread = np.zeros_like(price, dtype=np.float32)
        delta = np.ones(H, dtype=np.float32)
        vega = np.zeros(H, dtype=np.float32)
        rel_spread = np.zeros(H, dtype=np.float32)
        k_arr = np.zeros(H, dtype=np.float32)
        tau_arr = np.zeros(H, dtype=np.float32)
        type_code = np.zeros(H, dtype=np.float32)
        liq_arr = np.ones(H, dtype=np.float32)
    else:
        price, half_spread = [], []
        delta, vega, rel_spread, k_arr, tau_arr, type_code, liq_arr = [], [], [], [], [], [], []
        for idx, d in enumerate(path_dates):
            rows = []
            for exdate, strike_key, cp, qty in spec["legs"]:
                r = _hs_leg_row(contract_index, d, pd.Timestamp(exdate), int(strike_key), cp)
                if r is None:
                    return None
                if _safe_float(r.get("best_bid"), 0.0) < cfg.min_path_bid:
                    return None
                if _safe_float(r.get("rel_spread"), 999.0) > cfg.max_path_rel_spread:
                    return None
                rows.append((r, float(qty)))
            p_t = float(sum(q * float(r["mid"]) for r, q in rows))
            hs_t = float(sum(abs(q) * 0.5 * (float(r["best_offer"]) - float(r["best_bid"])) for r, q in rows))
            if not np.isfinite(p_t) or p_t <= 0:
                return None
            price.append(p_t)
            half_spread.append(max(hs_t, 0.0))
            if idx < H:
                delta.append(float(sum(q * float(r["delta_bs"]) for r, q in rows)))
                vega.append(float(sum(q * float(r["vega_bs"]) for r, q in rows)))
                rel_spread.append(float(np.mean([float(r["rel_spread"]) for r, _q in rows])))
                k_arr.append(float(np.mean([float(r["k"]) for r, _q in rows])))
                tau_arr.append(float(np.mean([float(r["tau"]) for r, _q in rows])))
                type_code.append(float(spec.get("type_id", 0.0)))
                liq_arr.append(float(sum(abs(q) * float(r.get("liq_score", 0.0)) for r, q in rows)))
        price = np.asarray(price, dtype=np.float32)
        half_spread = np.asarray(half_spread, dtype=np.float32)
        delta = np.asarray(delta, dtype=np.float32)
        vega = np.asarray(vega, dtype=np.float32)
        rel_spread = np.asarray(rel_spread, dtype=np.float32)
        k_arr = np.asarray(k_arr, dtype=np.float32)
        tau_arr = np.asarray(tau_arr, dtype=np.float32)
        type_code = np.asarray(type_code, dtype=np.float32)
        liq_arr = np.asarray(liq_arr, dtype=np.float32)

    # Per-time instrument features.  These are normalized by a training normalizer.
    mid_t = np.maximum(price[:-1], 1e-6)
    feat = np.column_stack([
        (spec["kind"] == "underlying") * np.ones(H),
        (type_code == 1.0).astype(float),
        (type_code == -1.0).astype(float),
        (type_code == 2.0).astype(float),
        k_arr,
        tau_arr,
        delta,
        vega / 100.0,
        rel_spread,
        np.log1p(mid_t) / 10.0,
        np.log1p(np.maximum(liq_arr, 0.0)) / 10.0,
        np.asarray(S_path[:-1], dtype=np.float32) / max(float(S_path[0]), 1e-6) - 1.0,
    ]).astype(np.float32)
    return {
        "name": str(spec["name"]),
        "kind": str(spec["kind"]),
        "price": price.astype(np.float32),
        "half_spread": half_spread.astype(np.float32),
        "delta": delta.astype(np.float32),
        "vega": vega.astype(np.float32),
        "features": feat,
    }


def _hs_build_one_episode(
    family: str,
    path_dates: List[pd.Timestamp],
    chains_by_date: Dict[pd.Timestamp, pd.DataFrame],
    contract_index: pd.DataFrame,
    state_by_date: Dict[str, pd.Series],
    grid_map: Dict[str, np.ndarray],
    latent_cols: List[str],
    setonly_cols: Optional[List[str]],
    cfg: HedgeSetConfig,
    data_cfg: DataConfig = DATA_CFG,
) -> Optional[dict]:
    t0 = path_dates[0]
    start_df = chains_by_date.get(t0)
    if start_df is None or len(start_df) == 0:
        return None

    if family in {"atm_call", "put25"}:
        sel = _select_single_liability(start_df, family, cfg)  # cfg has compatible attrs
        if sel is None:
            return None
        exdate = pd.Timestamp(sel["exdate"])
        strike_key = int(sel["strike_key"])
        cp = str(sel["cp_flag"])
        cp_code = 1.0 if cp == "C" else -1.0
        liability_legs = [(exdate, strike_key, cp, 1.0)]
    elif family == "atm_straddle":
        sel = _select_straddle_liability(start_df, cfg)
        if sel is None:
            return None
        exdate = pd.Timestamp(sel["exdate"])
        strike_key = int(sel["strike_key"])
        cp_code = 0.0
        liability_legs = [(exdate, strike_key, "C", 1.0), (exdate, strike_key, "P", 1.0)]
    else:
        raise ValueError(f"Unsupported hedgeset liability family: {family}")

    liability_keys = _hs_liability_keys(sel, family)
    S_path, L_path = [], []
    liab_delta, liab_vega, liab_k, liab_tau = [], [], [], []
    latent_states, latent_summary_states, summary_states, grid_states, setonly_states = [], [], [], [], []
    regime_raw: Dict[str, float] = {}
    L0 = None
    for idx, d in enumerate(path_dates):
        rows = []
        for exd, sk, cp_leg, qty in liability_legs:
            r = _hs_leg_row(contract_index, d, exd, sk, cp_leg)
            if r is None:
                return None
            if _safe_float(r.get("best_bid"), 0.0) < cfg.min_path_bid:
                return None
            if _safe_float(r.get("rel_spread"), 999.0) > cfg.max_path_rel_spread:
                return None
            rows.append((r, float(qty)))
        S_t = float(rows[0][0]["S"])
        L_t = float(sum(q * float(r["mid"]) for r, q in rows))
        if not np.isfinite(L_t) or L_t <= 0:
            return None
        if L0 is None:
            L0 = L_t
        S_path.append(S_t)
        L_path.append(L_t)
        if idx < len(path_dates) - 1:
            dstr = str(pd.Timestamp(d).date())
            if dstr not in state_by_date or dstr not in grid_map:
                return None
            st = state_by_date[dstr]
            delta_t = float(sum(q * float(r["delta_bs"]) for r, q in rows))
            vega_t = float(sum(q * float(r["vega_bs"]) for r, q in rows))
            k_t = float(np.mean([float(r["k"]) for r, _q in rows]))
            tau_t = float(np.mean([float(r["tau"]) for r, _q in rows]))
            liab_delta.append(delta_t); liab_vega.append(vega_t); liab_k.append(k_t); liab_tau.append(tau_t)
            z = np.asarray([_safe_float(st.get(c), 0.0) for c in latent_cols], dtype=np.float32)
            side = _scaled_side_from_row(st)
            summary = _state_row_to_summary_vec(st)
            grid = np.asarray(grid_map[dstr], dtype=np.float32).reshape(-1)
            if idx == 0:
                # extract_latents stores the day-level quote count as N_quotes.
                # Older caches or diagnostic panels may store quote_density.
                # Use either name so the regime score's quote-coverage term is
                # actually active instead of being silently imputed to the train median.
                qden = st.get("quote_density", st.get("N_quotes", np.nan))
                regime_raw = {
                    "VIX": _safe_float(st.get("VIX"), np.nan),
                    "VXV": _safe_float(st.get("VXV"), np.nan),
                    "atm_iv": _safe_float(st.get("atm_iv"), np.nan),
                    "short_iv": _safe_float(st.get("short_iv"), np.nan),
                    "term_slope": _safe_float(st.get("term_slope"), np.nan),
                    "skew_proxy": _safe_float(st.get("skew_proxy"), np.nan),
                    "quote_density": _safe_float(qden, np.nan),
                    "rv5_back": _safe_float(st.get("rv5_back"), np.nan),
                    "rv21_back": _safe_float(st.get("rv21_back"), np.nan),
                    "dsv5_back": _safe_float(st.get("dsv5_back"), np.nan),
                }
            liab = _liability_feat(k_t, tau_t, cp_code, delta_t, vega_t, float(L0), L_t)
            latent_states.append(np.concatenate([z, side, z * side[0], z * side[1], liab], axis=0).astype(np.float32))
            latent_summary_states.append(
                np.concatenate(
                    [z, summary, side, z * side[0], z * side[1], summary * side[0], summary * side[1], liab],
                    axis=0,
                ).astype(np.float32)
            )
            summary_states.append(np.concatenate([summary, side, liab], axis=0).astype(np.float32))
            grid_states.append(np.concatenate([grid, side, liab], axis=0).astype(np.float32))
            if setonly_cols:
                z_set = np.asarray([_safe_float(st.get(c), 0.0) for c in setonly_cols], dtype=np.float32)
                # Same side and liability conditioning as the proposed latent state.
                setonly_states.append(np.concatenate([z_set, side, z_set * side[0], z_set * side[1], liab], axis=0).astype(np.float32))

    H = cfg.horizon
    if len(S_path) != H + 1 or len(L_path) != H + 1 or len(liab_delta) != H:
        return None

    specs = _hs_select_candidate_specs(start_df, liability_keys, cfg)
    inst_paths = []
    for spec in specs:
        p = _hs_build_instrument_path(spec, path_dates, contract_index, S_path, cfg)
        if p is not None:
            inst_paths.append(p)
    if len(inst_paths) < int(cfg.min_instruments):
        return None
    inst_paths = inst_paths[: int(cfg.max_instruments)]

    M = len(inst_paths)
    inst_price = np.stack([p["price"] for p in inst_paths], axis=1).astype(np.float32)          # [H+1,M]
    inst_hs = np.stack([p["half_spread"] for p in inst_paths], axis=1).astype(np.float32)       # [H+1,M]
    inst_delta = np.stack([p["delta"] for p in inst_paths], axis=1).astype(np.float32)          # [H,M]
    inst_vega = np.stack([p["vega"] for p in inst_paths], axis=1).astype(np.float32)            # [H,M]
    inst_feat = np.stack([p["features"] for p in inst_paths], axis=1).astype(np.float32)        # [H,M,D]
    inst_names = [p["name"] for p in inst_paths]
    inst_kinds = [p["kind"] for p in inst_paths]

    return {
        "family": family,
        "start_date": str(path_dates[0].date()),
        "end_date": str(path_dates[-1].date()),
        "exdate": str(pd.Timestamp(exdate).date()),
        "strike_key": int(strike_key),
        "cp_code": float(cp_code),
        "H": H,
        "M": M,
        "S": np.asarray(S_path, dtype=np.float32),
        "L": np.asarray(L_path, dtype=np.float32),
        "liab_delta": np.asarray(liab_delta, dtype=np.float32),
        "liab_vega": np.asarray(liab_vega, dtype=np.float32),
        "liab_k": np.asarray(liab_k, dtype=np.float32),
        "liab_tau": np.asarray(liab_tau, dtype=np.float32),
        "inst_price": inst_price,
        "inst_half_spread": inst_hs,
        "inst_delta": inst_delta,
        "inst_vega": inst_vega,
        "inst_features": inst_feat,
        "inst_names": inst_names,
        "inst_kinds": inst_kinds,
        "scale": float(max(float(L0), 1.0)),
        "latent_state": np.asarray(latent_states, dtype=np.float32),
        "latent_summary_state": np.asarray(latent_summary_states, dtype=np.float32),
        "summary_state": np.asarray(summary_states, dtype=np.float32),
        "grid_state": np.asarray(grid_states, dtype=np.float32),
        **({"setonly_state": np.asarray(setonly_states, dtype=np.float32),
            "nosurface_state": np.asarray(setonly_states, dtype=np.float32)} if len(setonly_states) == H else {}),
        "dates": [str(pd.Timestamp(d).date()) for d in path_dates],
        "regime_features_raw": regime_raw,
        "regime_score": np.nan,
        "regime_id": -1,
        "regime_name": "unassigned",
        "split": _date_split_from_timestamp(path_dates[0], data_cfg),
    }


def _hedgeset_regime_vector(ep: dict) -> np.ndarray:
    """Option-implied regime features used for ambiguity-aware hedging.

    The vector is built from the episode start date only and later standardized
    using train-split statistics.  It intentionally uses absolute skew and term
    slope magnitudes, because both steep negative skew and large term-structure
    dislocations can indicate stressed option-implied regimes.  Quote density is
    multiplied by -1 so sparse quote coverage contributes positively to stress.
    """
    raw = ep.get("regime_features_raw", {}) or {}
    def val(k: str, default: float = np.nan) -> float:
        return _safe_float(raw.get(k), default)
    return np.asarray([
        val("VIX"),
        val("VXV"),
        val("atm_iv"),
        val("short_iv"),
        abs(val("skew_proxy")),
        abs(val("term_slope")),
        val("rv5_back"),
        val("rv21_back"),
        val("dsv5_back"),
        -val("quote_density"),
    ], dtype=float)


def _assign_hedgeset_regimes(episodes: List[dict], cfg: HedgeSetConfig) -> Dict[str, Any]:
    """Assign train-calibrated option-implied regime labels to HedgeSet episodes.

    No test information is used.  Medians, scales, and score quantiles are fit
    only on training episodes, then applied unchanged to validation and test.
    This supports ambiguity-aware robust CVaR training/evaluation without
    leaking future regime distribution information.
    """
    if len(episodes) == 0:
        return {"regime_bins": int(getattr(cfg, "regime_bins", 3)), "regime_counts": {}}
    n_bins = max(2, int(getattr(cfg, "regime_bins", 3)))
    X = np.vstack([_hedgeset_regime_vector(ep) for ep in episodes])
    train_mask = np.asarray([ep.get("split") == "train" for ep in episodes], dtype=bool)
    X_train = X[train_mask] if train_mask.any() else X
    med = np.nanmedian(X_train, axis=0)
    scale = np.nanstd(X_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    Z = (np.where(np.isfinite(X), X, med) - med) / scale
    # Downweight noisy density and term-structure components relative to direct
    # volatility/skew/backward-risk evidence.
    weights = np.asarray([1.0, 0.8, 1.0, 0.7, 0.8, 0.5, 0.8, 0.8, 0.7, 0.35], dtype=float)
    scores = np.nanmean(Z * weights.reshape(1, -1), axis=1)
    train_scores = scores[train_mask] if train_mask.any() else scores
    train_scores = train_scores[np.isfinite(train_scores)]
    if train_scores.size < n_bins:
        cutpoints = np.linspace(np.nanmin(scores), np.nanmax(scores), n_bins + 1)[1:-1]
    else:
        probs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        cutpoints = np.nanquantile(train_scores, probs)
    cutpoints = np.asarray(cutpoints, dtype=float)
    # Ensure monotonic cutpoints even if train scores are nearly tied.
    if cutpoints.size > 1:
        for i in range(1, cutpoints.size):
            if cutpoints[i] <= cutpoints[i - 1]:
                cutpoints[i] = cutpoints[i - 1] + 1e-8
    names3 = ["calm", "middle", "stress"]
    counts: Dict[str, int] = {}
    for ep, score in zip(episodes, scores):
        rid = int(np.searchsorted(cutpoints, score, side="right"))
        rid = max(0, min(rid, n_bins - 1))
        if n_bins == 3:
            name = names3[rid]
        else:
            name = f"regime_{rid}"
        ep["regime_score"] = float(score) if np.isfinite(score) else 0.0
        ep["regime_id"] = int(rid)
        ep["regime_name"] = name
        counts[name] = counts.get(name, 0) + 1
    return {
        "regime_bins": int(n_bins),
        "regime_cutpoints": [float(x) for x in cutpoints],
        "regime_feature_median": [float(x) for x in med],
        "regime_feature_scale": [float(x) for x in scale],
        "regime_counts": counts,
        "regime_note": "Regime medians/scales/cutpoints fit on train episodes only.",
    }


def _stable_hedgeset_rules_tag(cfg: HedgeSetConfig) -> str:
    payload = {
        "candidate_rules": [str(x).lower().strip() for x in cfg.candidate_rules],
        "max_instruments": int(cfg.max_instruments),
        "min_instruments": int(cfg.min_instruments),
        "target_tau_days": int(cfg.target_tau_days),
        "tau_tol_days": int(cfg.tau_tol_days),
        "min_liability_bid": float(cfg.min_liability_bid),
        "max_liability_rel_spread": float(cfg.max_liability_rel_spread),
        "max_path_rel_spread": float(cfg.max_path_rel_spread),
        "min_path_bid": float(cfg.min_path_bid),
        "path_k_min": float(cfg.path_k_min),
        "path_k_max": float(cfg.path_k_max),
        # Include the regime assignment design in the cache key.  Earlier
        # HedgeSet caches did not contain train-calibrated regime labels; using
        # the same filename would silently reuse regime_id=-1 episodes and make
        # the ambiguity-aware objective ineffective.
        "regime_assignment_version": "train_quantile_score_v1",
        "regime_bins": int(getattr(cfg, "regime_bins", 3)),
        # Liability state now includes delta, vega and normalized liability MTM.
        # This must be part of the cache key; otherwise old episode caches with
        # the shorter liability vector would be silently reused.
        "liability_state_version": "delta_vega_lratio_v2",
    }
    txt = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(txt.encode("utf-8")).hexdigest()[:10]


def build_hedgeset_episodes(
    chains_path: str,
    state_cache_path: str,
    family: str,
    data_cfg: DataConfig = DATA_CFG,
    cfg: HedgeSetConfig = HedgeSetConfig(),
    force_rebuild: bool = False,
) -> str:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    # Use a stable cache tag. Python's built-in hash() is randomized across
    # processes, which would silently create different cache filenames for the
    # same candidate rules. Include the construction filters and state-cache
    # basename so stale episodes are not reused after changing filters or the
    # encoder checkpoint preference.
    rules_tag = _stable_hedgeset_rules_tag(cfg)
    state_tag = Path(state_cache_path).stem.replace("hedging_state_cache_", "state_")
    out_path = os.path.join(
        cfg.cache_dir,
        f"hedgeset_{family}_{state_tag}_H{cfg.horizon}_stride{cfg.start_stride}_M{cfg.max_instruments}_{rules_tag}.pkl"
    )
    if os.path.exists(out_path) and not force_rebuild:
        try:
            with open(out_path, "rb") as f:
                cached_payload = pickle.load(f)
            meta = cached_payload.get("meta", {})
            eps0 = cached_payload.get("episodes", [])
            has_regime_meta = isinstance(meta, dict) and ("regime_meta" in meta)
            has_valid_regime = bool(eps0) and all(
                ("regime_id" in ep and int(ep.get("regime_id", -1)) >= 0 and ep.get("regime_name", "unassigned") != "unassigned")
                for ep in eps0[: min(len(eps0), 20)]
            )
            if has_regime_meta and has_valid_regime:
                logger.info(f"HedgeSet episode cache found: {out_path}")
                return out_path
            logger.info("Existing HedgeSet episode cache lacks valid regime labels; rebuilding.")
        except Exception:
            logger.info("Could not inspect existing HedgeSet episode cache; rebuilding.")

    chains = pd.read_parquet(chains_path)
    chains["date"] = pd.to_datetime(chains["date"])
    chains["exdate"] = pd.to_datetime(chains["exdate"])
    chains.sort_values(["date", "exdate", "strike_key", "cp_flag"], inplace=True)
    chains_by_date = {pd.Timestamp(d): g.copy() for d, g in chains.groupby("date")}
    contract_index = chains.set_index(["date", "exdate", "strike_key", "cp_flag"], drop=False)

    with open(state_cache_path, "rb") as f:
        payload = pickle.load(f)
    state_df = payload["state_df"].copy()
    state_df["date"] = pd.to_datetime(state_df["date"])
    state_df["date_str"] = state_df["date"].dt.strftime("%Y-%m-%d")
    grid_map = payload["grid_map"]
    latent_cols = sorted([c for c in state_df.columns if c.startswith("z_")], key=lambda s: int(s.split("_")[1]))
    setonly_cols = sorted([c for c in state_df.columns if c.startswith("setonly_z_")], key=lambda s: int(s.split("_")[2]))
    if len(setonly_cols) == 0:
        setonly_cols = None
    state_by_date = {row["date_str"]: row for _, row in state_df.iterrows()}
    available_dates = sorted([pd.Timestamp(d) for d in state_df["date"].unique()])

    episodes, skipped = [], 0
    for pos in tqdm(range(0, len(available_dates) - cfg.horizon, cfg.start_stride), desc=f"hedgeset episodes {family}"):
        path_dates = available_dates[pos:pos + cfg.horizon + 1]
        if len(path_dates) != cfg.horizon + 1:
            continue
        splits = {_date_split_from_timestamp(d, data_cfg) for d in path_dates}
        if len(splits) != 1:
            skipped += 1
            continue
        ep = _hs_build_one_episode(family, path_dates, chains_by_date, contract_index, state_by_date, grid_map, latent_cols, setonly_cols, cfg, data_cfg=data_cfg)
        if ep is None:
            skipped += 1
            continue
        episodes.append(ep)

    if len(episodes) == 0:
        raise ValueError(f"No HedgeSet episodes produced for family={family}.")
    regime_meta = _assign_hedgeset_regimes(episodes, cfg)
    meta = {
        "family": family,
        "horizon": cfg.horizon,
        "start_stride": cfg.start_stride,
        "candidate_rules": cfg.candidate_rules,
        "candidate_rules_tag": rules_tag,
        "candidate_set_rebalance": "fixed_at_episode_start",
        "max_instruments": cfg.max_instruments,
        "min_instruments": cfg.min_instruments,
        "n_episodes": len(episodes),
        "n_skipped": skipped,
        "split_counts": pd.Series([ep["split"] for ep in episodes]).value_counts().to_dict(),
        "instrument_count_summary": pd.Series([ep["M"] for ep in episodes]).describe().to_dict(),
        "regime_meta": regime_meta,
        "regime_counts_by_split": {
            str(sp): pd.Series([ep["regime_name"] for ep in episodes if ep.get("split") == sp]).value_counts().to_dict()
            for sp in sorted(set(ep.get("split") for ep in episodes))
        },
    }
    with open(out_path, "wb") as f:
        pickle.dump({"episodes": episodes, "meta": meta}, f, protocol=4)
    logger.info(f"Saved HedgeSet episodes: {out_path} | {meta}")
    return out_path


class HedgeSetEpisodeDataset(Dataset):
    def __init__(self, episodes_path: str, split: str, state_mode: str):
        with open(episodes_path, "rb") as f:
            payload = pickle.load(f)
        self.meta = payload.get("meta", {})
        self.episodes = [ep for ep in payload["episodes"] if ep.get("split") == split]
        self.split = split
        self.state_mode = state_mode
        self.key = f"{state_mode}_state"
        if len(self.episodes) == 0:
            raise ValueError(f"No HedgeSet episodes for split={split}. Meta={self.meta}")
        if self.key not in self.episodes[0]:
            raise ValueError(f"State mode {state_mode} unavailable in HedgeSet episodes.")

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict:
        ep = self.episodes[idx]
        return {
            "global_state": torch.tensor(ep[self.key], dtype=torch.float32),
            "inst_features": torch.tensor(ep["inst_features"], dtype=torch.float32),
            "inst_price": torch.tensor(ep["inst_price"], dtype=torch.float32),
            "inst_half_spread": torch.tensor(ep["inst_half_spread"], dtype=torch.float32),
            "inst_delta": torch.tensor(ep["inst_delta"], dtype=torch.float32),
            "inst_vega": torch.tensor(ep["inst_vega"], dtype=torch.float32),
            "L": torch.tensor(ep["L"], dtype=torch.float32),
            "liab_delta": torch.tensor(ep["liab_delta"], dtype=torch.float32),
            "liab_vega": torch.tensor(ep["liab_vega"], dtype=torch.float32),
            "scale": torch.tensor(float(ep["scale"]), dtype=torch.float32),
            "start_date": ep["start_date"],
            "family": ep["family"],
            "inst_names": ep["inst_names"],
            "regime_id": torch.tensor(int(ep.get("regime_id", -1)), dtype=torch.long),
            "regime_score": torch.tensor(float(ep.get("regime_score", 0.0)), dtype=torch.float32),
            "regime_name": str(ep.get("regime_name", "unassigned")),
            "M": int(ep["M"]),
        }


def hedgeset_collate_fn(batch: List[dict]) -> dict:
    B = len(batch)
    H = batch[0]["global_state"].shape[0]
    gs_dim = batch[0]["global_state"].shape[-1]
    inst_dim = batch[0]["inst_features"].shape[-1]
    max_M = max(int(b["M"]) for b in batch)
    global_state = torch.zeros(B, H, gs_dim)
    inst_features = torch.zeros(B, H, max_M, inst_dim)
    inst_price = torch.zeros(B, H + 1, max_M)
    inst_half_spread = torch.zeros(B, H + 1, max_M)
    inst_delta = torch.zeros(B, H, max_M)
    inst_vega = torch.zeros(B, H, max_M)
    inst_mask = torch.zeros(B, H, max_M, dtype=torch.bool)
    L = torch.zeros(B, H + 1)
    liab_delta = torch.zeros(B, H)
    liab_vega = torch.zeros(B, H)
    scale = torch.zeros(B)
    regime_id = torch.full((B,), -1, dtype=torch.long)
    regime_score = torch.zeros(B)
    start_dates, families, inst_names, regime_names = [], [], [], []
    for i, b in enumerate(batch):
        M = int(b["M"])
        global_state[i] = b["global_state"]
        inst_features[i, :, :M, :] = b["inst_features"]
        inst_price[i, :, :M] = b["inst_price"]
        inst_half_spread[i, :, :M] = b["inst_half_spread"]
        inst_delta[i, :, :M] = b["inst_delta"]
        inst_vega[i, :, :M] = b["inst_vega"]
        inst_mask[i, :, :M] = True
        L[i] = b["L"]
        liab_delta[i] = b["liab_delta"]
        liab_vega[i] = b["liab_vega"]
        scale[i] = b["scale"]
        regime_id[i] = b.get("regime_id", torch.tensor(-1, dtype=torch.long))
        regime_score[i] = b.get("regime_score", torch.tensor(0.0, dtype=torch.float32))
        start_dates.append(b["start_date"])
        families.append(b["family"])
        inst_names.append(b["inst_names"])
        regime_names.append(b.get("regime_name", "unassigned"))
    return {
        "global_state": global_state,
        "inst_features": inst_features,
        "inst_price": inst_price,
        "inst_half_spread": inst_half_spread,
        "inst_delta": inst_delta,
        "inst_vega": inst_vega,
        "inst_mask": inst_mask,
        "L": L,
        "liab_delta": liab_delta,
        "liab_vega": liab_vega,
        "scale": scale,
        "regime_id": regime_id,
        "regime_score": regime_score,
        "start_date": start_dates,
        "family": families,
        "inst_names": inst_names,
        "regime_name": regime_names,
    }


class DualFeatureNormalizer(nn.Module):
    def __init__(self, g_mean: torch.Tensor, g_std: torch.Tensor, i_mean: torch.Tensor, i_std: torch.Tensor):
        super().__init__()
        self.register_buffer("g_mean", g_mean.float())
        self.register_buffer("g_std", g_std.float().clamp(min=1e-6))
        self.register_buffer("i_mean", i_mean.float())
        self.register_buffer("i_std", i_std.float().clamp(min=1e-6))

    def norm_global(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.g_mean) / self.g_std

    def norm_inst(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.i_mean) / self.i_std


def fit_hedgeset_normalizer(ds: HedgeSetEpisodeDataset, device: torch.device) -> DualFeatureNormalizer:
    gs, ins = [], []
    for ep in ds.episodes:
        gs.append(np.asarray(ep[f"{ds.state_mode}_state"], dtype=np.float32))
        ins.append(np.asarray(ep["inst_features"], dtype=np.float32).reshape(-1, np.asarray(ep["inst_features"]).shape[-1]))
    G = np.concatenate(gs, axis=0)
    I = np.concatenate(ins, axis=0)
    gm = np.nanmean(G, axis=0); gs_ = np.nanstd(G, axis=0)
    im = np.nanmean(I, axis=0); is_ = np.nanstd(I, axis=0)
    gm = np.where(np.isfinite(gm), gm, 0.0).astype(np.float32)
    gs_ = np.where(np.isfinite(gs_) & (gs_ > 1e-6), gs_, 1.0).astype(np.float32)
    im = np.where(np.isfinite(im), im, 0.0).astype(np.float32)
    is_ = np.where(np.isfinite(is_) & (is_ > 1e-6), is_, 1.0).astype(np.float32)
    return DualFeatureNormalizer(torch.tensor(gm), torch.tensor(gs_), torch.tensor(im), torch.tensor(is_)).to(device)


class HedgeSetCrossAttentionPolicy(nn.Module):
    """Cross-attention policy over a variable-size hedge set.

    Default mode is hard residual control around a classical hedge anchor:
        q_t = q_base_t + r_t.

    The no-trade band and sparse gate act only on the residual state r_t, never
    on q_base_t.  Therefore r_t == 0 gives exactly the classical hedge path,
    including the same transaction-cost accounting in the rollout.

    Absolute target mode is retained for ablations by setting residual_base to
    none/absolute/raw.
    """
    def __init__(self, global_dim: int, inst_dim: int, cfg: HedgeSetConfig):
        super().__init__()
        d = int(cfg.embed_dim)
        self.global_proj = nn.Sequential(nn.Linear(global_dim, d), nn.GELU(), nn.LayerNorm(d))
        # Instrument token includes current residual or total position plus the
        # classical base position.  In residual mode the first inventory channel
        # is r_prev / residual_scale; in absolute mode it is q_prev / q_max.
        self.inst_proj = nn.Sequential(nn.Linear(inst_dim + 2, d), nn.GELU(), nn.LayerNorm(d))
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.norm1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.norm2 = nn.LayerNorm(d)
        layers = []
        cur = d
        for _ in range(max(1, cfg.hidden_layers - 1)):
            layers += [nn.Linear(cur, cfg.hidden_dim), nn.GELU(), nn.LayerNorm(cfg.hidden_dim)]
            cur = cfg.hidden_dim
        layers.append(nn.Linear(cur, 3))
        self.out = nn.Sequential(*layers)

        # Global residual trust and auxiliary transition/risk prediction heads.
        # The trust gate is intentionally global, not per-instrument: it decides
        # whether the neural overlay should be trusted at all.  If trust -> 0,
        # q_t reverts to the hard classical base exactly.
        self.global_context_norm = nn.LayerNorm(2 * d)
        self.global_trust_gate = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1)
        )
        self.aux_head = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 4)
        )
        try:
            nn.init.constant_(self.global_trust_gate[-1].bias, float(getattr(cfg, "trust_gate_init_bias", -2.0)))
            nn.init.zeros_(self.global_trust_gate[-1].weight)
        except Exception:
            pass

        self.q_max = float(cfg.q_max)
        self.trade_max = float(cfg.trade_max)
        self.band_max = float(cfg.band_max)
        self.smooth_temp = float(cfg.smooth_temp)
        self.gate_temp = float(cfg.gate_temp)
        self.trust_gate_temp = float(getattr(cfg, "trust_gate_temp", 1.0))
        self.residual_scale = float(getattr(cfg, "residual_scale", 0.15))
        self.use_no_trade_band = bool(getattr(cfg, "use_no_trade_band", True))
        self.use_sparse_gate = bool(getattr(cfg, "use_sparse_gate", True))
        self.use_global_trust_gate = bool(getattr(cfg, "use_global_trust_gate", True))

    def forward(
        self,
        g_norm: torch.Tensor,
        inst_norm: torch.Tensor,
        q_prev: torch.Tensor,
        mask: torch.Tensor,
        q_anchor: Optional[torch.Tensor] = None,
        residual_enabled: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, M, _ = inst_norm.shape
        if q_anchor is None:
            q_anchor = torch.zeros_like(q_prev)
        q_anchor = torch.where(mask, q_anchor, torch.zeros_like(q_anchor))

        g_tok = self.global_proj(g_norm).unsqueeze(1)
        if residual_enabled:
            inv_scale = max(self.residual_scale, 1e-6)
        else:
            inv_scale = max(self.q_max, 1e-6)
        q_prev_scaled = (q_prev / inv_scale).unsqueeze(-1)
        q_anchor_scaled = (q_anchor / max(self.q_max, 1e-6)).unsqueeze(-1)
        inst_tok = self.inst_proj(torch.cat([inst_norm, q_prev_scaled, q_anchor_scaled], dim=-1))
        tokens = torch.cat([g_tok, inst_tok], dim=1)
        valid_tokens = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1)

        Q = self.q_proj(inst_tok)
        K = self.k_proj(tokens)
        V = self.v_proj(tokens)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(max(Q.shape[-1], 1))
        scores = scores.masked_fill(~valid_tokens.unsqueeze(1), -1e9)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.bmm(attn, V)
        h = self.norm1(inst_tok + ctx)
        h = self.norm2(h + self.ff(h))
        raw = self.out(h)

        mask_f = mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        pooled_inst = (h * mask_f).sum(dim=1) / denom
        global_ctx = self.global_context_norm(torch.cat([g_tok.squeeze(1), pooled_inst], dim=-1))
        aux_pred = self.aux_head(global_ctx)
        global_trust = torch.sigmoid(self.global_trust_gate(global_ctx).squeeze(-1) / max(self.trust_gate_temp, 1e-6))
        if not self.use_global_trust_gate:
            global_trust = torch.ones_like(global_trust)

        if residual_enabled:
            r_prev = torch.where(mask, q_prev, torch.zeros_like(q_prev))
            # Hard residual target on the configured residual scale.
            r_target = self.residual_scale * torch.tanh(raw[..., 0])
            r_target = torch.where(mask, r_target, torch.zeros_like(r_target))

            if self.use_no_trade_band:
                # Residual band is on the same absolute scale as residual_scale.
                band = self.residual_scale * torch.sigmoid(raw[..., 1])
                gap = r_target - r_prev
                abs_gap = torch.abs(gap)
                active_mag = self.smooth_temp * F.softplus((abs_gap - band) / max(self.smooth_temp, 1e-6))
                residual_trade = torch.sign(gap) * active_mag
                residual_trade = torch.clamp(residual_trade, -self.trade_max, self.trade_max)
                r_proposed = torch.clamp(r_prev + residual_trade, -self.residual_scale, self.residual_scale)
            else:
                band = torch.zeros_like(r_target)
                gap = r_target - r_prev
                residual_trade = torch.clamp(gap, -self.trade_max, self.trade_max)
                r_proposed = torch.clamp(r_prev + residual_trade, -self.residual_scale, self.residual_scale)

            gate = torch.sigmoid(raw[..., 2] / max(self.gate_temp, 1e-6))
            if not self.use_sparse_gate:
                gate = torch.ones_like(gate)
            r_new = torch.clamp(r_prev + gate * (r_proposed - r_prev), -self.residual_scale, self.residual_scale)
            r_new = torch.where(mask, r_new, torch.zeros_like(r_new))
            q_new = torch.clamp(q_anchor + r_new, -self.q_max, self.q_max)
            q_new = torch.where(mask, q_new, torch.zeros_like(q_new))
            return q_new, {
                "mu": q_anchor + r_target,
                "band": band,
                "gate": gate,
                "attn": attn,
                "q_anchor": q_anchor,
                "residual_prev": r_prev,
                "residual_target": r_target,
                "residual_new": r_new,
                "residual_trade": r_new - r_prev,
                "global_trust": global_trust,
                "aux_pred": aux_pred,
            }

        # Absolute target ablation.  This preserves the previous from-scratch
        # behavior and is intentionally not protected by the LS fallback.
        raw_target = self.q_max * torch.tanh(raw[..., 0])
        mu = raw_target
        band = self.band_max * torch.sigmoid(raw[..., 1])
        gate = torch.sigmoid(raw[..., 2] / max(self.gate_temp, 1e-6))
        gap = mu - q_prev
        if self.use_no_trade_band:
            abs_gap = torch.abs(gap)
            active_mag = self.smooth_temp * F.softplus((abs_gap - band) / max(self.smooth_temp, 1e-6))
            trade = torch.sign(gap) * active_mag
            trade = torch.clamp(trade, -self.trade_max, self.trade_max)
            q_proposed = torch.clamp(q_prev + trade, -self.q_max, self.q_max)
        else:
            trade = torch.clamp(gap, -self.trade_max, self.trade_max)
            q_proposed = torch.clamp(q_prev + trade, -self.q_max, self.q_max)
            band = torch.zeros_like(band)
        if not self.use_sparse_gate:
            gate = torch.ones_like(gate)
        q_new = torch.clamp(q_prev + gate * (q_proposed - q_prev), -self.q_max, self.q_max)
        q_new = torch.where(mask, q_new, torch.zeros_like(q_new))
        return q_new, {
            "mu": mu,
            "band": band,
            "gate": gate,
            "attn": attn,
            "q_anchor": q_anchor,
            "residual_prev": torch.zeros_like(q_new),
            "residual_target": torch.zeros_like(q_new),
            "residual_new": torch.zeros_like(q_new),
            "residual_trade": torch.zeros_like(q_new),
            "global_trust": torch.ones(q_new.shape[0], device=q_new.device, dtype=q_new.dtype),
            "aux_pred": aux_pred,
        }


class HedgeSetAgent(nn.Module):
    def __init__(self, global_dim: int, inst_dim: int, normalizer: DualFeatureNormalizer, cfg: HedgeSetConfig):
        super().__init__()
        self.normalizer = normalizer
        self.policy = HedgeSetCrossAttentionPolicy(global_dim, inst_dim, cfg)
        self.cvar_eta = nn.Parameter(torch.tensor(0.0))
        self.cfg = cfg

    def act(
        self,
        global_state_t: torch.Tensor,
        inst_features_t: torch.Tensor,
        q_prev: torch.Tensor,
        mask_t: torch.Tensor,
        q_anchor_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        g = self.normalizer.norm_global(global_state_t)
        i = self.normalizer.norm_inst(inst_features_t)
        residual_base = str(getattr(self.cfg, "residual_base", "ls_deltavega")).lower().strip()
        residual_enabled = residual_base not in {"", "none", "absolute", "raw"}
        return self.policy(g, i, q_prev, mask_t, q_anchor=q_anchor_t, residual_enabled=residual_enabled)



def _hs_ls_delta_vega_target(
    batch: dict,
    t: int,
    q_max: float,
    ridge: float = 1e-3,
    vega_scale: float = 100.0,
    spread_penalty: float = 0.05,
) -> torch.Tensor:
    """Cost-aware least-squares delta-vega hedge over the candidate set.

    For each episode, solve a small ridge-regularized problem
        min_q ||A q - y||_2^2 + q' R q,
    where A contains instrument delta and scaled vega.  The diagonal penalty R
    includes both a numerical ridge and a half-spread penalty, so illiquid option
    hedges are not selected too aggressively.  This is a substantially stronger
    classical baseline than the greedy single-option vega hedge.
    """
    mask_t = batch["inst_mask"][:, t, :]
    delta_inst = batch["inst_delta"][:, t, :]
    vega_inst = batch["inst_vega"][:, t, :] / max(float(vega_scale), 1e-6)
    y_delta = batch["liab_delta"][:, t]
    y_vega = batch["liab_vega"][:, t] / max(float(vega_scale), 1e-6)
    half_spread = batch.get("inst_half_spread", torch.zeros_like(delta_inst))[:, t, :]
    scale = batch.get("scale", torch.ones(delta_inst.shape[0], device=delta_inst.device, dtype=delta_inst.dtype)).to(delta_inst.device)
    B, M = delta_inst.shape
    q_out = torch.zeros(B, M, device=delta_inst.device, dtype=delta_inst.dtype)
    for b in range(B):
        valid = mask_t[b]
        if not bool(valid.any()):
            continue
        A = torch.stack([delta_inst[b, valid], vega_inst[b, valid]], dim=0)  # [2,m]
        y = torch.stack([y_delta[b], y_vega[b]], dim=0)                    # [2]
        m = A.shape[1]
        hs_norm = half_spread[b, valid] / torch.clamp(scale[b], min=1.0)
        diag_pen = float(ridge) + float(spread_penalty) * torch.clamp(hs_norm, min=0.0)
        lhs = A.transpose(0, 1) @ A + torch.diag(diag_pen + 1e-8)
        rhs = A.transpose(0, 1) @ y
        try:
            q_valid = torch.linalg.solve(lhs, rhs)
        except RuntimeError:
            q_valid = torch.linalg.pinv(lhs) @ rhs
        q_valid = torch.clamp(q_valid, -float(q_max), float(q_max))
        q_out[b, valid] = q_valid
    return q_out


def _hs_no_trade_delta_target(batch: dict, t: int, q_prev: torch.Tensor, cfg: HedgeSetConfig) -> torch.Tensor:
    q_target = torch.zeros_like(q_prev)
    q_target[:, 0] = torch.clamp(batch["liab_delta"][:, t].to(q_prev.device), -cfg.q_max, cfg.q_max)
    gap0 = torch.abs(q_target[:, 0] - q_prev[:, 0])
    trade_mask = gap0 > float(getattr(cfg, "delta_no_trade_band", 0.10))
    q_new = q_prev.clone()
    q_new[:, 0] = torch.where(trade_mask, q_target[:, 0], q_prev[:, 0])
    return q_new


def _hs_base_target(batch: dict, t: int, q_prev: torch.Tensor, cfg: HedgeSetConfig, base_name: Optional[str] = None) -> torch.Tensor:
    """Classical anchor used by residual HedgeSet policies."""
    base = str(base_name if base_name is not None else getattr(cfg, "residual_base", "ls_deltavega")).lower().strip()
    if base in {"", "none", "absolute", "raw"}:
        return torch.zeros_like(q_prev)
    if base in {"delta", "smiled_delta"}:
        q = torch.zeros_like(q_prev)
        q[:, 0] = torch.clamp(batch["liab_delta"][:, t].to(q_prev.device), -cfg.q_max, cfg.q_max)
        return q
    if base in {"delta_nt", "notrade_delta", "no_trade_delta"}:
        return _hs_no_trade_delta_target(batch, t, q_prev, cfg)
    if base in {"greedy", "greedy_deltavega", "deltavega"}:
        tmp_cfg = copy.copy(cfg)
        tmp_cfg.current_action_scope = getattr(cfg, "current_action_scope", "full")
        # reuse the same greedy logic by constructing one-step target here
        mask_t = batch["inst_mask"][:, t, :].to(q_prev.device)
        q_new = torch.zeros_like(q_prev)
        vega_inst = batch["inst_vega"][:, t, :].to(q_prev.device)
        delta_inst = batch["inst_delta"][:, t, :].to(q_prev.device)
        valid_opt = mask_t.clone()
        if valid_opt.shape[1] > 0:
            valid_opt[:, 0] = False
        score = torch.where(valid_opt, torch.abs(vega_inst), torch.zeros_like(vega_inst))
        j = torch.argmax(score, dim=1)
        liab_v = batch["liab_vega"][:, t].to(q_prev.device)
        liab_d = batch["liab_delta"][:, t].to(q_prev.device)
        for b in range(q_prev.shape[0]):
            jb = int(j[b].item())
            if bool(valid_opt[b, jb]) and abs(float(vega_inst[b, jb])) > 1e-6:
                qopt = torch.clamp(liab_v[b] / vega_inst[b, jb], -cfg.q_max, cfg.q_max)
                q_new[b, jb] = qopt
                q_new[b, 0] = torch.clamp(liab_d[b] - qopt * delta_inst[b, jb], -cfg.q_max, cfg.q_max)
            else:
                q_new[b, 0] = torch.clamp(liab_d[b], -cfg.q_max, cfg.q_max)
        return q_new
    if base in {"ls", "lsdv", "ls_deltavega", "lsdelta_vega", "lsdeltavega"}:
        return _hs_ls_delta_vega_target(
            batch, t, q_max=cfg.q_max, ridge=getattr(cfg, "ls_ridge", 1e-3),
            vega_scale=getattr(cfg, "ls_vega_scale", 100.0),
            spread_penalty=getattr(cfg, "ls_spread_penalty", 0.05),
        ).to(q_prev.device)
    raise ValueError(f"Unknown residual_base={base}")


def _hs_project_factor_neutral_residual(
    r_raw: torch.Tensor,
    batch: dict,
    t: int,
    mask_t: torch.Tensor,
    cfg: HedgeSetConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Project residual positions toward the delta/vega null space.

    The classical LSDeltaVega base already spans liability delta and vega.  The
    neural overlay should therefore express liquidity, skew, term-structure, or
    regime views without rebuilding first-order exposures.  We use the batched
    projection
        r_perp = (I - A'(A A' + rho I)^{-1} A) r,
    with A=[delta; vega/vega_scale].  The operation is differentiable and
    preserves the exact LS fallback because r_raw==0 implies r_perp==0.
    """
    device = r_raw.device
    mask_f = mask_t.to(device=device, dtype=r_raw.dtype)
    r0 = torch.where(mask_t.to(device), r_raw, torch.zeros_like(r_raw))
    delta = batch["inst_delta"][:, t, :].to(device=device, dtype=r_raw.dtype) * mask_f
    vega = batch["inst_vega"][:, t, :].to(device=device, dtype=r_raw.dtype) / max(float(getattr(cfg, "ls_vega_scale", 100.0)), 1e-6)
    vega = vega * mask_f
    A = torch.stack([delta, vega], dim=1)  # [B,2,M]

    if not bool(getattr(cfg, "factor_neutral_residual", True)):
        r = r0
    else:
        rho = max(float(getattr(cfg, "factor_neutral_ridge", 1e-4)), 1e-8)
        eye = torch.eye(2, device=device, dtype=r_raw.dtype).unsqueeze(0).expand(r_raw.shape[0], -1, -1)
        gram = torch.bmm(A, A.transpose(1, 2)) + rho * eye
        rhs = torch.bmm(A, r0.unsqueeze(-1))
        lam = torch.linalg.solve(gram, rhs)
        correction = torch.bmm(A.transpose(1, 2), lam).squeeze(-1)
        r = torch.where(mask_t.to(device), r0 - correction, torch.zeros_like(r0))
        # Preserve factor neutrality while enforcing the conservative residual
        # scale at the portfolio level.  Scaling the whole residual vector keeps
        # A r approximately zero, unlike per-coordinate clipping.
        cap = max(float(getattr(cfg, "residual_scale", 0.15)), 1e-8)
        max_abs = torch.amax(torch.abs(r), dim=1, keepdim=True).clamp(min=cap)
        shrink = torch.clamp(cap / max_abs, max=1.0)
        r = r * shrink

    delta_expo = torch.abs((r * delta).sum(dim=1))
    vega_expo = torch.abs((r * vega).sum(dim=1))
    factor_l2 = delta_expo.pow(2) + vega_expo.pow(2)
    return r, {
        "residual_delta_exposure": delta_expo,
        "residual_vega_exposure": vega_expo,
        "residual_factor_l2": factor_l2,
    }


def _rollout_hedgeset(agent: Optional[HedgeSetAgent], batch: dict, cfg: HedgeSetConfig, cost_bps: float, mode: str = "policy") -> Dict[str, torch.Tensor]:
    G = batch["global_state"]
    IF = batch["inst_features"]
    P = batch["inst_price"]
    HS = batch["inst_half_spread"]
    mask = batch["inst_mask"]
    action_scope = str(getattr(cfg, "current_action_scope", "full")).lower().strip()
    if action_scope in {"underlying", "uonly", "underlying_only", "u_only"}:
        mask = mask.clone()
        if mask.shape[-1] > 1:
            mask[:, :, 1:] = False
    L = batch["L"]
    scale = batch["scale"].clamp(min=1.0)
    B, H, M = mask.shape
    device = G.device

    # q_prev is the actual total hedge inventory.  r_prev is the neural residual
    # inventory.  q_base_prev is the standalone classical base inventory used to
    # compute base PnL and no-trade base rules without contamination from the
    # neural residual.  This is what enforces r == 0 => exact base path.
    q_prev = torch.zeros(B, M, device=device)
    r_prev = torch.zeros(B, M, device=device)
    q_base_prev = torch.zeros(B, M, device=device)

    pnl = torch.zeros(B, device=device)
    tc_total = torch.zeros(B, device=device)
    turnover = torch.zeros(B, device=device)
    pos_l2 = torch.zeros(B, device=device)
    sparse = torch.zeros(B, device=device)

    base_pnl = torch.zeros(B, device=device)
    base_tc_total = torch.zeros(B, device=device)
    base_turnover = torch.zeros(B, device=device)

    residual_l2 = torch.zeros(B, device=device)
    residual_turnover = torch.zeros(B, device=device)
    residual_delta_exposure = torch.zeros(B, device=device)
    residual_vega_exposure = torch.zeros(B, device=device)
    residual_factor_l2 = torch.zeros(B, device=device)
    global_trust_sum = torch.zeros(B, device=device)

    q_hist, q_base_hist, r_hist = [], [], []
    aux_preds, aux_targets = [], []

    residual_base = str(getattr(cfg, "residual_base", "ls_deltavega")).lower().strip()
    residual_enabled = residual_base not in {"", "none", "absolute", "raw"}

    for t in range(H):
        mask_t = mask[:, t, :]
        if mode == "no_hedge":
            q_new = torch.zeros_like(q_prev)
            q_base = q_new
            r_new = torch.zeros_like(q_prev)
        elif mode == "delta":
            q_new = _hs_base_target(batch, t, q_prev, cfg, base_name="delta")
            q_base = q_new
            r_new = torch.zeros_like(q_prev)
        elif mode == "delta_nt":
            q_new = _hs_base_target(batch, t, q_prev, cfg, base_name="delta_nt")
            q_base = q_new
            r_new = torch.zeros_like(q_prev)
        elif mode == "deltavega":
            q_new = _hs_base_target(batch, t, q_prev, cfg, base_name="greedy_deltavega")
            q_base = q_new
            r_new = torch.zeros_like(q_prev)
        elif mode == "ls_deltavega":
            q_new = _hs_base_target(batch, t, q_prev, cfg, base_name="ls_deltavega")
            q_base = q_new
            r_new = torch.zeros_like(q_prev)
        else:
            if agent is None:
                raise ValueError("agent is required for policy mode")
            if residual_enabled:
                # Classical base path is computed from q_base_prev, not q_prev.
                # For LSDeltaVega this is path-independent; for delta_nt it is
                # crucial because no-trade decisions must be those of the base.
                q_base = _hs_base_target(batch, t, q_base_prev, cfg)
                q_new_raw, aux = agent.act(G[:, t, :], IF[:, t, :, :], r_prev, mask_t, q_anchor_t=q_base)
                r_raw = aux.get("residual_new", q_new_raw - q_base)
                r_projected, factor_diag = _hs_project_factor_neutral_residual(r_raw, batch, t, mask_t, cfg)
                trust = aux.get("global_trust", torch.ones(B, device=device, dtype=r_projected.dtype))
                trust = trust.to(device=device, dtype=r_projected.dtype).clamp(0.0, 1.0)
                r_new = trust.unsqueeze(1) * r_projected
                r_new = torch.where(mask_t, r_new, torch.zeros_like(r_new))
                q_new = torch.clamp(q_base + r_new, -cfg.q_max, cfg.q_max)
                q_new = torch.where(mask_t, q_new, torch.zeros_like(q_new))
                residual_delta_exposure = residual_delta_exposure + factor_diag["residual_delta_exposure"]
                residual_vega_exposure = residual_vega_exposure + factor_diag["residual_vega_exposure"]
                residual_factor_l2 = residual_factor_l2 + factor_diag["residual_factor_l2"]
                global_trust_sum = global_trust_sum + trust
                if "aux_pred" in aux:
                    aux_preds.append(aux["aux_pred"])
            else:
                q_base = torch.zeros_like(q_prev)
                q_new, aux = agent.act(G[:, t, :], IF[:, t, :, :], q_prev, mask_t, q_anchor_t=q_base)
                r_new = torch.zeros_like(q_prev)

        q_new = torch.where(mask_t, q_new, torch.zeros_like(q_new))
        q_base = torch.where(mask_t, q_base, torch.zeros_like(q_base))
        r_new = torch.where(mask_t, r_new, torch.zeros_like(r_new))

        dP = P[:, t + 1, :] - P[:, t, :]
        dL = L[:, t + 1] - L[:, t]

        # Actual policy PnL/cost.
        dq = q_new - q_prev
        prop_cost = (float(cost_bps) / 10000.0) * P[:, t, :] * torch.abs(dq)
        spread_cost = HS[:, t, :] * torch.abs(dq)
        tc = torch.where(mask_t, prop_cost + spread_cost, torch.zeros_like(prop_cost)).sum(dim=1)
        pnl = pnl + (q_new * dP).sum(dim=1) - dL - tc
        tc_total = tc_total + tc
        turnover = turnover + torch.abs(dq).sum(dim=1)
        pos_l2 = pos_l2 + (q_new ** 2).sum(dim=1)
        sparse = sparse + torch.log1p(torch.abs(dq) / 1e-3).sum(dim=1)

        # Standalone base PnL/cost.  If residual is zero, q_new == q_base and
        # these quantities exactly match the policy quantities.
        dq_base = q_base - q_base_prev
        prop_cost_b = (float(cost_bps) / 10000.0) * P[:, t, :] * torch.abs(dq_base)
        spread_cost_b = HS[:, t, :] * torch.abs(dq_base)
        tc_b = torch.where(mask_t, prop_cost_b + spread_cost_b, torch.zeros_like(prop_cost_b)).sum(dim=1)
        base_step_pnl = (q_base * dP).sum(dim=1) - dL - tc_b
        base_pnl = base_pnl + base_step_pnl
        base_tc_total = base_tc_total + tc_b
        base_turnover = base_turnover + torch.abs(dq_base).sum(dim=1)

        if residual_enabled and len(aux_preds) > len(aux_targets):
            # Dense auxiliary targets use only realized one-step quantities and
            # the standalone base hedge.  They are scaled to be O(1) and do not
            # leak future information into evaluation because they are training
            # losses computed inside historical episodes.
            base_step_loss_norm = (-base_step_pnl) / scale
            liab_step_loss_norm = dL / scale
            underlier_ret_abs = torch.abs(dP[:, 0] / P[:, t, 0].clamp(min=1e-6)) if P.shape[-1] > 0 else torch.zeros_like(base_step_loss_norm)
            base_tc_step_norm = tc_b / scale
            aux_targets.append(torch.stack([base_step_loss_norm, liab_step_loss_norm, underlier_ret_abs, base_tc_step_norm], dim=1).detach())

        residual_l2 = residual_l2 + (r_new ** 2).sum(dim=1)
        residual_turnover = residual_turnover + torch.abs(r_new - r_prev).sum(dim=1)

        q_prev = q_new
        q_base_prev = q_base
        r_prev = r_new if residual_enabled else torch.zeros_like(r_prev)
        q_hist.append(q_new)
        q_base_hist.append(q_base)
        r_hist.append(r_new)

    if cfg.terminal_liquidation:
        mask_last = mask[:, -1, :]
        prop_close = (float(cost_bps) / 10000.0) * P[:, -1, :] * torch.abs(q_prev)
        spread_close = HS[:, -1, :] * torch.abs(q_prev)
        tc_close = torch.where(mask_last, prop_close + spread_close, torch.zeros_like(prop_close)).sum(dim=1)
        pnl = pnl - tc_close
        tc_total = tc_total + tc_close
        turnover = turnover + torch.abs(q_prev).sum(dim=1)

        prop_close_b = (float(cost_bps) / 10000.0) * P[:, -1, :] * torch.abs(q_base_prev)
        spread_close_b = HS[:, -1, :] * torch.abs(q_base_prev)
        tc_close_b = torch.where(mask_last, prop_close_b + spread_close_b, torch.zeros_like(prop_close_b)).sum(dim=1)
        base_pnl = base_pnl - tc_close_b
        base_tc_total = base_tc_total + tc_close_b
        base_turnover = base_turnover + torch.abs(q_base_prev).sum(dim=1)

    loss_raw = -pnl
    loss_norm = loss_raw / scale
    base_loss_raw = -base_pnl
    base_loss_norm = base_loss_raw / scale
    excess_loss_norm = loss_norm - base_loss_norm.detach()

    q_stack = torch.stack(q_hist, dim=1) if q_hist else torch.zeros(B, H, M, device=device)
    qb_stack = torch.stack(q_base_hist, dim=1) if q_base_hist else torch.zeros(B, H, M, device=device)
    r_stack = torch.stack(r_hist, dim=1) if r_hist else torch.zeros(B, H, M, device=device)
    if len(aux_preds) > 0 and len(aux_targets) == len(aux_preds):
        aux_pred_stack = torch.stack(aux_preds, dim=1)
        aux_target_stack = torch.stack(aux_targets, dim=1)
        aux_risk_loss = F.huber_loss(aux_pred_stack, aux_target_stack, delta=0.25, reduction="mean")
    else:
        aux_pred_stack = torch.zeros(B, 0, 4, device=device)
        aux_target_stack = torch.zeros(B, 0, 4, device=device)
        aux_risk_loss = torch.zeros((), device=device, dtype=loss_norm.dtype)

    return {
        "loss_raw": loss_raw,
        "loss_norm": loss_norm,
        "pnl_raw": pnl,
        "tc_norm": tc_total / scale,
        "turnover": turnover,
        "pos_l2": pos_l2 / max(float(H), 1.0),
        "sparse": sparse / max(float(H), 1.0),
        "q_hist": q_stack,
        "q_base_hist": qb_stack,
        "residual_hist": r_stack,
        "base_loss_raw": base_loss_raw.detach(),
        "base_loss_norm": base_loss_norm.detach(),
        "base_pnl_raw": base_pnl.detach(),
        "base_tc_norm": (base_tc_total / scale).detach(),
        "base_turnover": base_turnover.detach(),
        "excess_loss_norm": excess_loss_norm,
        "excess_relu": torch.relu(excess_loss_norm),
        "residual_l2": residual_l2 / max(float(H), 1.0),
        "residual_turnover": residual_turnover,
        "residual_delta_exposure": residual_delta_exposure / max(float(H), 1.0),
        "residual_vega_exposure": residual_vega_exposure / max(float(H), 1.0),
        "residual_factor_l2": residual_factor_l2 / max(float(H), 1.0),
        "global_trust_mean": global_trust_sum / max(float(H), 1.0),
        "aux_pred": aux_pred_stack,
        "aux_target": aux_target_stack,
        "aux_risk_loss": aux_risk_loss,
        "inst_mask_used": mask,
    }


def _hedgeset_imitation_loss(out: Dict[str, torch.Tensor], batch: dict) -> torch.Tensor:
    q = out["q_hist"]
    qb = out.get("q_base_hist", torch.zeros_like(q))
    mask = out.get("inst_mask_used", batch["inst_mask"].to(q.device)).float()
    return (((q - qb) ** 2) * mask).sum() / (mask.sum() + 1e-8)



def _regime_soft_worst_cvar(values: torch.Tensor, regime_id: Optional[torch.Tensor], cfg: HedgeSetConfig) -> torch.Tensor:
    if regime_id is None:
        return torch.zeros((), device=values.device, dtype=values.dtype)
    reg = regime_id.to(values.device)
    vals = []
    for rid in torch.unique(reg):
        mask_r = reg == rid
        if int(mask_r.sum().item()) >= int(getattr(cfg, "ambiguity_min_count", 2)):
            vals.append(_cvar_torch(values[mask_r], cfg.cvar_alpha, None))
    if not vals:
        return torch.zeros((), device=values.device, dtype=values.dtype)
    eta = max(float(getattr(cfg, "regime_excess_eta", getattr(cfg, "ambiguity_eta", 8.0))), 1e-6)
    stacked = torch.stack(vals)
    return torch.logsumexp(eta * stacked, dim=0) / eta


def hedgeset_training_loss(agent: HedgeSetAgent, batch: dict, cfg: HedgeSetConfig, cost_bps: float) -> Tuple[torch.Tensor, dict]:
    out = _rollout_hedgeset(agent, batch, cfg, cost_bps, mode="policy")
    losses = out["loss_norm"]
    cvar = _cvar_torch(losses, cfg.cvar_alpha, agent.cvar_eta)
    mean_loss = losses.mean()
    tc_pen = out["tc_norm"].mean()
    turn_pen = out["turnover"].mean()
    pos_pen = out["pos_l2"].mean()
    sparse_pen = out["sparse"].mean()
    imit_pen = _hedgeset_imitation_loss(out, batch)

    excess = out.get("excess_loss_norm", torch.zeros_like(losses))
    excess_cvar = _cvar_torch(excess, cfg.cvar_alpha, None)
    excess_relu = torch.relu(excess).mean()
    residual_l2_pen = out.get("residual_l2", torch.zeros_like(losses)).mean()
    residual_trade_pen = out.get("residual_turnover", torch.zeros_like(losses)).mean()
    residual_factor_pen = out.get("residual_factor_l2", torch.zeros_like(losses)).mean()
    trust_pen = out.get("global_trust_mean", torch.zeros_like(losses)).mean()
    aux_risk_pen = out.get("aux_risk_loss", torch.zeros((), device=losses.device, dtype=losses.dtype))
    regime_excess_cvar = _regime_soft_worst_cvar(excess, batch.get("regime_id", None), cfg)

    ambiguity_pen = torch.zeros((), device=losses.device, dtype=losses.dtype)
    if bool(getattr(cfg, "ambiguity_aware", False)) and float(getattr(cfg, "lambda_ambiguity", 0.0)) > 0:
        reg = batch.get("regime_id", None)
        if reg is not None:
            reg = reg.to(losses.device)
            vals = []
            for rid in torch.unique(reg):
                mask_r = reg == rid
                if int(mask_r.sum().item()) >= int(getattr(cfg, "ambiguity_min_count", 2)):
                    vals.append(_cvar_torch(losses[mask_r], cfg.cvar_alpha, None))
            if len(vals) > 0:
                reg_cvars = torch.stack(vals)
                eta = max(float(getattr(cfg, "ambiguity_eta", 8.0)), 1e-6)
                ambiguity_pen = torch.logsumexp(eta * reg_cvars, dim=0) / eta

    phase = str(getattr(cfg, "current_training_phase", "cvar")).lower()
    if phase == "imitation":
        # Warm-start to residual zero.  Because the base is hard-added in the
        # rollout, imitation is exactly residual shrinkage toward the LS path.
        total = (
            float(getattr(cfg, "imitation_lambda", 1.0)) * imit_pen
            + 0.05 * turn_pen
            + 0.01 * pos_pen
            + 0.01 * sparse_pen
            + float(getattr(cfg, "lambda_residual_l2", 0.05)) * residual_l2_pen
            + float(getattr(cfg, "lambda_residual_trade", 0.02)) * residual_trade_pen
            + float(getattr(cfg, "lambda_factor_neutral", 0.10)) * residual_factor_pen
            + float(getattr(cfg, "lambda_trust_l1", 0.01)) * trust_pen
            + float(getattr(cfg, "lambda_aux_risk", 0.05)) * aux_risk_pen
        )
    else:
        total = (
            cvar
            + cfg.lambda_mean * mean_loss
            + cfg.lambda_tc * tc_pen
            + cfg.lambda_turnover * turn_pen
            + cfg.lambda_pos * pos_pen
            + cfg.lambda_sparse * sparse_pen
            + float(getattr(cfg, "lambda_ambiguity", 0.0)) * ambiguity_pen
            + float(getattr(cfg, "finetune_imitation_lambda", 0.02)) * imit_pen
            + float(getattr(cfg, "lambda_excess_cvar", 0.50)) * excess_cvar
            + float(getattr(cfg, "lambda_excess_relu", 0.25)) * excess_relu
            + float(getattr(cfg, "lambda_regime_excess_cvar", 0.0)) * regime_excess_cvar
            + float(getattr(cfg, "lambda_residual_l2", 0.05)) * residual_l2_pen
            + float(getattr(cfg, "lambda_residual_trade", 0.02)) * residual_trade_pen
            + float(getattr(cfg, "lambda_factor_neutral", 0.10)) * residual_factor_pen
            + float(getattr(cfg, "lambda_trust_l1", 0.01)) * trust_pen
            + float(getattr(cfg, "lambda_aux_risk", 0.05)) * aux_risk_pen
        )
    return total, {
        "total": float(total.detach().cpu()),
        "cvar": float(cvar.detach().cpu()),
        "ambiguity": float(ambiguity_pen.detach().cpu()),
        "mean_loss": float(mean_loss.detach().cpu()),
        "tc": float(tc_pen.detach().cpu()),
        "turnover": float(turn_pen.detach().cpu()),
        "pos": float(pos_pen.detach().cpu()),
        "sparse": float(sparse_pen.detach().cpu()),
        "imitation": float(imit_pen.detach().cpu()),
        "excess_cvar": float(excess_cvar.detach().cpu()),
        "excess_relu": float(excess_relu.detach().cpu()),
        "residual_l2": float(residual_l2_pen.detach().cpu()),
        "residual_turnover": float(residual_trade_pen.detach().cpu()),
        "residual_factor": float(residual_factor_pen.detach().cpu()),
        "trust": float(trust_pen.detach().cpu()),
        "regime_excess_cvar": float(regime_excess_cvar.detach().cpu()),
        "aux_risk": float(aux_risk_pen.detach().cpu()),
    }


def summarize_hedgeset_panel(
    loss_norm: np.ndarray,
    pnl_raw: np.ndarray,
    tc_norm: np.ndarray,
    turnover: np.ndarray,
    q_abs: np.ndarray,
    base_loss_norm: Optional[np.ndarray] = None,
    excess_loss_norm: Optional[np.ndarray] = None,
    residual_l2: Optional[np.ndarray] = None,
    residual_turnover: Optional[np.ndarray] = None,
    residual_delta_exposure: Optional[np.ndarray] = None,
    residual_vega_exposure: Optional[np.ndarray] = None,
    residual_factor_l2: Optional[np.ndarray] = None,
    global_trust_mean: Optional[np.ndarray] = None,
) -> dict:
    out = {
        "n": int(len(loss_norm)),
        "mean_loss_norm": float(np.nanmean(loss_norm)),
        "std_loss_norm": float(np.nanstd(loss_norm)),
        "rmse_loss_norm": float(np.sqrt(np.nanmean(np.square(loss_norm)))),
        "var_95": float(np.nanquantile(loss_norm, 0.95)),
        "cvar_95": float(np.nanmean(loss_norm[loss_norm >= np.nanquantile(loss_norm, 0.95)])),
        "var_99": float(np.nanquantile(loss_norm, 0.99)),
        "cvar_99": float(np.nanmean(loss_norm[loss_norm >= np.nanquantile(loss_norm, 0.99)])),
        "shortfall_prob": float(np.nanmean(loss_norm > 0.0)),
        "mean_pnl_raw": float(np.nanmean(pnl_raw)),
        "mean_tc_norm": float(np.nanmean(tc_norm)),
        "mean_turnover": float(np.nanmean(turnover)),
        "mean_abs_position": float(np.nanmean(q_abs)),
    }
    if base_loss_norm is not None:
        b = np.asarray(base_loss_norm, dtype=float)
        out["mean_base_loss_norm"] = float(np.nanmean(b)) if np.isfinite(b).any() else np.nan
        out["base_cvar_95"] = float(np.nanmean(b[b >= np.nanquantile(b, 0.95)])) if np.isfinite(b).any() else np.nan
        out["base_cvar_99"] = float(np.nanmean(b[b >= np.nanquantile(b, 0.99)])) if np.isfinite(b).any() else np.nan
    if excess_loss_norm is not None:
        e = np.asarray(excess_loss_norm, dtype=float)
        out["mean_excess_loss_norm"] = float(np.nanmean(e)) if np.isfinite(e).any() else np.nan
        out["mean_relu_excess_loss_norm"] = float(np.nanmean(np.maximum(e, 0.0))) if np.isfinite(e).any() else np.nan
        if np.isfinite(e).any():
            q95 = np.nanquantile(e, 0.95)
            q99 = np.nanquantile(e, 0.99)
            out["excess_cvar_95"] = float(np.nanmean(e[e >= q95]))
            out["excess_cvar_99"] = float(np.nanmean(e[e >= q99]))
            out["excess_var_95"] = float(q95)
            out["excess_var_99"] = float(q99)
        else:
            out["excess_cvar_95"] = np.nan
            out["excess_cvar_99"] = np.nan
            out["excess_var_95"] = np.nan
            out["excess_var_99"] = np.nan
    if residual_l2 is not None:
        r2 = np.asarray(residual_l2, dtype=float)
        out["mean_residual_l2"] = float(np.nanmean(r2)) if np.isfinite(r2).any() else np.nan
    if residual_turnover is not None:
        rt = np.asarray(residual_turnover, dtype=float)
        out["mean_residual_turnover"] = float(np.nanmean(rt)) if np.isfinite(rt).any() else np.nan
    if residual_delta_exposure is not None:
        rd = np.asarray(residual_delta_exposure, dtype=float)
        out["mean_abs_residual_delta_exposure"] = float(np.nanmean(rd)) if np.isfinite(rd).any() else np.nan
        out["tail_abs_residual_delta_exposure"] = float(np.nanmean(rd[rd >= np.nanquantile(rd, 0.95)])) if np.isfinite(rd).any() else np.nan
    if residual_vega_exposure is not None:
        rv = np.asarray(residual_vega_exposure, dtype=float)
        out["mean_abs_residual_vega_exposure"] = float(np.nanmean(rv)) if np.isfinite(rv).any() else np.nan
        out["tail_abs_residual_vega_exposure"] = float(np.nanmean(rv[rv >= np.nanquantile(rv, 0.95)])) if np.isfinite(rv).any() else np.nan
    if residual_factor_l2 is not None:
        rf = np.asarray(residual_factor_l2, dtype=float)
        out["mean_residual_factor_l2"] = float(np.nanmean(rf)) if np.isfinite(rf).any() else np.nan
    if global_trust_mean is not None:
        gt = np.asarray(global_trust_mean, dtype=float)
        out["mean_global_trust"] = float(np.nanmean(gt)) if np.isfinite(gt).any() else np.nan
        out["q95_global_trust"] = float(np.nanquantile(gt, 0.95)) if np.isfinite(gt).any() else np.nan
    return out


@torch.no_grad()
def evaluate_hedgeset_agent(agent: Optional[HedgeSetAgent], loader: DataLoader, cfg: HedgeSetConfig, cost_bps: float, model_label: str, family: str, state_mode: str, mode: str = "policy") -> Tuple[dict, pd.DataFrame]:
    if agent is not None:
        agent.eval()
        device = next(agent.parameters()).device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for batch in loader:
        batch_t = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = _rollout_hedgeset(agent, batch_t, cfg, cost_bps, mode=mode)
        ln = out["loss_norm"].detach().cpu().numpy()
        pr = out["pnl_raw"].detach().cpu().numpy()
        tc = out["tc_norm"].detach().cpu().numpy()
        to = out["turnover"].detach().cpu().numpy()
        base_ln = out.get("base_loss_norm", torch.full_like(out["loss_norm"], float("nan"))).detach().cpu().numpy()
        excess_ln = out.get("excess_loss_norm", torch.full_like(out["loss_norm"], float("nan"))).detach().cpu().numpy()
        resid_l2 = out.get("residual_l2", torch.full_like(out["loss_norm"], float("nan"))).detach().cpu().numpy()
        resid_turn = out.get("residual_turnover", torch.full_like(out["loss_norm"], float("nan"))).detach().cpu().numpy()
        resid_delta = out.get("residual_delta_exposure", torch.full_like(out["loss_norm"], float("nan"))).detach().cpu().numpy()
        resid_vega = out.get("residual_vega_exposure", torch.full_like(out["loss_norm"], float("nan"))).detach().cpu().numpy()
        resid_factor = out.get("residual_factor_l2", torch.full_like(out["loss_norm"], float("nan"))).detach().cpu().numpy()
        trust_mean = out.get("global_trust_mean", torch.full_like(out["loss_norm"], float("nan"))).detach().cpu().numpy()
        q_np = out["q_hist"].detach().abs().cpu().numpy()
        # Use the actual mask used by the rollout.  This matters for
        # action-scope ablations such as underlying-only: PnL was already
        # computed with the restricted mask, so position diagnostics should use
        # the same denominator rather than the full original candidate set.
        mask_np = out.get("inst_mask_used", batch_t["inst_mask"]).detach().cpu().numpy().astype(bool)
        denom = np.maximum(mask_np.sum(axis=(1, 2)), 1)
        q_abs = (q_np * mask_np).sum(axis=(1, 2)) / denom
        q_nonzero = ((q_np > 1e-4) & mask_np).sum(axis=(1, 2)) / denom
        for i in range(len(ln)):
            rows.append({
                "model": model_label,
                "family": family,
                "state_mode": state_mode,
                "cost_bps": cost_bps,
                "start_date": batch["start_date"][i],
                "loss_norm": float(ln[i]),
                "pnl_raw": float(pr[i]),
                "tc_norm": float(tc[i]),
                "turnover": float(to[i]),
                "base_loss_norm": float(base_ln[i]),
                "excess_loss_norm": float(excess_ln[i]),
                "relu_excess_loss_norm": float(max(excess_ln[i], 0.0)) if np.isfinite(excess_ln[i]) else np.nan,
                "residual_l2": float(resid_l2[i]),
                "residual_turnover": float(resid_turn[i]),
                "residual_delta_exposure": float(resid_delta[i]),
                "residual_vega_exposure": float(resid_vega[i]),
                "residual_factor_l2": float(resid_factor[i]),
                "global_trust_mean": float(trust_mean[i]),
                "mean_abs_position": float(q_abs[i]),
                "nonzero_position_rate": float(q_nonzero[i]),
                "n_instruments": int(len(batch["inst_names"][i])),
                "instrument_names": ";".join(batch["inst_names"][i]),
                "regime_id": int(batch["regime_id"][i].detach().cpu().item()) if "regime_id" in batch_t else -1,
                "regime_score": float(batch["regime_score"][i].detach().cpu().item()) if "regime_score" in batch_t else np.nan,
                "regime_name": str(batch.get("regime_name", ["unassigned"] * len(ln))[i]) if isinstance(batch.get("regime_name", None), list) else "unassigned",
                "seed": getattr(cfg, "current_seed", np.nan),
                "policy_variant": getattr(cfg, "current_policy_variant", "static" if agent is None else "full"),
                "candidate_run": getattr(cfg, "run_name", "main"),
                "candidate_rules_tag": getattr(cfg, "candidate_rules_tag", ""),
                "action_scope": getattr(cfg, "current_action_scope", "full"),
                "residual_base": getattr(cfg, "residual_base", "none"),
                "residual_scale": getattr(cfg, "residual_scale", np.nan),
                "imitation_epochs": getattr(cfg, "imitation_epochs", np.nan),
            })
    panel = pd.DataFrame(rows)
    metrics = summarize_hedgeset_panel(
        panel["loss_norm"].values,
        panel["pnl_raw"].values,
        panel["tc_norm"].values,
        panel["turnover"].values,
        panel["mean_abs_position"].values,
        base_loss_norm=panel["base_loss_norm"].values if "base_loss_norm" in panel.columns else None,
        excess_loss_norm=panel["excess_loss_norm"].values if "excess_loss_norm" in panel.columns else None,
        residual_l2=panel["residual_l2"].values if "residual_l2" in panel.columns else None,
        residual_turnover=panel["residual_turnover"].values if "residual_turnover" in panel.columns else None,
        residual_delta_exposure=panel["residual_delta_exposure"].values if "residual_delta_exposure" in panel.columns else None,
        residual_vega_exposure=panel["residual_vega_exposure"].values if "residual_vega_exposure" in panel.columns else None,
        residual_factor_l2=panel["residual_factor_l2"].values if "residual_factor_l2" in panel.columns else None,
        global_trust_mean=panel["global_trust_mean"].values if "global_trust_mean" in panel.columns else None,
    )
    if "regime_id" in panel.columns and len(panel) > 0:
        reg_cvars, reg_shortfalls = [], []
        for rid, g in panel.groupby("regime_id", dropna=False):
            vals = pd.to_numeric(g["loss_norm"], errors="coerce").dropna().values
            if vals.size == 0:
                continue
            q95 = np.nanquantile(vals, 0.95)
            c95 = float(np.nanmean(vals[vals >= q95]))
            sf = float(np.nanmean(vals > 0.0))
            rid_int = int(rid) if pd.notna(rid) else -1
            metrics[f"regime_{rid_int}_n"] = int(vals.size)
            metrics[f"regime_{rid_int}_cvar_95"] = c95
            metrics[f"regime_{rid_int}_shortfall_prob"] = sf
            if "excess_loss_norm" in g.columns:
                ex_vals = pd.to_numeric(g["excess_loss_norm"], errors="coerce").dropna().values
                if ex_vals.size > 0:
                    qex = np.nanquantile(ex_vals, 0.95)
                    ex_cvar = float(np.nanmean(ex_vals[ex_vals >= qex]))
                    ex_relu = float(np.nanmean(np.maximum(ex_vals, 0.0)))
                    metrics[f"regime_{rid_int}_excess_cvar_95"] = ex_cvar
                    metrics[f"regime_{rid_int}_relu_excess_mean"] = ex_relu
                    metrics.setdefault("_regime_excess_cvars", []).append(ex_cvar)
                    metrics.setdefault("_regime_relu_excess", []).append(ex_relu)
            reg_cvars.append(c95)
            reg_shortfalls.append(sf)
        if reg_cvars:
            metrics["worst_regime_cvar_95"] = float(np.nanmax(reg_cvars))
            metrics["worst_regime_shortfall_prob"] = float(np.nanmax(reg_shortfalls)) if reg_shortfalls else np.nan
            if "_regime_excess_cvars" in metrics:
                metrics["worst_regime_excess_cvar_95"] = float(np.nanmax(metrics.pop("_regime_excess_cvars")))
            if "_regime_relu_excess" in metrics:
                metrics["worst_regime_relu_excess_mean"] = float(np.nanmax(metrics.pop("_regime_relu_excess")))
    metrics.update({
        "model": model_label,
        "family": family,
        "state_mode": state_mode,
        "cost_bps": cost_bps,
        "seed": getattr(cfg, "current_seed", np.nan),
        "policy_variant": getattr(cfg, "current_policy_variant", "static" if agent is None else "full"),
        "candidate_run": getattr(cfg, "run_name", "main"),
        "candidate_rules_tag": getattr(cfg, "candidate_rules_tag", ""),
        "action_scope": getattr(cfg, "current_action_scope", "full"),
        "residual_base": getattr(cfg, "residual_base", "none"),
        "residual_scale": getattr(cfg, "residual_scale", np.nan),
        "imitation_epochs": getattr(cfg, "imitation_epochs", np.nan),
        "factor_neutral_residual": getattr(cfg, "factor_neutral_residual", False),
        "use_global_trust_gate": getattr(cfg, "use_global_trust_gate", False),
        "lambda_regime_excess_cvar": getattr(cfg, "lambda_regime_excess_cvar", np.nan),
        "lambda_aux_risk": getattr(cfg, "lambda_aux_risk", np.nan),
    })
    return metrics, panel


def _hedgeset_validation_score(metrics: dict, cfg: HedgeSetConfig) -> float:
    score = 0.0
    score += float(getattr(cfg, "val_score_cvar95_weight", 1.0)) * float(metrics.get("cvar_95", np.inf))
    score += float(getattr(cfg, "val_score_cvar99_weight", 0.25)) * float(metrics.get("cvar_99", metrics.get("cvar_95", np.inf)))
    score += float(getattr(cfg, "val_score_mean_weight", 0.10)) * float(metrics.get("mean_loss_norm", 0.0))
    score += float(getattr(cfg, "val_score_turnover_weight", 0.02)) * float(metrics.get("mean_turnover", 0.0))
    score += float(getattr(cfg, "val_score_tc_weight", 0.05)) * float(metrics.get("mean_tc_norm", 0.0))
    score += float(getattr(cfg, "val_score_excess_cvar95_weight", 1.0)) * float(metrics.get("excess_cvar_95", 0.0))
    score += float(getattr(cfg, "val_score_excess_cvar99_weight", 0.25)) * float(metrics.get("excess_cvar_99", metrics.get("excess_cvar_95", 0.0)))
    score += float(getattr(cfg, "val_score_excess_relu_weight", 0.50)) * float(metrics.get("mean_relu_excess_loss_norm", 0.0))
    score += float(getattr(cfg, "val_score_worst_regime_excess_weight", 0.50)) * float(metrics.get("worst_regime_excess_cvar_95", metrics.get("excess_cvar_95", 0.0)))
    score += float(getattr(cfg, "val_score_factor_weight", 0.10)) * float(metrics.get("mean_residual_factor_l2", 0.0))
    score += float(getattr(cfg, "val_score_trust_weight", 0.02)) * float(metrics.get("mean_global_trust", 0.0))
    score += float(getattr(cfg, "val_score_worst_regime_weight", 0.25)) * float(metrics.get("worst_regime_cvar_95", metrics.get("cvar_95", np.inf)))
    return float(score)


def _hedgeset_feasible_checkpoint(metrics: dict, cfg: HedgeSetConfig, excess_cvar95_max: Optional[float] = None) -> bool:
    excess_max = float(getattr(cfg, "val_feasible_excess_cvar95_max", 0.012) if excess_cvar95_max is None else excess_cvar95_max)
    checks = [
        float(metrics.get("excess_cvar_95", np.inf)) <= excess_max,
        float(metrics.get("mean_abs_residual_delta_exposure", np.inf)) <= float(getattr(cfg, "val_feasible_delta_exposure_max", 1e-4)),
        float(metrics.get("mean_abs_residual_vega_exposure", np.inf)) <= float(getattr(cfg, "val_feasible_vega_exposure_max", 1e-4)),
    ]
    turnover_max = getattr(cfg, "val_feasible_turnover_max", None)
    if turnover_max is not None:
        checks.append(float(metrics.get("mean_turnover", np.inf)) <= float(turnover_max))
    tc_norm_max = getattr(cfg, "val_feasible_tc_norm_max", None)
    if tc_norm_max is not None:
        checks.append(float(metrics.get("mean_tc_norm", np.inf)) <= float(tc_norm_max))
    return all(bool(x) for x in checks)


def _hedgeset_feasible_risk_score(metrics: dict) -> float:
    return float(metrics.get("cvar_95", np.inf)) + 0.25 * float(metrics.get("cvar_99", metrics.get("cvar_95", np.inf))) + 0.25 * float(metrics.get("worst_regime_cvar_95", metrics.get("cvar_95", np.inf)))


def _hedgeset_optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _hedgeset_threshold_tag(value: float) -> str:
    return f"{float(value):.3f}".replace(".", "p")


def train_hedgeset_policy_one(episodes_path: str, family: str, state_mode: str, cost_bps: float, cfg: HedgeSetConfig, out_dir: str, seed: int = 42, policy_variant: str = "full", action_scope: str = "full", candidate_run: Optional[str] = None, rules_tag: Optional[str] = None) -> Tuple[dict, pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)
    cfg.current_seed = int(seed)
    cfg.current_policy_variant = str(policy_variant)
    cfg.current_action_scope = str(action_scope).lower().strip()
    if candidate_run is not None:
        cfg.run_name = str(candidate_run)
    if rules_tag is not None:
        cfg.candidate_rules_tag = str(rules_tag)
    device = torch.device(TRAIN_CFG.device if torch.cuda.is_available() else "cpu")
    train_ds = HedgeSetEpisodeDataset(episodes_path, "train", state_mode)
    val_ds = HedgeSetEpisodeDataset(episodes_path, "val", state_mode)
    test_ds = HedgeSetEpisodeDataset(episodes_path, "test", state_mode)
    if bool(getattr(cfg, "ambiguity_aware", False)):
        required = int(getattr(cfg, "regime_bins", 3)) * int(getattr(cfg, "ambiguity_min_count", 2))
        if int(cfg.batch_size) < required:
            logger.warning(
                f"Ambiguity-aware training batch_size={cfg.batch_size} is smaller than "
                f"regime_bins*ambiguity_min_count={required}. The worst-regime CVaR term may be inactive in many batches."
            )
    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=hedgeset_collate_fn, drop_last=False)
    val_ld = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=hedgeset_collate_fn)
    test_ld = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=hedgeset_collate_fn)
    normer = fit_hedgeset_normalizer(train_ds, device)
    global_dim = train_ds[0]["global_state"].shape[-1]
    inst_dim = train_ds[0]["inst_features"].shape[-1]
    agent = HedgeSetAgent(global_dim, inst_dim, normer, cfg).to(device)
    opt = torch.optim.AdamW(agent.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.05)
    best_val = float("inf")
    best_state = None
    best_feasible_val = float("inf")
    best_feasible_state = None
    best_feasible_metrics = None
    best_feasible_epoch = None
    feasible_thresholds = sorted({float(x) for x in getattr(cfg, "val_feasible_excess_cvar95_grid", [0.012]) if float(x) > 0})
    best_feasible_by_threshold = {
        th: {"score": float("inf"), "state": None, "metrics": None, "epoch": None}
        for th in feasible_thresholds
    }
    bad_epochs = 0
    history = []
    for epoch in range(cfg.epochs):
        cfg.current_training_phase = "imitation" if epoch < int(getattr(cfg, "imitation_epochs", 0)) else "cvar"
        agent.train()
        acc = {"total": 0.0, "cvar": 0.0, "ambiguity": 0.0, "mean_loss": 0.0, "tc": 0.0, "turnover": 0.0, "pos": 0.0, "sparse": 0.0, "imitation": 0.0}
        n_batches = 0
        for batch in train_ld:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            loss, info = hedgeset_training_loss(agent, batch, cfg, cost_bps)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            opt.step()
            for k in acc:
                acc[k] += info.get(k, 0.0)
            n_batches += 1
        scheduler.step()
        for k in acc:
            acc[k] /= max(n_batches, 1)

        agent.eval()
        val_metrics, _ = evaluate_hedgeset_agent(agent, val_ld, cfg, cost_bps, "Policy", family, state_mode, mode="policy")
        val_score = _hedgeset_validation_score(val_metrics, cfg)
        val_feasible = _hedgeset_feasible_checkpoint(val_metrics, cfg)
        val_feasible_score = _hedgeset_feasible_risk_score(val_metrics)
        row = {"epoch": epoch + 1, "phase": cfg.current_training_phase, "lr": opt.param_groups[0]["lr"], **{f"train_{k}": v for k, v in acc.items()}, **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))}, "val_selection_score": val_score, "val_feasible": val_feasible, "val_feasible_risk_score": val_feasible_score}
        for th in feasible_thresholds:
            row[f"val_feasible_excess_{_hedgeset_threshold_tag(th)}"] = _hedgeset_feasible_checkpoint(val_metrics, cfg, excess_cvar95_max=th)
        history.append(row)
        if (epoch + 1) % 25 == 0 or epoch == 0 or cfg.current_training_phase == "imitation":
            logger.info(
                f"[hedgeset {family}/{state_mode}/cost={cost_bps:g} ep {epoch+1:03d} {cfg.current_training_phase}] "
                f"train_cvar={acc['cvar']:.4f} imit={acc['imitation']:.4f} val_score={val_score:.4f} val_cvar95={val_metrics.get('cvar_95', np.nan):.4f}"
            )
        if val_feasible and val_feasible_score < best_feasible_val - cfg.min_delta:
            best_feasible_val = val_feasible_score
            best_feasible_state = copy.deepcopy(agent.state_dict())
            best_feasible_metrics = dict(val_metrics)
            best_feasible_epoch = epoch + 1
            torch.save(
                {
                    "agent": best_feasible_state,
                    "epoch": best_feasible_epoch,
                    "best_feasible_score": best_feasible_val,
                    "best_val_metrics": best_feasible_metrics,
                    "feasible_constraints": {
                        "excess_cvar_95": float(getattr(cfg, "val_feasible_excess_cvar95_max", 0.012)),
                        "mean_turnover": _hedgeset_optional_float(getattr(cfg, "val_feasible_turnover_max", None)),
                        "mean_tc_norm": _hedgeset_optional_float(getattr(cfg, "val_feasible_tc_norm_max", None)),
                        "mean_abs_residual_delta_exposure": float(getattr(cfg, "val_feasible_delta_exposure_max", 1e-4)),
                        "mean_abs_residual_vega_exposure": float(getattr(cfg, "val_feasible_vega_exposure_max", 1e-4)),
                    },
                },
                os.path.join(out_dir, "best_feasible_policy.pt"),
            )
        for th in feasible_thresholds:
            if not _hedgeset_feasible_checkpoint(val_metrics, cfg, excess_cvar95_max=th):
                continue
            slot = best_feasible_by_threshold[th]
            if val_feasible_score < float(slot["score"]) - cfg.min_delta:
                slot["score"] = val_feasible_score
                slot["state"] = copy.deepcopy(agent.state_dict())
                slot["metrics"] = dict(val_metrics)
                slot["epoch"] = epoch + 1
                torch.save(
                    {
                        "agent": slot["state"],
                        "epoch": slot["epoch"],
                        "best_feasible_score": slot["score"],
                        "best_val_metrics": slot["metrics"],
                        "feasible_constraints": {
                            "excess_cvar_95": th,
                            "mean_turnover": _hedgeset_optional_float(getattr(cfg, "val_feasible_turnover_max", None)),
                            "mean_tc_norm": _hedgeset_optional_float(getattr(cfg, "val_feasible_tc_norm_max", None)),
                            "mean_abs_residual_delta_exposure": float(getattr(cfg, "val_feasible_delta_exposure_max", 1e-4)),
                            "mean_abs_residual_vega_exposure": float(getattr(cfg, "val_feasible_vega_exposure_max", 1e-4)),
                        },
                    },
                    os.path.join(out_dir, f"best_feasible_policy_excess_{_hedgeset_threshold_tag(th)}.pt"),
                )
        if val_score < best_val - cfg.min_delta:
            best_val = val_score
            best_state = copy.deepcopy(agent.state_dict())
            bad_epochs = 0
            torch.save({"agent": best_state, "epoch": epoch + 1, "best_val_score": best_val, "best_val_metrics": val_metrics}, os.path.join(out_dir, "best_policy.pt"))
            torch.save({"agent": best_state, "epoch": epoch + 1, "best_val_score": best_val, "best_val_metrics": val_metrics}, os.path.join(out_dir, "best_score_policy.pt"))
        else:
            # Do not early-stop during imitation warm-start.
            if cfg.current_training_phase != "imitation":
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    logger.info(f"Early stopping HedgeSet policy at epoch {epoch+1}; best validation score={best_val:.6f}")
                    break
    if best_state is not None:
        agent.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "training_history.csv"), index=False)

    prefix = "AmbHedgeSet" if bool(getattr(cfg, "ambiguity_aware", False)) else "HedgeSet"
    pv = str(policy_variant)
    if bool(getattr(cfg, "ambiguity_aware", False)) and pv in {"amb", "ambiguous", "amb_full", "robust", "ambiguity"}:
        suffix = ""
    else:
        suffix = "" if pv == "full" else f"-{pv}"
    # Keep the main model label stable for pairwise bootstrap compatibility.
    # The selected classical anchor is saved in residual_base / run_config.json.
    scope_suffix = "" if str(action_scope).lower().strip() in {"full", "main", "all"} else f"-{str(action_scope).lower().strip()}"
    model_label = getattr(cfg, "model_label_override", None)
    if not model_label:
        model_label = f"{prefix}-{state_mode}{suffix}{scope_suffix}"
    test_metrics, test_panel = evaluate_hedgeset_agent(agent, test_ld, cfg, cost_bps, model_label, family, state_mode, mode="policy")
    test_metrics["checkpoint_selection"] = "best_score"
    test_panel["checkpoint_selection"] = "best_score"
    test_panel.to_csv(os.path.join(out_dir, "test_episode_panel.csv"), index=False)
    pd.DataFrame([test_metrics]).to_csv(os.path.join(out_dir, "test_metrics.csv"), index=False)
    checkpoint_rows = [dict(test_metrics)]
    if best_feasible_state is not None:
        agent.load_state_dict(best_feasible_state)
        feasible_metrics, feasible_panel = evaluate_hedgeset_agent(agent, test_ld, cfg, cost_bps, model_label, family, state_mode, mode="policy")
        feasible_metrics["checkpoint_selection"] = "best_feasible"
        feasible_metrics["best_feasible_epoch"] = best_feasible_epoch
        feasible_metrics["best_feasible_val_risk_score"] = best_feasible_val
        feasible_panel["checkpoint_selection"] = "best_feasible"
        feasible_panel.to_csv(os.path.join(out_dir, "test_episode_panel_best_feasible.csv"), index=False)
        pd.DataFrame([feasible_metrics]).to_csv(os.path.join(out_dir, "test_metrics_best_feasible.csv"), index=False)
        checkpoint_rows.append(dict(feasible_metrics))
    for th in feasible_thresholds:
        slot = best_feasible_by_threshold.get(th, {})
        if slot.get("state") is None:
            continue
        agent.load_state_dict(slot["state"])
        th_metrics, th_panel = evaluate_hedgeset_agent(agent, test_ld, cfg, cost_bps, model_label, family, state_mode, mode="policy")
        tag = _hedgeset_threshold_tag(th)
        th_metrics["checkpoint_selection"] = f"best_feasible_excess_{tag}"
        th_metrics["best_feasible_epoch"] = slot.get("epoch")
        th_metrics["best_feasible_val_risk_score"] = slot.get("score")
        th_metrics["best_feasible_excess_cvar95_max"] = th
        th_panel["checkpoint_selection"] = f"best_feasible_excess_{tag}"
        th_panel.to_csv(os.path.join(out_dir, f"test_episode_panel_best_feasible_excess_{tag}.csv"), index=False)
        pd.DataFrame([th_metrics]).to_csv(os.path.join(out_dir, f"test_metrics_best_feasible_excess_{tag}.csv"), index=False)
        checkpoint_rows.append(dict(th_metrics))
    pd.DataFrame(checkpoint_rows).to_csv(os.path.join(out_dir, "checkpoint_comparison_metrics.csv"), index=False)
    return test_metrics, test_panel


# -----------------------------------------------------------------------------
# Modern HedgeSet baselines
# -----------------------------------------------------------------------------
# These baselines use the same HedgeSet evaluation stack, historical
# quote-marked episodes, transaction costs, candidate instruments, validation
# selection, and paired bootstrap code.


def _normalize_modern_baseline_names(names: Optional[List[str] | str]) -> List[str]:
    if names is None:
        return []
    if isinstance(names, str):
        raw = _parse_list_arg(names)
    else:
        raw = [str(x).strip() for x in names if str(x).strip()]
    out: List[str] = []
    alias = {
        "all": ["ivgrid", "ntb_ls", "cvar_from_scratch", "qiao_residual_ls"],
        "surface": ["ivgrid"],
        "iv_surface": ["ivgrid"],
        "ivgrid": ["ivgrid"],
        "grid": ["ivgrid"],
        "fixed_grid": ["ivgrid"],
        "ntb": ["ntb_ls"],
        "ntb_ls": ["ntb_ls"],
        "ntb-ls": ["ntb_ls"],
        "no_transaction_band": ["ntb_ls"],
        "cvar": ["cvar_from_scratch"],
        "tail_rl": ["cvar_from_scratch"],
        "from_scratch": ["cvar_from_scratch"],
        "cvar_from_scratch": ["cvar_from_scratch"],
        "cvar-hedgeset": ["cvar_from_scratch"],
        "qiao": ["qiao_residual_ls"],
        "qiao_residual": ["qiao_residual_ls"],
        "qiao_residual_ls": ["qiao_residual_ls"],
        "qiao-residual-ls": ["qiao_residual_ls"],
        "qiaoresidual_ls": ["qiao_residual_ls"],
        "qiaoresidual-ls": ["qiao_residual_ls"],
        "plain_residual": ["qiao_residual_ls"],
        "plain_residual_ls": ["qiao_residual_ls"],
    }
    for x in raw:
        key = x.lower().strip().replace(" ", "_")
        vals = alias.get(key)
        if vals is None:
            raise ValueError(
                f"Unknown modern HedgeSet baseline: {x}. "
                "Use ivgrid, ntb_ls, cvar_from_scratch, qiao_residual_ls, or all."
            )
        for v in vals:
            if v not in out:
                out.append(v)
    return out

class NTBLSAgent(nn.Module):
    """Learned no-transaction-band baseline around LSDeltaVega.

    The baseline target is not learned:
        q_base_t = LSDeltaVega(batch, t).
    The network only learns state-dependent bands b_t.  The smooth gate
        sigmoid((|q_base_t - q_{t-1}| - b_t) / temp)
    decides how much of the base rebalance to execute.  There is no residual
    alpha overlay, no factor-neutral projection, and no trust gate.
    """
    def __init__(self, global_dim: int, inst_dim: int, normalizer: DualFeatureNormalizer, cfg: HedgeSetConfig):
        super().__init__()
        self.normalizer = normalizer
        d = int(cfg.embed_dim)
        self.global_proj = nn.Sequential(nn.Linear(global_dim, d), nn.GELU(), nn.LayerNorm(d))
        self.inst_proj = nn.Sequential(nn.Linear(inst_dim, d), nn.GELU(), nn.LayerNorm(d))
        self.band_head = nn.Sequential(
            nn.Linear(2 * d + 3, int(cfg.hidden_dim)), nn.GELU(), nn.LayerNorm(int(cfg.hidden_dim)),
            nn.Linear(int(cfg.hidden_dim), 1),
        )
        self.q_max = float(cfg.q_max)
        self.band_max = float(getattr(cfg, "ntb_ls_band_max", 1.0))
        self.smooth_temp = float(cfg.smooth_temp)
        self.use_smooth_gate = bool(getattr(cfg, "ntb_ls_use_smooth_gate", True))

    def act(self, global_state_t: torch.Tensor, inst_features_t: torch.Tensor,
            q_prev: torch.Tensor, q_base_t: torch.Tensor, mask_t: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        g = self.normalizer.norm_global(global_state_t)
        i = self.normalizer.norm_inst(inst_features_t)
        B, M, _ = i.shape
        g_tok = self.global_proj(g).unsqueeze(1).expand(-1, M, -1)
        i_tok = self.inst_proj(i)
        q_prev_s = (q_prev / max(self.q_max, 1e-6)).unsqueeze(-1)
        q_base_s = (q_base_t / max(self.q_max, 1e-6)).unsqueeze(-1)
        gap = q_base_t - q_prev
        gap_s = (torch.abs(gap) / max(self.q_max, 1e-6)).unsqueeze(-1)
        h = torch.cat([g_tok, i_tok, q_prev_s, q_base_s, gap_s], dim=-1)
        band = self.band_max * torch.sigmoid(self.band_head(h).squeeze(-1))
        if self.use_smooth_gate:
            gate = torch.sigmoid((torch.abs(gap) - band) / max(self.smooth_temp, 1e-6))
        else:
            gate = (torch.abs(gap) > band).to(gap.dtype)
        q_new = torch.clamp(q_prev + gate * gap, -self.q_max, self.q_max)
        q_new = torch.where(mask_t, q_new, torch.zeros_like(q_new))
        band = torch.where(mask_t, band, torch.zeros_like(band))
        gate = torch.where(mask_t, gate, torch.zeros_like(gate))
        return q_new, {"band": band, "gate": gate, "q_base": q_base_t}


def _rollout_ntb_ls(agent: NTBLSAgent, batch: dict, cfg: HedgeSetConfig, cost_bps: float) -> Dict[str, torch.Tensor]:
    G = batch["global_state"]
    IF = batch["inst_features"]
    P = batch["inst_price"]
    HS = batch["inst_half_spread"]
    mask = batch["inst_mask"]
    action_scope = str(getattr(cfg, "current_action_scope", "full")).lower().strip()
    if action_scope in {"underlying", "uonly", "underlying_only", "u_only"}:
        mask = mask.clone()
        if mask.shape[-1] > 1:
            mask[:, :, 1:] = False
    L = batch["L"]
    scale = batch["scale"].clamp(min=1.0)
    B, H, M = mask.shape
    device = G.device

    q_prev = torch.zeros(B, M, device=device)
    q_base_prev = torch.zeros(B, M, device=device)
    pnl = torch.zeros(B, device=device)
    tc_total = torch.zeros(B, device=device)
    turnover = torch.zeros(B, device=device)
    pos_l2 = torch.zeros(B, device=device)

    base_pnl = torch.zeros(B, device=device)
    base_tc_total = torch.zeros(B, device=device)
    base_turnover = torch.zeros(B, device=device)

    band_sum = torch.zeros(B, device=device)
    gate_sum = torch.zeros(B, device=device)
    q_hist, q_base_hist = [], []

    for t in range(H):
        mask_t = mask[:, t, :]
        q_base = _hs_base_target(batch, t, q_base_prev, cfg, base_name="ls_deltavega")
        q_base = torch.where(mask_t, q_base, torch.zeros_like(q_base))
        q_new, aux = agent.act(G[:, t, :], IF[:, t, :, :], q_prev, q_base, mask_t)
        q_new = torch.where(mask_t, q_new, torch.zeros_like(q_new))

        dP = P[:, t + 1, :] - P[:, t, :]
        dL = L[:, t + 1] - L[:, t]

        dq = q_new - q_prev
        prop_cost = (float(cost_bps) / 10000.0) * P[:, t, :] * torch.abs(dq)
        spread_cost = HS[:, t, :] * torch.abs(dq)
        tc = torch.where(mask_t, prop_cost + spread_cost, torch.zeros_like(prop_cost)).sum(dim=1)
        pnl = pnl + (q_new * dP).sum(dim=1) - dL - tc
        tc_total = tc_total + tc
        turnover = turnover + torch.abs(dq).sum(dim=1)
        pos_l2 = pos_l2 + (q_new ** 2).sum(dim=1)

        dq_base = q_base - q_base_prev
        prop_cost_b = (float(cost_bps) / 10000.0) * P[:, t, :] * torch.abs(dq_base)
        spread_cost_b = HS[:, t, :] * torch.abs(dq_base)
        tc_b = torch.where(mask_t, prop_cost_b + spread_cost_b, torch.zeros_like(prop_cost_b)).sum(dim=1)
        base_pnl = base_pnl + (q_base * dP).sum(dim=1) - dL - tc_b
        base_tc_total = base_tc_total + tc_b
        base_turnover = base_turnover + torch.abs(dq_base).sum(dim=1)

        m_float = mask_t.float()
        denom = m_float.sum(dim=1).clamp(min=1.0)
        band_sum = band_sum + (aux["band"] * m_float).sum(dim=1) / denom
        gate_sum = gate_sum + (aux["gate"] * m_float).sum(dim=1) / denom

        q_prev = q_new
        q_base_prev = q_base
        q_hist.append(q_new)
        q_base_hist.append(q_base)

    if cfg.terminal_liquidation:
        mask_last = mask[:, -1, :]
        prop_close = (float(cost_bps) / 10000.0) * P[:, -1, :] * torch.abs(q_prev)
        spread_close = HS[:, -1, :] * torch.abs(q_prev)
        tc_close = torch.where(mask_last, prop_close + spread_close, torch.zeros_like(prop_close)).sum(dim=1)
        pnl = pnl - tc_close
        tc_total = tc_total + tc_close
        turnover = turnover + torch.abs(q_prev).sum(dim=1)

        prop_close_b = (float(cost_bps) / 10000.0) * P[:, -1, :] * torch.abs(q_base_prev)
        spread_close_b = HS[:, -1, :] * torch.abs(q_base_prev)
        tc_close_b = torch.where(mask_last, prop_close_b + spread_close_b, torch.zeros_like(prop_close_b)).sum(dim=1)
        base_pnl = base_pnl - tc_close_b
        base_tc_total = base_tc_total + tc_close_b
        base_turnover = base_turnover + torch.abs(q_base_prev).sum(dim=1)

    loss_raw = -pnl
    loss_norm = loss_raw / scale
    base_loss_raw = -base_pnl
    base_loss_norm = base_loss_raw / scale
    excess_loss_norm = loss_norm - base_loss_norm.detach()
    q_stack = torch.stack(q_hist, dim=1) if q_hist else torch.zeros(B, H, M, device=device)
    qb_stack = torch.stack(q_base_hist, dim=1) if q_base_hist else torch.zeros(B, H, M, device=device)
    return {
        "loss_raw": loss_raw,
        "loss_norm": loss_norm,
        "pnl_raw": pnl,
        "tc_norm": tc_total / scale,
        "turnover": turnover,
        "pos_l2": pos_l2 / max(float(H), 1.0),
        "q_hist": q_stack,
        "q_base_hist": qb_stack,
        "base_loss_raw": base_loss_raw.detach(),
        "base_loss_norm": base_loss_norm.detach(),
        "base_pnl_raw": base_pnl.detach(),
        "base_tc_norm": (base_tc_total / scale).detach(),
        "base_turnover": base_turnover.detach(),
        "excess_loss_norm": excess_loss_norm,
        "excess_relu": torch.relu(excess_loss_norm),
        "ntb_band_mean": band_sum / max(float(H), 1.0),
        "ntb_gate_mean": gate_sum / max(float(H), 1.0),
        "inst_mask_used": mask,
    }


def ntb_ls_training_loss(agent: NTBLSAgent, batch: dict, cfg: HedgeSetConfig, cost_bps: float) -> Tuple[torch.Tensor, dict]:
    out = _rollout_ntb_ls(agent, batch, cfg, cost_bps)
    losses = out["loss_norm"]
    cvar = _cvar_torch(losses, cfg.cvar_alpha, None)
    mean_loss = losses.mean()
    tc_pen = out["tc_norm"].mean()
    turn_pen = out["turnover"].mean()
    pos_pen = out["pos_l2"].mean()
    excess = out.get("excess_loss_norm", torch.zeros_like(losses))
    excess_cvar = _cvar_torch(excess, cfg.cvar_alpha, None)
    excess_relu = torch.relu(excess).mean()
    band_l1 = out.get("ntb_band_mean", torch.zeros_like(losses)).mean()
    total = (
        cvar
        + cfg.lambda_mean * mean_loss
        + cfg.lambda_tc * tc_pen
        + cfg.lambda_turnover * turn_pen
        + cfg.lambda_pos * pos_pen
        + float(getattr(cfg, "lambda_excess_cvar", 0.50)) * excess_cvar
        + float(getattr(cfg, "lambda_excess_relu", 0.25)) * excess_relu
        + float(getattr(cfg, "ntb_ls_lambda_band_l1", 0.001)) * band_l1
    )
    return total, {
        "total": float(total.detach().cpu()),
        "cvar": float(cvar.detach().cpu()),
        "mean_loss": float(mean_loss.detach().cpu()),
        "tc": float(tc_pen.detach().cpu()),
        "turnover": float(turn_pen.detach().cpu()),
        "pos": float(pos_pen.detach().cpu()),
        "excess_cvar": float(excess_cvar.detach().cpu()),
        "excess_relu": float(excess_relu.detach().cpu()),
        "band": float(band_l1.detach().cpu()),
    }


@torch.no_grad()
def evaluate_ntb_ls_agent(agent: NTBLSAgent, loader: DataLoader, cfg: HedgeSetConfig, cost_bps: float,
                          model_label: str, family: str, state_mode: str) -> Tuple[dict, pd.DataFrame]:
    agent.eval()
    device = next(agent.parameters()).device
    rows = []
    for batch in loader:
        batch_t = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = _rollout_ntb_ls(agent, batch_t, cfg, cost_bps)
        ln = out["loss_norm"].detach().cpu().numpy()
        pr = out["pnl_raw"].detach().cpu().numpy()
        tc = out["tc_norm"].detach().cpu().numpy()
        to = out["turnover"].detach().cpu().numpy()
        base_ln = out["base_loss_norm"].detach().cpu().numpy()
        excess_ln = out["excess_loss_norm"].detach().cpu().numpy()
        band_m = out["ntb_band_mean"].detach().cpu().numpy()
        gate_m = out["ntb_gate_mean"].detach().cpu().numpy()
        q_np = out["q_hist"].detach().abs().cpu().numpy()
        mask_np = out.get("inst_mask_used", batch_t["inst_mask"]).detach().cpu().numpy().astype(bool)
        denom = np.maximum(mask_np.sum(axis=(1, 2)), 1)
        q_abs = (q_np * mask_np).sum(axis=(1, 2)) / denom
        q_nonzero = ((q_np > 1e-4) & mask_np).sum(axis=(1, 2)) / denom
        for i in range(len(ln)):
            rows.append({
                "model": model_label,
                "family": family,
                "state_mode": state_mode,
                "cost_bps": cost_bps,
                "start_date": batch["start_date"][i],
                "loss_norm": float(ln[i]),
                "pnl_raw": float(pr[i]),
                "tc_norm": float(tc[i]),
                "turnover": float(to[i]),
                "base_loss_norm": float(base_ln[i]),
                "excess_loss_norm": float(excess_ln[i]),
                "relu_excess_loss_norm": float(max(excess_ln[i], 0.0)) if np.isfinite(excess_ln[i]) else np.nan,
                "mean_abs_position": float(q_abs[i]),
                "nonzero_position_rate": float(q_nonzero[i]),
                "ntb_band_mean": float(band_m[i]),
                "ntb_gate_mean": float(gate_m[i]),
                "n_instruments": int(len(batch["inst_names"][i])),
                "instrument_names": ";".join(batch["inst_names"][i]),
                "regime_id": int(batch["regime_id"][i].detach().cpu().item()) if "regime_id" in batch_t else -1,
                "regime_score": float(batch["regime_score"][i].detach().cpu().item()) if "regime_score" in batch_t else np.nan,
                "regime_name": str(batch.get("regime_name", ["unassigned"] * len(ln))[i]) if isinstance(batch.get("regime_name", None), list) else "unassigned",
                "seed": getattr(cfg, "current_seed", np.nan),
                "policy_variant": "ntb_ls",
                "candidate_run": getattr(cfg, "run_name", "main"),
                "candidate_rules_tag": getattr(cfg, "candidate_rules_tag", ""),
                "action_scope": getattr(cfg, "current_action_scope", "full"),
                "residual_base": "ls_deltavega",
                "residual_scale": 0.0,
                "imitation_epochs": 0,
            })
    panel = pd.DataFrame(rows)
    metrics = summarize_hedgeset_panel(
        panel["loss_norm"].values, panel["pnl_raw"].values, panel["tc_norm"].values,
        panel["turnover"].values, panel["mean_abs_position"].values,
        base_loss_norm=panel["base_loss_norm"].values if "base_loss_norm" in panel.columns else None,
        excess_loss_norm=panel["excess_loss_norm"].values if "excess_loss_norm" in panel.columns else None,
    )
    if "regime_id" in panel.columns and len(panel) > 0:
        reg_cvars, reg_shortfalls, reg_ex_cvars = [], [], []
        for rid, g in panel.groupby("regime_id", dropna=False):
            vals = pd.to_numeric(g["loss_norm"], errors="coerce").dropna().values
            if vals.size == 0:
                continue
            q95 = np.nanquantile(vals, 0.95)
            reg_cvars.append(float(np.nanmean(vals[vals >= q95])))
            reg_shortfalls.append(float(np.nanmean(vals > 0.0)))
            ex_vals = pd.to_numeric(g["excess_loss_norm"], errors="coerce").dropna().values
            if ex_vals.size > 0:
                qex = np.nanquantile(ex_vals, 0.95)
                reg_ex_cvars.append(float(np.nanmean(ex_vals[ex_vals >= qex])))
        if reg_cvars:
            metrics["worst_regime_cvar_95"] = float(np.nanmax(reg_cvars))
            metrics["worst_regime_shortfall_prob"] = float(np.nanmax(reg_shortfalls)) if reg_shortfalls else np.nan
        if reg_ex_cvars:
            metrics["worst_regime_excess_cvar_95"] = float(np.nanmax(reg_ex_cvars))
    metrics.update({
        "model": model_label,
        "family": family,
        "state_mode": state_mode,
        "cost_bps": cost_bps,
        "seed": getattr(cfg, "current_seed", np.nan),
        "policy_variant": "ntb_ls",
        "candidate_run": getattr(cfg, "run_name", "main"),
        "candidate_rules_tag": getattr(cfg, "candidate_rules_tag", ""),
        "action_scope": getattr(cfg, "current_action_scope", "full"),
        "residual_base": "ls_deltavega",
        "residual_scale": 0.0,
        "imitation_epochs": 0,
        "mean_ntb_band": float(np.nanmean(panel["ntb_band_mean"].values)) if "ntb_band_mean" in panel.columns else np.nan,
        "mean_ntb_gate": float(np.nanmean(panel["ntb_gate_mean"].values)) if "ntb_gate_mean" in panel.columns else np.nan,
    })
    return metrics, panel


def train_ntb_ls_baseline_one(episodes_path: str, family: str, state_mode: str, cost_bps: float,
                              cfg: HedgeSetConfig, out_dir: str, seed: int = 42,
                              action_scope: str = "full", candidate_run: Optional[str] = None,
                              rules_tag: Optional[str] = None) -> Tuple[dict, pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)
    cfg.current_seed = int(seed)
    cfg.current_policy_variant = "ntb_ls"
    cfg.current_action_scope = str(action_scope).lower().strip()
    if candidate_run is not None:
        cfg.run_name = str(candidate_run)
    if rules_tag is not None:
        cfg.candidate_rules_tag = str(rules_tag)
    device = torch.device(TRAIN_CFG.device if torch.cuda.is_available() else "cpu")
    train_ds = HedgeSetEpisodeDataset(episodes_path, "train", state_mode)
    val_ds = HedgeSetEpisodeDataset(episodes_path, "val", state_mode)
    test_ds = HedgeSetEpisodeDataset(episodes_path, "test", state_mode)
    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=hedgeset_collate_fn, drop_last=False)
    val_ld = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=hedgeset_collate_fn)
    test_ld = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=hedgeset_collate_fn)
    normer = fit_hedgeset_normalizer(train_ds, device)
    global_dim = train_ds[0]["global_state"].shape[-1]
    inst_dim = train_ds[0]["inst_features"].shape[-1]
    agent = NTBLSAgent(global_dim, inst_dim, normer, cfg).to(device)
    opt = torch.optim.AdamW(agent.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.05)
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history = []
    for epoch in range(cfg.epochs):
        agent.train()
        acc = {"total": 0.0, "cvar": 0.0, "mean_loss": 0.0, "tc": 0.0, "turnover": 0.0, "pos": 0.0, "excess_cvar": 0.0, "excess_relu": 0.0, "band": 0.0}
        n_batches = 0
        for batch in train_ld:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            loss, info = ntb_ls_training_loss(agent, batch, cfg, cost_bps)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            opt.step()
            for k in acc:
                acc[k] += info.get(k, 0.0)
            n_batches += 1
        scheduler.step()
        for k in acc:
            acc[k] /= max(n_batches, 1)
        agent.eval()
        val_metrics, _ = evaluate_ntb_ls_agent(agent, val_ld, cfg, cost_bps, "NTB-LS", family, state_mode)
        val_score = _hedgeset_validation_score(val_metrics, cfg)
        row = {"epoch": epoch + 1, "lr": opt.param_groups[0]["lr"], **{f"train_{k}": v for k, v in acc.items()}, **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))}, "val_selection_score": val_score}
        history.append(row)
        if (epoch + 1) % 25 == 0 or epoch == 0:
            logger.info(f"[ntb-ls {family}/{state_mode}/cost={cost_bps:g} ep {epoch+1:03d}] train_cvar={acc['cvar']:.4f} val_score={val_score:.4f} val_cvar95={val_metrics.get('cvar_95', np.nan):.4f}")
        if val_score < best_val - cfg.min_delta:
            best_val = val_score
            best_state = copy.deepcopy(agent.state_dict())
            bad_epochs = 0
            torch.save({"agent": best_state, "epoch": epoch + 1, "best_val_score": best_val, "best_val_metrics": val_metrics}, os.path.join(out_dir, "best_policy.pt"))
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                logger.info(f"Early stopping NTB-LS baseline at epoch {epoch+1}; best validation score={best_val:.6f}")
                break
    if best_state is not None:
        agent.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "training_history.csv"), index=False)
    test_metrics, test_panel = evaluate_ntb_ls_agent(agent, test_ld, cfg, cost_bps, "NTB-LS", family, state_mode)
    test_panel.to_csv(os.path.join(out_dir, "test_episode_panel.csv"), index=False)
    pd.DataFrame([test_metrics]).to_csv(os.path.join(out_dir, "test_metrics.csv"), index=False)
    return test_metrics, test_panel




class QiaoResidualLSAgent(nn.Module):
    """Plain anchored residual-learning baseline inspired by Qiao-style residual hedging.

    The action rule is
        q_t = q_t^0 + r_theta(s_t, H_t),
    where q_t^0 is the LSDeltaVega anchor and r_theta is a bounded MLP residual.

    This baseline uses no FNUG-specific residual controls:
      * factor-neutral residual projection,
      * global trust gate,
      * excess-CVaR training,
      * worst-regime excess penalty,
      * auxiliary dense risk loss.

    Training uses the Qiao-style local squared hedging-error objective generalized
    to the multi-instrument HedgeSet setting:
        (Delta L_t - q_t' Delta P_t)^2,
    evaluated on historical quote-marked paths. Full-episode CVaR is used for
    validation checkpoint selection and final reporting.
    """
    def __init__(self, global_dim: int, inst_dim: int, normalizer: DualFeatureNormalizer, cfg: HedgeSetConfig):
        super().__init__()
        self.normalizer = normalizer
        h = int(getattr(cfg, "qiao_hidden_dim", getattr(cfg, "hidden_dim", 96)))
        n_layers = max(1, int(getattr(cfg, "qiao_hidden_layers", getattr(cfg, "hidden_layers", 3))))
        self.global_proj = nn.Sequential(nn.Linear(global_dim, h), nn.GELU(), nn.LayerNorm(h))

        # Per-instrument MLP.  This is a plain residual learner, not a set-attention
        # controller.  It receives the normalized instrument features, a global
        # market-state embedding, previous total position, LS base position, and
        # the base rebalance gap.
        in_dim = h + inst_dim + 3
        layers = []
        cur = in_dim
        for _ in range(max(1, n_layers - 1)):
            layers += [nn.Linear(cur, h), nn.GELU(), nn.LayerNorm(h)]
            cur = h
        layers.append(nn.Linear(cur, 1))
        self.residual_net = nn.Sequential(*layers)

        self.q_max = float(getattr(cfg, "q_max", 3.0))
        self.residual_scale = float(getattr(cfg, "qiao_residual_scale", getattr(cfg, "residual_scale", 0.15)))

    def act(
        self,
        global_state_t: torch.Tensor,
        inst_features_t: torch.Tensor,
        q_prev: torch.Tensor,
        q_base_t: torch.Tensor,
        mask_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        g = self.normalizer.norm_global(global_state_t)
        i = self.normalizer.norm_inst(inst_features_t)
        B, M, _ = i.shape
        g_h = self.global_proj(g).unsqueeze(1).expand(-1, M, -1)
        q_prev_s = (q_prev / max(self.q_max, 1e-6)).unsqueeze(-1)
        q_base_s = (q_base_t / max(self.q_max, 1e-6)).unsqueeze(-1)
        gap_s = ((q_base_t - q_prev) / max(self.q_max, 1e-6)).unsqueeze(-1)
        x = torch.cat([g_h, i, q_prev_s, q_base_s, gap_s], dim=-1)
        residual_target = self.residual_scale * torch.tanh(self.residual_net(x).squeeze(-1))
        residual_target = torch.where(mask_t, residual_target, torch.zeros_like(residual_target))
        q_new = torch.clamp(q_base_t + residual_target, -self.q_max, self.q_max)
        q_new = torch.where(mask_t, q_new, torch.zeros_like(q_new))
        # Diagnostics and residual penalties should reflect the actually applied
        # residual after position clipping, not the pre-clipping target.
        residual_applied = torch.where(mask_t, q_new - q_base_t, torch.zeros_like(q_new))
        return q_new, {"residual": residual_applied, "residual_target": residual_target, "q_base": q_base_t}


def _rollout_qiao_residual_ls(
    agent: QiaoResidualLSAgent,
    batch: dict,
    cfg: HedgeSetConfig,
    cost_bps: float,
) -> Dict[str, torch.Tensor]:
    G = batch["global_state"]
    IF = batch["inst_features"]
    P = batch["inst_price"]
    HS = batch["inst_half_spread"]
    mask = batch["inst_mask"]
    action_scope = str(getattr(cfg, "current_action_scope", "full")).lower().strip()
    if action_scope in {"underlying", "uonly", "underlying_only", "u_only"}:
        mask = mask.clone()
        if mask.shape[-1] > 1:
            mask[:, :, 1:] = False

    L = batch["L"]
    scale = batch["scale"].clamp(min=1.0)
    B, H, M = mask.shape
    device = G.device

    q_prev = torch.zeros(B, M, device=device)
    q_base_prev = torch.zeros(B, M, device=device)
    r_prev = torch.zeros(B, M, device=device)

    pnl = torch.zeros(B, device=device)
    tc_total = torch.zeros(B, device=device)
    turnover = torch.zeros(B, device=device)
    pos_l2 = torch.zeros(B, device=device)

    base_pnl = torch.zeros(B, device=device)
    base_tc_total = torch.zeros(B, device=device)
    base_turnover = torch.zeros(B, device=device)

    local_sq = torch.zeros(B, device=device)
    residual_l2 = torch.zeros(B, device=device)
    residual_turnover = torch.zeros(B, device=device)
    residual_delta_exposure = torch.zeros(B, device=device)
    residual_vega_exposure = torch.zeros(B, device=device)
    residual_factor_l2 = torch.zeros(B, device=device)

    q_hist, q_base_hist, r_hist = [], [], []
    base_name = str(getattr(cfg, "qiao_residual_base", "ls_deltavega")).lower().strip() or "ls_deltavega"

    for t in range(H):
        mask_t = mask[:, t, :]
        q_base = _hs_base_target(batch, t, q_base_prev, cfg, base_name=base_name)
        q_base = torch.where(mask_t, q_base, torch.zeros_like(q_base))
        q_new, aux = agent.act(G[:, t, :], IF[:, t, :, :], q_prev, q_base, mask_t)
        q_new = torch.where(mask_t, q_new, torch.zeros_like(q_new))
        r_new = torch.where(mask_t, aux["residual"], torch.zeros_like(q_prev))

        dP = P[:, t + 1, :] - P[:, t, :]
        dL = L[:, t + 1] - L[:, t]

        # Qiao-style local squared hedging error. Transaction costs are charged
        # in the rollout and final evaluation.
        local_err = dL - (q_new * dP).sum(dim=1)
        local_sq = local_sq + (local_err / scale).pow(2)

        dq = q_new - q_prev
        prop_cost = (float(cost_bps) / 10000.0) * P[:, t, :] * torch.abs(dq)
        spread_cost = HS[:, t, :] * torch.abs(dq)
        tc = torch.where(mask_t, prop_cost + spread_cost, torch.zeros_like(prop_cost)).sum(dim=1)
        pnl = pnl + (q_new * dP).sum(dim=1) - dL - tc
        tc_total = tc_total + tc
        turnover = turnover + torch.abs(dq).sum(dim=1)
        pos_l2 = pos_l2 + (q_new ** 2).sum(dim=1)

        # Standalone LS-base PnL for excess-loss diagnostics.
        dq_base = q_base - q_base_prev
        prop_cost_b = (float(cost_bps) / 10000.0) * P[:, t, :] * torch.abs(dq_base)
        spread_cost_b = HS[:, t, :] * torch.abs(dq_base)
        tc_b = torch.where(mask_t, prop_cost_b + spread_cost_b, torch.zeros_like(prop_cost_b)).sum(dim=1)
        base_pnl = base_pnl + (q_base * dP).sum(dim=1) - dL - tc_b
        base_tc_total = base_tc_total + tc_b
        base_turnover = base_turnover + torch.abs(dq_base).sum(dim=1)

        residual_l2 = residual_l2 + (r_new ** 2).sum(dim=1)
        residual_turnover = residual_turnover + torch.abs(r_new - r_prev).sum(dim=1)

        delta_inst = batch["inst_delta"][:, t, :].to(device=device, dtype=r_new.dtype)
        vega_inst = batch["inst_vega"][:, t, :].to(device=device, dtype=r_new.dtype) / max(float(getattr(cfg, "ls_vega_scale", 100.0)), 1e-6)
        rd = torch.abs((r_new * delta_inst).sum(dim=1))
        rv = torch.abs((r_new * vega_inst).sum(dim=1))
        residual_delta_exposure = residual_delta_exposure + rd
        residual_vega_exposure = residual_vega_exposure + rv
        residual_factor_l2 = residual_factor_l2 + rd.pow(2) + rv.pow(2)

        q_prev = q_new
        q_base_prev = q_base
        r_prev = r_new
        q_hist.append(q_new)
        q_base_hist.append(q_base)
        r_hist.append(r_new)

    if cfg.terminal_liquidation:
        mask_last = mask[:, -1, :]
        prop_close = (float(cost_bps) / 10000.0) * P[:, -1, :] * torch.abs(q_prev)
        spread_close = HS[:, -1, :] * torch.abs(q_prev)
        tc_close = torch.where(mask_last, prop_close + spread_close, torch.zeros_like(prop_close)).sum(dim=1)
        pnl = pnl - tc_close
        tc_total = tc_total + tc_close
        turnover = turnover + torch.abs(q_prev).sum(dim=1)

        prop_close_b = (float(cost_bps) / 10000.0) * P[:, -1, :] * torch.abs(q_base_prev)
        spread_close_b = HS[:, -1, :] * torch.abs(q_base_prev)
        tc_close_b = torch.where(mask_last, prop_close_b + spread_close_b, torch.zeros_like(prop_close_b)).sum(dim=1)
        base_pnl = base_pnl - tc_close_b
        base_tc_total = base_tc_total + tc_close_b
        base_turnover = base_turnover + torch.abs(q_base_prev).sum(dim=1)

    loss_raw = -pnl
    loss_norm = loss_raw / scale
    base_loss_raw = -base_pnl
    base_loss_norm = base_loss_raw / scale
    excess_loss_norm = loss_norm - base_loss_norm.detach()

    q_stack = torch.stack(q_hist, dim=1) if q_hist else torch.zeros(B, H, M, device=device)
    qb_stack = torch.stack(q_base_hist, dim=1) if q_base_hist else torch.zeros(B, H, M, device=device)
    r_stack = torch.stack(r_hist, dim=1) if r_hist else torch.zeros(B, H, M, device=device)

    return {
        "loss_raw": loss_raw,
        "loss_norm": loss_norm,
        "pnl_raw": pnl,
        "tc_norm": tc_total / scale,
        "turnover": turnover,
        "pos_l2": pos_l2 / max(float(H), 1.0),
        "q_hist": q_stack,
        "q_base_hist": qb_stack,
        "residual_hist": r_stack,
        "base_loss_raw": base_loss_raw.detach(),
        "base_loss_norm": base_loss_norm.detach(),
        "base_pnl_raw": base_pnl.detach(),
        "base_tc_norm": (base_tc_total / scale).detach(),
        "base_turnover": base_turnover.detach(),
        "excess_loss_norm": excess_loss_norm,
        "excess_relu": torch.relu(excess_loss_norm),
        "local_mse": local_sq / max(float(H), 1.0),
        "residual_l2": residual_l2 / max(float(H), 1.0),
        "residual_turnover": residual_turnover,
        "residual_delta_exposure": residual_delta_exposure / max(float(H), 1.0),
        "residual_vega_exposure": residual_vega_exposure / max(float(H), 1.0),
        "residual_factor_l2": residual_factor_l2 / max(float(H), 1.0),
        "inst_mask_used": mask,
    }


def qiao_residual_ls_training_loss(agent: QiaoResidualLSAgent, batch: dict, cfg: HedgeSetConfig, cost_bps: float) -> Tuple[torch.Tensor, dict]:
    out = _rollout_qiao_residual_ls(agent, batch, cfg, cost_bps)
    local_mse = out["local_mse"].mean()
    residual_l2 = out["residual_l2"].mean()
    residual_trade = out["residual_turnover"].mean()
    tc_pen = out["tc_norm"].mean()
    turn_pen = out["turnover"].mean()
    total = (
        float(getattr(cfg, "qiao_lambda_local_mse", 1.0)) * local_mse
        + float(getattr(cfg, "qiao_lambda_residual_l2", 0.02)) * residual_l2
        + float(getattr(cfg, "qiao_lambda_residual_trade", 0.01)) * residual_trade
        + float(getattr(cfg, "qiao_lambda_tc", 0.0)) * tc_pen
        + float(getattr(cfg, "qiao_lambda_turnover", 0.0)) * turn_pen
    )
    return total, {
        "total": float(total.detach().cpu()),
        "local_mse": float(local_mse.detach().cpu()),
        "residual_l2": float(residual_l2.detach().cpu()),
        "residual_turnover": float(residual_trade.detach().cpu()),
        "tc": float(tc_pen.detach().cpu()),
        "turnover": float(turn_pen.detach().cpu()),
    }


def _qiao_residual_ls_validation_score(metrics: dict, cfg: HedgeSetConfig) -> float:
    """Validation selection score for the plain Qiao-style residual baseline.

    Unlike FNUG, this baseline is not selected using excess-risk, trust, or
    factor-neutral metrics.  This keeps the comparator a plain anchored residual
    learner while still using validation-only episode risk to select checkpoints.
    """
    c95 = float(metrics.get("cvar_95", np.inf))
    c99 = float(metrics.get("cvar_99", c95))
    mean = float(metrics.get("mean_loss_norm", 0.0))
    turnover = float(metrics.get("mean_turnover", 0.0))
    tc = float(metrics.get("mean_tc_norm", 0.0))
    return (
        float(getattr(cfg, "val_score_cvar95_weight", 1.0)) * c95
        + float(getattr(cfg, "val_score_cvar99_weight", 0.25)) * c99
        + float(getattr(cfg, "val_score_mean_weight", 0.10)) * mean
        + float(getattr(cfg, "val_score_turnover_weight", 0.02)) * turnover
        + float(getattr(cfg, "val_score_tc_weight", 0.05)) * tc
    )


@torch.no_grad()
def evaluate_qiao_residual_ls_agent(agent: QiaoResidualLSAgent, loader: DataLoader, cfg: HedgeSetConfig, cost_bps: float,
                                    model_label: str, family: str, state_mode: str) -> Tuple[dict, pd.DataFrame]:
    agent.eval()
    device = next(agent.parameters()).device
    rows = []
    for batch in loader:
        batch_t = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = _rollout_qiao_residual_ls(agent, batch_t, cfg, cost_bps)
        ln = out["loss_norm"].detach().cpu().numpy()
        pr = out["pnl_raw"].detach().cpu().numpy()
        tc = out["tc_norm"].detach().cpu().numpy()
        to = out["turnover"].detach().cpu().numpy()
        base_ln = out["base_loss_norm"].detach().cpu().numpy()
        excess_ln = out["excess_loss_norm"].detach().cpu().numpy()
        resid_l2 = out["residual_l2"].detach().cpu().numpy()
        resid_turn = out["residual_turnover"].detach().cpu().numpy()
        resid_delta = out["residual_delta_exposure"].detach().cpu().numpy()
        resid_vega = out["residual_vega_exposure"].detach().cpu().numpy()
        resid_factor = out["residual_factor_l2"].detach().cpu().numpy()
        local_mse = out["local_mse"].detach().cpu().numpy()
        q_np = out["q_hist"].detach().abs().cpu().numpy()
        mask_np = out.get("inst_mask_used", batch_t["inst_mask"]).detach().cpu().numpy().astype(bool)
        denom = np.maximum(mask_np.sum(axis=(1, 2)), 1)
        q_abs = (q_np * mask_np).sum(axis=(1, 2)) / denom
        q_nonzero = ((q_np > 1e-4) & mask_np).sum(axis=(1, 2)) / denom
        for i in range(len(ln)):
            rows.append({
                "model": model_label,
                "family": family,
                "state_mode": state_mode,
                "cost_bps": cost_bps,
                "start_date": batch["start_date"][i],
                "loss_norm": float(ln[i]),
                "pnl_raw": float(pr[i]),
                "tc_norm": float(tc[i]),
                "turnover": float(to[i]),
                "base_loss_norm": float(base_ln[i]),
                "excess_loss_norm": float(excess_ln[i]),
                "relu_excess_loss_norm": float(max(excess_ln[i], 0.0)) if np.isfinite(excess_ln[i]) else np.nan,
                "local_mse": float(local_mse[i]),
                "residual_l2": float(resid_l2[i]),
                "residual_turnover": float(resid_turn[i]),
                "residual_delta_exposure": float(resid_delta[i]),
                "residual_vega_exposure": float(resid_vega[i]),
                "residual_factor_l2": float(resid_factor[i]),
                "global_trust_mean": np.nan,
                "mean_abs_position": float(q_abs[i]),
                "nonzero_position_rate": float(q_nonzero[i]),
                "n_instruments": int(len(batch["inst_names"][i])),
                "instrument_names": ";".join(batch["inst_names"][i]),
                "regime_id": int(batch["regime_id"][i].detach().cpu().item()) if "regime_id" in batch_t else -1,
                "regime_score": float(batch["regime_score"][i].detach().cpu().item()) if "regime_score" in batch_t else np.nan,
                "regime_name": str(batch.get("regime_name", ["unassigned"] * len(ln))[i]) if isinstance(batch.get("regime_name", None), list) else "unassigned",
                "seed": getattr(cfg, "current_seed", np.nan),
                "policy_variant": "qiao_residual_ls",
                "candidate_run": getattr(cfg, "run_name", "main"),
                "candidate_rules_tag": getattr(cfg, "candidate_rules_tag", ""),
                "action_scope": getattr(cfg, "current_action_scope", "full"),
                "residual_base": getattr(cfg, "qiao_residual_base", "ls_deltavega"),
                "residual_scale": getattr(cfg, "qiao_residual_scale", getattr(cfg, "residual_scale", np.nan)),
                "imitation_epochs": 0,
            })
    panel = pd.DataFrame(rows)
    metrics = summarize_hedgeset_panel(
        panel["loss_norm"].values,
        panel["pnl_raw"].values,
        panel["tc_norm"].values,
        panel["turnover"].values,
        panel["mean_abs_position"].values,
        base_loss_norm=panel["base_loss_norm"].values if "base_loss_norm" in panel.columns else None,
        excess_loss_norm=panel["excess_loss_norm"].values if "excess_loss_norm" in panel.columns else None,
        residual_l2=panel["residual_l2"].values if "residual_l2" in panel.columns else None,
        residual_turnover=panel["residual_turnover"].values if "residual_turnover" in panel.columns else None,
        residual_delta_exposure=panel["residual_delta_exposure"].values if "residual_delta_exposure" in panel.columns else None,
        residual_vega_exposure=panel["residual_vega_exposure"].values if "residual_vega_exposure" in panel.columns else None,
        residual_factor_l2=panel["residual_factor_l2"].values if "residual_factor_l2" in panel.columns else None,
        global_trust_mean=None,
    )
    if "local_mse" in panel.columns:
        metrics["mean_local_mse"] = float(np.nanmean(panel["local_mse"].values))
    if "regime_id" in panel.columns and len(panel) > 0:
        reg_cvars, reg_shortfalls, reg_ex_cvars = [], [], []
        for rid, g in panel.groupby("regime_id", dropna=False):
            vals = pd.to_numeric(g["loss_norm"], errors="coerce").dropna().values
            if vals.size == 0:
                continue
            q95 = np.nanquantile(vals, 0.95)
            reg_cvars.append(float(np.nanmean(vals[vals >= q95])))
            reg_shortfalls.append(float(np.nanmean(vals > 0.0)))
            ex_vals = pd.to_numeric(g["excess_loss_norm"], errors="coerce").dropna().values
            if ex_vals.size > 0:
                qex = np.nanquantile(ex_vals, 0.95)
                reg_ex_cvars.append(float(np.nanmean(ex_vals[ex_vals >= qex])))
        if reg_cvars:
            metrics["worst_regime_cvar_95"] = float(np.nanmax(reg_cvars))
            metrics["worst_regime_shortfall_prob"] = float(np.nanmax(reg_shortfalls)) if reg_shortfalls else np.nan
        if reg_ex_cvars:
            metrics["worst_regime_excess_cvar_95"] = float(np.nanmax(reg_ex_cvars))
    metrics.update({
        "model": model_label,
        "family": family,
        "state_mode": state_mode,
        "cost_bps": cost_bps,
        "seed": getattr(cfg, "current_seed", np.nan),
        "policy_variant": "qiao_residual_ls",
        "candidate_run": getattr(cfg, "run_name", "main"),
        "candidate_rules_tag": getattr(cfg, "candidate_rules_tag", ""),
        "action_scope": getattr(cfg, "current_action_scope", "full"),
        "residual_base": getattr(cfg, "qiao_residual_base", "ls_deltavega"),
        "residual_scale": getattr(cfg, "qiao_residual_scale", getattr(cfg, "residual_scale", np.nan)),
        "imitation_epochs": 0,
        "factor_neutral_residual": False,
        "use_global_trust_gate": False,
        "lambda_regime_excess_cvar": 0.0,
        "lambda_aux_risk": 0.0,
    })
    return metrics, panel


def train_qiao_residual_ls_baseline_one(
    episodes_path: str,
    family: str,
    state_mode: str,
    cost_bps: float,
    cfg: HedgeSetConfig,
    out_dir: str,
    seed: int = 42,
    action_scope: str = "full",
    candidate_run: Optional[str] = None,
    rules_tag: Optional[str] = None,
) -> Tuple[dict, pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)
    cfg.current_seed = int(seed)
    cfg.current_policy_variant = "qiao_residual_ls"
    cfg.current_action_scope = str(action_scope).lower().strip()
    if candidate_run is not None:
        cfg.run_name = str(candidate_run)
    if rules_tag is not None:
        cfg.candidate_rules_tag = str(rules_tag)
    device = torch.device(TRAIN_CFG.device if torch.cuda.is_available() else "cpu")
    train_ds = HedgeSetEpisodeDataset(episodes_path, "train", state_mode)
    val_ds = HedgeSetEpisodeDataset(episodes_path, "val", state_mode)
    test_ds = HedgeSetEpisodeDataset(episodes_path, "test", state_mode)
    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=hedgeset_collate_fn, drop_last=False)
    val_ld = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=hedgeset_collate_fn)
    test_ld = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=hedgeset_collate_fn)
    normer = fit_hedgeset_normalizer(train_ds, device)
    global_dim = train_ds[0]["global_state"].shape[-1]
    inst_dim = train_ds[0]["inst_features"].shape[-1]
    agent = QiaoResidualLSAgent(global_dim, inst_dim, normer, cfg).to(device)
    opt = torch.optim.AdamW(agent.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.05)
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history = []
    for epoch in range(cfg.epochs):
        agent.train()
        acc = {"total": 0.0, "local_mse": 0.0, "residual_l2": 0.0, "residual_turnover": 0.0, "tc": 0.0, "turnover": 0.0}
        n_batches = 0
        for batch in train_ld:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            loss, info = qiao_residual_ls_training_loss(agent, batch, cfg, cost_bps)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            opt.step()
            for k in acc:
                acc[k] += info.get(k, 0.0)
            n_batches += 1
        scheduler.step()
        for k in acc:
            acc[k] /= max(n_batches, 1)
        agent.eval()
        val_metrics, _ = evaluate_qiao_residual_ls_agent(agent, val_ld, cfg, cost_bps, "QiaoResidual-LS", family, state_mode)
        val_score = _qiao_residual_ls_validation_score(val_metrics, cfg)
        row = {"epoch": epoch + 1, "lr": opt.param_groups[0]["lr"], **{f"train_{k}": v for k, v in acc.items()}, **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))}, "val_selection_score": val_score}
        history.append(row)
        if (epoch + 1) % 25 == 0 or epoch == 0:
            logger.info(
                f"[qiao-residual-ls {family}/{state_mode}/cost={cost_bps:g} ep {epoch+1:03d}] "
                f"train_local_mse={acc['local_mse']:.4f} val_score={val_score:.4f} val_cvar95={val_metrics.get('cvar_95', np.nan):.4f}"
            )
        if val_score < best_val - cfg.min_delta:
            best_val = val_score
            best_state = copy.deepcopy(agent.state_dict())
            bad_epochs = 0
            torch.save({"agent": best_state, "epoch": epoch + 1, "best_val_score": best_val, "best_val_metrics": val_metrics}, os.path.join(out_dir, "best_policy.pt"))
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                logger.info(f"Early stopping QiaoResidual-LS baseline at epoch {epoch+1}; best validation score={best_val:.6f}")
                break
    if best_state is not None:
        agent.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "training_history.csv"), index=False)
    test_metrics, test_panel = evaluate_qiao_residual_ls_agent(agent, test_ld, cfg, cost_bps, "QiaoResidual-LS", family, state_mode)
    test_panel.to_csv(os.path.join(out_dir, "test_episode_panel.csv"), index=False)
    pd.DataFrame([test_metrics]).to_csv(os.path.join(out_dir, "test_metrics.csv"), index=False)
    return test_metrics, test_panel

def run_hedgeset_modern_baselines(
    episodes_path: str,
    family: str,
    cost_bps: float,
    cfg: HedgeSetConfig,
    action_scopes: List[str],
) -> Tuple[List[dict], List[pd.DataFrame]]:
    names = _normalize_modern_baseline_names(getattr(cfg, "modern_baselines", []))
    if not names:
        return [], []
    scopes = [s for s in action_scopes if str(s).lower().strip() in {"full", "main", "all"}]
    if not scopes:
        scopes = ["full"]
    metrics: List[dict] = []
    panels: List[pd.DataFrame] = []
    for scope in scopes:
        for sd in cfg.seeds:
            if "ivgrid" in names:
                mode = str(getattr(cfg, "ivgrid_baseline_state_mode", "grid"))
                try:
                    _ = HedgeSetEpisodeDataset(episodes_path, "train", mode)
                    cfg_iv = _apply_hedgeset_policy_variant(cfg, "full")
                    cfg_iv.model_label_override = "IVGrid-HedgeSet"
                    out_dir = os.path.join(cfg.results_dir, family, f"cost_{cost_bps:g}bps", "modern_baselines", "ivgrid", f"scope_{scope}", f"seed_{sd}")
                    m, p = train_hedgeset_policy_one(episodes_path, family, mode, cost_bps, cfg_iv, out_dir, seed=int(sd), policy_variant="ivgrid", action_scope=scope, candidate_run=getattr(cfg, "run_name", "main"), rules_tag=getattr(cfg, "candidate_rules_tag", ""))
                    metrics.append(m); panels.append(p)
                except Exception as exc:
                    logger.warning(f"Skipping IVGrid-HedgeSet baseline for family={family}: {exc}")
            if "qiao_residual_ls" in names:
                mode = str(getattr(cfg, "qiao_residual_state_mode", "latent"))
                try:
                    _ = HedgeSetEpisodeDataset(episodes_path, "train", mode)
                    cfg_qiao = copy.deepcopy(cfg)
                    cfg_qiao.current_policy_variant = "qiao_residual_ls"
                    cfg_qiao.qiao_residual_base = str(getattr(cfg, "qiao_residual_base", "ls_deltavega"))
                    cfg_qiao.factor_neutral_residual = False
                    cfg_qiao.use_global_trust_gate = False
                    cfg_qiao.lambda_factor_neutral = 0.0
                    cfg_qiao.lambda_trust_l1 = 0.0
                    cfg_qiao.lambda_aux_risk = 0.0
                    cfg_qiao.lambda_regime_excess_cvar = 0.0
                    cfg_qiao.lambda_excess_cvar = 0.0
                    cfg_qiao.lambda_excess_relu = 0.0
                    cfg_qiao.imitation_epochs = 0
                    cfg_qiao.finetune_imitation_lambda = 0.0
                    out_dir = os.path.join(cfg.results_dir, family, f"cost_{cost_bps:g}bps", "modern_baselines", "qiao_residual_ls", f"scope_{scope}", f"seed_{sd}")
                    m, p = train_qiao_residual_ls_baseline_one(episodes_path, family, mode, cost_bps, cfg_qiao, out_dir, seed=int(sd), action_scope=scope, candidate_run=getattr(cfg, "run_name", "main"), rules_tag=getattr(cfg, "candidate_rules_tag", ""))
                    metrics.append(m); panels.append(p)
                except Exception as exc:
                    logger.warning(f"Skipping QiaoResidual-LS baseline for family={family}: {exc}")
            if "ntb_ls" in names:
                mode = str(getattr(cfg, "ntb_ls_state_mode", "summary"))
                try:
                    _ = HedgeSetEpisodeDataset(episodes_path, "train", mode)
                    cfg_ntb = copy.deepcopy(cfg)
                    cfg_ntb.current_policy_variant = "ntb_ls"
                    cfg_ntb.residual_base = "ls_deltavega"
                    cfg_ntb.factor_neutral_residual = False
                    cfg_ntb.use_global_trust_gate = False
                    cfg_ntb.lambda_factor_neutral = 0.0
                    cfg_ntb.lambda_trust_l1 = 0.0
                    cfg_ntb.lambda_aux_risk = 0.0
                    cfg_ntb.imitation_epochs = 0
                    out_dir = os.path.join(cfg.results_dir, family, f"cost_{cost_bps:g}bps", "modern_baselines", "ntb_ls", f"scope_{scope}", f"seed_{sd}")
                    m, p = train_ntb_ls_baseline_one(episodes_path, family, mode, cost_bps, cfg_ntb, out_dir, seed=int(sd), action_scope=scope, candidate_run=getattr(cfg, "run_name", "main"), rules_tag=getattr(cfg, "candidate_rules_tag", ""))
                    metrics.append(m); panels.append(p)
                except Exception as exc:
                    logger.warning(f"Skipping NTB-LS baseline for family={family}: {exc}")
            if "cvar_from_scratch" in names:
                mode = str(getattr(cfg, "cvar_from_scratch_state_mode", "latent"))
                try:
                    _ = HedgeSetEpisodeDataset(episodes_path, "train", mode)
                    cfg_cv = _apply_hedgeset_policy_variant(cfg, "from_scratch")
                    cfg_cv.model_label_override = "CVaR-HedgeSet"
                    cfg_cv.residual_base = "none"
                    cfg_cv.factor_neutral_residual = False
                    cfg_cv.use_global_trust_gate = False
                    cfg_cv.lambda_factor_neutral = 0.0
                    cfg_cv.lambda_trust_l1 = 0.0
                    cfg_cv.lambda_aux_risk = 0.0
                    cfg_cv.lambda_regime_excess_cvar = 0.0
                    cfg_cv.lambda_excess_cvar = 0.0
                    cfg_cv.lambda_excess_relu = 0.0
                    cfg_cv.lambda_residual_l2 = 0.0
                    cfg_cv.lambda_residual_trade = 0.0
                    cfg_cv.imitation_epochs = 0
                    cfg_cv.finetune_imitation_lambda = 0.0
                    # This baseline is a pure tail-risk RL/deep-hedging baseline:
                    # select checkpoints by absolute policy loss, not by excess
                    # loss versus a hard classical base.
                    cfg_cv.val_score_excess_cvar95_weight = 0.0
                    cfg_cv.val_score_excess_cvar99_weight = 0.0
                    cfg_cv.val_score_excess_relu_weight = 0.0
                    cfg_cv.val_score_worst_regime_excess_weight = 0.0
                    cfg_cv.val_score_factor_weight = 0.0
                    cfg_cv.val_score_trust_weight = 0.0
                    out_dir = os.path.join(cfg.results_dir, family, f"cost_{cost_bps:g}bps", "modern_baselines", "cvar_from_scratch", f"scope_{scope}", f"seed_{sd}")
                    m, p = train_hedgeset_policy_one(episodes_path, family, mode, cost_bps, cfg_cv, out_dir, seed=int(sd), policy_variant="cvar_from_scratch", action_scope=scope, candidate_run=getattr(cfg, "run_name", "main"), rules_tag=getattr(cfg, "candidate_rules_tag", ""))
                    metrics.append(m); panels.append(p)
                except Exception as exc:
                    logger.warning(f"Skipping CVaR-HedgeSet-from-scratch baseline for family={family}: {exc}")
    return metrics, panels

def _select_no_trade_delta_band(episodes_path: str, state_mode_for_loader: str, cost_bps: float, cfg: HedgeSetConfig) -> float:
    """Select the no-trade delta threshold on the validation split by CVaR95."""
    val_ds = HedgeSetEpisodeDataset(episodes_path, "val", state_mode_for_loader)
    val_ld = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=hedgeset_collate_fn)
    best_band = float(getattr(cfg, "delta_no_trade_band", 0.10))
    best_score = float("inf")
    for band in list(getattr(cfg, "delta_no_trade_band_grid", [0.0, 0.05, 0.10, 0.20])):
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.delta_no_trade_band = float(band)
        cfg_tmp.current_policy_variant = "static"
        cfg_tmp.current_seed = np.nan
        m, _ = evaluate_hedgeset_agent(None, val_ld, cfg_tmp, cost_bps, "NoTradeDelta-val", val_ds.episodes[0]["family"], "static", mode="delta_nt")
        score = float(m.get("cvar_95", np.inf))
        if score < best_score:
            best_score = score
            best_band = float(band)
    return best_band


def evaluate_static_hedgeset_baselines(episodes_path: str, family: str, state_mode_for_loader: str, cost_bps: float, cfg: HedgeSetConfig, out_dir: str, delta_thresholds: Optional[List[float]] = None) -> Tuple[List[dict], pd.DataFrame]:
    """Evaluate stronger static baselines on the test split.

    Includes:
      * NoHedge
      * SmiledDelta: daily underlying delta hedge
      * NoTradeDelta: validation-selected no-trade delta band
      * GreedyDeltaVega: simple one-option vega heuristic
      * LSDeltaVega: least-squares delta-vega hedge over the full candidate set
    """
    os.makedirs(out_dir, exist_ok=True)
    test_ds = HedgeSetEpisodeDataset(episodes_path, "test", state_mode_for_loader)
    test_ld = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=hedgeset_collate_fn)
    metrics, panels = [], []

    # Select no-trade delta threshold on validation only.
    if delta_thresholds is not None and len(delta_thresholds) > 0:
        cfg.delta_no_trade_band_grid = [float(x) for x in delta_thresholds]
    nt_band = _select_no_trade_delta_band(episodes_path, state_mode_for_loader, cost_bps, cfg)
    cfg.delta_no_trade_band = float(nt_band)
    pd.DataFrame([{"selected_delta_no_trade_band": nt_band, "cost_bps": cost_bps, "family": family}]).to_csv(
        os.path.join(out_dir, "no_trade_delta_selected_band.csv"), index=False
    )

    static_specs = [
        ("NoHedge", "no_hedge"),
        ("SmiledDelta", "delta"),
        ("NoTradeDelta", "delta_nt"),
        ("GreedyDeltaVega", "deltavega"),
        ("LSDeltaVega", "ls_deltavega"),
    ]
    cfg.current_policy_variant = "static"
    cfg.current_seed = np.nan
    # Static baselines should always use the candidate set defined by the
    # current sweep.  Do not inherit a neural policy action-scope ablation
    # such as underlying-only from a previous run.
    cfg.current_action_scope = "full"
    for label, mode in static_specs:
        m, p = evaluate_hedgeset_agent(None, test_ld, cfg, cost_bps, label, family, "static", mode=mode)
        metrics.append(m); panels.append(p)
    all_panel = pd.concat(panels, ignore_index=True)
    all_panel.to_csv(os.path.join(out_dir, "static_baseline_episode_panel.csv"), index=False)
    pd.DataFrame(metrics).to_csv(os.path.join(out_dir, "static_baseline_metrics.csv"), index=False)
    return metrics, all_panel

def _sanitize_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(text)).strip("_") or "run"


def _hedgeset_preset_rules(name: str, default_rules: List[str]) -> Tuple[str, List[str], Optional[int], Optional[int]]:
    """Return (run_name, candidate_rules, max_instruments, min_instruments)."""
    key = str(name).strip().lower()
    if key in {"main", "full", "default"}:
        return "full", list(default_rules), None, None
    if key in {"underlying", "underlying_only", "uonly", "u_only"}:
        return "underlying_only", ["underlying"], 1, 1
    if key in {"liquid", "liquid_options", "u_liquid"}:
        return "liquid", ["underlying", "atm_call30", "atm_put30", "put25_30", "call25_30", "atm_straddle30"], 8, 2
    if key in {"tail", "tailhedge", "puts"}:
        return "tail_puts", ["underlying", "put25_30", "put10_30", "put25_60", "put10_60"], 8, 2
    raise ValueError(f"Unknown HedgeSet candidate sweep preset: {name}")


def _apply_hedgeset_policy_variant(cfg: HedgeSetConfig, variant: str) -> HedgeSetConfig:
    v = str(variant).strip().lower()
    cfg_v = copy.deepcopy(cfg)
    cfg_v.current_policy_variant = v
    if v in {"full", "main", "default"}:
        cfg_v.use_no_trade_band = True
        cfg_v.use_sparse_gate = True
        cfg_v.lambda_sparse = cfg.lambda_sparse
    elif v in {"no_band", "raw_target"}:
        cfg_v.use_no_trade_band = False
        cfg_v.use_sparse_gate = True
        cfg_v.lambda_sparse = cfg.lambda_sparse
    elif v in {"absolute", "no_residual", "from_scratch"}:
        cfg_v.use_no_trade_band = True
        cfg_v.use_sparse_gate = True
        cfg_v.lambda_sparse = cfg.lambda_sparse
        cfg_v.residual_base = "none"
        cfg_v.imitation_epochs = 0
        cfg_v.finetune_imitation_lambda = 0.0
    elif v in {"no_gate", "no_sparse_gate"}:
        cfg_v.use_no_trade_band = True
        cfg_v.use_sparse_gate = False
        cfg_v.lambda_sparse = 0.0
    elif v in {"no_band_no_gate", "raw_no_gate"}:
        cfg_v.use_no_trade_band = False
        cfg_v.use_sparse_gate = False
        cfg_v.lambda_sparse = 0.0
    elif v in {"amb", "ambiguous", "amb_full", "robust", "ambiguity"}:
        cfg_v.use_no_trade_band = True
        cfg_v.use_sparse_gate = True
        cfg_v.lambda_sparse = cfg.lambda_sparse
        cfg_v.ambiguity_aware = True
    elif v in {"amb_no_band", "ambiguous_no_band"}:
        cfg_v.use_no_trade_band = False
        cfg_v.use_sparse_gate = True
        cfg_v.lambda_sparse = cfg.lambda_sparse
        cfg_v.ambiguity_aware = True
    elif v in {"amb_no_gate", "ambiguous_no_gate"}:
        cfg_v.use_no_trade_band = True
        cfg_v.use_sparse_gate = False
        cfg_v.lambda_sparse = 0.0
        cfg_v.ambiguity_aware = True
    elif v in {"amb_no_band_no_gate", "ambiguous_no_band_no_gate"}:
        cfg_v.use_no_trade_band = False
        cfg_v.use_sparse_gate = False
        cfg_v.lambda_sparse = 0.0
        cfg_v.ambiguity_aware = True
    else:
        raise ValueError(f"Unknown HedgeSet policy variant: {variant}")
    return cfg_v


def _hedgeset_seed_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty or "seed" not in metrics_df.columns:
        return pd.DataFrame()
    dyn = metrics_df[pd.to_numeric(metrics_df["seed"], errors="coerce").notna()].copy()
    if dyn.empty:
        return pd.DataFrame()
    group_cols = [c for c in ["candidate_run", "family", "cost_bps", "model", "state_mode", "policy_variant", "action_scope", "residual_base"] if c in dyn.columns]
    metric_cols = [c for c in [
        "mean_loss_norm", "cvar_95", "cvar_99", "shortfall_prob",
        "worst_regime_cvar_95", "worst_regime_shortfall_prob", "worst_regime_excess_cvar_95",
        "mean_base_loss_norm", "mean_excess_loss_norm", "mean_relu_excess_loss_norm",
        "excess_cvar_95", "excess_cvar_99",
        "mean_residual_l2", "mean_residual_turnover",
        "mean_abs_residual_delta_exposure", "mean_abs_residual_vega_exposure", "mean_residual_factor_l2",
        "mean_global_trust", "q95_global_trust",
        "mean_ntb_band", "mean_ntb_gate",
        "mean_local_mse",
        "mean_tc_norm", "mean_turnover", "mean_abs_position"
    ] if c in dyn.columns]
    rows = []
    for key, g in dyn.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row["n_seeds"] = int(g["seed"].nunique())
        for c in metric_cols:
            vals = pd.to_numeric(g[c], errors="coerce").values
            row[f"{c}_mean"] = float(np.nanmean(vals))
            row[f"{c}_std"] = float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def run_hedgeset_pipeline(
    families: List[str],
    state_modes: List[str],
    cost_bps_list: List[float],
    candidate_rules: List[str],
    horizon: int = 5,
    stride: int = 5,
    checkpoint_preference: str = "forecast",
    rebuild_chains: bool = False,
    rebuild_states: bool = False,
    rebuild_episodes: bool = False,
    seed: int = 42,
    seeds: Optional[List[int]] = None,
    policy_variants: Optional[List[str]] = None,
    action_scopes: Optional[List[str]] = None,
    bootstrap_metrics: Optional[List[str]] = None,
    delta_thresholds: Optional[List[float]] = None,
    setonly_checkpoint: Optional[str] = None,
    run_name: str = "main",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    patience: Optional[int] = None,
    max_instruments: Optional[int] = None,
    min_instruments: Optional[int] = None,
    terminal_liquidation: bool = True,
    ambiguity_lambda: Optional[float] = None,
    ambiguity_eta: Optional[float] = None,
    ambiguity_min_count: Optional[int] = None,
    regime_bins: Optional[int] = None,
    residual_base: Optional[str] = None,
    residual_scale: Optional[float] = None,
    imitation_epochs: Optional[int] = None,
    imitation_lambda: Optional[float] = None,
    finetune_imitation_lambda: Optional[float] = None,
    lambda_excess_cvar: Optional[float] = None,
    lambda_excess_relu: Optional[float] = None,
    lambda_residual_l2: Optional[float] = None,
    lambda_residual_trade: Optional[float] = None,
    factor_neutral_residual: Optional[bool] = None,
    factor_neutral_ridge: Optional[float] = None,
    lambda_factor_neutral: Optional[float] = None,
    use_global_trust_gate: Optional[bool] = None,
    trust_gate_init_bias: Optional[float] = None,
    trust_gate_temp: Optional[float] = None,
    lambda_trust_l1: Optional[float] = None,
    lambda_regime_excess_cvar: Optional[float] = None,
    lambda_aux_risk: Optional[float] = None,
    val_score_worst_regime_excess_weight: Optional[float] = None,
    val_feasible_excess_cvar95_max: Optional[float] = None,
    val_feasible_excess_cvar95_grid: Optional[List[float]] = None,
    val_feasible_turnover_max: Optional[float] = None,
    val_feasible_tc_norm_max: Optional[float] = None,
    val_feasible_delta_exposure_max: Optional[float] = None,
    val_feasible_vega_exposure_max: Optional[float] = None,
    val_score_excess_cvar95_weight: Optional[float] = None,
    val_score_excess_cvar99_weight: Optional[float] = None,
    val_score_excess_relu_weight: Optional[float] = None,
    modern_baselines: Optional[List[str]] = None,
    ivgrid_state_mode: str = "grid",
    ntb_state_mode: str = "summary",
    cvar_from_scratch_state_mode: str = "latent",
    ntb_band_max: Optional[float] = None,
    ntb_lambda_band_l1: Optional[float] = None,
    qiao_state_mode: str = "latent",
    qiao_residual_scale: Optional[float] = None,
    qiao_lambda_local_mse: Optional[float] = None,
    qiao_lambda_residual_l2: Optional[float] = None,
    qiao_lambda_residual_trade: Optional[float] = None,
) -> None:
    # HedgeSet policies operate on many small tensors.  On CPU, excessive BLAS
    # threading can make these small matrix operations dramatically slower.
    if not torch.cuda.is_available():
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    cfg = HedgeSetConfig()
    cfg.horizon = int(horizon)
    cfg.start_stride = int(stride)
    cfg.families = families
    cfg.state_modes = state_modes
    cfg.cost_bps_list = cost_bps_list
    cfg.candidate_rules = candidate_rules
    cfg.terminal_liquidation = bool(terminal_liquidation)
    if ambiguity_lambda is not None:
        cfg.lambda_ambiguity = float(ambiguity_lambda)
    if ambiguity_eta is not None:
        cfg.ambiguity_eta = float(ambiguity_eta)
    if ambiguity_min_count is not None:
        cfg.ambiguity_min_count = int(ambiguity_min_count)
    if regime_bins is not None:
        cfg.regime_bins = int(regime_bins)
    if residual_base is not None:
        cfg.residual_base = str(residual_base)
    if residual_scale is not None:
        cfg.residual_scale = float(residual_scale)
    if imitation_epochs is not None:
        cfg.imitation_epochs = int(imitation_epochs)
    if imitation_lambda is not None:
        cfg.imitation_lambda = float(imitation_lambda)
    if finetune_imitation_lambda is not None:
        cfg.finetune_imitation_lambda = float(finetune_imitation_lambda)
    if lambda_excess_cvar is not None:
        cfg.lambda_excess_cvar = float(lambda_excess_cvar)
    if lambda_excess_relu is not None:
        cfg.lambda_excess_relu = float(lambda_excess_relu)
    if lambda_residual_l2 is not None:
        cfg.lambda_residual_l2 = float(lambda_residual_l2)
    if lambda_residual_trade is not None:
        cfg.lambda_residual_trade = float(lambda_residual_trade)
    if factor_neutral_residual is not None:
        cfg.factor_neutral_residual = bool(factor_neutral_residual)
    if factor_neutral_ridge is not None:
        cfg.factor_neutral_ridge = float(factor_neutral_ridge)
    if lambda_factor_neutral is not None:
        cfg.lambda_factor_neutral = float(lambda_factor_neutral)
    if use_global_trust_gate is not None:
        cfg.use_global_trust_gate = bool(use_global_trust_gate)
    if trust_gate_init_bias is not None:
        cfg.trust_gate_init_bias = float(trust_gate_init_bias)
    if trust_gate_temp is not None:
        cfg.trust_gate_temp = float(trust_gate_temp)
    if lambda_trust_l1 is not None:
        cfg.lambda_trust_l1 = float(lambda_trust_l1)
    if lambda_regime_excess_cvar is not None:
        cfg.lambda_regime_excess_cvar = float(lambda_regime_excess_cvar)
    if lambda_aux_risk is not None:
        cfg.lambda_aux_risk = float(lambda_aux_risk)
    if val_score_worst_regime_excess_weight is not None:
        cfg.val_score_worst_regime_excess_weight = float(val_score_worst_regime_excess_weight)
    if val_feasible_excess_cvar95_max is not None:
        cfg.val_feasible_excess_cvar95_max = float(val_feasible_excess_cvar95_max)
    if val_feasible_excess_cvar95_grid:
        cfg.val_feasible_excess_cvar95_grid = [float(x) for x in val_feasible_excess_cvar95_grid]
    if val_feasible_turnover_max is not None:
        cfg.val_feasible_turnover_max = float(val_feasible_turnover_max)
    if val_feasible_tc_norm_max is not None:
        cfg.val_feasible_tc_norm_max = float(val_feasible_tc_norm_max)
    if val_feasible_delta_exposure_max is not None:
        cfg.val_feasible_delta_exposure_max = float(val_feasible_delta_exposure_max)
    if val_feasible_vega_exposure_max is not None:
        cfg.val_feasible_vega_exposure_max = float(val_feasible_vega_exposure_max)
    if val_score_excess_cvar95_weight is not None:
        cfg.val_score_excess_cvar95_weight = float(val_score_excess_cvar95_weight)
    if val_score_excess_cvar99_weight is not None:
        cfg.val_score_excess_cvar99_weight = float(val_score_excess_cvar99_weight)
    if val_score_excess_relu_weight is not None:
        cfg.val_score_excess_relu_weight = float(val_score_excess_relu_weight)
    cfg.modern_baselines = _normalize_modern_baseline_names(modern_baselines or [])
    cfg.ivgrid_baseline_state_mode = str(ivgrid_state_mode)
    cfg.ntb_ls_state_mode = str(ntb_state_mode)
    cfg.cvar_from_scratch_state_mode = str(cvar_from_scratch_state_mode)
    cfg.qiao_residual_state_mode = str(qiao_state_mode)
    if qiao_residual_scale is not None:
        cfg.qiao_residual_scale = float(qiao_residual_scale)
    if qiao_lambda_local_mse is not None:
        cfg.qiao_lambda_local_mse = float(qiao_lambda_local_mse)
    if qiao_lambda_residual_l2 is not None:
        cfg.qiao_lambda_residual_l2 = float(qiao_lambda_residual_l2)
    if qiao_lambda_residual_trade is not None:
        cfg.qiao_lambda_residual_trade = float(qiao_lambda_residual_trade)
    if ntb_band_max is not None:
        cfg.ntb_ls_band_max = float(ntb_band_max)
    if ntb_lambda_band_l1 is not None:
        cfg.ntb_ls_lambda_band_l1 = float(ntb_lambda_band_l1)
    cfg.run_name = _sanitize_tag(run_name)
    cfg.setonly_checkpoint_path = str(setonly_checkpoint or "")
    cfg.require_setonly_state = any(str(m).lower().strip() in {"setonly", "nosurface", "no_surface"} for m in state_modes)
    if bootstrap_metrics is not None and len(bootstrap_metrics) > 0:
        cfg.bootstrap_stats = [str(x).strip() for x in bootstrap_metrics if str(x).strip()]
    if epochs is not None: cfg.epochs = int(epochs)
    if batch_size is not None: cfg.batch_size = int(batch_size)
    if lr is not None: cfg.lr = float(lr)
    if patience is not None: cfg.patience = int(patience)
    if max_instruments is not None: cfg.max_instruments = int(max_instruments)
    if min_instruments is not None: cfg.min_instruments = int(min_instruments)
    if len([r for r in cfg.candidate_rules if str(r).strip().lower() != "underlying"]) == 0:
        cfg.max_instruments = 1
        cfg.min_instruments = 1
    cfg.seeds = list(seeds if seeds is not None else [seed])
    cfg.policy_variants = list(policy_variants if policy_variants is not None else ["full"])
    action_scopes = [str(x).lower().strip() for x in (action_scopes or ["full", "underlying"]) if str(x).strip()]
    bootstrap_metrics = [str(x).lower().strip() for x in (bootstrap_metrics or ["mean_loss", "cvar_95", "shortfall_prob"]) if str(x).strip()]
    if setonly_checkpoint:
        cfg.setonly_checkpoint_path = str(setonly_checkpoint)
    if not cfg.state_modes:
        raise ValueError("At least one hedge state mode is required for --hedgeset.")

    # Non-overwriting output path: candidate-run name + deterministic rule tag.
    cfg.candidate_rules_tag = _stable_hedgeset_rules_tag(cfg)
    base_results_dir = os.path.join(DATA_CFG.results_dir, "hedgeset")
    cfg.results_dir = os.path.join(base_results_dir, f"{cfg.run_name}_{cfg.candidate_rules_tag}")
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.cache_dir, exist_ok=True)
    with open(os.path.join(cfg.results_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "run_name": cfg.run_name,
            "candidate_rules": cfg.candidate_rules,
            "candidate_rules_tag": cfg.candidate_rules_tag,
            "families": families,
            "state_modes": state_modes,
            "cost_bps_list": cost_bps_list,
            "seeds": cfg.seeds,
            "policy_variants": cfg.policy_variants,
            "horizon": cfg.horizon,
            "stride": cfg.start_stride,
            "max_instruments": cfg.max_instruments,
            "min_instruments": cfg.min_instruments,
            "lambda_ambiguity": cfg.lambda_ambiguity,
            "ambiguity_eta": cfg.ambiguity_eta,
            "ambiguity_min_count": cfg.ambiguity_min_count,
            "regime_bins": cfg.regime_bins,
            "residual_base": cfg.residual_base,
            "residual_scale": cfg.residual_scale,
            "imitation_epochs": cfg.imitation_epochs,
            "imitation_lambda": cfg.imitation_lambda,
            "finetune_imitation_lambda": cfg.finetune_imitation_lambda,
            "lambda_excess_cvar": cfg.lambda_excess_cvar,
            "lambda_excess_relu": cfg.lambda_excess_relu,
            "lambda_residual_l2": cfg.lambda_residual_l2,
            "lambda_residual_trade": cfg.lambda_residual_trade,
            "factor_neutral_residual": cfg.factor_neutral_residual,
            "factor_neutral_ridge": cfg.factor_neutral_ridge,
            "lambda_factor_neutral": cfg.lambda_factor_neutral,
            "use_global_trust_gate": cfg.use_global_trust_gate,
            "trust_gate_init_bias": cfg.trust_gate_init_bias,
            "trust_gate_temp": cfg.trust_gate_temp,
            "lambda_trust_l1": cfg.lambda_trust_l1,
            "lambda_regime_excess_cvar": cfg.lambda_regime_excess_cvar,
            "lambda_aux_risk": cfg.lambda_aux_risk,
            "val_feasible_excess_cvar95_max": cfg.val_feasible_excess_cvar95_max,
            "val_feasible_excess_cvar95_grid": cfg.val_feasible_excess_cvar95_grid,
            "val_feasible_turnover_max": cfg.val_feasible_turnover_max,
            "val_feasible_tc_norm_max": cfg.val_feasible_tc_norm_max,
            "val_feasible_delta_exposure_max": cfg.val_feasible_delta_exposure_max,
            "val_feasible_vega_exposure_max": cfg.val_feasible_vega_exposure_max,
            "modern_baselines": cfg.modern_baselines,
            "ivgrid_baseline_state_mode": cfg.ivgrid_baseline_state_mode,
            "ntb_ls_state_mode": cfg.ntb_ls_state_mode,
            "cvar_from_scratch_state_mode": cfg.cvar_from_scratch_state_mode,
            "ntb_ls_band_max": cfg.ntb_ls_band_max,
            "ntb_ls_lambda_band_l1": cfg.ntb_ls_lambda_band_l1,
            "qiao_residual_state_mode": cfg.qiao_residual_state_mode,
            "qiao_residual_base": cfg.qiao_residual_base,
            "qiao_residual_scale": cfg.qiao_residual_scale,
            "qiao_lambda_local_mse": cfg.qiao_lambda_local_mse,
            "qiao_lambda_residual_l2": cfg.qiao_lambda_residual_l2,
            "qiao_lambda_residual_trade": cfg.qiao_lambda_residual_trade,
            "val_score_weights": {
                "cvar95": cfg.val_score_cvar95_weight,
                "cvar99": cfg.val_score_cvar99_weight,
                "mean": cfg.val_score_mean_weight,
                "turnover": cfg.val_score_turnover_weight,
                "tc": cfg.val_score_tc_weight,
                "worst_regime": cfg.val_score_worst_regime_weight,
                "excess_cvar95": cfg.val_score_excess_cvar95_weight,
                "excess_cvar99": cfg.val_score_excess_cvar99_weight,
                "excess_relu": cfg.val_score_excess_relu_weight,
                "worst_regime_excess": cfg.val_score_worst_regime_excess_weight,
            },
        }, f, indent=2)

    cache_path = os.path.join(DATA_CFG.cache_dir, "daily_quotes.pkl")
    if not os.path.exists(cache_path):
        logger.info("Forecasting cache missing; preprocessing first.")
        preprocess_all(DATA_CFG, force_rebuild=False)

    # Reuse exact-contract chain table and daily state cache from the underlying-only extension.
    hedge_chain_cfg = HedgingConfig()
    hedge_chain_cfg.cache_dir = cfg.cache_dir
    hedge_chain_cfg.results_dir = cfg.results_dir
    hedge_chain_cfg.horizon = cfg.horizon
    hedge_chain_cfg.start_stride = cfg.start_stride
    hedge_chain_cfg.target_tau_days = cfg.target_tau_days
    hedge_chain_cfg.tau_tol_days = cfg.tau_tol_days
    hedge_chain_cfg.min_liability_bid = cfg.min_liability_bid
    hedge_chain_cfg.max_liability_rel_spread = cfg.max_liability_rel_spread
    hedge_chain_cfg.max_path_rel_spread = cfg.max_path_rel_spread
    hedge_chain_cfg.min_path_bid = cfg.min_path_bid
    hedge_chain_cfg.path_k_min = cfg.path_k_min
    hedge_chain_cfg.path_k_max = cfg.path_k_max
    hedge_chain_cfg.setonly_checkpoint_path = getattr(cfg, "setonly_checkpoint_path", "")
    hedge_chain_cfg.require_setonly_state = any(str(m).lower() in {"setonly", "nosurface", "no_surface"} for m in state_modes)
    setattr(hedge_chain_cfg, "require_setonly_state", cfg.require_setonly_state)
    setattr(hedge_chain_cfg, "setonly_checkpoint_path", cfg.setonly_checkpoint_path)
    chains_path = preprocess_hedging_chains(DATA_CFG, hedge_chain_cfg, force_rebuild=rebuild_chains)
    state_cache_path = build_hedging_state_cache(cache_path, DATA_CFG, MODEL_CFG, hedge_chain_cfg, checkpoint_preference=checkpoint_preference, force_rebuild=bool(rebuild_states or hedge_chain_cfg.require_setonly_state))
    rebuild_episodes_effective = bool(rebuild_episodes or rebuild_chains or rebuild_states or hedge_chain_cfg.require_setonly_state)

    all_metrics, all_panels, all_boot = [], [], []
    for family in families:
        ep_path = build_hedgeset_episodes(chains_path, state_cache_path, family, DATA_CFG, cfg, force_rebuild=rebuild_episodes_effective)
        for cost_bps in cost_bps_list:
            static_dir = os.path.join(cfg.results_dir, family, f"cost_{cost_bps:g}bps", "static")
            cfg.current_seed = np.nan
            cfg.current_policy_variant = "static"
            static_metrics, static_panel = evaluate_static_hedgeset_baselines(ep_path, family, state_modes[0], cost_bps, cfg, static_dir, delta_thresholds=delta_thresholds)
            all_metrics.extend(static_metrics); all_panels.append(static_panel)

            policy_panels = []
            for variant in cfg.policy_variants:
                for mode in state_modes:
                    # If no set-only checkpoint exists, this state mode will be unavailable.
                    try:
                        _ = HedgeSetEpisodeDataset(ep_path, "train", mode)
                    except Exception as exc:
                        logger.warning(f"Skipping HedgeSet state mode={mode}: {exc}")
                        continue
                    variants_for_mode = [variant] if mode == "latent" else (["full"] if variant == cfg.policy_variants[0] else [])
                    for v in variants_for_mode:
                        for scope in action_scopes:
                            if scope in {"underlying", "uonly", "underlying_only"} and mode != "latent":
                                continue
                            for sd in cfg.seeds:
                                cfg_var = _apply_hedgeset_policy_variant(cfg, v)
                                out_dir = os.path.join(cfg.results_dir, family, f"cost_{cost_bps:g}bps", f"variant_{v}", f"scope_{scope}", f"policy_{mode}", f"seed_{sd}")
                                m, p = train_hedgeset_policy_one(ep_path, family, mode, cost_bps, cfg_var, out_dir, seed=int(sd), policy_variant=v, action_scope=scope, candidate_run=getattr(cfg, "run_name", "main"), rules_tag=getattr(cfg, "candidate_rules_tag", ""))
                                all_metrics.append(m); all_panels.append(p); policy_panels.append(p)

            if cfg.modern_baselines:
                mb_metrics, mb_panels = run_hedgeset_modern_baselines(ep_path, family, cost_bps, cfg, action_scopes)
                all_metrics.extend(mb_metrics)
                all_panels.extend(mb_panels)
                policy_panels.extend(mb_panels)

            panel_here = pd.concat([static_panel] + policy_panels, ignore_index=True) if policy_panels else static_panel.copy()
            # Pairwise bootstrap for tail-risk and turnover claims.  Comparisons
            # are run within each seed/variant when applicable.
            for variant in sorted(set(panel_here.get("policy_variant", pd.Series(["full"])).dropna())):
                amb_variants_loop = {"amb", "ambiguous", "amb_full", "robust", "ambiguity"}
                modern_comp_models = {"IVGrid-HedgeSet", "NTB-LS", "CVaR-HedgeSet", "QiaoResidual-LS"}
                if "policy_variant" in panel_here and str(variant) in amb_variants_loop:
                    sub_v = panel_here[panel_here["policy_variant"].isin([variant, "full", "static"]) | panel_here["model"].isin(modern_comp_models)]
                elif str(variant) == "full" and "policy_variant" in panel_here:
                    sub_v = panel_here[panel_here["policy_variant"].isin(["full", "static"]) | panel_here["model"].isin(modern_comp_models)]
                else:
                    sub_v = panel_here[panel_here.get("policy_variant", "static").isin([variant, "static"])] if "policy_variant" in panel_here else panel_here
                for sd in sorted(set(pd.to_numeric(sub_v.get("seed", pd.Series([np.nan])), errors="coerce").dropna())):
                    sub = sub_v[(pd.to_numeric(sub_v.get("seed", pd.Series(np.nan)), errors="coerce").eq(sd)) | (sub_v.get("seed", pd.Series(np.nan)).isna())]
                    amb_variants = {"amb", "ambiguous", "amb_full", "robust", "ambiguity"}
                    if str(variant) in amb_variants:
                        model_a = "AmbHedgeSet-latent"
                        suff = ""
                        # Compare the ambiguity-aware model against the standard
                        # full HedgeSet as well as classical and representation baselines.
                        comps = [
                            "HedgeSet-latent", "SmiledDelta", "NoTradeDelta", "LSDeltaVega", "GreedyDeltaVega",
                            "IVGrid-HedgeSet", "NTB-LS", "CVaR-HedgeSet", "QiaoResidual-LS",
                            "HedgeSet-summary", "HedgeSet-grid", "HedgeSet-setonly", "HedgeSet-nosurface",
                        ]
                    else:
                        model_a = "HedgeSet-latent" if variant == "full" else f"HedgeSet-latent-{variant}"
                        suff = "" if variant == "full" else f"-{variant}"
                        comps = [
                            "SmiledDelta", "NoTradeDelta", "LSDeltaVega", "GreedyDeltaVega",
                            "IVGrid-HedgeSet", "NTB-LS", "CVaR-HedgeSet", "QiaoResidual-LS",
                            f"HedgeSet-summary{suff}", f"HedgeSet-grid{suff}",
                            f"HedgeSet-setonly{suff}", f"HedgeSet-nosurface{suff}",
                        ]
                    if model_a not in set(sub["model"]):
                        continue
                    # Additional latent-policy architecture ablations are only
                    # meaningful when the target model is the full latent policy.
                    if variant == "full":
                        comps.extend(["HedgeSet-latent-no_band", "HedgeSet-latent-no_gate", "HedgeSet-latent-no_band_no_gate", "HedgeSet-latent-underlying"])
                    if str(variant) in amb_variants:
                        comps.extend(["HedgeSet-latent-no_band", "HedgeSet-latent-no_gate", "HedgeSet-latent-underlying"])
                    for comp in comps:
                        if comp not in set(sub["model"]):
                            continue
                        for metric_name in cfg.bootstrap_stats:
                            row = paired_bootstrap_hedging_stat(sub, model_a, comp, seed=int(sd), metric=metric_name, n_boot=int(cfg.bootstrap_reps))
                            row.update({"family": family, "cost_bps": cost_bps, "seed": int(sd), "policy_variant": variant, "candidate_run": cfg.run_name})
                            all_boot.append(row)

    metrics_df = pd.DataFrame(all_metrics)
    panels_df = pd.concat(all_panels, ignore_index=True) if all_panels else pd.DataFrame()
    boot_df = pd.DataFrame(all_boot)
    seed_summary_df = _hedgeset_seed_summary(metrics_df)
    metrics_path = os.path.join(cfg.results_dir, "hedgeset_test_metrics_all.csv")
    panel_path = os.path.join(cfg.results_dir, "hedgeset_test_episode_panel_all.csv")
    boot_path = os.path.join(cfg.results_dir, "hedgeset_pairwise_bootstrap.csv")
    seed_summary_path = os.path.join(cfg.results_dir, "hedgeset_seed_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)
    panels_df.to_csv(panel_path, index=False)
    boot_df.to_csv(boot_path, index=False)
    seed_summary_df.to_csv(seed_summary_path, index=False)
    logger.info(f"HedgeSet metrics saved -> {metrics_path}")
    logger.info(f"HedgeSet episode panel saved -> {panel_path}")
    logger.info(f"HedgeSet bootstrap saved -> {boot_path}")
    logger.info(f"HedgeSet seed summary saved -> {seed_summary_path}")
    keep = ["candidate_run", "model", "family", "state_mode", "policy_variant", "action_scope", "residual_base", "residual_scale", "imitation_epochs", "factor_neutral_residual", "use_global_trust_gate", "seed", "cost_bps", "n", "mean_loss_norm", "cvar_95", "cvar_99", "worst_regime_cvar_95", "worst_regime_excess_cvar_95", "shortfall_prob", "worst_regime_shortfall_prob", "mean_base_loss_norm", "mean_excess_loss_norm", "mean_relu_excess_loss_norm", "excess_cvar_95", "excess_cvar_99", "mean_residual_l2", "mean_residual_turnover", "mean_abs_residual_delta_exposure", "mean_abs_residual_vega_exposure", "mean_residual_factor_l2", "mean_global_trust", "mean_ntb_band", "mean_ntb_gate", "mean_local_mse", "mean_tc_norm", "mean_turnover", "mean_abs_position"]
    present = [c for c in keep if c in metrics_df.columns]
    if present:
        logger.info("\n" + metrics_df[present].sort_values(["family", "cost_bps", "cvar_95"]).to_string(index=False))


def run_hedgeset_candidate_sweeps(
    families: List[str],
    state_modes: List[str],
    cost_bps_list: List[float],
    default_candidate_rules: List[str],
    sweep_names: List[str],
    **kwargs,
) -> None:
    for sw in sweep_names:
        run_name, rules, max_inst, min_inst = _hedgeset_preset_rules(sw, default_candidate_rules)
        logger.info("=" * 80)
        logger.info(f"Running HedgeSet candidate sweep: {run_name} | rules={rules}")
        logger.info("=" * 80)
        run_hedgeset_pipeline(
            families=families,
            state_modes=state_modes,
            cost_bps_list=cost_bps_list,
            candidate_rules=rules,
            run_name=run_name,
            max_instruments=max_inst if max_inst is not None else kwargs.get("max_instruments"),
            min_instruments=min_inst,
            **{k: v for k, v in kwargs.items() if k not in {"max_instruments"}},
        )

def main():
    parser = argparse.ArgumentParser(description="SPX Option Risk-State Learning Pipeline + HedgeSet modern baselines")
    parser.add_argument("--steps", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                        help="Steps to run: 0=preprocess 1=stage1 2=stage2 3=stage3 4=eval")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing, use existing cache")
    parser.add_argument("--rebuild", action="store_true", help="Force cache rebuild even if cache exists")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--ablation", action="store_true", help="Run original surface-loss ablation variants")
    parser.add_argument("--set-forecast-only", action="store_true",
                        help="Run the clean set-encoder-only forecasting ablation.")
    parser.add_argument("--multi-seed", action="store_true",
                        help="Run multi-seed neural experiments in separate results folders.")
    parser.add_argument("--seeds", type=str, default="1,42,2026",
                        help="Comma-separated seeds for --multi-seed or --set-forecast-only.")
    parser.add_argument("--multi-seed-no-full", action="store_true",
                        help="With --multi-seed, skip full staged model runs.")
    parser.add_argument("--multi-seed-no-setonly", action="store_true",
                        help="With --multi-seed, skip set-forecast-only ablation runs.")
    parser.add_argument("--no-grid-appendix", action="store_true",
                        help="Do not train/export Grid-CNN and Grid-Transformer appendix baselines during evaluation.")
    parser.add_argument("--set-only-side-only", action="store_true",
                        help="With --set-forecast-only, run only the side-info version and skip the no-side variant.")
    parser.add_argument("--paper-diagnostics", action="store_true",
                        help="Run diagnostic-only audits without changing trained models or main results.")
    parser.add_argument("--diag-csv-parquet", action="store_true",
                        help="Audit CSV/Parquet value equivalence for raw data files.")
    parser.add_argument("--diag-dsv5", action="store_true",
                        help="Rebuild DSV5 floor-sensitivity diagnostics from the prediction panel.")
    parser.add_argument("--diag-iv-failures", action="store_true",
                        help="Audit midpoint-IV inversion failure rates from raw option files.")
    parser.add_argument("--diag-max-rows-per-year", type=int, default=None,
                        help="Optional row cap per option year for IV inversion diagnostics.")
    parser.add_argument("--grid-side-appendix", action="store_true",
                        help="Train/export side-information-matched Grid-CNN+Side and Grid-Transformer+Side appendix baselines.")

    # Hedging/control extension.
    parser.add_argument("--hedging", action="store_true",
                        help="Run option-implied belief-state deep hedging experiments.")
    parser.add_argument("--hedge-families", type=str, default="atm_call,put25,atm_straddle",
                        help="Comma-separated liability families: atm_call,put25,atm_straddle.")
    parser.add_argument("--hedge-state-modes", type=str, default="latent,summary,grid,nosurface",
                        help="Comma-separated policy state modes. Use latent for the proposed belief state. nosurface is enabled when set_forecast_only_best.pt exists.")
    parser.add_argument("--hedge-cost-bps-list", type=str, default="1,2,5",
                        help="Comma-separated proportional hedge trading costs in basis points.")
    parser.add_argument("--hedge-horizon", type=int, default=5,
                        help="Number of trading days in each hedging episode.")
    parser.add_argument("--hedge-stride", type=int, default=5,
                        help="Stride between episode start dates in trading days.")
    parser.add_argument("--hedge-checkpoint", type=str, default="forecast",
                        choices=["forecast", "joint", "surface"],
                        help="Stage-3 checkpoint preference used to extract latent states.")
    parser.add_argument("--hedge-rebuild-chains", action="store_true",
                        help="Force rebuild of raw contract-level hedging chain cache.")
    parser.add_argument("--hedge-rebuild-states", action="store_true",
                        help="Force rebuild of latent/summary/grid state cache for hedging.")
    parser.add_argument("--hedge-rebuild-episodes", action="store_true",
                        help="Force rebuild of hedging episode caches.")
    parser.add_argument("--hedge-epochs", type=int, default=None,
                        help="Override hedging policy training epochs.")
    parser.add_argument("--hedge-batch-size", type=int, default=None,
                        help="Override hedging policy batch size.")
    parser.add_argument("--hedge-lr", type=float, default=None,
                        help="Override hedging policy learning rate.")
    parser.add_argument("--hedge-patience", type=int, default=None,
                        help="Override hedging policy early-stopping patience.")
    parser.add_argument("--hedge-no-terminal-liquidation", action="store_true",
                        help="Do not charge terminal liquidation cost for the remaining hedge position.")

    # QuoteSet-to-HedgeSet extension: variable-size hedge-instrument set policy.
    parser.add_argument("--hedgeset", action="store_true",
                        help="Run QuoteSet-to-HedgeSet deep hedging with a variable-size candidate hedge instrument set.")
    parser.add_argument("--hedgeset-candidate-rules", type=str,
                        default="underlying,atm_call30,atm_put30,put25_30,put10_30,call25_30,call10_30,atm_straddle30,put25_60,call25_60,atm_straddle60",
                        help="Comma-separated candidate hedge rules. Include underlying plus option rules such as put25_30 or atm_straddle60.")
    parser.add_argument("--hedgeset-max-instruments", type=int, default=12,
                        help="Maximum number of candidate hedge instruments retained per episode, including underlying.")
    parser.add_argument("--hedgeset-run-name", type=str, default="main",
                        help="Non-overwriting run label appended to the HedgeSet results directory.")
    parser.add_argument("--hedgeset-candidate-sweeps", type=str, default="main,underlying",
                        help="Optional comma-separated candidate-set sweeps: main,underlying,liquid,tail. Each gets a separate output directory.")
    parser.add_argument("--hedgeset-policy-variants", type=str, default="full,no_band,no_gate",
                        help="Comma-separated policy ablations: full,no_band,no_gate,no_band_no_gate.")
    parser.add_argument("--hedgeset-action-scopes", type=str, default="full,underlying",
                        help="Comma-separated action-set scopes for learned policies: full,underlying. Underlying gives the underlying-only action-set baseline.")
    parser.add_argument("--hedgeset-delta-thresholds", type=str, default="0,0.025,0.05,0.10,0.15,0.20,0.30",
                        help="Validation grid for NoTradeDelta baseline threshold.")
    parser.add_argument("--hedgeset-setonly-checkpoint", type=str, default=None,
                        help="Optional path to set_forecast_only_best.pt for setonly/nosurface state ablations. Can also use HEDGESET_SETONLY_CKPT.")
    parser.add_argument("--hedgeset-bootstrap-metrics", type=str, default="mean_loss,cvar_95,cvar_99,worst_regime_cvar_95,shortfall_prob,worst_regime_shortfall_prob,mean_turnover,mean_tc_norm",
                        help="Comma-separated paired-bootstrap metrics for HedgeSet comparisons.")
    parser.add_argument("--hedgeset-modern-baselines", type=str, default="",
                        help="Comma-separated modern baselines: ivgrid,ntb_ls,cvar_from_scratch,qiao_residual_ls,all. These run in the same episode/cost/bootstrap stack.")
    parser.add_argument("--hedgeset-ivgrid-state-mode", type=str, default="grid",
                        help="State mode for fixed-grid IV Surface Deep Hedging baseline. Default: grid.")
    parser.add_argument("--hedgeset-ntb-state-mode", type=str, default="summary",
                        help="State mode for NTB-LS learned no-transaction-band baseline. Default: summary, i.e. no QuoteSet latent.")
    parser.add_argument("--hedgeset-cvar-fromscratch-state-mode", type=str, default="latent",
                        help="State mode for CVaR-HedgeSet-from-scratch baseline. Default: latent.")
    parser.add_argument("--hedgeset-qiao-state-mode", type=str, default="latent",
                        help="State mode for QiaoResidual-LS plain anchored residual baseline. Default: latent.")
    parser.add_argument("--hedgeset-qiao-residual-scale", type=float, default=0.15,
                        help="Absolute residual cap for QiaoResidual-LS around the LSDeltaVega anchor.")
    parser.add_argument("--hedgeset-qiao-lambda-local-mse", type=float, default=1.0,
                        help="Weight on Qiao-style local squared hedging-error loss.")
    parser.add_argument("--hedgeset-qiao-lambda-residual-l2", type=float, default=0.02,
                        help="Weight on QiaoResidual-LS residual inventory penalty.")
    parser.add_argument("--hedgeset-qiao-lambda-residual-trade", type=float, default=0.01,
                        help="Weight on QiaoResidual-LS residual turnover penalty.")
    parser.add_argument("--hedgeset-ntb-band-max", type=float, default=1.0,
                        help="Maximum learned no-transaction band for NTB-LS baseline.")
    parser.add_argument("--hedgeset-ntb-lambda-band-l1", type=float, default=0.001,
                        help="Small L1 penalty on NTB-LS learned band size.")
    parser.add_argument("--hedgeset-paper-suite", action="store_true",
                        help="Convenience switch: run candidate sweeps main,underlying,liquid and policy variants full,no_band,no_gate.")
    parser.add_argument("--hedgeset-ambiguity-aware", action="store_true",
                        help="Append the ambiguity-aware regime-robust HedgeSet variant to the run.")
    parser.add_argument("--hedgeset-ambiguity-lambda", type=float, default=0.25,
                        help="Weight on smooth worst-regime CVaR in ambiguity-aware HedgeSet training.")
    parser.add_argument("--hedgeset-ambiguity-eta", type=float, default=8.0,
                        help="Log-sum-exp sharpness for smooth worst-regime CVaR.")
    parser.add_argument("--hedgeset-ambiguity-min-count", type=int, default=2,
                        help="Minimum number of batch episodes required for a regime CVaR term.")
    parser.add_argument("--hedgeset-regime-bins", type=int, default=3,
                        help="Number of train-calibrated option-implied regime bins.")
    parser.add_argument("--hedgeset-residual-base", type=str, default="ls_deltavega",
                        choices=["none", "delta", "delta_nt", "greedy_deltavega", "ls_deltavega"],
                        help="Classical hedge anchor for residual HedgeSet policy. Use none for from-scratch absolute target policy.")
    parser.add_argument("--hedgeset-residual-scale", type=float, default=0.15,
                        help="Absolute cap for residual correction around the classical hedge anchor. Recommended sweep: 0.10,0.15,0.25.")
    parser.add_argument("--hedgeset-imitation-epochs", type=int, default=10,
                        help="Number of warm-start epochs that imitate the selected classical hedge anchor before CVaR fine-tuning.")
    parser.add_argument("--hedgeset-imitation-lambda", type=float, default=1.0,
                        help="Weight on imitation loss during warm-start.")
    parser.add_argument("--hedgeset-finetune-imitation-lambda", type=float, default=0.02,
                        help="Small imitation regularizer retained during CVaR fine-tuning.")
    parser.add_argument("--hedgeset-lambda-excess-cvar", type=float, default=0.50,
                        help="Weight on CVaR(policy loss minus residual-base loss).")
    parser.add_argument("--hedgeset-lambda-excess-relu", type=float, default=0.25,
                        help="Weight on mean positive excess loss versus residual base.")
    parser.add_argument("--hedgeset-lambda-residual-l2", type=float, default=0.05,
                        help="Weight on squared residual inventory penalty.")
    parser.add_argument("--hedgeset-lambda-residual-trade", type=float, default=0.02,
                        help="Weight on residual turnover penalty.")
    parser.add_argument("--hedgeset-factor-neutral-residual", dest="hedgeset_factor_neutral_residual", action="store_true", default=True,
                        help="Project residual positions toward the delta/vega null space of the candidate set.")
    parser.add_argument("--hedgeset-no-factor-neutral-residual", dest="hedgeset_factor_neutral_residual", action="store_false",
                        help="Disable factor-neutral residual projection.")
    parser.add_argument("--hedgeset-factor-neutral-ridge", type=float, default=1e-4,
                        help="Ridge used in the residual null-space projection.")
    parser.add_argument("--hedgeset-lambda-factor-neutral", type=float, default=0.10,
                        help="Penalty on residual delta/vega factor exposure after projection.")
    parser.add_argument("--hedgeset-global-trust-gate", dest="hedgeset_global_trust_gate", action="store_true", default=True,
                        help="Use a global trust gate to shrink the learned residual back to the LS base.")
    parser.add_argument("--hedgeset-no-global-trust-gate", dest="hedgeset_global_trust_gate", action="store_false",
                        help="Disable global residual trust gate.")
    parser.add_argument("--hedgeset-trust-gate-init-bias", type=float, default=-2.0,
                        help="Initial bias of the global trust gate; negative values start close to the LS base.")
    parser.add_argument("--hedgeset-trust-gate-temp", type=float, default=1.0,
                        help="Temperature of the global trust gate sigmoid.")
    parser.add_argument("--hedgeset-lambda-trust-l1", type=float, default=0.01,
                        help="Penalty on average residual trust, encouraging LS fallback unless residual value is earned.")
    parser.add_argument("--hedgeset-lambda-regime-excess-cvar", type=float, default=0.0,
                        help="Penalty on static worst-regime CVaR of excess loss over the LS base. Default 0 for the final specification.")
    parser.add_argument("--hedgeset-val-score-worst-regime-excess-weight", type=float, default=0.0,
                        help="Validation-score weight on worst-regime excess CVaR. Default 0 for the final no-static-regime-excess design.")
    parser.add_argument("--hedgeset-val-feasible-excess-cvar95-max", type=float, default=None,
                        help="Maximum validation excess CVaR95 for the generic feasible checkpoint.")
    parser.add_argument("--hedgeset-val-feasible-excess-cvar95-grid", type=str, default="0.007,0.008,0.009,0.012",
                        help="Comma-separated validation excess CVaR95 budgets for feasible checkpoint frontier saving.")
    parser.add_argument("--hedgeset-val-feasible-turnover-max", type=float, default=None,
                        help="Maximum validation turnover for feasible checkpoint selection.")
    parser.add_argument("--hedgeset-val-feasible-tc-norm-max", type=float, default=None,
                        help="Maximum normalized validation transaction cost for feasible checkpoint selection.")
    parser.add_argument("--hedgeset-val-feasible-delta-exposure-max", type=float, default=None,
                        help="Maximum validation residual delta exposure for feasible checkpoint selection.")
    parser.add_argument("--hedgeset-val-feasible-vega-exposure-max", type=float, default=None,
                        help="Maximum validation residual vega exposure for feasible checkpoint selection.")
    parser.add_argument("--hedgeset-val-score-excess-cvar95-weight", type=float, default=None,
                        help="Validation-score weight on excess CVaR95.")
    parser.add_argument("--hedgeset-val-score-excess-cvar99-weight", type=float, default=None,
                        help="Validation-score weight on excess CVaR99.")
    parser.add_argument("--hedgeset-val-score-excess-relu-weight", type=float, default=None,
                        help="Validation-score weight on mean positive excess loss.")
    parser.add_argument("--hedgeset-lambda-aux-risk", type=float, default=0.05,
                        help="Weight on auxiliary transition/base-risk prediction loss.")
    args = parser.parse_args()

    seeds = _parse_seed_list(args.seeds)

    if args.paper_diagnostics:
        run_paper_diagnostics(
            diag_csv_parquet=args.diag_csv_parquet,
            diag_dsv5=args.diag_dsv5,
            diag_iv_failures=args.diag_iv_failures,
            max_rows_per_year=args.diag_max_rows_per_year,
        )
    elif args.hedgeset:
        default_rules = _parse_list_arg(args.hedgeset_candidate_rules)
        if args.hedgeset_ambiguity_aware:
            pv = [x.strip() for x in str(args.hedgeset_policy_variants).split(",") if x.strip()]
            if not any(x in {"amb", "ambiguous", "amb_full", "robust", "ambiguity"} for x in pv):
                pv.append("ambiguous")
                args.hedgeset_policy_variants = ",".join(pv)
        if args.hedgeset_paper_suite:
            # Paper suite should include the liquid candidate-set sweep by default.
            # Preserve an explicitly custom --hedgeset-candidate-sweeps value,
            # but upgrade the parser default "main,underlying" to the full suite.
            _sweeps_txt = str(args.hedgeset_candidate_sweeps).strip()
            if (not _sweeps_txt) or (_sweeps_txt == "main,underlying"):
                args.hedgeset_candidate_sweeps = "main,underlying,liquid"
            if str(args.hedgeset_policy_variants).strip() == "full":
                args.hedgeset_policy_variants = "full,no_band,no_gate"
        common_kwargs = dict(
            horizon=args.hedge_horizon,
            stride=args.hedge_stride,
            checkpoint_preference=args.hedge_checkpoint,
            rebuild_chains=args.hedge_rebuild_chains,
            rebuild_states=args.hedge_rebuild_states,
            rebuild_episodes=args.hedge_rebuild_episodes,
            seed=TRAIN_CFG.seed,
            seeds=seeds,
            policy_variants=_parse_list_arg(args.hedgeset_policy_variants),
            epochs=args.hedge_epochs,
            batch_size=args.hedge_batch_size,
            lr=args.hedge_lr,
            patience=args.hedge_patience,
            max_instruments=args.hedgeset_max_instruments,
            terminal_liquidation=not args.hedge_no_terminal_liquidation,
            setonly_checkpoint=args.hedgeset_setonly_checkpoint,
            bootstrap_metrics=_parse_list_arg(args.hedgeset_bootstrap_metrics),
            action_scopes=_parse_list_arg(args.hedgeset_action_scopes),
            delta_thresholds=_parse_float_list(args.hedgeset_delta_thresholds),
            ambiguity_lambda=args.hedgeset_ambiguity_lambda,
            ambiguity_eta=args.hedgeset_ambiguity_eta,
            ambiguity_min_count=args.hedgeset_ambiguity_min_count,
            regime_bins=args.hedgeset_regime_bins,
            residual_base=args.hedgeset_residual_base,
            residual_scale=args.hedgeset_residual_scale,
            imitation_epochs=args.hedgeset_imitation_epochs,
            imitation_lambda=args.hedgeset_imitation_lambda,
            finetune_imitation_lambda=args.hedgeset_finetune_imitation_lambda,
            lambda_excess_cvar=args.hedgeset_lambda_excess_cvar,
            lambda_excess_relu=args.hedgeset_lambda_excess_relu,
            lambda_residual_l2=args.hedgeset_lambda_residual_l2,
            lambda_residual_trade=args.hedgeset_lambda_residual_trade,
            factor_neutral_residual=args.hedgeset_factor_neutral_residual,
            factor_neutral_ridge=args.hedgeset_factor_neutral_ridge,
            lambda_factor_neutral=args.hedgeset_lambda_factor_neutral,
            use_global_trust_gate=args.hedgeset_global_trust_gate,
            trust_gate_init_bias=args.hedgeset_trust_gate_init_bias,
            trust_gate_temp=args.hedgeset_trust_gate_temp,
            lambda_trust_l1=args.hedgeset_lambda_trust_l1,
            lambda_regime_excess_cvar=args.hedgeset_lambda_regime_excess_cvar,
            lambda_aux_risk=args.hedgeset_lambda_aux_risk,
            val_score_worst_regime_excess_weight=args.hedgeset_val_score_worst_regime_excess_weight,
            val_feasible_excess_cvar95_max=args.hedgeset_val_feasible_excess_cvar95_max,
            val_feasible_excess_cvar95_grid=_parse_float_list(args.hedgeset_val_feasible_excess_cvar95_grid),
            val_feasible_turnover_max=args.hedgeset_val_feasible_turnover_max,
            val_feasible_tc_norm_max=args.hedgeset_val_feasible_tc_norm_max,
            val_feasible_delta_exposure_max=args.hedgeset_val_feasible_delta_exposure_max,
            val_feasible_vega_exposure_max=args.hedgeset_val_feasible_vega_exposure_max,
            val_score_excess_cvar95_weight=args.hedgeset_val_score_excess_cvar95_weight,
            val_score_excess_cvar99_weight=args.hedgeset_val_score_excess_cvar99_weight,
            val_score_excess_relu_weight=args.hedgeset_val_score_excess_relu_weight,
            modern_baselines=_parse_list_arg(args.hedgeset_modern_baselines),
            ivgrid_state_mode=args.hedgeset_ivgrid_state_mode,
            ntb_state_mode=args.hedgeset_ntb_state_mode,
            cvar_from_scratch_state_mode=args.hedgeset_cvar_fromscratch_state_mode,
            ntb_band_max=args.hedgeset_ntb_band_max,
            ntb_lambda_band_l1=args.hedgeset_ntb_lambda_band_l1,
            qiao_state_mode=args.hedgeset_qiao_state_mode,
            qiao_residual_scale=args.hedgeset_qiao_residual_scale,
            qiao_lambda_local_mse=args.hedgeset_qiao_lambda_local_mse,
            qiao_lambda_residual_l2=args.hedgeset_qiao_lambda_residual_l2,
            qiao_lambda_residual_trade=args.hedgeset_qiao_lambda_residual_trade,
        )
        candidate_sweeps_text = str(args.hedgeset_candidate_sweeps).strip()
        if candidate_sweeps_text.lower() in {"none", "no", "off", "false", "0"}:
            candidate_sweeps_text = ""
        if candidate_sweeps_text:
            run_hedgeset_candidate_sweeps(
                families=_parse_list_arg(args.hedge_families),
                state_modes=_parse_list_arg(args.hedge_state_modes),
                cost_bps_list=_parse_float_list(args.hedge_cost_bps_list),
                default_candidate_rules=default_rules,
                sweep_names=_parse_list_arg(candidate_sweeps_text),
                **common_kwargs,
            )
        else:
            run_hedgeset_pipeline(
                families=_parse_list_arg(args.hedge_families),
                state_modes=_parse_list_arg(args.hedge_state_modes),
                cost_bps_list=_parse_float_list(args.hedge_cost_bps_list),
                candidate_rules=default_rules,
                run_name=args.hedgeset_run_name,
                **common_kwargs,
            )
    elif args.hedging:
        run_hedging_pipeline(
            families=_parse_list_arg(args.hedge_families),
            state_modes=_parse_list_arg(args.hedge_state_modes),
            cost_bps_list=_parse_float_list(args.hedge_cost_bps_list),
            horizon=args.hedge_horizon,
            stride=args.hedge_stride,
            checkpoint_preference=args.hedge_checkpoint,
            rebuild_chains=args.hedge_rebuild_chains,
            rebuild_states=args.hedge_rebuild_states,
            rebuild_episodes=args.hedge_rebuild_episodes,
            seed=TRAIN_CFG.seed,
            epochs=args.hedge_epochs,
            batch_size=args.hedge_batch_size,
            lr=args.hedge_lr,
            patience=args.hedge_patience,
            terminal_liquidation=not args.hedge_no_terminal_liquidation,
        )
    elif args.multi_seed:
        run_multi_seed(
            seeds=seeds,
            steps=args.steps,
            skip_preprocess=True if args.skip_preprocess else args.skip_preprocess,
            rebuild=args.rebuild,
            resume=args.resume,
            run_full=not args.multi_seed_no_full,
            run_setonly=not args.multi_seed_no_setonly,
            include_grid_appendix=not args.no_grid_appendix,
        )
    elif args.set_forecast_only:
        run_set_forecast_only_experiments(
            seeds=seeds,
            include_no_side=not args.set_only_side_only,
            include_grid_appendix=not args.no_grid_appendix,
        )
    elif args.ablation:
        run_ablation()
    else:
        run_pipeline(
            args.steps,
            skip_preprocess=args.skip_preprocess,
            rebuild=args.rebuild,
            resume=args.resume,
            include_grid_appendix=not args.no_grid_appendix,
            include_grid_side_appendix=args.grid_side_appendix,
        )


if __name__ == "__main__":
    main()
