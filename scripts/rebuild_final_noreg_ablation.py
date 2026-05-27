from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FINAL_RUN = "fnug_final_noreg_fam3_cost3_seed3_1eaa24b3f8"
ABLATION_RUNS = {
    "No factor-neutral": "Aplus_no_factorneutral_1eaa24b3f8",
    "No trust gate": "Aplus_no_trustgate_1eaa24b3f8",
    "No auxiliary risk": "Aplus_no_auxrisk_1eaa24b3f8",
}

METRICS = {
    "mean_loss": "loss_norm",
    "cvar_95": "loss_norm",
    "cvar_99": "loss_norm",
    "worst_regime_cvar_95": "loss_norm",
    "excess_cvar_95": "excess_loss_norm",
    "mean_positive_excess": "relu_excess_loss_norm",
    "mean_residual_l2": "residual_l2",
    "mean_residual_delta_exposure": "residual_delta_exposure",
    "mean_residual_vega_exposure": "residual_vega_exposure",
    "mean_residual_factor_l2": "residual_factor_l2",
    "mean_global_trust": "global_trust_mean",
    "mean_turnover": "turnover",
    "mean_tc_norm": "tc_norm",
}


def read_panel(root: Path, run_name: str, label: str) -> pd.DataFrame:
    path = root / "results" / "hedgeset" / run_name / "hedgeset_test_episode_panel_all.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    panel = pd.read_csv(path)
    panel = panel[panel["model"].eq("HedgeSet-latent")].copy()
    panel["model"] = label
    return panel


def filter_case(panel: pd.DataFrame, label: str) -> pd.DataFrame:
    seeds = pd.to_numeric(panel["seed"], errors="coerce")
    costs = pd.to_numeric(panel["cost_bps"], errors="coerce")
    return panel[
        panel["model"].eq(label)
        & panel["family"].eq("atm_call")
        & np.isclose(costs, 2.0)
        & np.isclose(seeds, 42.0)
    ].copy()


def stat(values: np.ndarray, metric: str, regimes: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    values = values[valid]
    if regimes is not None:
        regimes = np.asarray(regimes)[valid]
    if values.size == 0:
        return math.nan
    if metric.startswith("mean_"):
        return float(np.mean(values))
    if metric == "cvar_95" or metric == "excess_cvar_95":
        q = np.quantile(values, 0.95)
        return float(np.mean(values[values >= q]))
    if metric == "cvar_99":
        q = np.quantile(values, 0.99)
        return float(np.mean(values[values >= q]))
    if metric == "worst_regime_cvar_95":
        if regimes is None:
            return math.nan
        vals = []
        for rid in pd.unique(pd.Series(regimes).dropna()):
            xr = values[regimes == rid]
            if xr.size == 0:
                continue
            q = np.quantile(xr, 0.95)
            vals.append(float(np.mean(xr[xr >= q])))
        return float(np.nanmax(vals)) if vals else math.nan
    raise ValueError(metric)


def paired_bootstrap(
    full: pd.DataFrame,
    ablation: pd.DataFrame,
    metric: str,
    n_boot: int,
    rng_seed: int,
) -> dict[str, float | int | str]:
    col = METRICS[metric]
    cols = ["start_date", col]
    if "regime_id" in full.columns:
        cols.append("regime_id")
    a = full[cols].rename(columns={col: "full"})
    b = ablation[["start_date", col]].rename(columns={col: "ablation"})
    merged = a.merge(b, on="start_date", how="inner")
    if merged.empty:
        raise ValueError(f"No paired rows for {metric}")

    x_full = pd.to_numeric(merged["full"], errors="coerce").to_numpy(dtype=float)
    x_abl = pd.to_numeric(merged["ablation"], errors="coerce").to_numpy(dtype=float)
    regimes = merged["regime_id"].to_numpy() if "regime_id" in merged.columns else None

    obs = stat(x_full, metric, regimes) - stat(x_abl, metric, regimes)
    rng = np.random.default_rng(rng_seed)
    n = len(merged)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r = regimes[idx] if regimes is not None else None
        draws[i] = stat(x_full[idx], metric, r) - stat(x_abl[idx], metric, r)

    centered = draws - obs
    pval = float(np.mean(np.abs(centered) >= abs(obs)))
    return {
        "metric": metric,
        "n": int(n),
        "full": stat(x_full, metric, regimes),
        "ablation": stat(x_abl, metric, regimes),
        "diff_full_minus_ablation": float(obs),
        "ci95_low": float(np.nanquantile(draws, 0.025)),
        "ci95_high": float(np.nanquantile(draws, 0.975)),
        "p_two_sided": min(pval, 1.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild final no-regime ablation bootstrap.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    outdir = args.outdir or (root / "results" / "analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    full_panel = filter_case(read_panel(root, FINAL_RUN, "Full FNUG"), "Full FNUG")
    rows = []
    for abl_idx, (label, run_name) in enumerate(ABLATION_RUNS.items()):
        ablation_panel = filter_case(read_panel(root, run_name, label), label)
        for metric_idx, metric in enumerate(METRICS):
            row = paired_bootstrap(
                full_panel,
                ablation_panel,
                metric,
                n_boot=args.n_boot,
                rng_seed=20_000 + 101 * abl_idx + metric_idx,
            )
            row["comparison"] = f"Full FNUG vs {label}"
            rows.append(row)

    out = pd.DataFrame(rows)
    cols = ["comparison"] + [c for c in out.columns if c != "comparison"]
    out = out[cols]
    out.to_csv(outdir / "final_noreg_ablation_pairwise_bootstrap.csv", index=False)
    print(f"Wrote {len(out)} ablation bootstrap rows to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
