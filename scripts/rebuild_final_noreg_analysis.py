from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FINAL_RUN = "fnug_final_noreg_fam3_cost3_seed3_1eaa24b3f8"
LEGACY_MODERN_RUN = "fnug_modern_fam3_cost3_seed3_m4_1eaa24b3f8"
IVGRID_NOREG_RUN = "fnug_modern_noreg_fam3_cost3_seed3_m4_1eaa24b3f8"
QIAO_RUN = "qiao_ls_m4_1eaa24b3f8"

FNUG_MODEL = "HedgeSet-latent"
LEGACY_MODERN_MODELS = ["NTB-LS", "CVaR-HedgeSet"]
COMPARATORS = [
    "LSDeltaVega",
    "IVGrid-HedgeSet",
    "QiaoResidual-LS",
    "NTB-LS",
    "CVaR-HedgeSet",
    "SmiledDelta",
    "NoTradeDelta",
    "GreedyDeltaVega",
]
BOOT_METRICS = [
    "mean_loss",
    "cvar_95",
    "cvar_99",
    "worst_regime_cvar_95",
    "shortfall_prob",
    "mean_turnover",
    "mean_tc_norm",
    "excess_cvar_95",
    "mean_positive_excess",
]


def hedgeset_root(root: Path) -> Path:
    return root / "results" / "hedgeset"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_run(root: Path, run_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = hedgeset_root(root) / run_name
    metrics = read_csv(run_dir / "hedgeset_test_metrics_all.csv")
    panel = read_csv(run_dir / "hedgeset_test_episode_panel_all.csv")
    return metrics, panel


def normalize_no_regime_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "lambda_regime_excess_cvar" in df.columns:
        df["lambda_regime_excess_cvar"] = 0.0
    return df


def only_models(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    return df[df["model"].isin(models)].copy()


def assemble(
    root: Path,
    final_run: str,
    legacy_modern_run: str,
    ivgrid_noreg_run: str,
    qiao_run: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_metrics, final_panel = read_run(root, final_run)
    legacy_metrics, legacy_panel = read_run(root, legacy_modern_run)
    ivgrid_metrics, ivgrid_panel = read_run(root, ivgrid_noreg_run)
    qiao_metrics, qiao_panel = read_run(root, qiao_run)

    final_keep = [FNUG_MODEL, "NoHedge", "SmiledDelta", "NoTradeDelta", "GreedyDeltaVega", "LSDeltaVega"]
    final_metrics = only_models(final_metrics, final_keep)
    final_panel = only_models(final_panel, final_keep)

    legacy_metrics = only_models(legacy_metrics, LEGACY_MODERN_MODELS)
    legacy_panel = only_models(legacy_panel, LEGACY_MODERN_MODELS)

    ivgrid_metrics = only_models(ivgrid_metrics, ["IVGrid-HedgeSet"])
    ivgrid_panel = only_models(ivgrid_panel, ["IVGrid-HedgeSet"])

    qiao_metrics = only_models(qiao_metrics, ["QiaoResidual-LS"])
    qiao_panel = only_models(qiao_panel, ["QiaoResidual-LS"])

    metrics = pd.concat(
        [final_metrics, legacy_metrics, qiao_metrics, ivgrid_metrics],
        ignore_index=True,
        sort=False,
    )
    panel = pd.concat(
        [final_panel, legacy_panel, qiao_panel, ivgrid_panel],
        ignore_index=True,
        sort=False,
    )
    return normalize_no_regime_metadata(metrics), normalize_no_regime_metadata(panel)


def finite_array(x: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def stat_from_values(x: np.ndarray, metric: str, regimes: np.ndarray | None = None) -> float:
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    x = x[valid]
    if regimes is not None:
        regimes = np.asarray(regimes)[valid]
    if x.size == 0:
        return math.nan

    metric = metric.lower()
    if metric in {"mean_loss", "mean", "mean_turnover", "mean_tc_norm", "mean_positive_excess"}:
        return float(np.mean(x))
    if metric == "shortfall_prob":
        return float(np.mean(x > 0.0))
    if metric == "cvar_95" or metric == "excess_cvar_95":
        q = np.quantile(x, 0.95)
        return float(np.mean(x[x >= q]))
    if metric == "cvar_99":
        q = np.quantile(x, 0.99)
        return float(np.mean(x[x >= q]))
    if metric == "worst_regime_cvar_95":
        if regimes is None:
            return math.nan
        vals = []
        for rid in pd.unique(pd.Series(regimes).dropna()):
            xr = x[regimes == rid]
            if xr.size == 0:
                continue
            q = np.quantile(xr, 0.95)
            vals.append(float(np.mean(xr[xr >= q])))
        return float(np.nanmax(vals)) if vals else math.nan
    raise ValueError(f"Unsupported metric: {metric}")


def metric_column(metric: str) -> str:
    if metric == "mean_turnover":
        return "turnover"
    if metric == "mean_tc_norm":
        return "tc_norm"
    if metric == "excess_cvar_95":
        return "excess_loss_norm"
    if metric == "mean_positive_excess":
        return "relu_excess_loss_norm"
    return "loss_norm"


def filter_case(panel: pd.DataFrame, model: str, family: str, cost: float, seed: float | int) -> pd.DataFrame:
    m = panel[
        panel["model"].eq(model)
        & panel["family"].eq(family)
        & np.isclose(pd.to_numeric(panel["cost_bps"], errors="coerce"), float(cost))
    ].copy()
    if "seed" not in m.columns:
        return m
    seeds = pd.to_numeric(m["seed"], errors="coerce")
    exact = m[np.isclose(seeds, float(seed), equal_nan=False)].copy()
    if not exact.empty:
        return exact
    static = m[seeds.isna()].copy()
    return static if not static.empty else m


def paired_bootstrap(
    panel: pd.DataFrame,
    model_a: str,
    model_b: str,
    family: str,
    cost: float,
    seed: int,
    metric: str,
    n_boot: int,
    rng_seed: int,
) -> dict[str, object]:
    col = metric_column(metric)
    if col not in panel.columns:
        return {}
    a = filter_case(panel, model_a, family, cost, seed)
    b = filter_case(panel, model_b, family, cost, seed)
    if a.empty or b.empty or col not in a.columns or col not in b.columns:
        return {}

    a_cols = ["start_date", col]
    if "regime_id" in a.columns:
        a_cols.append("regime_id")
    aa = a[a_cols].rename(columns={col: "a"})
    bb = b[["start_date", col]].rename(columns={col: "b"})
    merged = aa.merge(bb, on="start_date", how="inner")
    if merged.empty:
        return {}

    xa = pd.to_numeric(merged["a"], errors="coerce").to_numpy(dtype=float)
    xb = pd.to_numeric(merged["b"], errors="coerce").to_numpy(dtype=float)
    regimes = merged["regime_id"].to_numpy() if "regime_id" in merged.columns else None
    obs = stat_from_values(xa, metric, regimes=regimes) - stat_from_values(xb, metric, regimes=regimes)
    if not np.isfinite(obs):
        return {}

    rng = np.random.default_rng(rng_seed)
    n = len(merged)
    boot = np.empty(n_boot, dtype=float)
    for r in range(n_boot):
        idx = rng.integers(0, n, size=n)
        reg_idx = regimes[idx] if regimes is not None else None
        boot[r] = stat_from_values(xa[idx], metric, reg_idx) - stat_from_values(xb[idx], metric, reg_idx)

    centered = boot - obs
    pval = float(np.mean(np.abs(centered) >= abs(obs)))
    return {
        "model_a": model_a,
        "model_b": model_b,
        "family": family,
        "cost_bps": float(cost),
        "seed": int(seed),
        "metric": metric,
        "n": int(n),
        "diff": float(obs),
        "mean_diff": float(obs),
        "ci95_low": float(np.nanquantile(boot, 0.025)),
        "ci95_high": float(np.nanquantile(boot, 0.975)),
        "p_two_sided": min(pval, 1.0),
    }


def build_bootstrap(panel: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    full = panel[panel["model"].eq(FNUG_MODEL)].copy()
    rows: list[dict[str, object]] = []
    cases = (
        full[["family", "cost_bps", "seed"]]
        .drop_duplicates()
        .sort_values(["family", "cost_bps", "seed"])
    )
    for case in cases.itertuples(index=False):
        family = str(case.family)
        cost = float(case.cost_bps)
        seed = int(case.seed)
        for comp_idx, comp in enumerate(COMPARATORS):
            if comp not in set(panel["model"]):
                continue
            for metric_idx, metric in enumerate(BOOT_METRICS):
                row = paired_bootstrap(
                    panel,
                    FNUG_MODEL,
                    comp,
                    family,
                    cost,
                    seed,
                    metric,
                    n_boot=n_boot,
                    rng_seed=10_000 + 101 * seed + 17 * comp_idx + metric_idx,
                )
                if row:
                    rows.append(row)
    return pd.DataFrame(rows)


def case_metrics(metrics: pd.DataFrame, family: str, cost: float, seed: int) -> pd.DataFrame:
    seeds = pd.to_numeric(metrics.get("seed", pd.Series(np.nan, index=metrics.index)), errors="coerce")
    costs = pd.to_numeric(metrics["cost_bps"], errors="coerce")
    exact_seed = np.isclose(seeds, float(seed), equal_nan=False)
    static_seed = seeds.isna()
    return metrics[
        metrics["family"].eq(family)
        & np.isclose(costs, float(cost))
        & (exact_seed | static_seed)
    ].copy()


def write_summary_tables(metrics: pd.DataFrame, outdir: Path) -> None:
    cols = [
        "model",
        "family",
        "cost_bps",
        "seed",
        "mean_loss_norm",
        "cvar_95",
        "cvar_99",
        "worst_regime_cvar_95",
        "excess_cvar_95",
        "mean_relu_excess_loss_norm",
        "mean_tc_norm",
        "mean_turnover",
        "mean_abs_residual_delta_exposure",
        "mean_abs_residual_vega_exposure",
        "mean_residual_factor_l2",
        "mean_global_trust",
    ]
    present = [c for c in cols if c in metrics.columns]
    main = case_metrics(metrics, "atm_call", 2.0, 42)
    main[present].to_csv(outdir / "final_noreg_main_cell_metrics.csv", index=False)

    rows = []
    for family in ["atm_call", "put25", "atm_straddle"]:
        for cost in [1.0, 2.0, 5.0]:
            ls = case_metrics(metrics, family, cost, 42)
            ls = ls[ls["model"].eq("LSDeltaVega")]
            if ls.empty:
                continue
            ls_cvar = float(ls.iloc[0]["cvar_95"])
            for seed in [1, 42, 2026]:
                fnug = case_metrics(metrics, family, cost, seed)
                fnug = fnug[fnug["model"].eq(FNUG_MODEL)]
                if fnug.empty:
                    continue
                rows.append({
                    "family": family,
                    "cost_bps": cost,
                    "seed": seed,
                    "fnug_cvar_95": float(fnug.iloc[0]["cvar_95"]),
                    "ls_cvar_95": ls_cvar,
                    "diff_fnug_minus_ls": float(fnug.iloc[0]["cvar_95"]) - ls_cvar,
                })
    pd.DataFrame(rows).to_csv(outdir / "final_noreg_heatmap_values.csv", index=False)

    fnug = main[main["model"].eq(FNUG_MODEL)].iloc[0]
    ls = main[main["model"].eq("LSDeltaVega")].iloc[0]
    red_rows = []
    for metric in ["cvar_95", "cvar_99", "worst_regime_cvar_95"]:
        old = float(ls[metric])
        new = float(fnug[metric])
        red_rows.append({
            "metric": metric,
            "fnug": new,
            "ls": old,
            "absolute_diff": new - old,
            "reduction_pct": 100.0 * (old - new) / old if old else math.nan,
        })
    pd.DataFrame(red_rows).to_csv(outdir / "final_noreg_vs_ls_reductions.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild final no-static-regime FNUG analysis tables.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root containing results/hedgeset.")
    parser.add_argument("--final-run", default=FINAL_RUN)
    parser.add_argument("--legacy-modern-run", default=LEGACY_MODERN_RUN)
    parser.add_argument("--ivgrid-noreg-run", default=IVGRID_NOREG_RUN)
    parser.add_argument("--qiao-run", default=QIAO_RUN)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    outdir = args.outdir or (root / "results" / "analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    metrics, panel = assemble(root, args.final_run, args.legacy_modern_run, args.ivgrid_noreg_run, args.qiao_run)
    metrics.to_csv(outdir / "final_noreg_combined_metrics.csv", index=False)
    panel.to_csv(outdir / "final_noreg_combined_episode_panel.csv", index=False)

    boot = build_bootstrap(panel, n_boot=args.n_boot)
    boot.to_csv(outdir / "final_noreg_pairwise_bootstrap.csv", index=False)
    write_summary_tables(metrics, outdir)

    print(f"Wrote combined metrics/panel and bootstrap to {outdir}")
    print(f"Bootstrap rows: {len(boot)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
