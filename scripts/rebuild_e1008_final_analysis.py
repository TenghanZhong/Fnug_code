from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

E1008_RUN = "fnug_e1008_final_fam3_cost3_seed3_1eaa24b3f8"
E1008_SELECTION = "best_feasible_excess_0p008"
SUPPORTING_MODERN_RUN = "fnug_modern_supporting_fam3_cost3_seed3_m4_1eaa24b3f8"
IVGRID_FINAL_OBJECTIVE_RUN = "fnug_modern_finalobjective_fam3_cost3_seed3_m4_1eaa24b3f8"
IVGRID_FINAL_OBJECTIVE_EXTRA_RUNS = ["fnug_modern_finalobjective_straddle_m4_1eaa24b3f8"]
QIAO_RUN_CANDIDATES = [
    "qiao_residual_ls_m4_1eaa24b3f8",
    "qiao_ls_m4_1eaa24b3f8",
]

FNUG_MODEL = "HedgeSet-latent"
SUPPORTING_MODERN_MODELS = ["NTB-LS", "CVaR-HedgeSet"]
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


def read_optional_run(root: Path, run_name: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    run_dir = hedgeset_root(root) / run_name
    metrics_path = run_dir / "hedgeset_test_metrics_all.csv"
    panel_path = run_dir / "hedgeset_test_episode_panel_all.csv"
    if not metrics_path.exists() or not panel_path.exists():
        return None
    return read_csv(metrics_path), read_csv(panel_path)


def only_models(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    return df[df["model"].isin(models)].copy()


def _with_case_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cost_bps_key"] = pd.to_numeric(out["cost_bps"], errors="coerce").round(8)
    out["seed_key"] = pd.to_numeric(out["seed"], errors="coerce").round(0).astype("Int64")
    return out


def build_ivgrid_matched_cell_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_names = ["cvar_95", "cvar_99", "worst_regime_cvar_95"]
    cols = ["family", "cost_bps_key", "seed_key", *metric_names]
    fnug = _with_case_key(metrics[metrics["model"].eq(FNUG_MODEL)])
    ivgrid = _with_case_key(metrics[metrics["model"].eq("IVGrid-HedgeSet")])
    matched = fnug[cols].merge(
        ivgrid[cols],
        on=["family", "cost_bps_key", "seed_key"],
        how="inner",
        suffixes=("_fnug", "_ivgrid"),
    )
    rows: list[dict[str, object]] = []
    families = ",".join(sorted(matched["family"].dropna().astype(str).unique()))
    for metric in metric_names:
        diff = pd.to_numeric(matched[f"{metric}_fnug"], errors="coerce") - pd.to_numeric(
            matched[f"{metric}_ivgrid"], errors="coerce"
        )
        diff = diff[np.isfinite(diff)]
        if diff.empty:
            continue
        rows.append({
            "metric": metric,
            "n_cells": int(diff.size),
            "families": families,
            "fnug_lower_cells": int((diff < 0).sum()),
            "ivgrid_lower_cells": int((diff > 0).sum()),
            "ties": int((diff == 0).sum()),
            "mean_diff_fnug_minus_ivgrid": float(diff.mean()),
            "median_diff_fnug_minus_ivgrid": float(diff.median()),
            "min_diff_fnug_minus_ivgrid": float(diff.min()),
            "max_diff_fnug_minus_ivgrid": float(diff.max()),
        })
    return pd.DataFrame(rows)


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
    if metric in {"cvar_95", "excess_cvar_95"}:
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
    cases = full[["family", "cost_bps", "seed"]].drop_duplicates().sort_values(["family", "cost_bps", "seed"])
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


def selected_file_suffix(selection: str) -> str:
    if selection == "best_score":
        return ""
    if selection == "best_feasible":
        return "_best_feasible"
    if selection.startswith("best_feasible_excess_"):
        return f"_{selection}"
    raise ValueError(f"Unsupported checkpoint selection: {selection}")


def read_selected_checkpoint_run(root: Path, run_name: str, selection: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = hedgeset_root(root) / run_name
    suffix = selected_file_suffix(selection)
    metric_name = f"test_metrics{suffix}.csv"
    panel_name = f"test_episode_panel{suffix}.csv"
    metric_files = sorted(run_dir.rglob(metric_name))
    panel_files = sorted(run_dir.rglob(panel_name))
    if not metric_files:
        raise FileNotFoundError(f"No selected metric files found under {run_dir} matching {metric_name}")
    if not panel_files:
        raise FileNotFoundError(f"No selected panel files found under {run_dir} matching {panel_name}")
    metrics = pd.concat([read_csv(p) for p in metric_files], ignore_index=True, sort=False)
    panel = pd.concat([read_csv(p) for p in panel_files], ignore_index=True, sort=False)
    metrics["checkpoint_selection"] = selection
    panel["checkpoint_selection"] = selection
    return metrics, panel


def resolve_qiao_run(root: Path, qiao_run: str | None) -> str:
    candidates = [qiao_run] if qiao_run else QIAO_RUN_CANDIDATES
    for name in candidates:
        if name and (hedgeset_root(root) / name / "hedgeset_test_metrics_all.csv").exists():
            return name
    raise FileNotFoundError(f"No Qiao run found. Tried: {candidates}")


def read_optional_run_from_roots(roots: list[Path], run_name: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    for root in roots:
        loaded = read_optional_run(root, run_name)
        if loaded is not None:
            return loaded
    return None


def read_best_run_from_roots(roots: list[Path], run_name: str, model: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    best: tuple[pd.DataFrame, pd.DataFrame] | None = None
    best_rows = -1
    for root in roots:
        loaded = read_optional_run(root, run_name)
        if loaded is None:
            continue
        metrics, panel = loaded
        count_df = metrics if model is None else metrics[metrics["model"].eq(model)]
        rows = len(count_df)
        if rows > best_rows:
            best = loaded
            best_rows = rows
    if best is None:
        raise FileNotFoundError(f"Run not found under any root: {run_name}")
    return best


def assemble_e1008(
    root: Path,
    aux_roots: list[Path],
    final_run: str,
    final_selection: str,
    supporting_modern_run: str,
    ivgrid_final_objective_run: str,
    ivgrid_final_objective_extra_runs: list[str],
    qiao_run: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_metrics_selected, final_panel_selected = read_selected_checkpoint_run(root, final_run, final_selection)
    final_metrics_all, final_panel_all = read_run(root, final_run)

    static_models = ["NoHedge", "SmiledDelta", "NoTradeDelta", "GreedyDeltaVega", "LSDeltaVega"]
    static_metrics = only_models(final_metrics_all, static_models)
    static_panel = only_models(final_panel_all, static_models)

    final_metrics = only_models(final_metrics_selected, [FNUG_MODEL])
    final_panel = only_models(final_panel_selected, [FNUG_MODEL])

    supporting_metrics, supporting_panel = read_run(root, supporting_modern_run)
    supporting_metrics = only_models(supporting_metrics, SUPPORTING_MODERN_MODELS)
    supporting_panel = only_models(supporting_panel, SUPPORTING_MODERN_MODELS)

    roots_for_extra = [root] + aux_roots
    ivgrid_metrics, ivgrid_panel = read_best_run_from_roots(
        roots_for_extra,
        ivgrid_final_objective_run,
        model="IVGrid-HedgeSet",
    )
    ivgrid_metric_parts = [only_models(ivgrid_metrics, ["IVGrid-HedgeSet"])]
    ivgrid_panel_parts = [only_models(ivgrid_panel, ["IVGrid-HedgeSet"])]
    for extra_run in ivgrid_final_objective_extra_runs:
        loaded = read_optional_run_from_roots(roots_for_extra, extra_run)
        if loaded is None:
            continue
        extra_metrics, extra_panel = loaded
        ivgrid_metric_parts.append(only_models(extra_metrics, ["IVGrid-HedgeSet"]))
        ivgrid_panel_parts.append(only_models(extra_panel, ["IVGrid-HedgeSet"]))
    ivgrid_metrics = pd.concat(ivgrid_metric_parts, ignore_index=True, sort=False)
    ivgrid_panel = pd.concat(ivgrid_panel_parts, ignore_index=True, sort=False)

    qiao_metrics, qiao_panel = read_run(root, resolve_qiao_run(root, qiao_run))
    qiao_metrics = only_models(qiao_metrics, ["QiaoResidual-LS"])
    qiao_panel = only_models(qiao_panel, ["QiaoResidual-LS"])

    metrics = pd.concat(
        [final_metrics, static_metrics, supporting_metrics, qiao_metrics, ivgrid_metrics],
        ignore_index=True,
        sort=False,
    )
    panel = pd.concat(
        [final_panel, static_panel, supporting_panel, qiao_panel, ivgrid_panel],
        ignore_index=True,
        sort=False,
    )
    if "lambda_regime_excess_cvar" in metrics.columns:
        metrics["lambda_regime_excess_cvar"] = 0.0
    if "lambda_regime_excess_cvar" in panel.columns:
        panel["lambda_regime_excess_cvar"] = 0.0
    return metrics, panel


def _case_metrics(metrics: pd.DataFrame, family: str, cost: float, seed: int) -> pd.DataFrame:
    seeds = pd.to_numeric(metrics.get("seed", pd.Series(np.nan, index=metrics.index)), errors="coerce")
    costs = pd.to_numeric(metrics["cost_bps"], errors="coerce")
    exact_seed = np.isclose(seeds, float(seed), equal_nan=False)
    static_seed = seeds.isna()
    return metrics[
        metrics["family"].eq(family)
        & np.isclose(costs, float(cost))
        & (exact_seed | static_seed)
    ].copy()


def write_e1008_summary_tables(metrics: pd.DataFrame, outdir: Path) -> None:
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
        "checkpoint_selection",
    ]
    present = [c for c in cols if c in metrics.columns]
    main = _case_metrics(metrics, "atm_call", 2.0, 42)
    main[present].to_csv(outdir / "e1008_final_main_cell_metrics.csv", index=False)

    rows = []
    for family in ["atm_call", "put25", "atm_straddle"]:
        for cost in [1.0, 2.0, 5.0]:
            ls = _case_metrics(metrics, family, cost, 42)
            ls = ls[ls["model"].eq("LSDeltaVega")]
            if ls.empty:
                continue
            ls_cvar = float(ls.iloc[0]["cvar_95"])
            for seed in [1, 42, 2026]:
                fnug = _case_metrics(metrics, family, cost, seed)
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
    pd.DataFrame(rows).to_csv(outdir / "e1008_final_heatmap_values.csv", index=False)

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
    pd.DataFrame(red_rows).to_csv(outdir / "e1008_final_vs_ls_reductions.csv", index=False)

    fnug_all = metrics[metrics["model"].eq(FNUG_MODEL)].copy()
    family_rows = []
    for family, grp in fnug_all.groupby("family"):
        family_rows.append({
            "family": family,
            "n_cells": int(len(grp)),
            "avg_cvar_95": float(grp["cvar_95"].mean()),
            "avg_cvar_99": float(grp["cvar_99"].mean()),
            "avg_worst_regime_cvar_95": float(grp["worst_regime_cvar_95"].mean()),
            "avg_excess_cvar_95": float(grp["excess_cvar_95"].mean()),
            "avg_mean_tc_norm": float(grp["mean_tc_norm"].mean()),
            "avg_mean_turnover": float(grp["mean_turnover"].mean()),
        })
    pd.DataFrame(family_rows).to_csv(outdir / "e1008_final_family_summary.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild E1@0.008 final FNUG analysis tables.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--aux-root", action="append", type=Path, default=[])
    parser.add_argument("--final-run", default=E1008_RUN)
    parser.add_argument("--final-selection", default=E1008_SELECTION)
    parser.add_argument("--supporting-modern-run", default=SUPPORTING_MODERN_RUN)
    parser.add_argument("--ivgrid-final-objective-run", default=IVGRID_FINAL_OBJECTIVE_RUN)
    parser.add_argument("--ivgrid-final-objective-extra-runs", default=",".join(IVGRID_FINAL_OBJECTIVE_EXTRA_RUNS))
    parser.add_argument("--qiao-run", default=None)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    aux_roots = [p.resolve() for p in args.aux_root]
    outdir = args.outdir or (root / "results" / "analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    extra_runs = [r.strip() for r in str(args.ivgrid_final_objective_extra_runs).split(",") if r.strip()]
    metrics, panel = assemble_e1008(
        root=root,
        aux_roots=aux_roots,
        final_run=args.final_run,
        final_selection=args.final_selection,
        supporting_modern_run=args.supporting_modern_run,
        ivgrid_final_objective_run=args.ivgrid_final_objective_run,
        ivgrid_final_objective_extra_runs=extra_runs,
        qiao_run=args.qiao_run,
    )
    metrics.to_csv(outdir / "e1008_final_combined_metrics.csv", index=False)
    panel.to_csv(outdir / "e1008_final_combined_episode_panel.csv", index=False)

    boot = build_bootstrap(panel, n_boot=args.n_boot)
    boot.to_csv(outdir / "e1008_final_pairwise_bootstrap.csv", index=False)
    write_e1008_summary_tables(metrics, outdir)
    ivgrid_summary = build_ivgrid_matched_cell_summary(metrics)
    ivgrid_summary.to_csv(outdir / "e1008_ivgrid_matched_cell_summary.csv", index=False)

    manifest = outdir / "e1008_final_artifact_manifest.md"
    manifest.write_text(
        "\n".join([
            "# E1008 final analysis manifest",
            "",
            f"- final_run: `{args.final_run}`",
            f"- final_selection: `{args.final_selection}`",
            f"- qiao_run: `{resolve_qiao_run(root, args.qiao_run)}`",
            f"- bootstrap_reps: `{args.n_boot}`",
            f"- combined_metrics_rows: `{len(metrics)}`",
            f"- combined_panel_rows: `{len(panel)}`",
            f"- bootstrap_rows: `{len(boot)}`",
            f"- ivgrid_summary_rows: `{len(ivgrid_summary)}`",
            "",
        ]),
        encoding="utf-8",
    )
    print(f"Wrote E1008 final metrics/panel/bootstrap to {outdir}")
    print(f"Bootstrap rows: {len(boot)}")
    print(f"IVGrid matched-cell summary rows: {len(ivgrid_summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
