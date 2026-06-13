from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from rebuild_final_noreg_analysis import (
    BOOT_METRICS,
    COMPARATORS,
    FNUG_MODEL,
    LEGACY_MODERN_MODELS,
    build_bootstrap,
    build_ivgrid_matched_cell_summary,
    hedgeset_root,
    only_models,
    read_csv,
    read_optional_run,
    read_run,
)


ROOT = Path(__file__).resolve().parents[1]

E1008_RUN = "budget_replay_E1_fam3_cost3_seed3_1eaa24b3f8"
E1008_SELECTION = "best_feasible_excess_0p008"
LEGACY_MODERN_RUN = "fnug_modern_fam3_cost3_seed3_m4_1eaa24b3f8"
IVGRID_NOREG_RUN = "fnug_modern_noreg_fam3_cost3_seed3_m4_1eaa24b3f8"
IVGRID_NOREG_EXTRA_RUNS = ["fnug_modern_noreg_straddle_m4_1eaa24b3f8"]
QIAO_RUN_CANDIDATES = [
    "paper_add2_qiao_m4_20260526_1eaa24b3f8",
    "qiao_ls_m4_1eaa24b3f8",
]


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
    legacy_modern_run: str,
    ivgrid_noreg_run: str,
    ivgrid_noreg_extra_runs: list[str],
    qiao_run: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_metrics_selected, final_panel_selected = read_selected_checkpoint_run(root, final_run, final_selection)
    final_metrics_all, final_panel_all = read_run(root, final_run)

    static_models = ["NoHedge", "SmiledDelta", "NoTradeDelta", "GreedyDeltaVega", "LSDeltaVega"]
    static_metrics = only_models(final_metrics_all, static_models)
    static_panel = only_models(final_panel_all, static_models)

    final_metrics = only_models(final_metrics_selected, [FNUG_MODEL])
    final_panel = only_models(final_panel_selected, [FNUG_MODEL])

    legacy_metrics, legacy_panel = read_run(root, legacy_modern_run)
    legacy_metrics = only_models(legacy_metrics, LEGACY_MODERN_MODELS)
    legacy_panel = only_models(legacy_panel, LEGACY_MODERN_MODELS)

    roots_for_extra = [root] + aux_roots
    ivgrid_metrics, ivgrid_panel = read_best_run_from_roots(roots_for_extra, ivgrid_noreg_run, model="IVGrid-HedgeSet")
    ivgrid_metric_parts = [only_models(ivgrid_metrics, ["IVGrid-HedgeSet"])]
    ivgrid_panel_parts = [only_models(ivgrid_panel, ["IVGrid-HedgeSet"])]
    for extra_run in ivgrid_noreg_extra_runs:
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
        [final_metrics, static_metrics, legacy_metrics, qiao_metrics, ivgrid_metrics],
        ignore_index=True,
        sort=False,
    )
    panel = pd.concat(
        [final_panel, static_panel, legacy_panel, qiao_panel, ivgrid_panel],
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
    parser.add_argument("--legacy-modern-run", default=LEGACY_MODERN_RUN)
    parser.add_argument("--ivgrid-noreg-run", default=IVGRID_NOREG_RUN)
    parser.add_argument("--ivgrid-noreg-extra-runs", default=",".join(IVGRID_NOREG_EXTRA_RUNS))
    parser.add_argument("--qiao-run", default=None)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    aux_roots = [p.resolve() for p in args.aux_root]
    outdir = args.outdir or (root / "results" / "analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    extra_runs = [r.strip() for r in str(args.ivgrid_noreg_extra_runs).split(",") if r.strip()]
    metrics, panel = assemble_e1008(
        root=root,
        aux_roots=aux_roots,
        final_run=args.final_run,
        final_selection=args.final_selection,
        legacy_modern_run=args.legacy_modern_run,
        ivgrid_noreg_run=args.ivgrid_noreg_run,
        ivgrid_noreg_extra_runs=extra_runs,
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
