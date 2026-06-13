from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL_RUN = "fnug_e1008_final_fam3_cost3_seed3_1eaa24b3f8"
FINAL_SELECTION = "best_feasible_excess_0p008"
AUDIT_SEEDS = [1, 42, 2026]
ABLATIONS = {
    "No factor-neutral constraint": ("e1008_ablation_no_factorneutral_1eaa24b3f8", "best_score"),
    "No trust gate": ("e1008_ablation_no_trustgate_1eaa24b3f8", FINAL_SELECTION),
    "No auxiliary risk": ("e1008_ablation_no_auxrisk_1eaa24b3f8", FINAL_SELECTION),
}
QIAO_RUN_CANDIDATES = [
    "qiao_residual_ls_m4_1eaa24b3f8",
    "qiao_ls_m4_1eaa24b3f8",
]

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


def suffix(selection: str) -> str:
    if selection == "best_score":
        return ""
    if selection == "best_feasible":
        return "_best_feasible"
    if selection.startswith("best_feasible_excess_"):
        return f"_{selection}"
    raise ValueError(selection)


def cell_dir(root: Path, run_name: str, seed: int = 42) -> Path:
    return (
        root / "results" / "hedgeset" / run_name / "atm_call" / "cost_2bps" /
        "variant_full" / "scope_full" / "policy_latent" / f"seed_{seed}"
    )


def read_selected_panel(root: Path, run_name: str, label: str, selection: str, seed: int = 42) -> pd.DataFrame:
    path = cell_dir(root, run_name, seed) / f"test_episode_panel{suffix(selection)}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    out = pd.read_csv(path)
    out = out[out["model"].eq("HedgeSet-latent")].copy()
    out["model"] = label
    out["checkpoint_selection"] = selection
    out["audit_seed"] = seed
    return out


def read_selected_metrics(root: Path, run_name: str, label: str, selection: str, seed: int = 42) -> pd.Series:
    path = cell_dir(root, run_name, seed) / f"test_metrics{suffix(selection)}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    out = pd.read_csv(path)
    row = out[out["model"].eq("HedgeSet-latent")].iloc[0].copy()
    row["model"] = label
    row["checkpoint_selection"] = selection
    row["audit_seed"] = seed
    return row


def resolve_qiao_run(root: Path) -> str:
    for run in QIAO_RUN_CANDIDATES:
        if (root / "results" / "hedgeset" / run / "hedgeset_test_episode_panel_all.csv").exists():
            return run
    raise FileNotFoundError(f"No Qiao run found under {root}")


def read_external_case(root: Path, run_name: str, model: str, label: str, seed: int = 42) -> tuple[pd.Series, pd.DataFrame]:
    run_dir = root / "results" / "hedgeset" / run_name
    metrics = pd.read_csv(run_dir / "hedgeset_test_metrics_all.csv")
    panel = pd.read_csv(run_dir / "hedgeset_test_episode_panel_all.csv")
    costs = pd.to_numeric(metrics["cost_bps"], errors="coerce")
    seeds = pd.to_numeric(metrics.get("seed", pd.Series(np.nan, index=metrics.index)), errors="coerce")
    m = metrics[
        metrics["model"].eq(model)
        & metrics["family"].eq("atm_call")
        & np.isclose(costs, 2.0)
        & (np.isclose(seeds, float(seed)) | seeds.isna())
    ].copy()
    if m.empty:
        raise ValueError(f"No metric row for {model} in {run_name}")
    p_costs = pd.to_numeric(panel["cost_bps"], errors="coerce")
    p_seeds = pd.to_numeric(panel.get("seed", pd.Series(np.nan, index=panel.index)), errors="coerce")
    p = panel[
        panel["model"].eq(model)
        & panel["family"].eq("atm_call")
        & np.isclose(p_costs, 2.0)
        & (np.isclose(p_seeds, float(seed)) | p_seeds.isna())
    ].copy()
    if p.empty:
        raise ValueError(f"No panel rows for {model} in {run_name}")
    row = m.iloc[0].copy()
    row["model"] = label
    row["checkpoint_selection"] = "external"
    row["seed"] = seed
    row["audit_seed"] = seed
    p["model"] = label
    p["checkpoint_selection"] = "external"
    p["seed"] = seed
    p["audit_seed"] = seed
    return row, p


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


def paired_bootstrap(full: pd.DataFrame, other: pd.DataFrame, metric: str, n_boot: int, rng_seed: int) -> dict[str, object]:
    col = METRICS[metric]
    keys = ["start_date"]
    if "seed" in full.columns and "seed" in other.columns:
        keys.append("seed")
    cols = keys + [col]
    if "regime_id" in full.columns:
        cols.append("regime_id")
    merged = (
        full[cols].rename(columns={col: "full"})
        .merge(other[keys + [col]].rename(columns={col: "other"}), on=keys, how="inner")
    )
    if merged.empty:
        raise ValueError(f"No paired rows for {metric}")
    x_full = pd.to_numeric(merged["full"], errors="coerce").to_numpy(dtype=float)
    x_other = pd.to_numeric(merged["other"], errors="coerce").to_numpy(dtype=float)
    regimes = merged["regime_id"].to_numpy() if "regime_id" in merged.columns else None
    obs = stat(x_full, metric, regimes) - stat(x_other, metric, regimes)
    rng = np.random.default_rng(rng_seed)
    n = len(merged)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r = regimes[idx] if regimes is not None else None
        draws[i] = stat(x_full[idx], metric, r) - stat(x_other[idx], metric, r)
    centered = draws - obs
    return {
        "metric": metric,
        "n": int(n),
        "full": stat(x_full, metric, regimes),
        "comparator": stat(x_other, metric, regimes),
        "diff_full_minus_comparator": float(obs),
        "ci95_low": float(np.nanquantile(draws, 0.025)),
        "ci95_high": float(np.nanquantile(draws, 0.975)),
        "p_two_sided": float(min(np.mean(np.abs(centered) >= abs(obs)), 1.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild E1008 final ablation table and paired bootstrap.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    outdir = args.outdir or (root / "results" / "analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    full_label = "Full FNUG"
    rows = [read_selected_metrics(root, FINAL_RUN, full_label, FINAL_SELECTION)]
    panels = {full_label: read_selected_panel(root, FINAL_RUN, full_label, FINAL_SELECTION)}

    qiao_run = resolve_qiao_run(root)
    qiao_metrics, qiao_panel = read_external_case(root, qiao_run, "QiaoResidual-LS", "QiaoResidual-LS")
    rows.append(qiao_metrics)
    panels["QiaoResidual-LS"] = qiao_panel

    ls_metrics, ls_panel = read_external_case(root, FINAL_RUN, "LSDeltaVega", "LS only")
    rows.append(ls_metrics)
    panels["LS only"] = ls_panel

    for label, (run_name, selection) in ABLATIONS.items():
        rows.append(read_selected_metrics(root, run_name, label, selection))
        panels[label] = read_selected_panel(root, run_name, label, selection)

    table = pd.DataFrame(rows)
    table.to_csv(outdir / "e1008_ablation_table.csv", index=False)

    audit_rows: list[pd.Series] = []
    audit_panels: dict[str, list[pd.DataFrame]] = {
        full_label: [],
        "QiaoResidual-LS": [],
        "LS only": [],
        **{label: [] for label in ABLATIONS},
    }
    for seed in AUDIT_SEEDS:
        audit_rows.append(read_selected_metrics(root, FINAL_RUN, full_label, FINAL_SELECTION, seed=seed))
        audit_panels[full_label].append(read_selected_panel(root, FINAL_RUN, full_label, FINAL_SELECTION, seed=seed))

        qiao_row, qiao_seed_panel = read_external_case(root, qiao_run, "QiaoResidual-LS", "QiaoResidual-LS", seed=seed)
        audit_rows.append(qiao_row)
        audit_panels["QiaoResidual-LS"].append(qiao_seed_panel)

        ls_row, ls_seed_panel = read_external_case(root, FINAL_RUN, "LSDeltaVega", "LS only", seed=seed)
        audit_rows.append(ls_row)
        audit_panels["LS only"].append(ls_seed_panel)

        for label, (run_name, selection) in ABLATIONS.items():
            audit_rows.append(read_selected_metrics(root, run_name, label, selection, seed=seed))
            audit_panels[label].append(read_selected_panel(root, run_name, label, selection, seed=seed))

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(outdir / "e1008_ablation_seed_audit.csv", index=False)
    summary_metrics = [
        "cvar_95",
        "cvar_99",
        "worst_regime_cvar_95",
        "excess_cvar_95",
        "mean_relu_excess_loss_norm",
        "mean_abs_residual_delta_exposure",
        "mean_abs_residual_vega_exposure",
        "mean_global_trust",
        "mean_tc_norm",
        "mean_turnover",
    ]
    present_summary_metrics = [c for c in summary_metrics if c in audit.columns]
    summary = (
        audit.groupby(["model", "checkpoint_selection"], dropna=False)[present_summary_metrics]
        .mean(numeric_only=True)
        .reset_index()
    )
    summary["n_seeds"] = (
        audit.groupby(["model", "checkpoint_selection"], dropna=False)["audit_seed"]
        .nunique()
        .to_numpy()
    )
    summary.to_csv(outdir / "e1008_ablation_seed_summary.csv", index=False)

    boot_rows: list[dict[str, object]] = []
    metrics = [
        "mean_loss",
        "cvar_95",
        "cvar_99",
        "worst_regime_cvar_95",
        "excess_cvar_95",
        "mean_positive_excess",
        "mean_residual_delta_exposure",
        "mean_residual_vega_exposure",
        "mean_global_trust",
        "mean_turnover",
        "mean_tc_norm",
    ]
    full_panel = panels[full_label]
    for comp_idx, (label, panel) in enumerate(panels.items()):
        if label == full_label:
            continue
        for metric_idx, metric in enumerate(metrics):
            row = paired_bootstrap(full_panel, panel, metric, args.n_boot, 30_000 + 101 * comp_idx + metric_idx)
            row["comparison"] = f"Full FNUG vs {label}"
            row["comparator_label"] = label
            boot_rows.append(row)
    boot = pd.DataFrame(boot_rows)
    cols = ["comparison", "comparator_label"] + [c for c in boot.columns if c not in {"comparison", "comparator_label"}]
    boot[cols].to_csv(outdir / "e1008_ablation_pairwise_bootstrap.csv", index=False)

    multiseed_panels = {
        label: pd.concat(parts, ignore_index=True, sort=False)
        for label, parts in audit_panels.items()
    }
    multiseed_rows: list[dict[str, object]] = []
    full_multiseed = multiseed_panels[full_label]
    for comp_idx, (label, panel) in enumerate(multiseed_panels.items()):
        if label == full_label:
            continue
        for metric_idx, metric in enumerate(metrics):
            row = paired_bootstrap(full_multiseed, panel, metric, args.n_boot, 40_000 + 101 * comp_idx + metric_idx)
            row["comparison"] = f"Full FNUG vs {label}"
            row["comparator_label"] = label
            multiseed_rows.append(row)
    multiseed_boot = pd.DataFrame(multiseed_rows)
    multiseed_boot[cols].to_csv(outdir / "e1008_ablation_multiseed_pairwise_bootstrap.csv", index=False)

    print(
        f"Wrote {len(table)} canonical ablation rows, {len(audit)} seed-audit rows, "
        f"{len(boot)} canonical bootstrap rows, and {len(multiseed_boot)} multiseed bootstrap rows to {outdir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
