# -*- coding: utf-8 -*-
"""Regenerate FNUG paper figures from the released E1008 analysis files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FIG2_MODELS = [
    "HedgeSet-latent",
    "LSDeltaVega",
    "IVGrid-HedgeSet",
    "NTB-LS",
    "CVaR-HedgeSet",
]
DISPLAY = {
    "HedgeSet-latent": "FNUG",
    "LSDeltaVega": "LSDeltaVega",
    "IVGrid-HedgeSet": "IVGrid-HedgeSet",
    "NTB-LS": "NTB-LS",
    "CVaR-HedgeSet": "CVaR-HedgeSet",
}
COLORS = {
    "HedgeSet-latent": "#0057B8",
    "LSDeltaVega": "#E69F00",
    "IVGrid-HedgeSet": "#009E73",
    "NTB-LS": "#D55E00",
    "CVaR-HedgeSet": "#7E57C2",
    "calm": "#1f77b4",
    "middle": "#ff7f0e",
    "stress": "#2ca02c",
    "unknown": "#7f7f7f",
}
LINESTYLES = {
    "HedgeSet-latent": "-",
    "LSDeltaVega": "--",
    "IVGrid-HedgeSet": "-.",
    "NTB-LS": ":",
    "CVaR-HedgeSet": "-",
}
LIABILITIES = [("ATM call", "atm_call"), ("25-delta put", "put25"), ("ATM straddle", "atm_straddle")]
COSTS = [1, 2, 5]
SEEDS = [1, 42, 2026]
REGIMES = ["calm", "middle", "stress"]


def resolve_results_dir(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {p}")
    if (p / "analysis").exists():
        return p
    if (p / "results" / "analysis").exists():
        return p / "results"
    if p.name == "analysis" and (p / "e1008_final_combined_episode_panel.csv").exists():
        return p.parent
    raise FileNotFoundError(
        f"Could not find an E1008 analysis directory under {p}. "
        "Use the repository root or the results directory."
    )


def analysis_path(results_dir: Path, name: str) -> Path:
    path = results_dir / "analysis" / name
    if not path.exists():
        raise FileNotFoundError(f"Missing analysis file: {path}")
    return path


def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, lineterminator="\n")


def var_cvar(values: Iterable[float], alpha: float = 0.95) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    var = float(np.quantile(arr, alpha))
    tail = arr[arr >= var]
    return var, float(tail.mean()) if tail.size else np.nan


def empirical_survival(values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(list(values), dtype=float)
    arr = np.sort(arr[np.isfinite(arr)])
    if arr.size == 0:
        return np.array([]), np.array([])
    return arr, (arr.size - np.arange(arr.size)) / (arr.size + 1.0)


def main_cell(panel: pd.DataFrame) -> pd.DataFrame:
    seed = pd.to_numeric(panel["seed"], errors="coerce")
    cost = pd.to_numeric(panel["cost_bps"], errors="coerce")
    mask = (
        panel["family"].eq("atm_call")
        & cost.eq(2.0)
        & (seed.eq(42.0) | seed.isna())
    )
    return panel.loc[mask].copy()


def make_figure2(results_dir: Path, outdir: Path) -> None:
    panel = pd.read_csv(analysis_path(results_dir, "e1008_final_combined_episode_panel.csv"), low_memory=False)
    df = main_cell(panel)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    summary_rows: list[dict[str, float | str | int]] = []

    for model in FIG2_MODELS:
        vals = pd.to_numeric(df.loc[df["model"].eq(model), "loss_norm"], errors="coerce").dropna().to_numpy()
        if vals.size == 0:
            continue
        x, y = empirical_survival(vals)
        ax.step(x, y, where="post", label=DISPLAY.get(model, model), color=COLORS.get(model), linestyle=LINESTYLES.get(model, "-"))
        var95, cvar95 = var_cvar(vals, 0.95)
        ax.scatter([var95], [0.05], color=COLORS.get(model), s=26, zorder=4)
        ax.vlines(cvar95, 0.0, 0.018, color=COLORS.get(model), linewidth=2.0)
        summary_rows.append({
            "model": DISPLAY.get(model, model),
            "n": int(vals.size),
            "var_95": var95,
            "cvar_95": cvar95,
            "mean_loss": float(np.mean(vals)),
        })

    all_tail = pd.to_numeric(df.loc[df["model"].isin(FIG2_MODELS), "loss_norm"], errors="coerce").dropna().to_numpy()
    if all_tail.size:
        lo = float(np.quantile(all_tail, 0.78))
        hi = float(np.quantile(all_tail, 0.995))
        ax.set_xlim(lo, hi)
        ax.set_ylim(0.0, 0.23)
    ax.set_xlabel("Normalized loss")
    ax.set_ylabel("Empirical survival probability")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper right", ncol=1)
    save_figure(fig, outdir, "figure2_tail_loss_survival")
    plt.close(fig)
    write_csv(pd.DataFrame(summary_rows), outdir / "figure2_tail_loss_summary.csv")


def make_figure3(results_dir: Path, outdir: Path) -> None:
    panel = pd.read_csv(analysis_path(results_dir, "e1008_final_combined_episode_panel.csv"), low_memory=False)
    df = main_cell(panel)
    fnug = df.loc[df["model"].eq("HedgeSet-latent"), ["start_date", "loss_norm", "regime_name"]].copy()
    ls = df.loc[df["model"].eq("LSDeltaVega"), ["start_date", "loss_norm"]].copy()
    fnug = fnug.rename(columns={"loss_norm": "fnug_loss", "regime_name": "regime"})
    ls = ls.rename(columns={"loss_norm": "ls_loss"})
    merged = fnug.merge(ls, on="start_date", how="inner")
    merged["excess_loss"] = pd.to_numeric(merged["fnug_loss"], errors="coerce") - pd.to_numeric(merged["ls_loss"], errors="coerce")
    merged = merged.sort_values("ls_loss").reset_index(drop=True)
    merged["rank"] = np.arange(1, len(merged) + 1)
    merged["rolling_mean_15"] = merged["excess_loss"].rolling(15, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for regime, grp in merged.groupby(merged["regime"].fillna("unknown")):
        ax.scatter(grp["rank"], grp["excess_loss"], s=18, alpha=0.75, label=str(regime), color=COLORS.get(str(regime), COLORS["unknown"]))
    ax.plot(merged["rank"], merged["rolling_mean_15"], color="black", linestyle="--", linewidth=1.5, label="15-episode mean")
    ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.6)
    ax.set_xlabel("Episodes sorted by LSDeltaVega loss")
    ax.set_ylabel("FNUG loss minus LSDeltaVega loss")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=8.5)
    save_figure(fig, outdir, "figure3_paired_excess_loss_vs_ls")
    plt.close(fig)

    write_csv(merged, outdir / "figure3_paired_excess_data.csv")
    write_csv(pd.DataFrame([{
        "n_episodes": len(merged),
        "mean_excess_loss": float(merged["excess_loss"].mean()),
        "median_excess_loss": float(merged["excess_loss"].median()),
        "share_fnug_better": float((merged["excess_loss"] < 0).mean()),
        "share_fnug_worse": float((merged["excess_loss"] > 0).mean()),
    }]), outdir / "figure3_paired_excess_summary.csv")


def make_figure4(results_dir: Path, outdir: Path) -> None:
    heat = pd.read_csv(analysis_path(results_dir, "e1008_final_heatmap_values.csv"))
    mats: dict[str, np.ndarray] = {}
    all_vals: list[float] = []
    for title, family in LIABILITIES:
        mat = np.full((len(COSTS), len(SEEDS)), np.nan)
        for i, cost in enumerate(COSTS):
            for j, seed in enumerate(SEEDS):
                row = heat.loc[
                    heat["family"].eq(family)
                    & pd.to_numeric(heat["cost_bps"], errors="coerce").eq(float(cost))
                    & pd.to_numeric(heat["seed"], errors="coerce").eq(float(seed))
                ]
                if not row.empty:
                    val = float(row.iloc[0]["diff_fnug_minus_ls"])
                    mat[i, j] = val
                    all_vals.append(val)
        mats[title] = mat

    vabs = max(abs(np.nanmin(all_vals)), abs(np.nanmax(all_vals))) if all_vals else 1.0
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.45), constrained_layout=True)
    image = None
    for ax, (title, _) in zip(axes, LIABILITIES):
        mat = mats[title]
        image = ax.imshow(mat, aspect="auto", cmap="RdBu_r", norm=norm)
        ax.set_xticks(np.arange(len(SEEDS)), [str(s) for s in SEEDS])
        ax.set_yticks(np.arange(len(COSTS)), [f"{c} bps" for c in COSTS])
        ax.set_xlabel("Seed")
        if ax is axes[0]:
            ax.set_ylabel("Transaction cost")
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                text_color = "white" if np.isfinite(val) and abs(val) > 0.62 * vabs else "black"
                ax.text(j, i, "NA" if not np.isfinite(val) else f"{val:.4f}", ha="center", va="center", color=text_color, fontsize=9.5)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.86, pad=0.025)
        cbar.set_label(r"$\Delta$CVaR$_{95}$ vs. LS")
    save_figure(fig, outdir, "figure4_cost_seed_robustness_heatmap")
    plt.close(fig)
    out = heat.rename(columns={
        "family": "liability",
        "fnug_cvar_95": "fnug_cvar95",
        "ls_cvar_95": "ls_cvar95_static",
        "diff_fnug_minus_ls": "cvar95_diff_fnug_minus_ls",
    }).copy()
    write_csv(out, outdir / "figure4_heatmap_values.csv")


def grouped_boxplot(ax: plt.Axes, df: pd.DataFrame, value_col: str, ylabel: str) -> None:
    groups = []
    labels = []
    for regime in REGIMES:
        vals = pd.to_numeric(df.loc[df["regime_name"].astype(str).str.lower().eq(regime), value_col], errors="coerce").dropna().to_numpy()
        if vals.size:
            groups.append(vals)
            labels.append(regime)
    if not groups:
        raise ValueError(f"No finite values for {value_col}")
    try:
        bp = ax.boxplot(
            groups,
            tick_labels=labels,
            patch_artist=True,
            showfliers=False,
            widths=0.62,
            medianprops={"linewidth": 1.5, "color": "black"},
        )
    except TypeError:
        bp = ax.boxplot(
            groups,
            labels=labels,
            patch_artist=True,
            showfliers=False,
            widths=0.62,
            medianprops={"linewidth": 1.5, "color": "black"},
        )
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(COLORS.get(label, COLORS["unknown"]))
        patch.set_alpha(0.42)
    ax.set_xlabel("Regime")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)


def make_figure5(results_dir: Path, outdir: Path) -> None:
    panel = pd.read_csv(analysis_path(results_dir, "e1008_final_combined_episode_panel.csv"), low_memory=False)
    df = panel.loc[panel["model"].eq("HedgeSet-latent")].copy()
    df["_residual_delta_scaled"] = pd.to_numeric(df["residual_delta_exposure"], errors="coerce") * 1e5
    df["_residual_vega_scaled"] = pd.to_numeric(df["residual_vega_exposure"], errors="coerce") * 1e6
    df["_global_trust"] = pd.to_numeric(df["global_trust_mean"], errors="coerce")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.35), constrained_layout=True)
    grouped_boxplot(axes[0], df, "_residual_delta_scaled", r"Residual delta exposure ($\times 10^{-5}$)")
    grouped_boxplot(axes[1], df, "_residual_vega_scaled", r"Residual vega exposure ($\times 10^{-6}$)")
    grouped_boxplot(axes[2], df, "_global_trust", "Global trust gate")
    axes[2].set_ylim(0.0, min(1.0, max(0.65, float(np.nanquantile(df["_global_trust"], 0.99)) + 0.05)))
    save_figure(fig, outdir, "figure5_residual_diagnostics")
    plt.close(fig)

    summary = df.groupby("regime_name").agg({
        "residual_delta_exposure": ["mean", "median", "std"],
        "residual_vega_exposure": ["mean", "median", "std"],
        "global_trust_mean": ["mean", "median", "std"],
        "residual_factor_l2": ["mean", "median", "std"],
        "excess_loss_norm": ["mean", "median", "std"],
    })
    summary.to_csv(outdir / "figure5_regime_diagnostic_summary.csv", lineterminator="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Repository root or results directory containing analysis/e1008_*.csv")
    parser.add_argument("--outdir", default="results/paper_figures", help="Output directory")
    args = parser.parse_args()

    results_dir = resolve_results_dir(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Resolved results directory: {results_dir}")
    print(f"Saving figures to: {outdir}")
    make_figure2(results_dir, outdir)
    print("Saved Figure 2")
    make_figure3(results_dir, outdir)
    print("Saved Figure 3")
    make_figure4(results_dir, outdir)
    print("Saved Figure 4")
    make_figure5(results_dir, outdir)
    print("Saved Figure 5")
    print("All done.")


if __name__ == "__main__":
    main()
