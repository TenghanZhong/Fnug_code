# -*- coding: utf-8 -*-
"""
Plotting script for FNUG paper Figures 2--5.

Output filenames are intentionally unchanged:
  figure2_tail_loss_survival.{pdf,png}
  figure3_paired_excess_loss_vs_ls.{pdf,png}
  figure4_cost_seed_robustness_heatmap.{pdf,png}
  figure5_residual_diagnostics.{pdf,png}

Windows PowerShell example:
python .\\src\\plot_fnug.py --input .\\results --outdir .\\results\\paper_figures
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# Paper-style plotting defaults
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.9,
    "lines.linewidth": 2.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# -----------------------------------------------------------------------------
# Run directory names
# -----------------------------------------------------------------------------
METHOD_DIRS = {
    "FNUG": "fnug_modern_fam3_cost3_seed3_m4_1eaa24b3f8",
    # Optional hard-LS-only run. Figure 2/3 use it if it exists.
    # Figure 4 deliberately reads LSDeltaVega from the static baseline panel
    # inside the main FNUG run.
    "LSDeltaVega": "A5_hardls_only_1eaa24b3f8",
}

# Internal labels are the keys below. Display labels are defined separately for
# plot legends.
MODERN_BASELINES = {
    "IVGrid-HedgeSet": "ivgrid",
    "NTB-LS": "ntb_ls",
    "CVaR-HedgeSet": "cvar_from_scratch",
}

DISPLAY_LABELS = {
    "FNUG": "FNUG",
    "LSDeltaVega": "LSDeltaVega",
    "IVGrid-HedgeSet": "IVGrid-HedgeSet",
    "NTB-LS": "NTB-LS",
    "CVaR-HedgeSet": "CVaR-HedgeSet (scratch)",
}

FIG2_ORDER = [
    "FNUG",
    "LSDeltaVega",
    "IVGrid-HedgeSet",
    "NTB-LS",
    "CVaR-HedgeSet",
]

LIABILITIES_FOR_HEATMAP = [
    ("ATM call", "atm_call"),
    ("put25", "put25"),
    ("ATM straddle", "atm_straddle"),
]
COSTS_FOR_HEATMAP = [1, 2, 5]
SEEDS_FOR_HEATMAP = [1, 42, 2026]
REGIME_ORDER = ["calm", "middle", "stress"]

COLORS = {
    "FNUG": "#0057B8",
    "LSDeltaVega": "#E69F00",
    "IVGrid-HedgeSet": "#009E73",
    "NTB-LS": "#D55E00",
    "CVaR-HedgeSet": "#7E57C2",
    "calm": "#1f77b4",
    "middle": "#ff7f0e",
    "stress": "#2ca02c",
    "unknown": "#7f7f7f",
    "Unknown": "#7f7f7f",
}

LINESTYLES = {
    "FNUG": "-",
    "LSDeltaVega": "--",
    "IVGrid-HedgeSet": "-.",
    "NTB-LS": ":",
    "CVaR-HedgeSet": "-",
}

# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def resolve_results_root(input_path: str | Path) -> Path:
    """Return project root that contains results/hedgeset."""
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {p}")

    if (p / "results" / "hedgeset").exists():
        return p
    if (p / "hedgeset").exists():
        return p.parent

    candidates = list(p.rglob("results/hedgeset")) if p.is_dir() else []
    if candidates:
        return candidates[0].parent.parent

    raise FileNotFoundError(
        f"Could not find results/hedgeset under: {p}\n"
        "Use either the project root or the results folder."
    )


def save_png_pdf(fig: plt.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")


def pick_col(df: pd.DataFrame, candidates: Sequence[str] | Iterable[str], required: bool = True) -> Optional[str]:
    candidates = list(candidates)
    cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in cols}
    for cand in candidates:
        if str(cand).lower() in lower_map:
            return lower_map[str(cand).lower()]
    for cand in candidates:
        cl = str(cand).lower()
        for c in cols:
            if cl in str(c).lower():
                return c
    if required:
        raise KeyError(f"None of candidate columns found: {candidates}\nAvailable columns: {cols}")
    return None


def get_loss_col(df: pd.DataFrame) -> str:
    return pick_col(df, [
        "loss_norm",
        "normalized_loss",
        "episode_loss_norm",
        "loss_normalized",
        "liability_loss_norm",
        "test_loss_norm",
        "loss",
    ], required=True)  # type: ignore[return-value]


def get_regime_col(df: pd.DataFrame, required: bool = False) -> Optional[str]:
    return pick_col(df, ["regime_name", "regime", "market_regime", "state_name", "state"], required=required)


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def compute_var_cvar(x: Iterable[float], alpha: float = 0.95) -> Tuple[float, float]:
    arr = np.asarray(list(x), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    q = float(np.quantile(arr, alpha))
    tail = arr[arr >= q]
    return q, float(np.mean(tail)) if tail.size else np.nan


def empirical_survival(x: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(list(x), dtype=float)
    arr = np.sort(arr[np.isfinite(arr)])
    n = len(arr)
    if n == 0:
        return np.array([]), np.array([])
    surv = (n - np.arange(n)) / (n + 1.0)
    return arr, surv


def model_normalized(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.lower()
        .str.replace(" ", "", regex=False)
        .str.replace("_", "", regex=False)
        .str.replace("-", "", regex=False)
    )


def is_ls_model_series(s: pd.Series) -> pd.Series:
    x = model_normalized(s)
    exact = x.isin(["lsdeltavega", "lsdelta", "ls"])
    substring = x.str.contains("ls", na=False) & x.str.contains("delta", na=False) & x.str.contains("vega", na=False)
    return exact | substring


def is_fnug_model_series(s: pd.Series) -> pd.Series:
    x = model_normalized(s)
    main = x.str.contains("hedgesetlatent", na=False) | x.str.contains("fnug", na=False)
    exclude = (
        x.str.contains("ivgrid", na=False)
        | x.str.contains("ntbls", na=False)
        | x.str.contains("cvar", na=False)
        | x.str.contains("smileddelta", na=False)
        | x.str.contains("notradedelta", na=False)
        | x.str.contains("greedydeltavega", na=False)
        | x.str.contains("lsdeltavega", na=False)
    )
    return main & (~exclude)


def get_id_cols_for_merge(df1: pd.DataFrame, df2: pd.DataFrame) -> Optional[list[str]]:
    candidate_sets = [
        ["episode_id"],
        ["start_date"],
        ["date"],
        ["trade_date"],
        ["test_date"],
        ["quote_date"],
        ["episode_date"],
        ["episode_id", "start_date"],
        ["episode_id", "date"],
    ]
    for cands in candidate_sets:
        if all(c in df1.columns for c in cands) and all(c in df2.columns for c in cands):
            return cands
    return None

# -----------------------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------------------
def fnug_case_panel_path(root: Path, liability: str, cost_bps: int, seed: int) -> Path:
    return (
        root / "results" / "hedgeset" / METHOD_DIRS["FNUG"] /
        liability / f"cost_{cost_bps}bps" / "variant_full" / "scope_full" /
        "policy_latent" / f"seed_{seed}" / "test_episode_panel.csv"
    )


def ls_case_panel_path(root: Path, liability: str, cost_bps: int, seed: int) -> Path:
    return (
        root / "results" / "hedgeset" / METHOD_DIRS["LSDeltaVega"] /
        liability / f"cost_{cost_bps}bps" / "variant_full" / "scope_full" /
        "policy_latent" / f"seed_{seed}" / "test_episode_panel.csv"
    )


def baseline_case_panel_path(root: Path, baseline_label: str, liability: str, cost_bps: int, seed: int) -> Path:
    baseline_dir = MODERN_BASELINES[baseline_label]
    return (
        root / "results" / "hedgeset" / METHOD_DIRS["FNUG"] /
        liability / f"cost_{cost_bps}bps" / "modern_baselines" /
        baseline_dir / "scope_full" / f"seed_{seed}" / "test_episode_panel.csv"
    )


def static_ls_panel_path(root: Path, liability: str, cost_bps: int) -> Path:
    return (
        root / "results" / "hedgeset" / METHOD_DIRS["FNUG"] /
        liability / f"cost_{cost_bps}bps" / "static" / "static_baseline_episode_panel.csv"
    )


def fnug_combined_panel_path(root: Path) -> Path:
    return root / "results" / "hedgeset" / METHOD_DIRS["FNUG"] / "hedgeset_test_episode_panel_all.csv"

# -----------------------------------------------------------------------------
# Data readers
# -----------------------------------------------------------------------------
def read_static_ls_panel(root: Path, liability: str, cost_bps: int) -> pd.DataFrame:
    path = static_ls_panel_path(root, liability, cost_bps)
    df = read_csv_safe(path)
    model_col = pick_col(df, ["model"], required=True)
    mask = is_ls_model_series(df[model_col])  # type: ignore[index]
    if not mask.any():
        raise ValueError(
            f"Cannot find LSDeltaVega rows in {path}. "
            f"Available models: {sorted(df[model_col].astype(str).unique())}"  # type: ignore[index]
        )
    return df.loc[mask].copy()


def read_ls_panel_for_case(root: Path, liability: str, cost_bps: int, seed: int) -> pd.DataFrame:
    direct = ls_case_panel_path(root, liability, cost_bps, seed)
    if direct.exists():
        return pd.read_csv(direct)
    return read_static_ls_panel(root, liability, cost_bps)


def get_fnug_case_cvar95(root: Path, liability: str, cost_bps: int, seed: int) -> float:
    df = read_csv_safe(fnug_case_panel_path(root, liability, cost_bps, seed))
    _, cvar95 = compute_var_cvar(df[get_loss_col(df)].to_numpy(), 0.95)
    return cvar95


def get_static_ls_cvar95(root: Path, liability: str, cost_bps: int) -> float:
    df = read_static_ls_panel(root, liability, cost_bps)
    _, cvar95 = compute_var_cvar(df[get_loss_col(df)].to_numpy(), 0.95)
    return cvar95

# -----------------------------------------------------------------------------
# Figure 2: tail survival
# -----------------------------------------------------------------------------
def make_figure2_tail_survival(root: Path, outdir: Path) -> None:
    """
    Optimized Figure 2.

    - Uses internal key "CVaR-HedgeSet" and display label "CVaR-HedgeSet (scratch)".
    - Shows a right-tail zoom so FNUG is visible against LSDeltaVega and IVGrid.
    - Draws FNUG last with a white underlay and thick blue line.
    - Marks VaR95 dots and CVaR95 vertical lines for FNUG / LS / IVGrid.
    """
    liability = "atm_call"
    cost_bps = 2
    seed = 42

    panels: Dict[str, np.ndarray] = {}

    df = read_csv_safe(fnug_case_panel_path(root, liability, cost_bps, seed))
    panels["FNUG"] = df[get_loss_col(df)].to_numpy()

    df = read_ls_panel_for_case(root, liability, cost_bps, seed)
    panels["LSDeltaVega"] = df[get_loss_col(df)].to_numpy()

    for label in MODERN_BASELINES:
        df = read_csv_safe(baseline_case_panel_path(root, label, liability, cost_bps, seed))
        panels[label] = df[get_loss_col(df)].to_numpy()

    draw_order = [
        "CVaR-HedgeSet",
        "NTB-LS",
        "LSDeltaVega",
        "IVGrid-HedgeSet",
        "FNUG",
    ]

    widths = {
        "FNUG": 3.2,
        "LSDeltaVega": 2.5,
        "IVGrid-HedgeSet": 2.5,
        "NTB-LS": 2.3,
        "CVaR-HedgeSet": 2.0,
    }
    alphas = {
        "FNUG": 1.0,
        "LSDeltaVega": 0.95,
        "IVGrid-HedgeSet": 0.95,
        "NTB-LS": 0.85,
        "CVaR-HedgeSet": 0.65,
    }

    fig, ax = plt.subplots(figsize=(8.8, 5.6))

    # Right-tail zoom: this is where CVaR95 differences are visible.
    ax.set_xlim(-0.005, 0.22)
    ax.set_ylim(6e-3, 1.2e-1)
    ax.set_yscale("log")

    for label in draw_order:
        x, surv = empirical_survival(panels[label])

        # White underlay makes FNUG visible even when curves overlap.
        if label == "FNUG":
            ax.step(
                x, surv,
                where="post",
                linewidth=5.2,
                linestyle="-",
                color="white",
                alpha=1.0,
                zorder=9,
            )

        ax.step(
            x, surv,
            where="post",
            linewidth=widths[label],
            linestyle=LINESTYLES[label],
            color=COLORS[label],
            alpha=alphas[label],
            label=DISPLAY_LABELS[label],
            zorder=10 if label == "FNUG" else 4,
        )

        var95, cvar95 = compute_var_cvar(panels[label], 0.95)

        ax.scatter(
            [var95],
            [0.05],
            s=42 if label == "FNUG" else 32,
            color=COLORS[label],
            edgecolor="white",
            linewidth=0.8,
            zorder=12 if label == "FNUG" else 8,
        )

        # CVaR95 vertical line for the three close competitors.
        if label in {"FNUG", "LSDeltaVega", "IVGrid-HedgeSet"}:
            ax.axvline(
                cvar95,
                color=COLORS[label],
                linestyle=LINESTYLES[label],
                linewidth=1.6,
                alpha=0.75,
                zorder=3,
            )

    ax.set_xlabel("Normalized loss")
    ax.set_ylabel("Tail survival probability")
    ax.grid(True, alpha=0.25, which="both")

    ax.text(
        0.003,
        7.3e-3,
        "dots: 95% VaR\nvertical lines: 95% CVaR",
        fontsize=9,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.75),
    )

    ax.legend(frameon=False, loc="upper right", ncol=1)

    save_png_pdf(fig, outdir, "figure2_tail_loss_survival")
    plt.close(fig)

    rows = []
    for label in FIG2_ORDER:
        var95, cvar95 = compute_var_cvar(panels[label], 0.95)
        var99, cvar99 = compute_var_cvar(panels[label], 0.99)
        rows.append({
            "method": DISPLAY_LABELS[label],
            "VaR95": var95,
            "CVaR95": cvar95,
            "VaR99": var99,
            "CVaR99": cvar99,
            "n": int(np.isfinite(panels[label]).sum()),
        })
    pd.DataFrame(rows).to_csv(outdir / "figure2_tail_loss_summary.csv", index=False)

# -----------------------------------------------------------------------------
# Figure 3: paired excess loss
# -----------------------------------------------------------------------------
def make_figure3_paired_excess_vs_ls(root: Path, outdir: Path, sort_mode: str = "ls_loss") -> None:
    liability, cost_bps, seed = "atm_call", 2, 42

    fnug = read_csv_safe(fnug_case_panel_path(root, liability, cost_bps, seed)).copy()
    ls = read_ls_panel_for_case(root, liability, cost_bps, seed).copy()

    loss_fnug = get_loss_col(fnug)
    loss_ls = get_loss_col(ls)
    regime_col = get_regime_col(fnug, required=False)
    merge_keys = get_id_cols_for_merge(fnug, ls)

    if merge_keys is not None:
        keep_fnug = merge_keys + [loss_fnug] + ([regime_col] if regime_col is not None else [])
        keep_ls = merge_keys + [loss_ls]
        tmp = fnug[keep_fnug].merge(ls[keep_ls], on=merge_keys, how="inner", suffixes=("_fnug", "_ls"))
        fnug_loss_col = f"{loss_fnug}_fnug" if f"{loss_fnug}_fnug" in tmp.columns else loss_fnug
        ls_loss_col = f"{loss_ls}_ls" if f"{loss_ls}_ls" in tmp.columns else loss_ls
    else:
        n = min(len(fnug), len(ls))
        tmp = pd.DataFrame({
            "row_id": np.arange(n),
            "loss_fnug": fnug[loss_fnug].iloc[:n].to_numpy(),
            "loss_ls": ls[loss_ls].iloc[:n].to_numpy(),
        })
        if regime_col is not None:
            tmp["regime_name"] = fnug[regime_col].iloc[:n].to_numpy()
        fnug_loss_col = "loss_fnug"
        ls_loss_col = "loss_ls"

    if tmp.empty:
        raise ValueError("FNUG and LS panels could not be paired. Check start_date or episode_id columns.")

    tmp["excess_loss"] = pd.to_numeric(tmp[fnug_loss_col], errors="coerce") - pd.to_numeric(tmp[ls_loss_col], errors="coerce")
    tmp = tmp.dropna(subset=["excess_loss", ls_loss_col]).copy()
    if sort_mode == "ls_loss":
        tmp = tmp.sort_values(ls_loss_col).reset_index(drop=True)
    else:
        tmp = tmp.reset_index(drop=True)

    x = np.arange(len(tmp))
    y = tmp["excess_loss"].to_numpy(dtype=float)
    regcol = "regime_name" if "regime_name" in tmp.columns else (regime_col if regime_col in tmp.columns else None)

    fig, ax = plt.subplots(figsize=(9.4, 5.6))

    if regcol is not None:
        reg_series = tmp[regcol].fillna("unknown").astype(str).str.lower()
        regimes = [r for r in REGIME_ORDER if r in set(reg_series)]
        regimes += [r for r in sorted(set(reg_series)) if r not in set(regimes)]
        for r in regimes:
            m = reg_series == r
            ax.scatter(
                x[m], y[m],
                s=22,
                alpha=0.78,
                label=r,
                color=COLORS.get(r, COLORS["unknown"]),
            )
    else:
        ax.scatter(x, y, s=22, alpha=0.78, label="episodes")

    ax.axhline(0.0, color="black", linewidth=2.4, zorder=1)
    if len(tmp) >= 15:
        roll = pd.Series(y).rolling(15, center=True, min_periods=5).mean()
        ax.plot(x, roll, linewidth=2.2, color="black", linestyle="--", alpha=0.9, label="15-episode mean")

    ax.set_xlabel("Test episodes sorted by LS loss")
    ax.set_ylabel(r"Excess loss: FNUG $-$ LSDeltaVega")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper right", ncol=4)

    ymin, ymax = np.nanmin(y), np.nanmax(y)
    pad = max(0.002, 0.08 * (ymax - ymin))
    ax.set_ylim(ymin - pad, ymax + pad)

    save_png_pdf(fig, outdir, "figure3_paired_excess_loss_vs_ls")
    plt.close(fig)

    pd.DataFrame([{
        "n_episodes": len(tmp),
        "mean_excess_loss": float(np.nanmean(y)),
        "median_excess_loss": float(np.nanmedian(y)),
        "share_fnug_better": float(np.mean(y < 0)),
        "share_fnug_worse": float(np.mean(y > 0)),
        "share_tie": float(np.mean(np.isclose(y, 0.0))),
    }]).to_csv(outdir / "figure3_paired_excess_summary.csv", index=False)
    tmp.to_csv(outdir / "figure3_paired_excess_data.csv", index=False)

# -----------------------------------------------------------------------------
# Figure 4: cost x seed robustness heatmap
# -----------------------------------------------------------------------------
def make_figure4_heatmap(root: Path, outdir: Path) -> None:
    mats: Dict[str, np.ndarray] = {}
    all_vals: list[float] = []
    rows: list[dict] = []

    for title, liab in LIABILITIES_FOR_HEATMAP:
        mat = np.full((len(COSTS_FOR_HEATMAP), len(SEEDS_FOR_HEATMAP)), np.nan)
        for i, cost in enumerate(COSTS_FOR_HEATMAP):
            try:
                ls_cvar = get_static_ls_cvar95(root, liab, cost)
            except Exception as exc:
                print(f"[WARN] Missing LS static baseline for liability={liab}, cost={cost}: {exc}")
                ls_cvar = np.nan

            for j, seed in enumerate(SEEDS_FOR_HEATMAP):
                try:
                    fnug_cvar = get_fnug_case_cvar95(root, liab, cost, seed)
                    diff = fnug_cvar - ls_cvar if np.isfinite(ls_cvar) else np.nan
                    mat[i, j] = diff
                    if np.isfinite(diff):
                        all_vals.append(float(diff))
                    rows.append({
                        "liability_display": title,
                        "liability": liab,
                        "cost_bps": cost,
                        "seed": seed,
                        "fnug_cvar95": fnug_cvar,
                        "ls_cvar95_static": ls_cvar,
                        "cvar95_diff_fnug_minus_ls": diff,
                    })
                except Exception as exc:
                    print(f"[WARN] Missing FNUG heatmap cell for liability={liab}, cost={cost}, seed={seed}: {exc}")
                    rows.append({
                        "liability_display": title,
                        "liability": liab,
                        "cost_bps": cost,
                        "seed": seed,
                        "fnug_cvar95": np.nan,
                        "ls_cvar95_static": ls_cvar,
                        "cvar95_diff_fnug_minus_ls": np.nan,
                    })
        mats[title] = mat

    vabs = max(abs(np.nanmin(all_vals)), abs(np.nanmax(all_vals))) if all_vals else 1.0
    vabs = max(vabs, 1e-6)
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.15), constrained_layout=True)
    im = None
    for ax, (title, _) in zip(axes, LIABILITIES_FOR_HEATMAP):
        mat = mats[title]
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", norm=norm)
        ax.set_xticks(np.arange(len(SEEDS_FOR_HEATMAP)))
        ax.set_xticklabels([str(s) for s in SEEDS_FOR_HEATMAP])
        ax.set_yticks(np.arange(len(COSTS_FOR_HEATMAP)))
        ax.set_yticklabels([f"{c} bps" for c in COSTS_FOR_HEATMAP])
        ax.set_xlabel("Seed")
        if ax is axes[0]:
            ax.set_ylabel("Transaction cost")
        ax.set_title(title)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                txt = "NA" if not np.isfinite(val) else f"{val:.4f}"
                text_color = "white" if np.isfinite(val) and abs(val) > 0.62 * vabs else "black"
                ax.text(j, i, txt, ha="center", va="center", color=text_color, fontsize=9.6)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.90)
        cbar.set_label(r"$\Delta$CVaR$_{95}$ vs. LS")

    save_png_pdf(fig, outdir, "figure4_cost_seed_robustness_heatmap")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(outdir / "figure4_heatmap_values.csv", index=False)

# -----------------------------------------------------------------------------
# Figure 5: residual diagnostics, improved but still better suited for appendix
# -----------------------------------------------------------------------------
def filter_fnug_diagnostics_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    model_col = pick_col(out, ["model"], required=False)
    state_col = pick_col(out, ["state_mode"], required=False)
    variant_col = pick_col(out, ["policy_variant"], required=False)

    if model_col is not None:
        out = out.loc[is_fnug_model_series(out[model_col])].copy()
    if state_col is not None and not out.empty:
        out = out.loc[out[state_col].astype(str).str.lower().eq("latent")].copy()
    if variant_col is not None and not out.empty:
        out = out.loc[out[variant_col].astype(str).str.lower().eq("full")].copy()

    for col in ["residual_delta_exposure", "residual_vega_exposure", "global_trust_mean"]:
        if col in out.columns:
            out = out.loc[pd.to_numeric(out[col], errors="coerce").notna()].copy()
    return out


def grouped_boxplot_fixed_order(ax: plt.Axes, df: pd.DataFrame, group_col: str, value_col: str, ylabel: str) -> None:
    grp = df[[group_col, value_col]].dropna().copy()
    grp[group_col] = grp[group_col].astype(str).str.lower()
    grp[value_col] = pd.to_numeric(grp[value_col], errors="coerce")
    grp = grp.dropna()
    if grp.empty:
        raise ValueError(f"No finite rows for grouped boxplot: {group_col}, {value_col}")

    order = [r for r in REGIME_ORDER if r in set(grp[group_col])]
    order += [r for r in sorted(set(grp[group_col])) if r not in set(order)]
    data = [grp.loc[grp[group_col] == r, value_col].to_numpy() for r in order]

    try:
        bp = ax.boxplot(
            data,
            tick_labels=order,
            patch_artist=True,
            showfliers=False,
            widths=0.62,
            medianprops=dict(linewidth=1.5, color="black"),
        )
    except TypeError:
        bp = ax.boxplot(
            data,
            labels=order,
            patch_artist=True,
            showfliers=False,
            widths=0.62,
            medianprops=dict(linewidth=1.5, color="black"),
        )

    for patch, r in zip(bp["boxes"], order):
        patch.set_facecolor(COLORS.get(r, COLORS["unknown"]))
        patch.set_alpha(0.42)

    ax.set_xlabel("Regime")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)


def make_figure5_residual_diagnostics(root: Path, outdir: Path) -> None:
    df = read_csv_safe(fnug_combined_panel_path(root)).copy()
    df = filter_fnug_diagnostics_rows(df)

    regime_col = pick_col(df, ["regime_name", "regime", "market_regime", "state_name"], required=True)
    delta_col = pick_col(df, ["residual_delta_exposure"], required=True)
    vega_col = pick_col(df, ["residual_vega_exposure"], required=True)
    trust_col = pick_col(df, ["global_trust_mean"], required=True)
    factor_l2_col = pick_col(df, ["residual_factor_l2"], required=False)
    excess_loss_col = pick_col(df, ["excess_loss_norm"], required=False)

    df = df.copy()
    df["_residual_delta_scaled"] = pd.to_numeric(df[delta_col], errors="coerce") * 1e5
    df["_residual_vega_scaled"] = pd.to_numeric(df[vega_col], errors="coerce") * 1e6
    df["_global_trust"] = pd.to_numeric(df[trust_col], errors="coerce")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.35), constrained_layout=True)
    grouped_boxplot_fixed_order(axes[0], df, regime_col, "_residual_delta_scaled", r"Residual delta exposure ($\times 10^{-5}$)")
    grouped_boxplot_fixed_order(axes[1], df, regime_col, "_residual_vega_scaled", r"Residual vega exposure ($\times 10^{-6}$)")
    grouped_boxplot_fixed_order(axes[2], df, regime_col, "_global_trust", "Global trust gate")
    axes[2].set_ylim(0.0, min(1.0, max(0.65, float(np.nanquantile(df["_global_trust"], 0.99)) + 0.05)))

    save_png_pdf(fig, outdir, "figure5_residual_diagnostics")
    plt.close(fig)

    agg_cols = {
        delta_col: ["mean", "median", "std"],
        vega_col: ["mean", "median", "std"],
        trust_col: ["mean", "median", "std"],
    }
    if factor_l2_col is not None:
        agg_cols[factor_l2_col] = ["mean", "median", "std"]
    if excess_loss_col is not None:
        agg_cols[excess_loss_col] = ["mean", "median", "std"]
    df.groupby(regime_col).agg(agg_cols).to_csv(outdir / "figure5_regime_diagnostic_summary.csv")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to project root containing results/hedgeset, or directly to results")
    parser.add_argument("--outdir", default="paper_figures", help="Output directory for figures")
    args = parser.parse_args()

    root = resolve_results_root(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Resolved root: {root}")
    print(f"Saving figures to: {outdir}")

    make_figure2_tail_survival(root, outdir)
    print("Saved Figure 2")

    make_figure3_paired_excess_vs_ls(root, outdir, sort_mode="ls_loss")
    print("Saved Figure 3")

    make_figure4_heatmap(root, outdir)
    print("Saved Figure 4")

    make_figure5_residual_diagnostics(root, outdir)
    print("Saved Figure 5")

    print("All done.")


if __name__ == "__main__":
    main()
