from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = ROOT / "results" / "analysis" / "e1008_final_combined_episode_panel.csv"
DEFAULT_OUTDIR = ROOT / "results" / "analysis"

FNUG_MODEL = "HedgeSet-latent"
COMPARISONS = [
    ("LSDeltaVega", "confirmatory_vs_ls"),
    ("QiaoResidual-LS", "confirmatory_vs_qiao"),
    ("IVGrid-HedgeSet", "diagnostic_vs_ivgrid"),
]


@dataclass(frozen=True)
class BootstrapResult:
    comparison: str
    model_a: str
    model_b: str
    family: str
    cost_bps: float
    seed: int
    metric: str
    alpha: float
    block_length: int
    n_episodes: int
    n_boot: int
    rng_seed: int
    observed_model_a: float
    observed_model_b: float
    observed_diff: float
    ci95_low: float
    ci95_high: float
    p_two_sided_centered: float
    bootstrap_mean_diff: float
    bootstrap_std_diff: float
    min_start_date: str
    max_start_date: str


def empirical_cvar(values: np.ndarray, alpha: float) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return math.nan
    threshold = np.quantile(x, alpha)
    tail = x[x >= threshold]
    if tail.size == 0:
        return math.nan
    return float(np.mean(tail))


def filter_case(panel: pd.DataFrame, model: str, family: str, cost_bps: float, seed: int) -> pd.DataFrame:
    costs = pd.to_numeric(panel["cost_bps"], errors="coerce")
    seeds = pd.to_numeric(panel["seed"], errors="coerce")
    out = panel[
        panel["model"].eq(model)
        & panel["family"].eq(family)
        & np.isclose(costs, float(cost_bps))
        & (np.isclose(seeds, float(seed)) | seeds.isna())
    ].copy()
    if out.empty:
        raise ValueError(f"No rows for model={model}, family={family}, cost_bps={cost_bps}, seed={seed}")
    out["start_date"] = pd.to_datetime(out["start_date"], errors="raise")
    out["loss_norm"] = pd.to_numeric(out["loss_norm"], errors="coerce")
    out = out.sort_values("start_date")
    if out["start_date"].duplicated().any():
        dupes = out.loc[out["start_date"].duplicated(), "start_date"].astype(str).tolist()
        raise ValueError(f"Duplicate start_date rows for {model}: {dupes[:5]}")
    return out[["start_date", "loss_norm"]]


def paired_losses(
    panel: pd.DataFrame,
    model_a: str,
    model_b: str,
    family: str,
    cost_bps: float,
    seed: int,
) -> pd.DataFrame:
    a = filter_case(panel, model_a, family, cost_bps, seed).rename(columns={"loss_norm": "loss_a"})
    b = filter_case(panel, model_b, family, cost_bps, seed).rename(columns={"loss_norm": "loss_b"})
    merged = a.merge(b, on="start_date", how="inner")
    if len(merged) != len(a) or len(merged) != len(b):
        missing_a = sorted(set(b["start_date"]) - set(merged["start_date"]))
        missing_b = sorted(set(a["start_date"]) - set(merged["start_date"]))
        raise ValueError(
            f"Unmatched paired dates for {model_a} vs {model_b}: "
            f"missing_a={missing_a[:3]}, missing_b={missing_b[:3]}"
        )
    if merged["loss_a"].isna().any() or merged["loss_b"].isna().any():
        raise ValueError(f"Non-finite losses for {model_a} vs {model_b}")
    return merged.sort_values("start_date").reset_index(drop=True)


def moving_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    if block_length > n:
        raise ValueError(f"block_length={block_length} exceeds n={n}")
    n_blocks = int(math.ceil(n / block_length))
    starts = rng.integers(0, n - block_length + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_length) for s in starts])
    return idx[:n]


def run_one(
    paired: pd.DataFrame,
    comparison_label: str,
    model_a: str,
    model_b: str,
    family: str,
    cost_bps: float,
    seed: int,
    alpha: float,
    block_length: int,
    n_boot: int,
    rng_seed: int,
) -> tuple[BootstrapResult, pd.DataFrame]:
    loss_a = paired["loss_a"].to_numpy(dtype=float)
    loss_b = paired["loss_b"].to_numpy(dtype=float)
    observed_a = empirical_cvar(loss_a, alpha)
    observed_b = empirical_cvar(loss_b, alpha)
    observed_diff = observed_a - observed_b
    if not np.isfinite(observed_diff):
        raise ValueError(f"Observed statistic is not finite for {model_a} vs {model_b}")

    rng = np.random.default_rng(rng_seed)
    draws = np.empty(n_boot, dtype=float)
    for r in range(n_boot):
        idx = moving_block_indices(len(paired), block_length, rng)
        draws[r] = empirical_cvar(loss_a[idx], alpha) - empirical_cvar(loss_b[idx], alpha)

    centered = draws - observed_diff
    p_value = float(np.mean(np.abs(centered) >= abs(observed_diff)))
    result = BootstrapResult(
        comparison=comparison_label,
        model_a=model_a,
        model_b=model_b,
        family=family,
        cost_bps=float(cost_bps),
        seed=int(seed),
        metric=f"cvar_{int(alpha * 100)}",
        alpha=float(alpha),
        block_length=int(block_length),
        n_episodes=int(len(paired)),
        n_boot=int(n_boot),
        rng_seed=int(rng_seed),
        observed_model_a=float(observed_a),
        observed_model_b=float(observed_b),
        observed_diff=float(observed_diff),
        ci95_low=float(np.quantile(draws, 0.025)),
        ci95_high=float(np.quantile(draws, 0.975)),
        p_two_sided_centered=min(p_value, 1.0),
        bootstrap_mean_diff=float(np.mean(draws)),
        bootstrap_std_diff=float(np.std(draws, ddof=1)),
        min_start_date=str(paired["start_date"].min().date()),
        max_start_date=str(paired["start_date"].max().date()),
    )
    draw_frame = pd.DataFrame({
        "comparison": comparison_label,
        "model_a": model_a,
        "model_b": model_b,
        "block_length": int(block_length),
        "draw": np.arange(n_boot, dtype=int),
        "diff": draws,
    })
    return result, draw_frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Run E1008 paired moving-block bootstrap for primary CVaR95 checks.")
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--family", default="atm_call")
    parser.add_argument("--cost-bps", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--block-lengths", default="3,5,10")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--rng-seed", type=int, default=20260622)
    parser.add_argument("--write-draws", action="store_true")
    args = parser.parse_args()

    def repo_path(path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return str(path)

    panel_path = args.panel.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if not panel_path.exists():
        raise FileNotFoundError(panel_path)

    panel = pd.read_csv(panel_path, low_memory=False)
    block_lengths = [int(x.strip()) for x in str(args.block_lengths).split(",") if x.strip()]
    if block_lengths != sorted(block_lengths):
        raise ValueError("Provide block lengths in ascending order for deterministic reporting.")

    results: list[BootstrapResult] = []
    draw_frames: list[pd.DataFrame] = []
    for comp_idx, (model_b, label) in enumerate(COMPARISONS):
        paired = paired_losses(panel, FNUG_MODEL, model_b, args.family, args.cost_bps, args.seed)
        for block_length in block_lengths:
            seed = int(args.rng_seed + 1000 * comp_idx + block_length)
            result, draws = run_one(
                paired=paired,
                comparison_label=label,
                model_a=FNUG_MODEL,
                model_b=model_b,
                family=args.family,
                cost_bps=args.cost_bps,
                seed=args.seed,
                alpha=args.alpha,
                block_length=block_length,
                n_boot=args.n_boot,
                rng_seed=seed,
            )
            results.append(result)
            if args.write_draws:
                draw_frames.append(draws)

    summary = pd.DataFrame([r.__dict__ for r in results])
    summary_path = outdir / "e1008_primary_moving_block_bootstrap.csv"
    summary.to_csv(summary_path, index=False)

    draws_path = None
    if args.write_draws:
        draws = pd.concat(draw_frames, ignore_index=True)
        draws_path = outdir / "e1008_primary_mbb_draws.csv.gz"
        draws.to_csv(draws_path, index=False, compression="gzip")

    manifest = {
        "description": "Paired moving-block bootstrap for primary ATM-call / 2 bps / seed 42 CVaR95 checks.",
        "panel": repo_path(panel_path),
        "summary": repo_path(summary_path),
        "draws": repo_path(draws_path),
        "model_a": FNUG_MODEL,
        "comparisons": [{"model_b": b, "label": label} for b, label in COMPARISONS],
        "family": args.family,
        "cost_bps": args.cost_bps,
        "seed": args.seed,
        "alpha": args.alpha,
        "block_lengths": block_lengths,
        "n_boot": args.n_boot,
        "rng_seed_base": args.rng_seed,
        "statistic": "CVaR_alpha(loss_model_a) - CVaR_alpha(loss_model_b), recomputed inside each bootstrap draw",
        "block_protocol": "Overlapping moving blocks sampled by start_date order; same block indices applied to both paired models.",
        "p_value": "Two-sided centered bootstrap p-value: mean(abs(draw - observed_diff) >= abs(observed_diff)).",
    }
    manifest_path = outdir / "e1008_primary_moving_block_bootstrap_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(summary.to_string(index=False))
    print(f"Wrote {summary_path}")
    if draws_path:
        print(f"Wrote {draws_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
