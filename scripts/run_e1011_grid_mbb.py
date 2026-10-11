from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts/run_e1008_moving_block_bootstrap.py'
PANEL = ROOT / 'results/analysis/e1008_final_combined_episode_panel.csv'
OUT = ROOT / 'results/analysis/e1011_robustness/grid_mbb'
ALPHA = 0.95
N_BOOT = 10000
BASE_SEED = 20260622
BLOCKS = (3, 5, 10)
FAMILIES = ('atm_call', 'put25', 'atm_straddle')
COSTS = (1.0, 2.0, 5.0)
SEEDS = (1, 42, 2026)

spec = importlib.util.spec_from_file_location('released_mbb', SOURCE)
mbb = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mbb
spec.loader.exec_module(mbb)


def indices(n: int, length: int, n_boot: int, rng_seed: int) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    starts = rng.integers(0, n - length + 1, size=(n_boot, math.ceil(n / length)))
    return (starts[:, :, None] + np.arange(length)[None, None, :]).reshape(n_boot, -1)[:, :n]


def cvar_rows(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    # Retain the released quantile threshold and include every tied tail value.
    draws = np.empty(len(idx), dtype=float)
    for start in range(0, len(idx), 500):
        x = values[idx[start:start + 500]]
        threshold = np.quantile(x, ALPHA, axis=1)
        mask = x >= threshold[:, None]
        draws[start:start + len(x)] = np.sum(np.where(mask, x, 0.0), axis=1) / mask.sum(axis=1)
    return draws


def summary(paired, model, label, family, cost, seed, length, draws, rng_seed):
    a = mbb.empirical_cvar(paired.loss_a.to_numpy(), ALPHA)
    b = mbb.empirical_cvar(paired.loss_b.to_numpy(), ALPHA)
    obs = a - b
    return dict(comparison=label, model_a=mbb.FNUG_MODEL, model_b=model,
                family=family, cost_bps=cost, seed=seed, metric='cvar_95', alpha=ALPHA,
                block_length=length, n_episodes=len(paired), n_boot=N_BOOT, rng_seed=rng_seed,
                observed_model_a=a, observed_model_b=b, observed_diff=obs,
                ci95_low=float(np.quantile(draws, .025)), ci95_high=float(np.quantile(draws, .975)),
                p_two_sided_centered=float(np.mean(np.abs(draws - obs) >= abs(obs))),
                bootstrap_mean_diff=float(np.mean(draws)),
                bootstrap_std_diff=float(np.std(draws, ddof=1)),
                min_start_date=str(paired.start_date.min().date()),
                max_start_date=str(paired.start_date.max().date()))


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind='stable')
    adjusted = np.minimum(1.0, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def validate(panel):
    OUT.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(ROOT / 'results/analysis/e1008_primary_moving_block_bootstrap.csv')
    checks = []
    for comp_i, (model, label) in enumerate(mbb.COMPARISONS):
        paired = mbb.paired_losses(panel, mbb.FNUG_MODEL, model, 'atm_call', 2.0, 42)
        for length in BLOCKS:
            rng_seed = BASE_SEED + comp_i * 1000 + length
            idx = indices(len(paired), length, N_BOOT, rng_seed)
            # Check RNG identity against the original loop before vectorization.
            rng = np.random.default_rng(rng_seed)
            scalar_idx = np.asarray([mbb.moving_block_indices(len(paired), length, rng) for _ in range(20)])
            assert np.array_equal(scalar_idx, idx[:20])
            vector_draws = cvar_rows(paired.loss_a.to_numpy(), idx) - cvar_rows(paired.loss_b.to_numpy(), idx)
            result, scalar_draws = mbb.run_one(paired, label, mbb.FNUG_MODEL, model,
                                              'atm_call', 2.0, 42, ALPHA, length, N_BOOT, rng_seed)
            assert np.allclose(vector_draws, scalar_draws['diff'].to_numpy(), atol=1e-14, rtol=1e-12)
            row = reference[(reference.model_b == model) & (reference.block_length == length)].iloc[0]
            fields = ['observed_model_a', 'observed_model_b', 'observed_diff', 'ci95_low',
                      'ci95_high', 'p_two_sided_centered', 'bootstrap_mean_diff', 'bootstrap_std_diff']
            for field in fields:
                assert np.isclose(getattr(result, field), float(row[field]), atol=1e-12, rtol=1e-10), (model, length, field)
            checks.append(dict(model=model, block_length=length, all_10000_draws_match=True,
                               released_summary_reproduced=True))
            print(f'Primary reproduction passed: {label}, block {length}', flush=True)
    (OUT / 'primary_reproduction_checks.json').write_text(json.dumps(checks, indent=2) + '\n')


def run(panel):
    OUT.mkdir(parents=True, exist_ok=True)
    assert (OUT / 'primary_reproduction_checks.json').exists(), 'Run --validate first'
    rows = []
    all_draws = {}
    cells = [(family, cost, seed) for family in FAMILIES for cost in COSTS for seed in SEEDS]
    common_dates = None
    for comp_i, (model, label) in enumerate(mbb.COMPARISONS):
        pairs = []
        for family, cost, seed in cells:
            paired = mbb.paired_losses(panel, mbb.FNUG_MODEL, model, family, cost, seed)
            assert len(paired) == 133
            assert np.isfinite(paired[['loss_a', 'loss_b']].to_numpy()).all()
            dates = paired.start_date.to_numpy()
            if common_dates is None:
                common_dates = dates
            assert np.array_equal(common_dates, dates), 'Grid has unmatched calendar dates'
            pairs.append(paired)
        for length in BLOCKS:
            rng_seed = BASE_SEED + comp_i * 1000 + length
            idx = indices(133, length, N_BOOT, rng_seed)
            comp_draws = []
            for (family, cost, seed), paired in zip(cells, pairs):
                d = cvar_rows(paired.loss_a.to_numpy(), idx) - cvar_rows(paired.loss_b.to_numpy(), idx)
                rows.append(summary(paired, model, label, family, cost, seed, length, d, rng_seed))
                comp_draws.append(d)
            all_draws[f'{label}_block_{length}'] = np.stack(comp_draws)
            pd.DataFrame(rows).to_csv(OUT / 'grid_mbb_partial.csv', index=False)
            print(f'27 cells completed: {label}, block {length}', flush=True)
    table = pd.DataFrame(rows)
    # Main multiplicity family: 27 cells x all three announced comparators.
    table['p_holm_all_81_within_block'] = np.nan
    table['p_holm_27_within_comparator_block'] = np.nan
    for length in BLOCKS:
        use = table.block_length.eq(length)
        table.loc[use, 'p_holm_all_81_within_block'] = holm(table.loc[use, 'p_two_sided_centered'])
        for _, frame in table[use].groupby('model_b', sort=False):
            table.loc[frame.index, 'p_holm_27_within_comparator_block'] = holm(frame.p_two_sided_centered)
    table['lower_point_estimate'] = table.observed_diff.lt(0)
    table['ci_excludes_zero_favorable'] = table.ci95_high.lt(0)
    table['holm81_significant_favorable'] = table.observed_diff.lt(0) & table.p_holm_all_81_within_block.lt(.05)
    table.to_csv(OUT / 'grid_mbb.csv', index=False)
    np.savez_compressed(OUT / 'grid_mbb_draws.npz', **all_draws)

    aggregate_rows = []
    count_rows = []
    for (model, length), frame in table.groupby(['model_b', 'block_length'], sort=False):
        label = frame.comparison.iloc[0]
        d = all_draws[f'{label}_block_{length}'].mean(axis=0)
        obs = float(frame.observed_diff.mean())
        aggregate_rows.append(dict(model_b=model, block_length=int(length),
                                   statistic='mean of 27 case CVaR95 differences; synchronized calendar blocks',
                                   n_cells=27, n_unique_episode_dates=133,
                                   observed_diff=obs, ci95_low=float(np.quantile(d, .025)),
                                   ci95_high=float(np.quantile(d, .975)),
                                   p_two_sided_centered=float(np.mean(np.abs(d - obs) >= abs(obs)))))
        count_rows.append(dict(model_b=model, block_length=int(length), n_cells=len(frame),
                               lower_point_estimate=int(frame.lower_point_estimate.sum()),
                               favorable_ci95=int(frame.ci_excludes_zero_favorable.sum()),
                               unfavorable_ci95=int(frame.ci95_low.gt(0).sum()),
                               centered_p_lt_005_favorable=int((frame.observed_diff.lt(0) & frame.p_two_sided_centered.lt(.05)).sum()),
                               favorable_holm_27=int((frame.observed_diff.lt(0) & frame.p_holm_27_within_comparator_block.lt(.05)).sum()),
                               favorable_holm_81=int(frame.holm81_significant_favorable.sum())))
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate['p_holm_three_comparators_within_block'] = np.nan
    for length, frame in aggregate.groupby('block_length'):
        aggregate.loc[frame.index, 'p_holm_three_comparators_within_block'] = holm(frame.p_two_sided_centered)
    aggregate.to_csv(OUT / 'grid_mbb_aggregate.csv', index=False)
    counts = pd.DataFrame(count_rows)
    counts.to_csv(OUT / 'grid_mbb_counts.csv', index=False)
    metadata = dict(code_revision='dfebfb50410f1d4ca7633ed18e5972e00be15d42',
                    panel_sha256=hashlib.sha256(PANEL.read_text(encoding='utf-8').encode('utf-8')).hexdigest(),
                    input_hash_format='SHA256 of UTF-8 text with LF line endings',
                    numpy_version=np.__version__, pandas_version=pd.__version__,
                    alpha=ALPHA, n_boot=N_BOOT, base_rng_seed=BASE_SEED, block_lengths=list(BLOCKS),
                    main_block_length=5, comparisons=mbb.COMPARISONS, cells=cells,
                    n_pairs=81, n_episodes_per_pair=133, all_pairs_have_exact_common_dates=True,
                    protocol='Released overlapping moving blocks; identical indices for both models and all grid cells within each comparison/block; no circular wrapping.',
                    statistic='Difference between two recomputed empirical tail means, using released quantile-and-ties definition.',
                    aggregate='Equal-weight mean of 27 case differences; synchronized calendar blocks preserve cross-case dependence; seeds are not additional independent episodes.',
                    multiplicity='Holm across 81 main-block comparisons; per-comparator 27-test Holm also reported; blocks 3 and 10 are sensitivity analyses.',
                    selection='No new model selection, tuning, retraining, or episode changes.',
                    primary_reproduction='All 9 released primary summaries and 90000 scalar draws reproduced before grid analysis.')
    (OUT / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(counts.to_string(index=False), flush=True)
    print(aggregate.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    panel = pd.read_csv(PANEL, low_memory=False)
    if args.validate:
        validate(panel)
    if args.run:
        run(panel)


if __name__ == '__main__':
    main()
