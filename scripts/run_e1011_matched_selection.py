from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_e1011_grid_mbb as grid

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/analysis/e1011_robustness'
INPUTS = BASE / 'inputs'
OUT = BASE / 'matched_selection'
SEEDS = (1, 42, 2026)


def selected_epoch(history, min_delta):
    best, epoch = np.inf, None
    for row in history.itertuples():
        if row.val_selection_score < best - min_delta:
            best, epoch = row.val_selection_score, int(row.epoch)
    return epoch, best


def recompute_metrics(panel):
    cvar = grid.mbb.empirical_cvar
    return dict(n=len(panel), mean_loss_norm=float(panel.loss_norm.mean()),
                cvar_95=cvar(panel.loss_norm, .95), cvar_99=cvar(panel.loss_norm, .99),
                excess_cvar_95=cvar(panel.excess_loss_norm, .95),
                mean_tc_norm=float(panel.tc_norm.mean()), mean_turnover=float(panel.turnover.mean()),
                mean_abs_residual_delta_exposure=float(panel.residual_delta_exposure.mean()),
                mean_abs_residual_vega_exposure=float(panel.residual_vega_exposure.mean()),
                mean_global_trust=float(panel.global_trust_mean.mean()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((INPUTS / 'checkpoint_metadata.json').read_text())
    histories = pd.read_csv(INPUTS / 'validation_history.csv', float_precision='round_trip')
    released_panel = pd.read_csv(INPUTS / 'matched_selection_episode_panel.csv', float_precision='round_trip')
    published = pd.read_csv(ROOT / 'results/analysis/e1008_ablation_seed_audit.csv')
    models = list(metadata['runs'])
    assert len(models) == 4 and len(released_panel) == 1596
    assert not histories.duplicated(['audit_model', 'audit_seed', 'epoch']).any()
    assert not released_panel.duplicated(['audit_model', 'audit_seed', 'start_date']).any()
    assert set(histories.audit_model) == set(released_panel.audit_model) == set(models)
    assert set(histories.audit_seed) == set(released_panel.audit_seed) == set(SEEDS)
    panels, selection_rows, metric_rows = {}, [], []
    common_dates = common_base = None
    for label in models:
        for seed in SEEDS:
            history = histories[histories.audit_model.eq(label) & histories.audit_seed.eq(seed)].copy()
            assert len(history) and history.epoch.is_monotonic_increasing
            score = sum(history[field] * weight for field, weight in metadata['score_terms'].items())
            max_error = float(np.max(np.abs(score - history.val_selection_score)))
            assert np.isfinite(score).all() and max_error < 1e-12
            epoch, val_score = selected_epoch(history, metadata['min_delta'])
            checkpoint = next(x for x in metadata['checkpoints'] if x['model'] == label and x['seed'] == seed)
            assert epoch == checkpoint['epoch'] and np.isclose(val_score, checkpoint['best_val_score'], atol=1e-12)
            chosen = history[history.epoch.eq(epoch)].iloc[0]
            eligible = history.val_excess_cvar_95.le(.008) & history.val_mean_abs_residual_delta_exposure.le(1e-4) & history.val_mean_abs_residual_vega_exposure.le(1e-4)
            selection_rows.append(dict(model=label, seed=seed, common_selection='best_score_no_filters',
                                       selected_epoch=epoch, validation_score=val_score,
                                       n_candidate_epochs=len(history), strict_full_feasible_epoch_count=int(eligible.sum()),
                                       selected_val_excess_cvar95=float(chosen.val_excess_cvar_95),
                                       selected_val_delta_exposure=float(chosen.val_mean_abs_residual_delta_exposure),
                                       selected_val_vega_exposure=float(chosen.val_mean_abs_residual_vega_exposure),
                                       max_recomputed_score_error=max_error))
            panel = released_panel[released_panel.audit_model.eq(label) & released_panel.audit_seed.eq(seed)].copy()
            panel.start_date = pd.to_datetime(panel.start_date)
            panel = panel.sort_values('start_date').reset_index(drop=True)
            assert len(panel) == 133 and set(panel.selected_epoch) == {epoch}
            assert np.isfinite(panel[['loss_norm', 'base_loss_norm', 'excess_loss_norm']].to_numpy()).all()
            assert np.allclose(panel.loss_norm - panel.base_loss_norm, panel.excess_loss_norm, atol=1e-12)
            if common_dates is None:
                common_dates, common_base = panel.start_date.to_numpy(), panel.base_loss_norm.to_numpy()
            assert np.array_equal(common_dates, panel.start_date.to_numpy())
            assert np.allclose(common_base, panel.base_loss_norm.to_numpy(), atol=1e-12)
            panels[(label, seed)] = panel
            metric_rows.append(dict(model=label, seed=seed, selected_epoch=epoch,
                                    checkpoint_selection='best_score_no_filters', **recompute_metrics(panel)))
    pd.DataFrame(selection_rows).to_csv(OUT / 'checkpoint_selection_audit.csv', index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / 'matched_selection_per_seed.csv', index=False)
    mean_metrics = metrics.groupby('model', sort=False)[list(recompute_metrics(next(iter(panels.values()))))].mean().reset_index()
    old_mean = published[published.model.isin(models)].groupby('model', sort=False).cvar_95.mean()
    mean_metrics['published_selection_cvar95'] = mean_metrics.model.map(old_mean)
    mean_metrics['new_minus_published_cvar95'] = mean_metrics.cvar_95 - mean_metrics.published_selection_cvar95
    mean_metrics.to_csv(OUT / 'matched_selection_mean_metrics.csv', index=False)
    rows, seed_rows, draws_archive = [], [], {}
    for comparator_i, label in enumerate(models[1:]):
        for length in grid.BLOCKS:
            rng_seed = 20261011 + comparator_i * 1000 + length
            idx = grid.indices(133, length, grid.N_BOOT, rng_seed)
            per_seed_draws, observed = [], []
            for seed in SEEDS:
                a, b = panels[(models[0], seed)].loss_norm.to_numpy(), panels[(label, seed)].loss_norm.to_numpy()
                d = grid.cvar_rows(a, idx) - grid.cvar_rows(b, idx)
                obs = grid.mbb.empirical_cvar(a, .95) - grid.mbb.empirical_cvar(b, .95)
                per_seed_draws.append(d)
                observed.append(obs)
                seed_rows.append(dict(comparator=label, seed=seed, block_length=length,
                                      diff_full_minus_comparator=obs,
                                      ci95_low=float(np.quantile(d, .025)), ci95_high=float(np.quantile(d, .975)),
                                      p_two_sided_centered=float(np.mean(np.abs(d - obs) >= abs(obs)))))
            seed_draws = np.stack(per_seed_draws)
            draws_archive[f'comparison_{comparator_i}_block_{length}'] = seed_draws
            d, obs = seed_draws.mean(axis=0), float(np.mean(observed))
            rows.append(dict(comparator=label, block_length=length, n_seeds=3, n_unique_episode_dates=133,
                             diff_full_minus_comparator=obs, ci95_low=float(np.quantile(d, .025)),
                             ci95_high=float(np.quantile(d, .975)),
                             p_two_sided_centered=float(np.mean(np.abs(d - obs) >= abs(obs))),
                             n_boot=grid.N_BOOT, rng_seed=rng_seed))
    bootstrap = pd.DataFrame(rows)
    bootstrap['p_holm_three_ablations_within_block'] = np.nan
    for _, frame in bootstrap.groupby('block_length'):
        bootstrap.loc[frame.index, 'p_holm_three_ablations_within_block'] = grid.holm(frame.p_two_sided_centered)
    bootstrap.to_csv(OUT / 'matched_selection_mbb.csv', index=False)
    pd.DataFrame(seed_rows).to_csv(OUT / 'matched_selection_per_seed_mbb.csv', index=False)
    np.savez_compressed(OUT / 'matched_selection_mbb_draws.npz', **draws_archive)
    manifest = dict(original_code_revision=metadata['original_code_revision'], runs=metadata['runs'],
                    score_terms=metadata['score_terms'], min_delta=metadata['min_delta'],
                    input_sha256={p.name:hashlib.sha256(p.read_text(encoding='utf-8').encode('utf-8')).hexdigest() for p in INPUTS.iterdir() if p.is_file()},
                    input_hash_format='SHA256 of UTF-8 text with LF line endings',
                    n_models=4, n_seeds=3, n_checkpoint_reselections=12,
                    n_unique_episode_dates=133, n_boot=grid.N_BOOT, block_lengths=list(grid.BLOCKS),
                    no_retraining=True, no_new_network_inference=True,
                    selection='Original recorded validation score and weights for all models, without feasibility thresholds; best-score update requires improvement greater than 1e-5.',
                    evaluation='Selections recomputed from complete released validation summaries, checked against original checkpoint metadata, and matched to saved selected-model episode losses.',
                    interpretation='Alternative common-selection robustness check of the same trained variants; changes both final FNUG score and thresholds and does not separate projection from its training exposure penalty.',
                    inference='Equal-weight mean of three seed-specific CVaR95 differences; shared date blocks; percentile confidence intervals; centered two-sided p values; Holm across the three component comparisons separately for each block length.')
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(mean_metrics[['model', 'cvar_95', 'published_selection_cvar95']].to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == '__main__':
    main()
