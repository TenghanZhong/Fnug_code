# Additional FNUG robustness analyses

This release supplies the two additional camera-ready analyses: the aggregate
CVaR95 comparison across 27 settings and the component comparison under a common
selection rule. It supplements the original released results without changing
their training runs or final selection protocol.

## Recompute

Install the existing repository requirements and run from the repository root:

```bash
python scripts/run_e1011_grid_mbb.py --validate --run
python scripts/run_e1011_matched_selection.py
```

Only NumPy and pandas are needed by these two scripts. The grid script imports
the original `scripts/run_e1008_moving_block_bootstrap.py` implementation.
Before running the full grid, `--validate` reproduces the nine original primary
MBB summaries and checks all 90000 scalar draws against the vectorized version.
Both scripts resolve paths relative to the repository, independently of the
shell's working directory.

## Aggregate comparison

The original released episode panel supplies 27 combinations of three liability
families, three transaction costs, and three seeds. Each combination has the
same 133 test episode start dates. CVaR95 is recomputed for each model and
setting in each bootstrap draw; FNUG-minus-comparator differences are then
averaged equally over the 27 settings. The same date blocks are applied across
all settings within each comparison. The seeds are not counted as additional
independent market episodes.

MBB uses 10000 draws and overlapping blocks without circular wrapping. The main
block length is five episodes; three and ten are sensitivity checks. Empirical
CVaR includes all values at or above the interpolated quantile, including ties.
Intervals are unadjusted 95% percentile intervals. Two-sided p values use the
centered bootstrap rule from the original implementation.

Files under `grid_mbb/`:

- `grid_mbb_aggregate.csv`: nine aggregate comparisons, including Holm-adjusted
  p values for the three baseline comparisons separately at each block length.
- `grid_mbb.csv`: 243 individual-setting comparisons, with additional Holm
  adjustments across 81 comparisons and across 27 settings per comparator.
- `grid_mbb_counts.csv`: counts of favorable estimates and inference results.
- `manifest.json` and `primary_reproduction_checks.json`: protocol and original
  primary-analysis reproduction checks.

The aggregate comparison has lower FNUG point estimates in 27/27 settings
against LSDeltaVega, 22/27 against IVGrid-HedgeSet, and 26/27 against
QiaoResidual-LS. At block length five, the LSDeltaVega mean difference is
-0.0200837880, with interval [-0.0340460641, -0.0086728272] and Holm p=0.0342.
The aggregate IVGrid-HedgeSet and QiaoResidual-LS intervals include zero.

## Common selection rule

This check uses the same FNUG, No FN, No gate, and No Aux training runs as the
original component audit, restricted to ATM calls at 2 bps and seeds 1, 42, and
2026. It applies the validation score already recorded during training to all
four models, with identical fixed weights and no feasibility thresholds.

The original final FNUG rule first applies the excess-CVaR95 and residual
exposure thresholds and then minimizes CVaR95 + 0.25 CVaR99 + 0.25 worst-regime
CVaR95. The common-rule check instead uses the composite score below. It is a
robustness check under an alternative common selection protocol; it does not
isolate only the removal of the hard thresholds, or separate the projection
from its training exposure penalty.

The score is the sum of these validation quantities multiplied by their weights:

| Validation quantity | Weight | Recorded field |
| --- | ---: | --- |
| Policy CVaR95 | 1.00 | `val_cvar_95` |
| Policy CVaR99 | 0.25 | `val_cvar_99` |
| Worst-regime policy CVaR95 | 0.25 | `val_worst_regime_cvar_95` |
| Mean normalized policy loss | 0.10 | `val_mean_loss_norm` |
| Mean normalized total transaction cost | 0.05 | `val_mean_tc_norm` |
| Mean episode turnover | 0.02 | `val_mean_turnover` |
| CVaR95 of loss relative to the anchor | 1.00 | `val_excess_cvar_95` |
| CVaR99 of loss relative to the anchor | 0.25 | `val_excess_cvar_99` |
| Mean positive part of loss relative to the anchor | 0.50 | `val_mean_relu_excess_loss_norm` |
| Mean squared residual factor exposure before gating | 0.10 | `val_mean_residual_factor_l2` |
| Mean scalar gate value | 0.02 | `val_mean_global_trust` |

Loss relative to the anchor is the policy's normalized episode loss minus the
LSDeltaVega loss on the same episode. Its CVaR is computed from those paired
differences, rather than by subtracting two CVaRs. Normalized losses and costs
use the initial-liability normalization in the paper. Episode turnover sums
absolute position changes, including liquidation. Residual factor exposure is
the squared norm of the residual delta and scaled-vega vector before the scalar
gate, with vega divided by 100; No FN uses the unprojected residual. Exposure and
gate summaries average over rebalance dates and then over validation episodes.
These two score terms remain present for all four models, including when the
corresponding training component is removed.

Model selection scans epochs in increasing order and updates the best model
only when the score improves by more than 1e-5, matching the original saved-score
rule. The released histories cover all epochs recorded before early stopping.
The script reconstructs the scores, verifies the selected epoch and score
against the original checkpoint metadata, and verifies that the released test
losses belong to that selected epoch.

The new `inputs/` files contain 843 validation-summary rows, 1596 selected-model
test episodes, and metadata for 12 saved checkpoints. They are derived summaries
and normalized losses, not raw option quotes or checkpoint tensors. The original
source-file hashes are retained in `checkpoint_metadata.json`. The public script
can reproduce the recorded selections and statistics, but cannot evaluate other
epochs because their policies and test losses are not redistributed.

Files under `matched_selection/`:

- `checkpoint_selection_audit.csv`: selected epochs and feasibility diagnostics.
- `matched_selection_per_seed.csv` and `matched_selection_mean_metrics.csv`:
  per-seed and equal-weight mean performance.
- `matched_selection_mbb.csv` and `matched_selection_per_seed_mbb.csv`:
  aggregate and per-seed paired bootstrap comparisons.
- `manifest.json`: fixed protocol and released input hashes.

Released input hashes use UTF-8 text with LF line endings, matching the Git
blobs independently of Windows checkout line-ending conversion. The original
source-file hashes in `checkpoint_metadata.json` describe the original files
before their columns were reduced for this release.

Each draw uses the same date blocks for both compared models and all three
seeds. Seed-specific CVaR95 differences are recomputed and averaged equally.
Holm adjustment covers the three component comparisons separately for each
block length; this family is separate from the three aggregate baseline
comparisons. At block length five, FNUG's mean CVaR95 is 0.0899201062 versus
0.1105862029 for No FN, a reduction of 18.6878%. The difference is -0.0206660967,
with interval [-0.0324699141, -0.0100454411] and Holm p=0.0009. The No gate and
No Aux intervals include zero. The same significance conclusions hold for
blocks of three and ten.

Bootstrap draws and the grid's intermediate CSV are regenerated by the scripts
and excluded from Git; the compact CSV summaries and inputs are checked in.
