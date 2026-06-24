# FNUG

This repository contains the public code and derived outputs for
**"FNUG: Factor-Neutral Update Gating for Anchored Option Hedging"**.

The released main specification is the budgeted FNUG policy selected under a
validation excess-CVaR95 budget of 0.008. It uses a latent quote-set state, a
hard LSDeltaVega anchor, residual scale 0.10, trust gate initialization -1.0,
trust penalty 0.003, and factor-neutral residual projection.

Raw SPX option quote data and model checkpoints are not redistributed. The
repository keeps derived analysis tables and figure files needed to reproduce
the released E1008 result set without publishing machine-specific paths or raw
training trees.

## Contents

- `src/fnug.py`: training, evaluation, baseline, and selector code.
- `src/plot_fnug.py`: regenerates the paper figures from released E1008
  analysis files.
- `configs/fnug_main.json`: released budgeted FNUG configuration.
- `configs/ablation_no_factorneutral.json`,
  `configs/ablation_no_trustgate.json`, and
  `configs/ablation_no_auxrisk.json`: E1008 ablation configurations.
- `configs/modern_baselines.json`, `configs/qiao_residual_ls.json`, and
  `configs/hard_ls_only.json`: comparator configurations.
- `results/analysis/e1008_final_*.csv`: final E1008 combined metrics,
  episode panel, heatmap values, reductions, and paired-bootstrap summaries.
- `results/analysis/e1008_ablation_*.csv`: ATM-call, 2 bps, three-seed
  ablation summaries and bootstrap outputs.
- `results/analysis/e1008_ivgrid_matched_cell_summary.csv`: matched IVGrid
  summary used in the manuscript.
- `results/paper_figures/`: CSV, PDF, and PNG outputs for the manuscript
  figures.
- `scripts/rebuild_e1008_final_analysis.py` and
  `scripts/rebuild_e1008_ablation.py`: rebuild the released analysis tables
  from local raw run outputs when those outputs are available.
- `scripts/check_artifact.py`: verifies the released artifact and checks that
  machine-specific references or non-release result files are not tracked.
- `data/README.md`: raw data availability and expected local file layout.

## Quick Check

From the repository root:

```bash
pip install -r requirements.txt
python scripts/check_artifact.py
```

The check verifies that only the E1008 public result set is tracked, confirms
the main ATM-call, 2 bps, seed-42 metrics, checks the released IVGrid matched-cell
summary, and audits the E1008 ablation seed panel.

## Reproduce Paper Figures

The figures can be regenerated from the released derived results:

```bash
pip install -r requirements.txt
python src/plot_fnug.py --input results --outdir results/paper_figures_rebuilt
```

Expected outputs:

- `figure2_tail_loss_survival.{pdf,png}`
- `figure3_paired_excess_loss_vs_ls.{pdf,png}`
- `figure4_cost_seed_robustness_heatmap.{pdf,png}`
- `figure5_residual_diagnostics.{pdf,png}`

## Rebuild Analysis Tables

If the local raw run outputs are available, rebuild the released analysis files
with:

```bash
python scripts/rebuild_e1008_final_analysis.py --root <local-results-root> --outdir results/analysis
python scripts/rebuild_e1008_ablation.py --root <local-results-root> --outdir results/analysis
```

The checked-in CSV files under `results/analysis/` are the released derived
outputs for this artifact.

## Raw Data

Raw option-chain data are not redistributed because they are subject to
data-licensing restrictions. To rerun training or evaluation from raw data,
place the required parquet files in a local data directory and set:

```bash
export FNUG_DATA_DIR=/path/to/raw/option/data
```

On Windows PowerShell:

```powershell
$env:FNUG_DATA_DIR="<path-to-raw-option-data>"
```
