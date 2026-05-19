# FNUG

This repository contains code and derived outputs for
**"FNUG: Factor-Neutral Update Gating for Anchored Option Hedging"**.

It provides scripts, configurations, and derived result files used to reproduce
the paper figures and reported metrics. It does not include raw SPX option quote
data, local cache files, model checkpoints, or local machine paths.

## Contents

- `src/fnug.py`: training, evaluation, and baseline code.
- `src/plot_fnug.py`: script for reproducing paper figures from released result
  files.
- `configs/`: reserved for extracted run configurations.
- `results/hedgeset/`: derived metric, bootstrap, and run-configuration files
  used by the paper.
- `results/paper_figures/`: CSV, PDF, and PNG outputs for the paper figures.
- `data/README.md`: raw data availability and expected file layout.
- `scripts/check_artifact.py`: sanity check for required files and local path
  tokens.

## Quick Check

From the repository root:

```bash
python scripts/check_artifact.py
```

This checks that the expected result files are present and that unexpected local
path tokens are absent from tracked text files.

## Reproduce Paper Figures

The paper figures can be regenerated from the released derived results:

```bash
python src/plot_fnug.py --input results --outdir results/paper_figures_rebuilt
```

Expected outputs:

- `figure2_tail_loss_survival.{pdf,png}`
- `figure3_paired_excess_loss_vs_ls.{pdf,png}`
- `figure4_cost_seed_robustness_heatmap.{pdf,png}`
- `figure5_residual_diagnostics.{pdf,png}`

## Raw Data

Raw option-chain data are not redistributed in this repository because they are
subject to data-licensing restrictions. The released CSV files are derived
evaluation outputs used to reproduce the reported tables, figures, and bootstrap
summaries.

To rerun training/evaluation from raw data, place the required parquet files in
a local data directory and set:

```bash
export FNUG_DATA_DIR=/path/to/raw/option/data
```

On Windows PowerShell:

```powershell
$env:FNUG_DATA_DIR="<path-to-raw-option-data>"
```

