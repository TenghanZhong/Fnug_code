# FNUG

This repository contains code and derived outputs for the anonymous
submission **"FNUG: Factor-Neutral Update Gating for Anchored Option Hedging"**.

The repository is prepared for review and artifact checking. It intentionally
does not include raw SPX option quote data, local cache files, model checkpoints,
or author-identifying paths.

## Contents

- `src/fnug.py`: training, evaluation, and baseline code.
- `src/plot_fnug.py`: script for reproducing paper figures from released result
  files.
- `configs/`: reserved for extracted run configurations.
- `results/hedgeset/`: derived metric, bootstrap, and run-configuration files
  used by the paper.
- `results/paper_figures/`: CSV, PDF, and PNG outputs for the paper figures.
- `data/README.md`: raw data availability and expected file layout.
- `scripts/check_artifact.py`: lightweight artifact and anonymity check.

## Quick Check

From the repository root:

```bash
python scripts/check_artifact.py
```

This checks that the expected result files are present and that common local
identity/path tokens are absent from tracked text files.

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

## Notes for Anonymous Review

- The repository is private while the submission is under review.
- Do not share a URL tied to a named personal GitHub account with reviewers.
- Before public release, mirror this repository to an anonymous artifact account
  or an approved conference artifact mechanism.
