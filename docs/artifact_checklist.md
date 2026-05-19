# Artifact Checklist

## Included

- Code for FNUG training, evaluation, baselines, and plotting.
- Run configurations for the main, modern-baseline, and diagnostic runs.
- Derived metric CSVs used by the paper.
- Paired-bootstrap tables.
- Figure source CSVs and rendered figure files.

## Excluded

- Raw option quote data.
- Local preprocessing caches.
- Model checkpoints.
- Compressed local backups.
- Author-identifying Git history.

## Release Procedure

1. Run `python scripts/check_artifact.py`.
2. Confirm no local path or author-identifying token is reported.
3. Push to a private repository during manuscript preparation.
4. Mirror to an anonymous repository or approved artifact system before review.

