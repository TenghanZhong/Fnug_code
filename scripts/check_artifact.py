from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_FILES = [
    "src/fnug.py",
    "src/plot_fnug.py",
    "results/paper_figures/figure4_heatmap_values.csv",
    "results/hedgeset/fnug_main_fam3_cost3_seed3_1eaa24b3f8/hedgeset_test_metrics_all.csv",
    "results/hedgeset/fnug_main_fam3_cost3_seed3_1eaa24b3f8/hedgeset_pairwise_bootstrap.csv",
    "results/hedgeset/fnug_modern_fam3_cost3_seed3_m4_1eaa24b3f8/hedgeset_test_metrics_all.csv",
    "results/hedgeset/A1_no_factorneutral_1eaa24b3f8/hedgeset_test_metrics_all.csv",
    "results/hedgeset/A2_no_trustgate_1eaa24b3f8/hedgeset_test_metrics_all.csv",
    "results/hedgeset/A3_no_regimeexcess_1eaa24b3f8/hedgeset_test_metrics_all.csv",
    "results/hedgeset/A4_no_auxrisk_1eaa24b3f8/hedgeset_test_metrics_all.csv",
    "results/hedgeset/A5_hardls_only_1eaa24b3f8/hedgeset_test_metrics_all.csv",
]

SENSITIVE_TOKENS = [
    "BaiduSyncdisk",
    "C:\\Users",
    "D:\\",
    "Overleaf",
    "CIKM_Option",
]

TEXT_EXTENSIONS = {".py", ".md", ".txt", ".json", ".csv", ".yml", ".yaml", ".toml", ".gitignore"}


def check_expected_files() -> list[str]:
    missing = []
    for rel in EXPECTED_FILES:
        if not (ROOT / rel).exists():
            missing.append(rel)
    return missing


def check_sensitive_tokens() -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        if path.name == "check_artifact.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            lower = line.lower()
            for token in SENSITIVE_TOKENS:
                if token.lower() in lower:
                    hits.append((str(path.relative_to(ROOT)), lineno, line.strip()))
                    break
    return hits


def check_main_metrics() -> dict[str, str]:
    path = ROOT / "results/hedgeset/fnug_main_fam3_cost3_seed3_1eaa24b3f8/hedgeset_test_metrics_all.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    target = None
    for row in rows:
        if (
            row.get("model") == "FNUG"
            and row.get("family") == "atm_call"
            and row.get("cost_bps") in {"2.0", "2"}
            and row.get("seed") == "42"
        ):
            target = row
            break
    if target is None:
        return {}
    return {
        "atm_call_cost2_seed42_cvar95": f"{float(target['cvar_95']):.4f}",
        "atm_call_cost2_seed42_cvar99": f"{float(target['cvar_99']):.4f}",
        "atm_call_cost2_seed42_worst_regime": f"{float(target['worst_regime_cvar_95']):.4f}",
    }


def main() -> int:
    missing = check_expected_files()
    sensitive = check_sensitive_tokens()
    metrics = check_main_metrics()

    if missing:
        print("Missing expected files:")
        for rel in missing:
            print(f"  - {rel}")

    if sensitive:
        print("Sensitive tokens found:")
        for rel, lineno, line in sensitive[:50]:
            print(f"  - {rel}:{lineno}: {line}")
        if len(sensitive) > 50:
            print(f"  ... {len(sensitive) - 50} more")

    if metrics:
        print("Metric smoke check:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    if missing or sensitive:
        return 1
    print("Artifact check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
