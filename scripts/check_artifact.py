from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_FILES = [
    "src/fnug.py",
    "src/plot_fnug.py",
    "configs/fnug_main.json",
    "configs/modern_baselines.json",
    "configs/qiao_residual_ls.json",
    "results/analysis/final_noreg_combined_metrics.csv",
    "results/analysis/final_noreg_combined_episode_panel.csv",
    "results/analysis/final_noreg_pairwise_bootstrap.csv",
    "results/analysis/ivgrid_noreg_maincell_bootstrap.csv",
    "results/analysis/final_noreg_artifact_manifest.md",
    "results/analysis/final_noreg_ablation_pairwise_bootstrap.csv",
    "results/analysis/final_noreg_main_cell_metrics.csv",
    "results/analysis/final_noreg_heatmap_values.csv",
    "results/analysis/final_noreg_vs_ls_reductions.csv",
    "results/paper_figures/figure4_heatmap_values.csv",
    "results/hedgeset/fnug_final_noreg_fam3_cost3_seed3_1eaa24b3f8/hedgeset_test_metrics_all.csv",
    "results/hedgeset/fnug_final_noreg_fam3_cost3_seed3_1eaa24b3f8/hedgeset_pairwise_bootstrap.csv",
    "results/hedgeset/fnug_final_noreg_fam3_cost3_seed3_1eaa24b3f8/run_config.json",
    "results/hedgeset/fnug_modern_noreg_fam3_cost3_seed3_m4_1eaa24b3f8/hedgeset_test_metrics_all.csv",
    "results/hedgeset/fnug_modern_noreg_fam3_cost3_seed3_m4_1eaa24b3f8/hedgeset_test_episode_panel_all.csv",
    "results/hedgeset/fnug_modern_noreg_fam3_cost3_seed3_m4_1eaa24b3f8/run_config.json",
    "results/hedgeset/fnug_modern_noreg_fam3_cost3_seed3_m4_1eaa24b3f8/atm_call/cost_2bps/modern_baselines/ivgrid/scope_full/seed_42/test_metrics.csv",
    "results/hedgeset/qiao_ls_m4_1eaa24b3f8/hedgeset_seed_summary.csv",
    "results/hedgeset/qiao_ls_m4_1eaa24b3f8/hedgeset_test_metrics_all.csv",
    "results/hedgeset/qiao_ls_m4_1eaa24b3f8/hedgeset_test_episode_panel_all.csv",
    "results/hedgeset/qiao_ls_m4_1eaa24b3f8/run_config.json",
    "results/hedgeset/Aplus_no_factorneutral_1eaa24b3f8/hedgeset_test_metrics_all.csv",
    "results/hedgeset/Aplus_no_trustgate_1eaa24b3f8/hedgeset_test_metrics_all.csv",
    "results/hedgeset/Aplus_no_auxrisk_1eaa24b3f8/hedgeset_test_metrics_all.csv",
    "results/hedgeset/A5_hardls_only_1eaa24b3f8/hedgeset_test_metrics_all.csv",
]

SENSITIVE_TOKENS = [
    "C:" + "\\" + "Users",
    "D:" + "\\",
    "Bai" + "du",
    "Over" + "leaf",
    "Chat" + "G" + "PT",
    "G" + "PT",
    "Co" + "dex",
    "review" + "er",
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
    path = ROOT / "results/hedgeset/fnug_final_noreg_fam3_cost3_seed3_1eaa24b3f8/hedgeset_test_metrics_all.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    target = None
    for row in rows:
        if (
            row.get("model") in {"FNUG", "HedgeSet-latent"}
            and row.get("family") == "atm_call"
            and row.get("cost_bps") in {"2.0", "2"}
            and row.get("seed") in {"42", "42.0"}
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


def check_regime_weights() -> list[tuple[str, str]]:
    bad: list[tuple[str, str]] = []
    config_paths = [
        ROOT / "configs/fnug_main.json",
        ROOT / "configs/modern_baselines.json",
        ROOT / "configs/qiao_residual_ls.json",
        ROOT / "results/hedgeset/fnug_final_noreg_fam3_cost3_seed3_1eaa24b3f8/run_config.json",
        ROOT / "results/hedgeset/fnug_modern_noreg_fam3_cost3_seed3_m4_1eaa24b3f8/run_config.json",
        ROOT / "results/hedgeset/qiao_ls_m4_1eaa24b3f8/run_config.json",
    ]
    pattern = re.compile(r'"lambda_regime_excess_cvar"\s*:\s*([^,\n}]+)')
    for path in config_paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        match = pattern.search(text)
        if match and float(match.group(1)) != 0.0:
            bad.append((str(path.relative_to(ROOT)), match.group(1).strip()))
    return bad


def check_qiao_metrics() -> dict[str, str]:
    path = ROOT / "results/hedgeset/qiao_ls_m4_1eaa24b3f8/hedgeset_test_metrics_all.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    target = None
    for row in rows:
        if (
            row.get("model") == "QiaoResidual-LS"
            and row.get("family") == "atm_call"
            and row.get("cost_bps") in {"2.0", "2"}
            and row.get("seed") in {"42", "42.0"}
        ):
            target = row
            break
    if target is None:
        return {}
    return {
        "qiao_atm_call_cost2_seed42_cvar95": f"{float(target['cvar_95']):.4f}",
        "qiao_atm_call_cost2_seed42_cvar99": f"{float(target['cvar_99']):.4f}",
        "qiao_atm_call_cost2_seed42_worst_regime": f"{float(target['worst_regime_cvar_95']):.4f}",
    }


def check_ivgrid_noreg_metrics() -> dict[str, str]:
    path = ROOT / "results/hedgeset/fnug_modern_noreg_fam3_cost3_seed3_m4_1eaa24b3f8/hedgeset_test_metrics_all.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    target = None
    for row in rows:
        if (
            row.get("model") == "IVGrid-HedgeSet"
            and row.get("family") == "atm_call"
            and row.get("cost_bps") in {"2.0", "2"}
            and row.get("seed") in {"42", "42.0"}
        ):
            target = row
            break
    if target is None:
        return {}
    return {
        "ivgrid_noreg_atm_call_cost2_seed42_cvar95": f"{float(target['cvar_95']):.4f}",
        "ivgrid_noreg_atm_call_cost2_seed42_cvar99": f"{float(target['cvar_99']):.4f}",
        "ivgrid_noreg_atm_call_cost2_seed42_worst_regime": f"{float(target['worst_regime_cvar_95']):.4f}",
        "ivgrid_noreg_atm_call_cost2_seed42_lambda_regime": f"{float(target['lambda_regime_excess_cvar']):.1f}",
    }


def check_combined_artifact_cleanliness() -> list[str]:
    problems: list[str] = []
    metrics_path = ROOT / "results/analysis/final_noreg_combined_metrics.csv"
    panel_path = ROOT / "results/analysis/final_noreg_combined_episode_panel.csv"
    if not metrics_path.exists() or not panel_path.exists():
        return ["combined analysis files are missing"]

    with metrics_path.open(newline="", encoding="utf-8") as f:
        metric_rows = list(csv.DictReader(f))
    with panel_path.open(newline="", encoding="utf-8") as f:
        panel_rows = list(csv.DictReader(f))

    for name, rows in [("metrics", metric_rows), ("episode_panel", panel_rows)]:
        legacy_ivgrid = [
            row for row in rows
            if row.get("model") == "IVGrid-HedgeSet"
            and row.get("candidate_run") == "fnug_modern_fam3_cost3_seed3_m4"
        ]
        if legacy_ivgrid:
            problems.append(f"{name} contains {len(legacy_ivgrid)} legacy IVGrid rows")

    ivgrid_noreg = [
        row for row in metric_rows
        if row.get("model") == "IVGrid-HedgeSet"
        and row.get("candidate_run") == "fnug_modern_noreg_fam3_cost3_seed3_m4"
    ]
    if not ivgrid_noreg:
        problems.append("combined metrics has no no-regime IVGrid rows")

    for row in metric_rows:
        value = row.get("lambda_regime_excess_cvar", "")
        if value not in {"", "0", "0.0"}:
            problems.append(
                f"combined metrics has nonzero regime weight for {row.get('model')} "
                f"({row.get('candidate_run')}): {value}"
            )
            break
    return problems


def main() -> int:
    missing = check_expected_files()
    sensitive = check_sensitive_tokens()
    bad_regime_weights = check_regime_weights()
    metrics = check_main_metrics()
    qiao_metrics = check_qiao_metrics()
    ivgrid_metrics = check_ivgrid_noreg_metrics()
    combined_problems = check_combined_artifact_cleanliness()

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

    if bad_regime_weights:
        print("Nonzero regime-excess weights found:")
        for rel, value in bad_regime_weights:
            print(f"  - {rel}: {value}")

    if metrics:
        print("Metric smoke check:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    else:
        print("Metric smoke check failed: expected ATM-call cost-2 seed-42 row was not found.")

    if qiao_metrics:
        print("QiaoResidual-LS smoke check:")
        for key, value in qiao_metrics.items():
            print(f"  {key}: {value}")
    else:
        print("QiaoResidual-LS smoke check failed: expected ATM-call cost-2 seed-42 row was not found.")

    if ivgrid_metrics:
        print("IVGrid-HedgeSet no-regime smoke check:")
        for key, value in ivgrid_metrics.items():
            print(f"  {key}: {value}")
    else:
        print("IVGrid-HedgeSet no-regime smoke check failed: expected ATM-call cost-2 seed-42 row was not found.")

    if combined_problems:
        print("Combined artifact cleanliness check failed:")
        for problem in combined_problems:
            print(f"  - {problem}")

    if missing or sensitive or bad_regime_weights or not metrics or not qiao_metrics or not ivgrid_metrics or combined_problems:
        return 1
    print("Artifact check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
