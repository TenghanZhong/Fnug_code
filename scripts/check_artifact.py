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
    "results/analysis/ivgrid_matched_cell_summary.csv",
    "results/analysis/ivgrid_noreg_maincell_bootstrap.csv",
    "results/analysis/final_noreg_artifact_manifest.md",
    "results/analysis/final_noreg_ablation_pairwise_bootstrap.csv",
    "results/analysis/final_noreg_main_cell_metrics.csv",
    "results/analysis/final_noreg_heatmap_values.csv",
    "results/analysis/final_noreg_vs_ls_reductions.csv",
    "results/analysis/e1008_final_combined_metrics.csv",
    "results/analysis/e1008_final_combined_episode_panel.csv",
    "results/analysis/e1008_final_pairwise_bootstrap.csv",
    "results/analysis/e1008_final_main_cell_metrics.csv",
    "results/analysis/e1008_final_heatmap_values.csv",
    "results/analysis/e1008_final_vs_ls_reductions.csv",
    "results/analysis/e1008_ivgrid_matched_cell_summary.csv",
    "results/analysis/e1008_ablation_table.csv",
    "results/analysis/e1008_ablation_pairwise_bootstrap.csv",
    "results/analysis/e1008_ablation_seed_audit.csv",
    "results/analysis/e1008_ablation_seed_summary.csv",
    "results/analysis/e1008_ablation_multiseed_pairwise_bootstrap.csv",
    "results/paper_figures/figure4_heatmap_values.csv",
    "results/hedgeset/fnug_final_noreg_fam3_cost3_seed3_1eaa24b3f8/hedgeset_test_metrics_all.csv",
    "results/hedgeset/fnug_final_noreg_fam3_cost3_seed3_1eaa24b3f8/hedgeset_pairwise_bootstrap.csv",
    "results/hedgeset/fnug_final_noreg_fam3_cost3_seed3_1eaa24b3f8/run_config.json",
    "results/hedgeset/fnug_modern_noreg_fam3_cost3_seed3_m4_1eaa24b3f8/hedgeset_test_metrics_all.csv",
    "results/hedgeset/fnug_modern_noreg_fam3_cost3_seed3_m4_1eaa24b3f8/hedgeset_test_episode_panel_all.csv",
    "results/hedgeset/fnug_modern_noreg_fam3_cost3_seed3_m4_1eaa24b3f8/run_config.json",
    "results/hedgeset/fnug_modern_noreg_fam3_cost3_seed3_m4_1eaa24b3f8/atm_call/cost_2bps/modern_baselines/ivgrid/scope_full/seed_42/test_metrics.csv",
    "results/hedgeset/fnug_modern_noreg_straddle_m4_1eaa24b3f8/hedgeset_test_metrics_all.csv",
    "results/hedgeset/fnug_modern_noreg_straddle_m4_1eaa24b3f8/hedgeset_test_episode_panel_all.csv",
    "results/hedgeset/fnug_modern_noreg_straddle_m4_1eaa24b3f8/run_config.json",
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
        ROOT / "results/hedgeset/fnug_modern_noreg_straddle_m4_1eaa24b3f8/run_config.json",
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
        and row.get("candidate_run") in {
            "fnug_modern_noreg_fam3_cost3_seed3_m4",
            "fnug_modern_noreg_straddle_m4",
        }
    ]
    if not ivgrid_noreg:
        problems.append("combined metrics has no no-regime IVGrid rows")
    elif len(ivgrid_noreg) != 24:
        problems.append(f"combined metrics has {len(ivgrid_noreg)} no-regime IVGrid rows; expected 24")

    for row in metric_rows:
        value = row.get("lambda_regime_excess_cvar", "")
        if value not in {"", "0", "0.0"}:
            problems.append(
                f"combined metrics has nonzero regime weight for {row.get('model')} "
                f"({row.get('candidate_run')}): {value}"
            )
            break
    return problems


def check_ivgrid_matched_summary() -> dict[str, str]:
    path = ROOT / "results/analysis/ivgrid_matched_cell_summary.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, str] = {}
    for row in rows:
        metric = row.get("metric", "")
        if metric in {"cvar_95", "cvar_99", "worst_regime_cvar_95"}:
            out[f"{metric}_n_cells"] = row.get("n_cells", "")
            out[f"{metric}_fnug_lower_cells"] = row.get("fnug_lower_cells", "")
            out[f"{metric}_mean_diff"] = f"{float(row['mean_diff_fnug_minus_ivgrid']):.4f}"
    expected = {
        "cvar_95_n_cells": "24",
        "cvar_95_fnug_lower_cells": "17",
        "cvar_99_n_cells": "24",
        "cvar_99_fnug_lower_cells": "17",
        "worst_regime_cvar_95_n_cells": "24",
        "worst_regime_cvar_95_fnug_lower_cells": "17",
    }
    for key, value in expected.items():
        if out.get(key) != value:
            return {}
    return out


def check_e1008_main_metrics() -> dict[str, str]:
    path = ROOT / "results/analysis/e1008_final_main_cell_metrics.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    target = None
    for row in rows:
        if row.get("model") in {"FNUG", "HedgeSet-latent"}:
            target = row
            break
    if target is None:
        return {}
    return {
        "e1008_atm_call_cost2_seed42_cvar95": f"{float(target['cvar_95']):.4f}",
        "e1008_atm_call_cost2_seed42_cvar99": f"{float(target['cvar_99']):.4f}",
        "e1008_atm_call_cost2_seed42_worst_regime": f"{float(target['worst_regime_cvar_95']):.4f}",
        "e1008_atm_call_cost2_seed42_excess_cvar95": f"{float(target['excess_cvar_95']):.4f}",
    }


def check_e1008_ivgrid_summary() -> dict[str, str]:
    path = ROOT / "results/analysis/e1008_ivgrid_matched_cell_summary.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, str] = {}
    for row in rows:
        metric = row.get("metric", "")
        if metric in {"cvar_95", "cvar_99", "worst_regime_cvar_95"}:
            out[f"e1008_{metric}_n_cells"] = row.get("n_cells", "")
            out[f"e1008_{metric}_fnug_lower_cells"] = row.get("fnug_lower_cells", "")
            out[f"e1008_{metric}_mean_diff"] = f"{float(row['mean_diff_fnug_minus_ivgrid']):.4f}"
    expected = {
        "e1008_cvar_95_n_cells": "24",
        "e1008_cvar_95_fnug_lower_cells": "19",
        "e1008_cvar_99_n_cells": "24",
        "e1008_cvar_99_fnug_lower_cells": "18",
        "e1008_worst_regime_cvar_95_n_cells": "24",
        "e1008_worst_regime_cvar_95_fnug_lower_cells": "18",
    }
    for key, value in expected.items():
        if out.get(key) != value:
            return {}
    return out


def check_e1008_ablation_seed_audit() -> dict[str, str]:
    path = ROOT / "results/analysis/e1008_ablation_seed_audit.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 18:
        return {}
    expected_models = {
        "Full FNUG": "best_feasible_excess_0p008",
        "QiaoResidual-LS": "external",
        "LS only": "external",
        "No factor-neutral constraint": "best_score",
        "No trust gate": "best_feasible_excess_0p008",
        "No auxiliary risk": "best_feasible_excess_0p008",
    }
    out: dict[str, str] = {}
    for model, expected_selection in expected_models.items():
        model_rows = [row for row in rows if row.get("model") == model]
        seeds = {row.get("audit_seed") for row in model_rows}
        selections = {row.get("checkpoint_selection") for row in model_rows}
        if seeds != {"1", "42", "2026"}:
            return {}
        if selections != {expected_selection}:
            return {}
        out[f"e1008_ablation_{model}_n_seeds"] = str(len(seeds))
    full_seed42 = [
        row for row in rows
        if row.get("model") == "Full FNUG" and row.get("audit_seed") == "42"
    ]
    if not full_seed42:
        return {}
    out["e1008_ablation_full_seed42_cvar95"] = f"{float(full_seed42[0]['cvar_95']):.4f}"
    return out


def main() -> int:
    missing = check_expected_files()
    sensitive = check_sensitive_tokens()
    bad_regime_weights = check_regime_weights()
    metrics = check_main_metrics()
    qiao_metrics = check_qiao_metrics()
    ivgrid_metrics = check_ivgrid_noreg_metrics()
    ivgrid_matched_summary = check_ivgrid_matched_summary()
    e1008_metrics = check_e1008_main_metrics()
    e1008_ivgrid_summary = check_e1008_ivgrid_summary()
    e1008_ablation_seed_audit = check_e1008_ablation_seed_audit()
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
        print("Legacy no-regime smoke check:")
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

    if ivgrid_matched_summary:
        print("Legacy IVGrid matched-cell summary check:")
        for key, value in ivgrid_matched_summary.items():
            print(f"  {key}: {value}")
    else:
        print("IVGrid matched-cell summary check failed.")

    if e1008_metrics:
        print("E1008 final smoke check:")
        for key, value in e1008_metrics.items():
            print(f"  {key}: {value}")
    else:
        print("E1008 final smoke check failed.")

    if e1008_ivgrid_summary:
        print("E1008 IVGrid matched-cell summary check:")
        for key, value in e1008_ivgrid_summary.items():
            print(f"  {key}: {value}")
    else:
        print("E1008 IVGrid matched-cell summary check failed.")

    if e1008_ablation_seed_audit:
        print("E1008 ablation seed-audit check:")
        for key, value in e1008_ablation_seed_audit.items():
            print(f"  {key}: {value}")
    else:
        print("E1008 ablation seed-audit check failed.")

    if combined_problems:
        print("Combined artifact cleanliness check failed:")
        for problem in combined_problems:
            print(f"  - {problem}")

    if (
        missing
        or sensitive
        or bad_regime_weights
        or not metrics
        or not qiao_metrics
        or not ivgrid_metrics
        or not ivgrid_matched_summary
        or not e1008_metrics
        or not e1008_ivgrid_summary
        or not e1008_ablation_seed_audit
        or combined_problems
    ):
        return 1
    print("Artifact check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
