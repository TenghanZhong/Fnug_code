from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_FILES = [
    "README.md",
    "requirements.txt",
    "src/fnug.py",
    "src/plot_fnug.py",
    "configs/fnug_main.json",
    "configs/modern_baselines.json",
    "configs/qiao_residual_ls.json",
    "configs/hard_ls_only.json",
    "configs/ablation_no_factorneutral.json",
    "configs/ablation_no_trustgate.json",
    "configs/ablation_no_auxrisk.json",
    "scripts/rebuild_e1008_final_analysis.py",
    "scripts/rebuild_e1008_ablation.py",
    "scripts/run_e1008_moving_block_bootstrap.py",
    "scripts/check_artifact.py",
    "tests/test_feasible_selector.py",
    "results/analysis/e1008_final_combined_metrics.csv",
    "results/analysis/e1008_final_combined_episode_panel.csv",
    "results/analysis/e1008_final_pairwise_bootstrap.csv",
    "results/analysis/e1008_final_main_cell_metrics.csv",
    "results/analysis/e1008_final_heatmap_values.csv",
    "results/analysis/e1008_final_vs_ls_reductions.csv",
    "results/analysis/e1008_final_family_summary.csv",
    "results/analysis/e1008_ivgrid_matched_cell_summary.csv",
    "results/analysis/e1008_primary_moving_block_bootstrap.csv",
    "results/analysis/e1008_primary_moving_block_bootstrap_manifest.json",
    "results/analysis/e1008_ablation_table.csv",
    "results/analysis/e1008_ablation_pairwise_bootstrap.csv",
    "results/analysis/e1008_ablation_seed_audit.csv",
    "results/analysis/e1008_ablation_seed_summary.csv",
    "results/analysis/e1008_ablation_multiseed_pairwise_bootstrap.csv",
    "results/analysis/e1008_final_artifact_manifest.md",
    "results/paper_figures/figure2_tail_loss_summary.csv",
    "results/paper_figures/figure2_tail_loss_survival.pdf",
    "results/paper_figures/figure2_tail_loss_survival.png",
    "results/paper_figures/figure3_paired_excess_data.csv",
    "results/paper_figures/figure3_paired_excess_loss_vs_ls.pdf",
    "results/paper_figures/figure3_paired_excess_loss_vs_ls.png",
    "results/paper_figures/figure3_paired_excess_summary.csv",
    "results/paper_figures/figure4_cost_seed_robustness_heatmap.pdf",
    "results/paper_figures/figure4_cost_seed_robustness_heatmap.png",
    "results/paper_figures/figure4_heatmap_values.csv",
    "results/paper_figures/figure5_regime_diagnostic_summary.csv",
    "results/paper_figures/figure5_residual_diagnostics.pdf",
    "results/paper_figures/figure5_residual_diagnostics.png",
]

FORBIDDEN_TRACKED_PATTERNS = [
    "results/hedgeset/",
    "final_" + "no" + "reg",
    "trust_" + "round",
    "run_" + "trust_",
    "ablation_no_" + "regimeexcess",
    "rebuild_final_" + "no" + "reg",
    "lambda_" + "regime_" + "excess_" + "cvar",
]

def _token(*parts: str | int) -> str:
    return "".join(chr(p) if isinstance(p, int) else p for p in parts)


SENSITIVE_TOKENS = [
    _token(67, ":", "\\", "Users"),
    _token(68, ":", "\\"),
    _token("Bai", "du"),
    _token("Over", "leaf"),
    _token("Chat", 71, 80, 84),
    _token(71, 80, 84),
    _token("Co", "dex"),
    _token("Open", 65, 73),
    _token("Clau", "de"),
    _token("Teng", "han"),
    _token("review", "er"),
]
TEXT_EXTENSIONS = {".py", ".md", ".txt", ".json", ".csv", ".yml", ".yaml", ".toml", ".gitignore"}


def tracked_files() -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return [line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]


def check_expected_files() -> list[str]:
    return [rel for rel in EXPECTED_FILES if not (ROOT / rel).exists()]


def check_forbidden_tracked(paths: list[str]) -> list[str]:
    bad: list[str] = []
    for rel in paths:
        for pattern in FORBIDDEN_TRACKED_PATTERNS:
            if pattern in rel:
                bad.append(rel)
                break
    return bad


def check_sensitive_tokens(paths: list[str]) -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    for rel in paths:
        path = ROOT / rel
        if path.name == "check_artifact.py" or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            lower = line.lower()
            for token in SENSITIVE_TOKENS:
                if token.lower() in lower:
                    hits.append((rel, lineno, line.strip()))
                    break
    return hits


def first_row(path: Path, predicate) -> dict[str, str] | None:
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if predicate(row):
                return row
    return None


def check_e1008_main_metrics() -> dict[str, str]:
    path = ROOT / "results/analysis/e1008_final_main_cell_metrics.csv"
    row = first_row(path, lambda r: r.get("model") in {"FNUG", "HedgeSet-latent"})
    if row is None:
        return {}
    out = {
        "cvar95": f"{float(row['cvar_95']):.4f}",
        "cvar99": f"{float(row['cvar_99']):.4f}",
        "worst_regime": f"{float(row['worst_regime_cvar_95']):.4f}",
        "excess_cvar95": f"{float(row['excess_cvar_95']):.4f}",
        "turnover": f"{float(row['mean_turnover']):.3f}",
    }
    expected = {
        "cvar95": "0.0945",
        "cvar99": "0.1282",
        "worst_regime": "0.1015",
        "excess_cvar95": "0.0102",
        "turnover": "2.105",
    }
    return out if out == expected else {}


def check_e1008_ivgrid_summary() -> dict[str, str]:
    path = ROOT / "results/analysis/e1008_ivgrid_matched_cell_summary.csv"
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
        "cvar_95_n_cells": "27",
        "cvar_95_fnug_lower_cells": "22",
        "cvar_95_mean_diff": "-0.0064",
        "cvar_99_n_cells": "27",
        "cvar_99_fnug_lower_cells": "21",
        "cvar_99_mean_diff": "-0.0109",
        "worst_regime_cvar_95_n_cells": "27",
        "worst_regime_cvar_95_fnug_lower_cells": "21",
        "worst_regime_cvar_95_mean_diff": "-0.0084",
    }
    return out if out == expected else {}


def check_e1008_mbb_summary() -> dict[str, str]:
    path = ROOT / "results/analysis/e1008_primary_moving_block_bootstrap.csv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    checks = {
        "confirmatory_vs_ls": ("-0.0193", "-0.0309", "-0.0093", "0.0020"),
        "confirmatory_vs_qiao": ("-0.0247", "-0.0344", "-0.0066", "0.0018"),
        "diagnostic_vs_ivgrid": ("-0.0014", "-0.0095", "0.0059", "0.6719"),
    }
    out: dict[str, str] = {}
    for label, expected in checks.items():
        match = [
            row for row in rows
            if row.get("comparison") == label and row.get("block_length") == "5"
        ]
        if len(match) != 1:
            return {}
        row = match[0]
        values = (
            f"{float(row['observed_diff']):.4f}",
            f"{float(row['ci95_low']):.4f}",
            f"{float(row['ci95_high']):.4f}",
            f"{float(row['p_two_sided_centered']):.4f}",
        )
        if values != expected:
            return {}
        out[label] = ",".join(values)
    return out


def check_e1008_ablation_seed_audit() -> dict[str, str]:
    path = ROOT / "results/analysis/e1008_ablation_seed_audit.csv"
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
    for model, expected_selection in expected_models.items():
        model_rows = [row for row in rows if row.get("model") == model]
        seeds = {row.get("audit_seed") for row in model_rows}
        selections = {row.get("checkpoint_selection") for row in model_rows}
        if seeds != {"1", "42", "2026"} or selections != {expected_selection}:
            return {}
    return {"rows": str(len(rows)), "models": str(len(expected_models))}


def check_config_budget() -> list[str]:
    bad: list[str] = []
    config_paths = [
        "configs/fnug_main.json",
        "configs/ablation_no_trustgate.json",
        "configs/ablation_no_auxrisk.json",
    ]
    for rel in config_paths:
        text = (ROOT / rel).read_text(encoding="utf-8")
        if '"final_checkpoint_selection": "best_feasible_excess_0p008"' not in text:
            bad.append(f"{rel}: final checkpoint is not E1008")
        if '"val_feasible_excess_cvar95_max": 0.008' not in text:
            bad.append(f"{rel}: feasible excess budget is not 0.008")
    factor_text = (ROOT / "configs/ablation_no_factorneutral.json").read_text(encoding="utf-8")
    if '"final_checkpoint_selection": "best_score"' not in factor_text:
        bad.append("configs/ablation_no_factorneutral.json: expected best_score diagnostic selection")
    if '"val_feasible_excess_cvar95_max": 0.008' not in factor_text:
        bad.append("configs/ablation_no_factorneutral.json: feasible excess budget is not 0.008")
    return bad


def main() -> int:
    paths = tracked_files()
    missing = check_expected_files()
    forbidden = check_forbidden_tracked(paths)
    sensitive = check_sensitive_tokens(paths)
    budget_config_problems = check_config_budget()
    main_metrics = check_e1008_main_metrics()
    ivgrid_summary = check_e1008_ivgrid_summary()
    mbb_summary = check_e1008_mbb_summary()
    ablation_audit = check_e1008_ablation_seed_audit()

    if missing:
        print("Missing expected files:")
        for rel in missing:
            print(f"  - {rel}")
    if forbidden:
        print("Forbidden tracked paths remain:")
        for rel in forbidden[:100]:
            print(f"  - {rel}")
        if len(forbidden) > 100:
            print(f"  ... {len(forbidden) - 100} more")
    if sensitive:
        print("Sensitive tokens found:")
        for rel, lineno, line in sensitive[:50]:
            print(f"  - {rel}:{lineno}: {line}")
        if len(sensitive) > 50:
            print(f"  ... {len(sensitive) - 50} more")
    if budget_config_problems:
        print("Configuration budget problems:")
        for problem in budget_config_problems:
            print(f"  - {problem}")

    print("E1008 main-cell smoke check:" if main_metrics else "E1008 main-cell smoke check failed.")
    for key, value in main_metrics.items():
        print(f"  {key}: {value}")
    print("E1008 IVGrid matched-cell check:" if ivgrid_summary else "E1008 IVGrid matched-cell check failed.")
    for key, value in ivgrid_summary.items():
        print(f"  {key}: {value}")
    print("E1008 primary MBB check:" if mbb_summary else "E1008 primary MBB check failed.")
    for key, value in mbb_summary.items():
        print(f"  {key}: {value}")
    print("E1008 ablation audit check:" if ablation_audit else "E1008 ablation audit check failed.")
    for key, value in ablation_audit.items():
        print(f"  {key}: {value}")

    if missing or forbidden or sensitive or budget_config_problems or not main_metrics or not ivgrid_summary or not mbb_summary or not ablation_audit:
        return 1
    print("Artifact check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
