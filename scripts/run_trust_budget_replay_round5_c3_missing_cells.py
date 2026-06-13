from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("FNUG_RESULTS_ROOT", REPO_ROOT))
CODE = Path(os.environ.get("FNUG_CODE_ENTRY", REPO_ROOT / "src" / "fnug.py"))
PYTHON = Path(os.environ.get("PYTHON", sys.executable))
RUN_NAME = "budget_replay_C3_fam3_cost3_seed3"
RUN_ROOT = ROOT / "results" / "hedgeset" / "budget_replay_C3_fam3_cost3_seed3_1eaa24b3f8"
LOG_DIR = ROOT / "logs" / "trust_budget_replay_round5" / "c3_missing_cells_py"

FAMILIES = ("atm_call", "put25", "atm_straddle")
COSTS = (1, 2, 5)
SEEDS = (1, 42, 2026)


def target_csv(family: str, cost: int, seed: int) -> Path:
    return (
        RUN_ROOT
        / family
        / f"cost_{cost}bps"
        / "variant_full"
        / "scope_full"
        / "policy_latent"
        / f"seed_{seed}"
        / "checkpoint_comparison_metrics.csv"
    )


def command(family: str, cost: int, seed: int) -> list[str]:
    return [
        str(PYTHON),
        str(CODE),
        "--hedgeset",
        "--hedgeset-candidate-sweeps",
        "none",
        "--hedge-families",
        family,
        "--hedge-state-modes",
        "latent",
        "--hedge-cost-bps-list",
        str(cost),
        "--seeds",
        str(seed),
        "--hedgeset-policy-variants",
        "full",
        "--hedgeset-action-scopes",
        "full",
        "--hedgeset-run-name",
        RUN_NAME,
        "--hedgeset-max-instruments",
        "4",
        "--hedgeset-residual-base",
        "ls_deltavega",
        "--hedgeset-residual-scale",
        "0.15",
        "--hedgeset-trust-gate-init-bias",
        "-1.0",
        "--hedgeset-lambda-trust-l1",
        "0.008",
        "--hedgeset-lambda-excess-cvar",
        "0.50",
        "--hedgeset-lambda-excess-relu",
        "0.25",
        "--hedgeset-lambda-factor-neutral",
        "0.10",
        "--hedgeset-lambda-regime-excess-cvar",
        "0.0",
        "--hedgeset-val-score-worst-regime-excess-weight",
        "0.0",
        "--hedgeset-val-feasible-excess-cvar95-grid",
        "0.007,0.008,0.009,0.012",
    ]


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["FNUG_HEDGE_CACHE_DIR"] = str(ROOT / "cache_hedging_trust_latentsummary")

    manifest = LOG_DIR / "orchestrator.log"
    with manifest.open("a", encoding="utf-8") as mf:
        mf.write(f"\n=== start {datetime.now().isoformat(timespec='seconds')} ===\n")
        for family in FAMILIES:
            for cost in COSTS:
                for seed in SEEDS:
                    target = target_csv(family, cost, seed)
                    if target.exists():
                        mf.write(f"SKIP {family} cost={cost} seed={seed}\n")
                        mf.flush()
                        continue

                    log_path = LOG_DIR / f"C3_{family}_cost{cost}_seed{seed}.log"
                    mf.write(f"RUN {family} cost={cost} seed={seed} -> {log_path}\n")
                    mf.flush()
                    with log_path.open("w", encoding="utf-8", errors="replace") as lf:
                        proc = subprocess.run(
                            command(family, cost, seed),
                            cwd=str(ROOT),
                            env=env,
                            stdout=lf,
                            stderr=subprocess.STDOUT,
                            text=True,
                        )
                    if proc.returncode != 0:
                        mf.write(f"FAIL returncode={proc.returncode} {family} cost={cost} seed={seed}\n")
                        mf.flush()
                        return proc.returncode
                    if not target.exists():
                        mf.write(f"FAIL missing target {target}\n")
                        mf.flush()
                        return 2
                    mf.write(f"DONE {family} cost={cost} seed={seed}\n")
                    mf.flush()
        mf.write(f"=== done {datetime.now().isoformat(timespec='seconds')} ===\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
