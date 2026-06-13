$root = if ($env:FNUG_RESULTS_ROOT) { $env:FNUG_RESULTS_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot "..")).Path }
$code = if ($env:FNUG_CODE_ENTRY) { $env:FNUG_CODE_ENTRY } else { Join-Path $PSScriptRoot "..\src\fnug.py" }
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$runName = "budget_replay_C3_fam3_cost3_seed3"
$runRoot = Join-Path $root "results\hedgeset\budget_replay_C3_fam3_cost3_seed3_1eaa24b3f8"
$logDir = Join-Path $root "logs\trust_budget_replay_round5\c3_missing_cells"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$env:FNUG_HEDGE_CACHE_DIR = Join-Path $root "cache_hedging_trust_latentsummary"
Set-Location $root

$families = @("atm_call", "put25", "atm_straddle")
$costs = @(1, 2, 5)
$seeds = @(1, 42, 2026)

foreach ($family in $families) {
    foreach ($cost in $costs) {
        foreach ($seed in $seeds) {
            $target = Join-Path $runRoot "$family\cost_${cost}bps\variant_full\scope_full\policy_latent\seed_$seed\checkpoint_comparison_metrics.csv"
            if (Test-Path -LiteralPath $target) {
                Write-Host "SKIP existing $family cost=$cost seed=$seed"
                continue
            }

            $log = Join-Path $logDir "C3_${family}_cost${cost}_seed${seed}.log"
            Write-Host "RUN missing C3 $family cost=$cost seed=$seed"
            $argsList = @(
                $code,
                "--hedgeset",
                "--hedgeset-candidate-sweeps", "none",
                "--hedge-families", $family,
                "--hedge-state-modes", "latent",
                "--hedge-cost-bps-list", "$cost",
                "--seeds", "$seed",
                "--hedgeset-policy-variants", "full",
                "--hedgeset-action-scopes", "full",
                "--hedgeset-run-name", $runName,
                "--hedgeset-max-instruments", "4",
                "--hedgeset-residual-base", "ls_deltavega",
                "--hedgeset-residual-scale", "0.15",
                "--hedgeset-trust-gate-init-bias", "-1.0",
                "--hedgeset-lambda-trust-l1", "0.008",
                "--hedgeset-lambda-excess-cvar", "0.50",
                "--hedgeset-lambda-excess-relu", "0.25",
                "--hedgeset-lambda-factor-neutral", "0.10",
                "--hedgeset-lambda-regime-excess-cvar", "0.0",
                "--hedgeset-val-score-worst-regime-excess-weight", "0.0",
                "--hedgeset-val-feasible-excess-cvar95-grid", "0.007,0.008,0.009,0.012"
            )

            & $py @argsList > $log 2>&1
            if ($LASTEXITCODE -ne 0) {
                throw "Run failed: C3 $family cost=$cost seed=$seed. See $log"
            }
            if (-not (Test-Path -LiteralPath $target)) {
                throw "Missing checkpoint comparison after run: $target"
            }
            Write-Host "DONE C3 $family cost=$cost seed=$seed"
        }
    }
}

Write-Host "DONE all missing C3 cells"
