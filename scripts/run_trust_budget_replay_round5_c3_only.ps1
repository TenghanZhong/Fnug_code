$root = if ($env:FNUG_RESULTS_ROOT) { $env:FNUG_RESULTS_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot "..")).Path }
$code = if ($env:FNUG_CODE_ENTRY) { $env:FNUG_CODE_ENTRY } else { Join-Path $PSScriptRoot "..\src\fnug.py" }
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$logDir = Join-Path $root "logs\trust_budget_replay_round5"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$env:FNUG_HEDGE_CACHE_DIR = Join-Path $root "cache_hedging_trust_latentsummary"
Set-Location $root

$log = Join-Path $logDir "budget_replay_C3_fam3_cost3_seed3_c3_only.log"
$argsList = @(
    $code,
    "--hedgeset",
    "--hedgeset-candidate-sweeps", "none",
    "--hedge-families", "atm_call,put25,atm_straddle",
    "--hedge-state-modes", "latent",
    "--hedge-cost-bps-list", "1,2,5",
    "--seeds", "1,42,2026",
    "--hedgeset-policy-variants", "full",
    "--hedgeset-action-scopes", "full",
    "--hedgeset-run-name", "budget_replay_C3_fam3_cost3_seed3",
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

Write-Host "=== RUN budget_replay_C3_fam3_cost3_seed3 ==="
& $py @argsList 2>&1 | Tee-Object -FilePath $log
if ($LASTEXITCODE -ne 0) {
    throw "Run failed: budget_replay_C3_fam3_cost3_seed3. See $log"
}
Write-Host "DONE budget_replay_C3_fam3_cost3_seed3 -> $log"
