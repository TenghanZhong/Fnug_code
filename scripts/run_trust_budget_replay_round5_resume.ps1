$root = if ($env:FNUG_RESULTS_ROOT) { $env:FNUG_RESULTS_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot "..")).Path }
$code = if ($env:FNUG_CODE_ENTRY) { $env:FNUG_CODE_ENTRY } else { Join-Path $PSScriptRoot "..\src\fnug.py" }
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$logDir = Join-Path $root "logs\trust_budget_replay_round5"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$env:FNUG_HEDGE_CACHE_DIR = Join-Path $root "cache_hedging_trust_latentsummary"
Set-Location $root

$runs = @(
    @{
        runName = "budget_replay_D3_fam3_cost3_seed3"
        logName = "budget_replay_D3_resume_atmstraddle.log"
        families = "atm_straddle"
        costs = "1,2,5"
        scale = "0.15"
        trust = "0.008"
        excessCvar = "0.75"
        excessRelu = "0.50"
        valWorstExcess = "0.25"
        feasibleMax = "0.008"
        feasibleGrid = "0.006,0.008,0.009,0.012"
    },
    @{
        runName = "budget_replay_C3_fam3_cost3_seed3"
        logName = "budget_replay_C3_fam3_cost3_seed3.log"
        families = "atm_call,put25,atm_straddle"
        costs = "1,2,5"
        scale = "0.15"
        trust = "0.008"
        excessCvar = "0.50"
        excessRelu = "0.25"
        valWorstExcess = "0.0"
        feasibleMax = ""
        feasibleGrid = "0.007,0.008,0.009,0.012"
    }
)

foreach ($cfg in $runs) {
    $log = Join-Path $logDir $cfg.logName
    Write-Host "=== RUN $($cfg.runName) / $($cfg.families) ==="
    $argsList = @(
        $code,
        "--hedgeset",
        "--hedgeset-candidate-sweeps", "none",
        "--hedge-families", $cfg.families,
        "--hedge-state-modes", "latent",
        "--hedge-cost-bps-list", $cfg.costs,
        "--seeds", "1,42,2026",
        "--hedgeset-policy-variants", "full",
        "--hedgeset-action-scopes", "full",
        "--hedgeset-run-name", $cfg.runName,
        "--hedgeset-max-instruments", "4",
        "--hedgeset-residual-base", "ls_deltavega",
        "--hedgeset-residual-scale", $cfg.scale,
        "--hedgeset-trust-gate-init-bias", "-1.0",
        "--hedgeset-lambda-trust-l1", $cfg.trust,
        "--hedgeset-lambda-excess-cvar", $cfg.excessCvar,
        "--hedgeset-lambda-excess-relu", $cfg.excessRelu,
        "--hedgeset-lambda-factor-neutral", "0.10",
        "--hedgeset-lambda-regime-excess-cvar", "0.0",
        "--hedgeset-val-score-worst-regime-excess-weight", $cfg.valWorstExcess,
        "--hedgeset-val-feasible-excess-cvar95-grid", $cfg.feasibleGrid
    )
    if ($cfg.feasibleMax -ne "") {
        $argsList += @("--hedgeset-val-feasible-excess-cvar95-max", $cfg.feasibleMax)
    }
    & $py @argsList 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) {
        throw "Run failed: $($cfg.runName). See $log"
    }
    Write-Host "DONE $($cfg.runName) -> $log"
}
