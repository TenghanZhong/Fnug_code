$root = if ($env:FNUG_RESULTS_ROOT) { $env:FNUG_RESULTS_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot "..")).Path }
$code = if ($env:FNUG_CODE_ENTRY) { $env:FNUG_CODE_ENTRY } else { Join-Path $PSScriptRoot "..\src\fnug.py" }
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$logDir = Join-Path $root "logs\trust_mini_externality_round4"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$env:FNUG_HEDGE_CACHE_DIR = Join-Path $root "cache_hedging_trust_latentsummary"
Set-Location $root

$configs = @(
    @{
        tag = "D3"
        scale = "0.15"
        trust = "0.008"
        excessCvar = "0.75"
        excessRelu = "0.50"
        valWorstExcess = "0.25"
        feasibleMax = "0.008"
        feasibleGrid = "0.006,0.008,0.010,0.012"
    },
    @{
        tag = "E1"
        scale = "0.10"
        trust = "0.003"
        excessCvar = "0.50"
        excessRelu = "0.25"
        valWorstExcess = "0.0"
        feasibleMax = ""
        feasibleGrid = "0.007,0.008,0.009,0.012"
    },
    @{
        tag = "C3"
        scale = "0.15"
        trust = "0.008"
        excessCvar = "0.50"
        excessRelu = "0.25"
        valWorstExcess = "0.0"
        feasibleMax = ""
        feasibleGrid = "0.007,0.008,0.009,0.012"
    }
)

$parts = @(
    @{
        suffix = "family_cost2"
        families = "atm_call,put25,atm_straddle"
        costs = "2"
    },
    @{
        suffix = "atm_cost15"
        families = "atm_call"
        costs = "1,5"
    }
)

foreach ($cfg in $configs) {
    foreach ($part in $parts) {
        $runName = "mini_ext_$($cfg.tag)_$($part.suffix)_seed3"
        $log = Join-Path $logDir ($runName + ".log")
        Write-Host "=== RUN $runName ==="
        $argsList = @(
            $code,
            "--hedgeset",
            "--hedgeset-candidate-sweeps", "none",
            "--hedge-families", $part.families,
            "--hedge-state-modes", "latent",
            "--hedge-cost-bps-list", $part.costs,
            "--seeds", "1,42,2026",
            "--hedgeset-policy-variants", "full",
            "--hedgeset-action-scopes", "full",
            "--hedgeset-run-name", $runName,
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
            throw "Run failed: $runName. See $log"
        }
        Write-Host "DONE $runName -> $log"
    }
}
