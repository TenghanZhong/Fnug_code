$root = if ($env:FNUG_RESULTS_ROOT) { $env:FNUG_RESULTS_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot "..")).Path }
$code = if ($env:FNUG_CODE_ENTRY) { $env:FNUG_CODE_ENTRY } else { Join-Path $PSScriptRoot "..\src\fnug.py" }
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$logDir = Join-Path $root "logs\trust_vscore_multiseed"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$env:FNUG_HEDGE_CACHE_DIR = Join-Path $root "cache_hedging_trust_latentsummary"
Set-Location $root

$configs = @(
    @{ name = "exp_vscore_A2_latent_bm1_l003_s015_seed3"; scale = "0.15"; trust = "0.003" },
    @{ name = "exp_vscore_A1_latent_bm1_l003_s020_seed3"; scale = "0.20"; trust = "0.003" },
    @{ name = "exp_vscore_C1_latent_bm1_l004_s015_seed3"; scale = "0.15"; trust = "0.004" }
)

foreach ($cfg in $configs) {
    $log = Join-Path $logDir ($cfg.name + ".log")
    Write-Host "=== RUN $($cfg.name) ==="
    & $py $code `
        --hedgeset `
        --hedgeset-candidate-sweeps none `
        --hedge-families atm_call `
        --hedge-state-modes latent `
        --hedge-cost-bps-list 2 `
        --seeds 1,42,2026 `
        --hedgeset-policy-variants full `
        --hedgeset-action-scopes full `
        --hedgeset-run-name $cfg.name `
        --hedgeset-max-instruments 4 `
        --hedgeset-residual-base ls_deltavega `
        --hedgeset-residual-scale $cfg.scale `
        --hedgeset-trust-gate-init-bias -1.0 `
        --hedgeset-lambda-trust-l1 $cfg.trust `
        --hedgeset-lambda-excess-cvar 0.50 `
        --hedgeset-lambda-excess-relu 0.25 `
        --hedgeset-lambda-factor-neutral 0.10 `
        --hedgeset-lambda-regime-excess-cvar 0.0 `
        --hedgeset-val-score-worst-regime-excess-weight 0.0 `
        2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) {
        throw "Run failed: $($cfg.name). See $log"
    }
    Write-Host "DONE $($cfg.name) -> $log"
}
