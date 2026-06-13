$root = if ($env:FNUG_RESULTS_ROOT) { $env:FNUG_RESULTS_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot "..")).Path }
$code = if ($env:FNUG_CODE_ENTRY) { $env:FNUG_CODE_ENTRY } else { Join-Path $PSScriptRoot "..\src\fnug.py" }
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$logDir = Join-Path $root "logs\trust_control_round2"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$env:FNUG_HEDGE_CACHE_DIR = Join-Path $root "cache_hedging_trust_latentsummary"
Set-Location $root

$configs = @(
    @{ name = "exp_ctrl_C2_latent_bm1_l006_s015_seed3"; scale = "0.15"; trust = "0.006"; excessCvar = "0.50"; excessRelu = "0.25" },
    @{ name = "exp_ctrl_C3_latent_bm1_l008_s015_seed3"; scale = "0.15"; trust = "0.008"; excessCvar = "0.50"; excessRelu = "0.25" },
    @{ name = "exp_ctrl_E1_latent_bm1_l003_s010_seed3"; scale = "0.10"; trust = "0.003"; excessCvar = "0.50"; excessRelu = "0.25" },
    @{ name = "exp_ctrl_F1_latent_bm1_l003_s015_ex075_seed3"; scale = "0.15"; trust = "0.003"; excessCvar = "0.75"; excessRelu = "0.25" },
    @{ name = "exp_ctrl_F2_latent_bm1_l003_s015_relu050_seed3"; scale = "0.15"; trust = "0.003"; excessCvar = "0.50"; excessRelu = "0.50" }
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
        --hedgeset-lambda-excess-cvar $cfg.excessCvar `
        --hedgeset-lambda-excess-relu $cfg.excessRelu `
        --hedgeset-lambda-factor-neutral 0.10 `
        --hedgeset-lambda-regime-excess-cvar 0.0 `
        --hedgeset-val-score-worst-regime-excess-weight 0.0 `
        2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) {
        throw "Run failed: $($cfg.name). See $log"
    }
    Write-Host "DONE $($cfg.name) -> $log"
}
