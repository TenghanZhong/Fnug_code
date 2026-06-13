import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fnug import HedgeSetConfig, _hedgeset_feasible_checkpoint


def _metrics(**overrides):
    base = {
        "excess_cvar_95": 0.0075,
        "mean_turnover": 2.35,
        "mean_tc_norm": 0.055,
        "mean_abs_residual_delta_exposure": 2e-6,
        "mean_abs_residual_vega_exposure": 3e-7,
    }
    base.update(overrides)
    return base


def test_feasible_selector_default_ignores_operating_caps():
    cfg = HedgeSetConfig()

    assert _hedgeset_feasible_checkpoint(_metrics(), cfg, excess_cvar95_max=0.008)


def test_feasible_selector_still_rejects_excess_and_exposure_breaches():
    cfg = HedgeSetConfig()

    assert not _hedgeset_feasible_checkpoint(
        _metrics(excess_cvar_95=0.009), cfg, excess_cvar95_max=0.008
    )
    assert not _hedgeset_feasible_checkpoint(
        _metrics(mean_abs_residual_delta_exposure=2e-4), cfg, excess_cvar95_max=0.008
    )
    assert not _hedgeset_feasible_checkpoint(
        _metrics(mean_abs_residual_vega_exposure=2e-4), cfg, excess_cvar95_max=0.008
    )


def test_feasible_selector_honors_explicit_operating_caps():
    cfg = HedgeSetConfig()
    cfg.val_feasible_turnover_max = 2.14
    cfg.val_feasible_tc_norm_max = 0.0266

    assert not _hedgeset_feasible_checkpoint(_metrics(), cfg, excess_cvar95_max=0.008)
    assert _hedgeset_feasible_checkpoint(
        _metrics(mean_turnover=2.10, mean_tc_norm=0.026), cfg, excess_cvar95_max=0.008
    )
