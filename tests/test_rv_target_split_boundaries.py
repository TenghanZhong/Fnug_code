import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fnug import DataConfig, RVTargetBuilder


def test_forward_targets_do_not_cross_chronological_splits():
    dates = pd.bdate_range("2021-10-01", "2023-02-15")
    prices = pd.DataFrame(
        {
            "date": dates,
            "return": np.linspace(-0.01, 0.01, len(dates)),
        }
    )
    cfg = DataConfig(train_end=2021, val_end=2022)

    targets = RVTargetBuilder(prices, cfg).build().set_index("date")

    for year in (2021, 2022):
        split_dates = dates[dates.year == year]

        assert targets.loc[split_dates[-5:], ["rv5", "dsv5"]].isna().all().all()
        assert targets.loc[split_dates[-21:], ["rv21", "stress21"]].isna().all().all()

        assert targets.loc[split_dates[-6], ["rv5", "dsv5"]].notna().all()
        assert targets.loc[split_dates[-22], ["rv21", "stress21"]].notna().all()
