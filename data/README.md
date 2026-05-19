# Data Availability

This artifact does not redistribute raw SPX option quote data. The raw data are
subject to licensing restrictions.

The training/evaluation code expects a directory containing:

- yearly option quote parquet files named `data_YYYY.parquet`;
- `security_prices.parquet`;
- `zero_coupon_yield.parquet`;
- `index_dividend_yield.parquet`;
- `VIXCLS.parquet`;
- `VXVCLS.parquet`.

Set the data directory with the `FNUG_DATA_DIR` environment variable before
running raw-data pipelines.

The repository includes derived result files under `results/` so that paper
tables, figures, and paired-bootstrap summaries can be checked without access to
the licensed raw data.

