"""End-to-end data pipeline: ingest -> quality checks -> log returns.

python -m flatminima.data.pipeline --source yahoo        # real data
python -m flatminima.data.pipeline --source synthetic    # offline, labelled synthetic
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flatminima.config import SEED, START_DATE, UNIVERSE
from flatminima.data import db, ingest, quality, returns, synthetic


def run(source: str, db_path: str | Path, start: str = START_DATE, n_synth: int = 5000) -> None:
    """Rebuild the database from scratch for the chosen source."""
    path = Path(db_path)
    if path.exists():
        path.unlink()
    con = db.connect(path)
    tickers = list(UNIVERSE)

    if source == "yahoo":
        df = ingest.download_yahoo(tickers, start=start)
    elif source == "synthetic":
        df = synthetic.simulate_panel(tickers, n=n_synth, start=start, seed=SEED)
        print("NOTE: SYNTHETIC data (GARCH-t simulation). Not market data.")
    else:
        raise ValueError(f"unknown source: {source}")

    n_rows = ingest.load_prices(con, df, source=source)
    report = quality.run_checks(con)
    n_ret = returns.build_returns(con)
    flagged = report[report["n_flagged"] > 0]
    print(f"prices: {n_rows} rows | returns: {n_ret} rows | tickers: {len(tickers)}")
    print(f"quality checks with flags: {len(flagged)} (see table data_quality)")
    if len(flagged):
        print(flagged.to_string(index=False))
    con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["yahoo", "synthetic"], required=True)
    parser.add_argument("--db", default="data/flatminima.duckdb")
    parser.add_argument("--start", default=START_DATE)
    args = parser.parse_args()
    run(args.source, args.db, start=args.start)


if __name__ == "__main__":
    main()
