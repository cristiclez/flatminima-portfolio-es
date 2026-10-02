"""Run the rolling risk engine on the stored data and save forecasts to DuckDB.

    python -m flatminima.risk.run                 # SPY and the equal-weight portfolio
    python -m flatminima.risk.run --series SPY --series TLT --window 1000

Output table ``risk_forecasts``: series, date, ret, sigma, var_99, var_975, es_975,
hs_var_99, hs_es_975 (positive losses, unit notional). Exception counts printed here are
purely descriptive; formal backtests (Kupiec, Christoffersen, Acerbi-Szekely) come next.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from flatminima.data import db, returns
from flatminima.risk.rolling import rolling_risk


def run(
    db_path: str | Path,
    series: Sequence[str] = ("SPY", "EW"),
    window: int = 1000,
    refit_every: int = 20,
    model: str = "gjr",
) -> pd.DataFrame:
    """Compute rolling forecasts for each series and store them in ``risk_forecasts``."""
    con = db.connect(db_path)
    wide = returns.wide_returns(con)
    frames = []
    for name in series:
        s = returns.equal_weight_return(wide) if name == "EW" else wide[name]
        out = rolling_risk(s, window=window, refit_every=refit_every, model=model)
        out = out.reset_index(names="date")
        out.insert(0, "series", name)
        frames.append(out)
        loss = -out["ret"]
        n = len(out)
        print(
            f"{name}: {n} OOS days | VaR99 exceptions {(loss > out['var_99']).sum()} "
            f"({(loss > out['var_99']).mean():.2%}, nominal 1%) | VaR97.5 exceptions "
            f"{(loss > out['var_975']).sum()} ({(loss > out['var_975']).mean():.2%}, nominal 2.5%)"
            f" | refits {out.attrs['n_refits']} (failed {out.attrs['n_failed_refits']})"
        )
    allf = pd.concat(frames, ignore_index=True)
    allf["model"] = model
    allf["window"] = window
    con.register("_rf", allf)
    con.execute("CREATE OR REPLACE TABLE risk_forecasts AS SELECT * FROM _rf")
    con.unregister("_rf")
    con.close()
    return allf


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="data/flatminima.duckdb")
    p.add_argument("--series", action="append", help="ticker or EW (default: SPY and EW)")
    p.add_argument("--window", type=int, default=1000)
    p.add_argument("--refit-every", type=int, default=20)
    p.add_argument("--model", choices=["garch", "gjr"], default="gjr")
    a = p.parse_args()
    run(a.db, a.series or ("SPY", "EW"), a.window, a.refit_every, a.model)


if __name__ == "__main__":
    main()
