"""Specification B: does Sharpness-Aware Minimization help a portfolio allocator out of sample?

    python -m flatminima.data.riskfree          # once, after `make data` (3-month T-bill, FRED)
    python -m flatminima.portfolio.experiment_b

Specification B was written AFTER seeing specification A's results
(``docs/portfolio_baselines.md``),
which were degenerate (the minimum-variance portfolio sat almost entirely in a short Treasury ETF).
It is therefore post hoc, and reported as such; specification A is kept unchanged.

Design (fixed before looking at any real-data result of B; hyperparameters were set on synthetic
data only):

* Returns are EXCESS log returns: asset log return minus log(1 + daily T-bill rate), so cash is
  worth zero and a Treasury ETF is no longer a free lunch. The daily rate is DTB3 / 100 / 252, an
  approximation of the true accrual.
* Long-only, fully invested, per-asset cap (default 10%) for every strategy.
* Every strategy sees the same 756 days; the policy network needs 126 more days of history to
  build its features, hence ``lookback = 882``.
* Adam and SAM get the same number of gradient evaluations (SAM takes half as many steps).
* SAM's radius rho is preregistered in {0.05, 0.2}; no other value is tried.
* The result may be that SAM does not help. That is reported as it comes out.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from flatminima.data import db, returns, riskfree
from flatminima.portfolio.metrics import mbb_sharpe_ci, summarize
from flatminima.portfolio.optimizers import equal_weight, min_cvar, min_variance
from flatminima.portfolio.walkforward import Strategy, walk_forward

RHOS = (0.05, 0.2)
EST_WINDOW = 756
WARMUP = 126


def build_strategies(
    cap: float,
    rhos: tuple[float, ...],
    seed: int,
    grad_budget: int,
    lr: float,
    analyse_every: int,
    families: tuple[str, ...] = ("convex", "direct", "policy"),
) -> dict[str, Strategy]:
    """Strategy dictionary for one seed (only the policy networks depend on the seed)."""
    from flatminima.covariance.estimators import sample_cov
    from flatminima.portfolio.torch_alloc import DirectStrategy, PolicyStrategy, TrainConfig

    out: dict[str, Strategy] = {}
    if "convex" in families:
        out["EqualWeight"] = lambda r: equal_weight(r.shape[1])
        out["MinVar-Sample(cap)"] = lambda r: min_variance(sample_cov(r[-EST_WINDOW:]), cap=cap)
        out["MinCVaR-97.5(cap)"] = lambda r: min_cvar(r[-EST_WINDOW:], 0.975, cap=cap)
    cfgs = {"Adam": TrainConfig("adam", lr=lr, grad_budget=grad_budget)}
    for rho in rhos:
        cfgs[f"SAM-{rho:g}"] = TrainConfig("sam", rho=rho, lr=lr, grad_budget=grad_budget)
    if "direct" in families:
        for name, cfg in cfgs.items():
            out[f"Direct-{name}"] = DirectStrategy(
                cfg, cap, EST_WINDOW, analyse_every, rho_ref=rhos[0]
            )
    if "policy" in families:
        for name, cfg in cfgs.items():
            out[f"Policy-{name}"] = PolicyStrategy(
                cfg, cap, retrain_every=3, seed=seed, analyse_every=analyse_every, rho_ref=rhos[0]
            )
    return out


def _hessian_table(strategies: dict[str, Strategy], dates, seed: int) -> pd.DataFrame:
    rows = []
    for name, s in strategies.items():
        for entry in getattr(s, "log", []):
            rows.append(
                {
                    "strategy": name,
                    "seed": seed,
                    "date": dates[entry["call"]],
                    **{k: v for k, v in entry.items() if k != "call"},
                }
            )
    return pd.DataFrame(rows)


def excess_log_returns(wide: pd.DataFrame, rf_annual_pct: pd.Series) -> pd.DataFrame:
    rf = riskfree.daily_rf(rf_annual_pct, wide.index)
    return wide.sub(np.log1p(rf.to_numpy()), axis=0)


def _md_table(df: pd.DataFrame, fmt: dict[str, str]) -> list[str]:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append(
            "| "
            + " | ".join(fmt.get(c, "{}").format(r[c]) if c in fmt else str(r[c]) for c in cols)
            + " |"
        )
    return lines


def run(
    db_path: str | Path,
    seeds: tuple[int, ...] = (0, 1, 2),
    rhos: tuple[float, ...] = RHOS,
    cap: float = 0.10,
    rebalance_every: int = 21,
    cost_bps: float = 5.0,
    grad_budget: int = 200,
    lr: float = 0.05,
    analyse_every: int = 6,
    n_boot: int = 5000,
    rf_constant_pct: float | None = None,
    report: str | Path | None = "docs/portfolio_specB.md",
    max_days: int | None = None,
) -> dict:
    """Run the experiment; returns the main table and robustness tables (also written to DuckDB)."""
    con = db.connect(db_path)
    wide = returns.wide_returns(con)
    source = con.execute("SELECT DISTINCT source FROM raw_prices").fetchall()[0][0]
    if rf_constant_pct is not None:
        rate = pd.Series(rf_constant_pct, index=wide.index, name="rate_pct")
        rf_label = f"constant {rf_constant_pct:.2f}% (SMOKE TEST ONLY)"
    else:
        try:
            rate = riskfree.read(con)
        except Exception as exc:  # table missing
            con.close()
            raise SystemExit(
                "Table 'riskfree' not found. Run `python -m flatminima.data.riskfree` first "
                "(after `make data`, which rebuilds the database from scratch)."
            ) from exc
        rf_label = f"FRED DTB3 ({rate.index.min().date()} to {rate.index.max().date()})"
    if max_days:
        wide = wide.iloc[-max_days:]
    x = excess_log_returns(wide, rate)
    lookback = EST_WINDOW + WARMUP

    results, hess, t0 = {}, [], time.time()
    for seed in seeds:
        fam = ("convex", "direct", "policy") if seed == seeds[0] else ("policy",)
        strats = build_strategies(cap, rhos, seed, grad_budget, lr, analyse_every, fam)
        res = walk_forward(
            x, strats, lookback=lookback, rebalance_every=rebalance_every, cost_bps=cost_bps
        )
        results[seed] = res
        hess.append(_hessian_table(strats, res.turnover.index, seed))
        print(f"seed {seed} done in {time.time() - t0:.0f}s", flush=True)

    main_res = results[seeds[0]]
    main = summarize(main_res, n_boot=n_boot)

    # robustness across seeds for the policy strategies
    rob_rows = []
    for seed, res in results.items():
        s = main if seed == seeds[0] else summarize(res, benchmark="Policy-Adam", n_boot=200)
        for r in s.itertuples():
            if r.strategy.startswith("Policy"):
                rob_rows.append(
                    {
                        "strategy": r.strategy,
                        "seed": seed,
                        "sharpe": r.sharpe_net,
                        "oos_cvar": r.oos_cvar_975,
                    }
                )
    rob = pd.DataFrame(rob_rows)
    hess_df = pd.concat([h for h in hess if len(h)], ignore_index=True)

    # paired Sharpe difference of each SAM variant against its Adam counterpart (main seed)
    paired = []
    for fam_name in ("Direct", "Policy"):
        base = f"{fam_name}-Adam"
        cols = [c for c in main_res.returns_net.columns if c.startswith(fam_name)]
        ci = mbb_sharpe_ci(main_res.returns_net[cols], benchmark=base, n_boot=n_boot)
        for c in cols:
            if c != base:
                paired.append(
                    {
                        "comparison": f"{c} - {base}",
                        "diff": ci.loc[c, "diff_vs_bench"],
                        "lo": ci.loc[c, "diff_ci_low"],
                        "hi": ci.loc[c, "diff_ci_high"],
                    }
                )
    paired = pd.DataFrame(paired)

    # store everything in DuckDB for the figures
    long = main_res.returns_net.stack().rename("ret").reset_index()
    long.columns = ["date", "strategy", "ret"]
    con.register("_r", long)
    con.execute("CREATE OR REPLACE TABLE specb_returns AS SELECT * FROM _r")
    con.unregister("_r")
    h = hess_df.copy()
    h["date"] = pd.to_datetime(h["date"])
    con.register("_h", h)
    con.execute("CREATE OR REPLACE TABLE specb_hessian AS SELECT * FROM _h")
    con.unregister("_h")
    con.close()

    out = {"main": main, "robustness": rob, "hessian": hess_df, "paired": paired}
    if report:
        _write_report(
            report,
            out,
            main_res,
            source,
            rf_label,
            cap,
            rhos,
            seeds,
            grad_budget,
            rebalance_every,
            cost_bps,
            lookback,
        )
    return out


def _write_report(path, out, res, source, rf_label, cap, rhos, seeds, budget, reb, cost, lb):
    main, rob, hess, paired = out["main"], out["robustness"], out["hessian"], out["paired"]
    d0, d1 = res.returns_net.index.min().date(), res.returns_net.index.max().date()
    L = [
        "# Specification B: Adam vs SAM allocators (post hoc)",
        "",
        "> **Specification B was defined after specification A's results were seen** "
        "(`docs/portfolio_baselines.md`), because A degenerated into a Treasury-ETF position. "
        "It is post hoc. A is kept unchanged. Nothing below is a claim of out-of-sample alpha.",
        "",
        f"Data: `{source}`; risk-free rate: {rf_label}, approximated as DTB3/100/252 per day. "
        f"All returns are excess log returns. Out-of-sample: {d0} to {d1} "
        f"({len(res.returns_net)} days). Lookback {lb} days (756 for estimation, 126 for the "
        f"policy features), rebalance every {reb} days, long-only, cap {cap:.0%} per asset, "
        f"cost {cost:g} bps per unit traded. Adam and SAM use the same number of gradient "
        f"evaluations ({budget}); SAM rho preregistered in {list(rhos)}. Policy networks: "
        f"seeds {list(seeds)}; the table below uses seed {seeds[0]}. Sharpe intervals: Moving "
        f"Block Bootstrap, block {main.attrs['block']}, {main.attrs['n_boot']} resamples. "
        "Not multiple-testing corrected.",
        "",
        "## Main table",
        "",
    ]
    t = main[
        [
            "strategy",
            "ann_excess_return",
            "ann_vol",
            "sharpe_net",
            "sharpe_ci_low",
            "sharpe_ci_high",
            "oos_cvar_975",
            "max_drawdown",
            "ann_turnover",
            "effective_n",
        ]
    ].copy()
    t["Sharpe (95% CI)"] = [
        f"{a:.2f} ({b:.2f}, {c:.2f})"
        for a, b, c in zip(t.sharpe_net, t.sharpe_ci_low, t.sharpe_ci_high, strict=True)
    ]
    t = t.drop(columns=["sharpe_net", "sharpe_ci_low", "sharpe_ci_high"])
    t = t[
        [
            "strategy",
            "ann_excess_return",
            "ann_vol",
            "Sharpe (95% CI)",
            "oos_cvar_975",
            "max_drawdown",
            "ann_turnover",
            "effective_n",
        ]
    ]
    L += _md_table(
        t,
        {
            "ann_excess_return": "{:.1%}",
            "ann_vol": "{:.1%}",
            "oos_cvar_975": "{:.2%}",
            "max_drawdown": "{:.1%}",
            "ann_turnover": "{:.2f}",
            "effective_n": "{:.1f}",
        },
    )
    L += [
        "",
        "`oos_cvar_975` = mean daily loss over the worst 2.5% of out-of-sample days.",
        "",
        "## Paired Sharpe difference, SAM minus Adam (main seed)",
        "",
    ]
    L += _md_table(paired, {"diff": "{:+.2f}", "lo": "{:+.2f}", "hi": "{:+.2f}"})
    L += ["", "## Robustness across policy seeds", ""]
    g = (
        rob.groupby("strategy")
        .agg(
            sharpe_mean=("sharpe", "mean"),
            sharpe_sd=("sharpe", "std"),
            cvar_mean=("oos_cvar", "mean"),
            cvar_sd=("oos_cvar", "std"),
            n=("seed", "count"),
        )
        .reset_index()
    )
    L += _md_table(
        g,
        {
            "sharpe_mean": "{:.2f}",
            "sharpe_sd": "{:.2f}",
            "cvar_mean": "{:.3%}",
            "cvar_sd": "{:.3%}",
        },
    )
    L += ["", "## Loss-surface curvature at the trained point (medians over refits)", ""]
    hs = (
        hess.groupby("strategy")
        .agg(
            train_loss=("train_loss", "median"),
            lambda_max=("lambda_max", "median"),
            trace=("trace", "median"),
            sharpness=("sharpness", "median"),
            n=("date", "count"),
        )
        .reset_index()
    )
    L += _md_table(
        hs,
        {"train_loss": "{:.5f}", "lambda_max": "{:.3g}", "trace": "{:.3g}", "sharpness": "{:.3g}"},
    )
    L += [
        "",
        "Curvature is measured on the training loss (in-sample CVaR) at the fitted "
        "parameters; sharpness is the loss increase for a one-step ascent of radius "
        f"{rhos[0]:g}.",
        "",
        "## Reading these results",
        "",
    ]
    L += _auto_reading(main, paired, hs, rob)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def _auto_reading(main, paired, hs, rob) -> list[str]:
    """Plain statements derived from the measured numbers; no narrative beyond them."""
    out = []
    for fam in ("Direct", "Policy"):
        a = hs[hs.strategy == f"{fam}-Adam"]
        for s in hs[hs.strategy.str.startswith(f"{fam}-SAM")].itertuples():
            if len(a):
                ratio = s.lambda_max / a.lambda_max.iloc[0]
                out.append(
                    f"- {s.strategy}: median lambda_max is {ratio:.2f}x that of {fam}-Adam "
                    f"({'flatter' if ratio < 1 else 'not flatter'} by this measure)."
                )
    for r in paired.itertuples():
        inc = r.lo <= 0 <= r.hi
        out.append(
            f"- {r.comparison}: Sharpe difference {r.diff:+.2f} (95% CI {r.lo:+.2f} to "
            f"{r.hi:+.2f}); the interval {'includes' if inc else 'excludes'} zero."
        )
    out += [
        "- The gradient-based allocators are trained on the in-sample CVaR; the baselines are "
        "not tuned on out-of-sample data and nothing here was selected on the out-of-sample "
        "result, but specification B itself was chosen after seeing A.",
        "- One out-of-sample path, one market, 28 ETFs from 2005 (survivorship bias). Intervals "
        "are not multiple-testing corrected.",
        "- Only the claims that the numbers above support may be quoted; if SAM shows no "
        "out-of-sample gain, that is the result.",
    ]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="data/flatminima.duckdb")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--cap", type=float, default=0.10)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--grad-budget", type=int, default=200)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument(
        "--rf-constant-pct",
        type=float,
        default=None,
        help="annual %% rate for a SMOKE TEST on synthetic data only",
    )
    p.add_argument("--max-days", type=int, default=None, help="use only the last N days (smoke)")
    p.add_argument("--report", default="docs/portfolio_specB.md")
    a = p.parse_args()
    out = run(
        a.db,
        tuple(a.seeds),
        cap=a.cap,
        cost_bps=a.cost_bps,
        grad_budget=a.grad_budget,
        n_boot=a.n_boot,
        rf_constant_pct=a.rf_constant_pct,
        report=a.report,
        max_days=a.max_days,
    )
    pd.set_option("display.width", 220)
    print(out["main"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
