"""Generate docs/CV_EVIDENCE.md: every figure that may appear on a CV, with how it was produced.

    python -m flatminima.report.evidence

All numbers are recomputed from the DuckDB tables (never typed by hand). Each entry states the
wording the evidence supports, the command that regenerates the source table, and the git
commit the evidence was generated at. If the working tree is dirty the file says so.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from pathlib import Path

from flatminima.backtest.tests import traffic_light_windows
from flatminima.data import db
from flatminima.portfolio.metrics import mbb_sharpe_ci


def git_state() -> tuple[str, bool]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _has(con, table: str) -> bool:
    return bool(
        con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]
        ).fetchall()[0][0]
    )


def backtest_claims(con) -> list[dict]:
    out = []
    for table, cmd in (
        ("risk_forecasts", "make risk && make backtest"),
        ("risk_forecasts_w500", "make robustness"),
    ):
        res_tab = f"backtest_results_{table}"
        if not (_has(con, res_tab) and _has(con, table)):
            continue
        res = con.execute(f'SELECT * FROM "{res_tab}"').df()
        fc = con.execute(f'SELECT * FROM "{table}" ORDER BY series, date').df()
        window = int(fc["window"].iloc[0])
        parts = []
        for name, g in fc.groupby("series", sort=False):
            loss = -g["ret"].to_numpy()
            n = len(g)
            e99 = int((loss > g["var_99"].to_numpy()).sum())
            tl = traffic_light_windows(loss > g["var_99"].to_numpy(), 250, 0.99)
            parts.append(
                f"{name}: {e99} VaR-99% exceptions in {n} out-of-sample days "
                f"({e99 / n:.2%} vs 1% nominal); Basel zones over rolling 250-day windows: "
                f"green {tl['share_green']:.1%}, yellow {tl['share_yellow']:.1%}, "
                f"red {tl['share_red']:.1%}"
            )
        out.append(
            {
                "id": f"BT-{window}",
                "claim": f"Rolling FHS-GJR-GARCH-t backtest, {window}-day window: "
                f"{len(res)} tests run, {int(res.reject_5pct.sum())} rejections at 5% "
                "(not multiple-testing corrected).",
                "detail": parts,
                "cmd": cmd,
                "source": f"table `{res_tab}`, `{table}`",
                "not_claim": "Do not call this FRTB compliance; it is ES at 97.5% with "
                "backtesting ('FRTB-style').",
            }
        )
    return out


def specb_claims(con, n_boot: int = 5000) -> list[dict]:
    if not _has(con, "specb_returns"):
        return []
    df = con.execute("SELECT date, strategy, ret FROM specb_returns").df()
    wide = df.pivot(index="date", columns="strategy", values="ret").sort_index()
    out = []
    for fam in ("Direct", "Policy"):
        base = f"{fam}-Adam"
        cols = [c for c in wide.columns if c.startswith(fam)]
        if base not in cols:
            continue
        ci = mbb_sharpe_ci(wide[cols], benchmark=base, n_boot=n_boot)
        for c in cols:
            if c == base:
                continue
            out.append(
                {
                    "id": f"SPECB-{c}",
                    "claim": f"{c} minus {base}: Sharpe difference "
                    f"{ci.loc[c, 'diff_vs_bench']:+.2f} (95% MBB CI "
                    f"{ci.loc[c, 'diff_ci_low']:+.2f} to {ci.loc[c, 'diff_ci_high']:+.2f}), "
                    f"block {ci.attrs['block']}, {n_boot} resamples, paired.",
                    "detail": [],
                    "cmd": "make data && make riskfree && make experiments",
                    "source": "table `specb_returns` (excess returns, net of 5 bps costs)",
                    "not_claim": "Specification B is post hoc. An interval that includes zero "
                    "is no evidence of an improvement.",
                }
            )
    if _has(con, "specb_hessian"):
        h = con.execute("SELECT * FROM specb_hessian").df()
        g = h.groupby("strategy")["lambda_max"].median()
        lines = [f"{k}: median lambda_max {v:.3g}" for k, v in g.items()]
        out.append(
            {
                "id": "SPECB-CURVATURE",
                "claim": "Median largest Hessian eigenvalue of the in-sample CVaR loss at the "
                "trained point, over walk-forward refits.",
                "detail": lines,
                "cmd": "make experiments",
                "source": "table `specb_hessian` (Lanczos on Hessian-vector products)",
                "not_claim": "Do not claim SAM finds flatter minima unless the ratio is clearly "
                "below 1 for both allocators; no uncertainty interval is available.",
            }
        )
    return out


def count_tests() -> int | None:
    try:
        r = subprocess.run(
            ["python", "-m", "pytest", "--collect-only", "-q"], capture_output=True, text=True
        )
        last = [ln for ln in r.stdout.splitlines() if "collected" in ln or "tests" in ln]
        return int(last[-1].split()[0]) if last else None
    except (OSError, ValueError, IndexError):
        return None


def build(db_path: str | Path) -> str:
    sha, dirty = git_state()
    con = db.connect(db_path)
    claims = backtest_claims(con) + specb_claims(con)
    con.close()
    n_tests = count_tests()
    L = [
        "# CV evidence",
        "",
        f"Generated {dt.date.today()} at commit `{sha}`"
        + (" (**working tree has uncommitted changes; commit before quoting**)" if dirty else "")
        + ". Every number below is recomputed from the database by "
        "`python -m flatminima.report.evidence`; none is typed by hand.",
        "",
        "Rule: a CV line may use only what is listed here, with its limits.",
        "",
    ]
    if n_tests:
        L += [f"- **TESTS**: {n_tests} automated tests (`pytest`), run in CI.", ""]
    for c in claims:
        L += [f"## {c['id']}", "", f"- Claim: {c['claim']}"]
        L += [f"  - {d}" for d in c["detail"]]
        L += [
            f"- Reproduce: `{c['cmd']}`",
            f"- Source: {c['source']}",
            f"- Do not say: {c['not_claim']}",
            "",
        ]
    if not claims:
        L += ["_No result tables found in the database; run the pipeline first._", ""]
    return "\n".join(L)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="data/flatminima.duckdb")
    p.add_argument("--out", default="docs/CV_EVIDENCE.md")
    a = p.parse_args()
    text = build(a.db)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text, encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
