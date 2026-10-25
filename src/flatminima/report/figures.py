"""Static figures for the README, built only from tables already stored in DuckDB.

    python -m flatminima.report.figures            # writes PNGs to docs/figures/

Figures whose source table is missing are skipped with a message, never faked. Needs the
``viz`` extra (matplotlib). Colours: the first slots of a colour-vision-checked categorical
palette, plus grey for the equal-weight benchmark; every series is also labelled directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from flatminima.data import db
from flatminima.portfolio.metrics import mbb_sharpe_ci

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GREY, RED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#e34948"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "legend.frameon": False,
        }
    )
    return plt


def _has(con, table: str) -> bool:
    return bool(
        con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]
        ).fetchall()[0][0]
    )


def _wide(con) -> pd.DataFrame:
    df = con.execute("SELECT date, strategy, ret FROM specb_returns").df()
    w = df.pivot(index="date", columns="strategy", values="ret").sort_index()
    w.index = pd.to_datetime(w.index)
    return w


SHOWN = [
    ("EqualWeight", "Equal weight", GREY),
    ("MinVar-Sample(cap)", "Min-variance (capped)", BLUE),
    ("Direct-Adam", "Direct allocator, Adam", ORANGE),
    ("Policy-Adam", "Policy network, Adam", AQUA),
]


def cumulative_and_drawdown(con, out: Path) -> list[Path]:
    plt = _plt()
    w = _wide(con)
    paths = []
    for kind in ("growth", "drawdown"):
        fig, ax = plt.subplots(figsize=(9, 4.6))
        ends = []
        for col, label, color in SHOWN:
            if col not in w:
                continue
            g = (1.0 + w[col]).cumprod()
            y = g if kind == "growth" else g / g.cummax() - 1.0
            ax.plot(
                y.index,
                y.to_numpy(),
                color=color,
                lw=2 if col != "EqualWeight" else 1.6,
                label=label,
            )
            ends.append((float(y.iloc[-1]), label, color))
        if kind == "growth":
            ax.set_title("Growth of one unit of excess return (walk-forward, net of costs)")
            ax.set_ylabel("cumulative excess return index")
        else:
            ax.set_title("Drawdown of the excess-return index")
            ax.set_ylabel("drawdown")
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=9)
        ax.set_xlabel("")
        p = out / ("fig3_cumulative_specB.png" if kind == "growth" else "specB_drawdown.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths


def curvature(con, out: Path) -> Path | None:
    if not _has(con, "specb_hessian"):
        return None
    plt = _plt()
    h = con.execute("SELECT * FROM specb_hessian").df()
    h["date"] = pd.to_datetime(h["date"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    colors = {"Adam": BLUE, "SAM-0.05": ORANGE, "SAM-0.2": AQUA}
    for ax, fam in zip(axes, ("Direct", "Policy"), strict=True):
        for opt, color in colors.items():
            d = h[(h.strategy == f"{fam}-{opt}") & (h.seed == h.seed.min())].sort_values("date")
            if len(d):
                ax.plot(d.date, d.lambda_max, color=color, lw=1.6, marker="o", ms=3, label=opt)
        ax.set_yscale("log")
        ax.set_title(f"{fam} allocator: largest Hessian eigenvalue")
        ax.set_ylabel("lambda_max of the training loss")
        ax.legend(fontsize=9)
    fig.tight_layout()
    p = out / "fig4_curvature_specB.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def paired_sharpe(con, out: Path, n_boot: int = 2000) -> Path:
    plt = _plt()
    w = _wide(con)
    rows = []
    for fam in ("Direct", "Policy"):
        base = f"{fam}-Adam"
        cols = [c for c in w.columns if c.startswith(fam)]
        if base not in cols:
            continue
        ci = mbb_sharpe_ci(w[cols], benchmark=base, n_boot=n_boot)
        for c in cols:
            if c != base:
                rows.append(
                    (
                        f"{c} minus {base}",
                        ci.loc[c, "diff_vs_bench"],
                        ci.loc[c, "diff_ci_low"],
                        ci.loc[c, "diff_ci_high"],
                    )
                )
    fig, ax = plt.subplots(figsize=(8, 3.0 + 0.1 * len(rows)))
    for i, (_name, d, lo, hi) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color=BLUE, lw=2.5, solid_capstyle="round")
        ax.plot([d], [i], "o", color=BLUE, ms=8, mec=SURFACE, mew=2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Sharpe difference (95% moving-block-bootstrap interval)")
    ax.set_title("SAM minus Adam, paired out-of-sample Sharpe")
    ax.grid(axis="y", visible=False)
    p = out / "specB_paired_sharpe.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def var_exceptions(con, out: Path, table: str = "risk_forecasts", series: str = "SPY") -> Path:
    plt = _plt()
    f = con.execute(f'SELECT * FROM "{table}" WHERE series = ?', [series]).df()
    f["date"] = pd.to_datetime(f["date"])
    loss = -f["ret"]
    exc = loss > f["var_99"]
    model = str(f["model"].iloc[0]).upper() if "model" in f else "model"
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(f.date, loss, color=GREY, lw=0.6, label="daily loss")
    ax.plot(f.date, f["var_99"], color=BLUE, lw=1.6, label=f"99% VaR forecast (FHS, {model})")
    ax.plot(f.date, f["es_975"], color=ORANGE, lw=1.4, label="97.5% ES forecast")
    ax.plot(
        f.date[exc], loss[exc], "o", color=RED, ms=4, label=f"VaR exceptions ({int(exc.sum())})"
    )
    ax.set_ylabel("loss (log-return units)")
    ax.set_title(f"{series}: out-of-sample losses against the risk forecasts")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    p = out / "fig1_var_es.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def traffic_light(con, out: Path) -> Path | None:
    """Share of rolling 250-day windows in each Basel zone, per series and window length."""
    from flatminima.backtest.tests import traffic_light_windows

    plt = _plt()
    rows = []
    for table in ("risk_forecasts", "risk_forecasts_w500"):
        if not _has(con, table):
            continue
        fc = con.execute(f'SELECT * FROM "{table}" ORDER BY series, date').df()
        win = int(fc["window"].iloc[0]) if "window" in fc else 0
        for name, g in fc.groupby("series", sort=False):
            hits = (-g["ret"].to_numpy()) > g["var_99"].to_numpy()
            if len(hits) >= 250:
                t = traffic_light_windows(hits, 250, 0.99)
                rows.append((f"{name}, {win}-day estimation window", t))
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(9, 1.3 + 0.7 * len(rows)))
    zone_colors = (
        ("share_green", "#008300", "green"),
        ("share_yellow", "#eda100", "yellow"),
        ("share_red", RED, "red"),
    )
    for i, (_label, t) in enumerate(rows):
        left = 0.0
        for key, color, zname in zone_colors:
            w = t[key]
            ax.barh(
                i,
                w,
                left=left,
                color=color,
                edgecolor=SURFACE,
                linewidth=2,
                label=zname if i == 0 else None,
            )
            if w > 0.06:
                ax.text(
                    left + w / 2,
                    i,
                    f"{w:.0%}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                    fontweight="bold",
                )
            left += w
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("share of rolling 250-day windows (99% VaR exceptions, Basel zones)")
    ax.set_title("Basel traffic light over rolling windows")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=9)
    p = out / "fig2_traffic_light.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def run(
    db_path: str | Path, out_dir: str | Path = "docs/figures", extended: bool = False
) -> list[Path]:
    """Build the four README figures from stored tables; ``extended`` adds drawdown and the
    paired Sharpe plot. Figures whose tables are missing are skipped."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    con = db.connect(db_path)
    made: list[Path] = []
    if _has(con, "risk_forecasts"):
        made.append(var_exceptions(con, out))
        tl = traffic_light(con, out)
        if tl:
            made.append(tl)
    else:
        print("skip VaR and traffic-light figures: run `make risk` first")
    if _has(con, "specb_returns"):
        growth, drawdown = cumulative_and_drawdown(con, out)
        made.append(growth)
        if extended:
            made += [drawdown, paired_sharpe(con, out)]
        c = curvature(con, out)
        if c:
            made.append(c)
    else:
        print("skip spec B figures: run `python -m flatminima.portfolio.experiment_b` first")
    con.close()
    if not extended:  # drawdown is always drawn together with growth; drop the extra file
        (out / "specB_drawdown.png").unlink(missing_ok=True)
    for p in made:
        print("wrote", p)
    return made


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/flatminima.duckdb")
    ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args()
    run(a.db, a.out, extended=True)


if __name__ == "__main__":
    main()
