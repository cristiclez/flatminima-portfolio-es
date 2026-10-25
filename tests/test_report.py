"""Figures and CV evidence on a small synthetic database."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("matplotlib")

from flatminima.backtest import run as bt  # noqa: E402
from flatminima.data import pipeline  # noqa: E402
from flatminima.portfolio import experiment_b as eb  # noqa: E402
from flatminima.report import evidence, figures  # noqa: E402
from flatminima.risk import run as risk  # noqa: E402


@pytest.fixture(scope="module")
def database(tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "s.duckdb"
    pipeline.run("synthetic", path, n_synth=1000)
    risk.run(path, series=("SPY",), window=500, refit_every=100)
    bt.run(path, report=None, n_sims=200)
    eb.run(
        path,
        seeds=(0,),
        grad_budget=10,
        analyse_every=2,
        n_boot=30,
        rf_constant_pct=2.0,
        report=None,
    )
    return path


def test_figures_are_written_from_tables(database, tmp_path):
    made = figures.run(database, tmp_path)
    assert {p.name for p in made} == {
        "fig1_var_es.png",
        "fig2_traffic_light.png",
        "fig3_cumulative_specB.png",
        "fig4_curvature_specB.png",
    }
    assert all(p.stat().st_size > 5_000 for p in made)


def test_figures_skip_missing_tables(tmp_path):
    path = tmp_path / "empty.duckdb"
    pipeline.run("synthetic", path, n_synth=300)
    assert figures.run(path, tmp_path / "f") == []


def test_evidence_is_recomputed_and_states_limits(database):
    text = evidence.build(database)
    assert "## BT-500" in text and "not multiple-testing corrected" in text
    assert "## SPECB-Direct-SAM-0.05" in text and "post hoc" in text
    assert "Do not say" in text and "FRTB compliance" in text
    assert "commit" in text
