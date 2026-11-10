"""Tier P scoring and family S6 (protocol sections 3.4 and 5.5) on fake outcomes, before the real ones exist.

Uses the sealed issue 2026-10-30 read-only: its contexts, tasks and forecasts are linked into a temporary tree, the
outcomes are synthetic (seasonal continuation of each context plus noise), and every output is written to the
temporary tree, so nothing sealed is touched.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tfmstudy import analysis as A
from tfmstudy.data import catalog
from tfmstudy.data import prospective as P

ROOT = Path(__file__).resolve().parents[1]
ISSUE = "2026-10-30"
PERIOD = P.period_name(ISSUE)
DATA = ROOT / "data" / "processed" / PERIOD
RUN = ROOT / "results" / "forecast" / PERIOD / "P"
pytestmark = pytest.mark.skipif(not (DATA / "tasks.parquet").exists() or not RUN.exists(),
                                reason="sealed Tier P issue 2026-10-30 not available locally")


def _fake_outcomes(tasks: pd.DataFrame, series: pd.DataFrame, rng) -> pd.DataFrame:
    """Seasonal continuation of each context over the forecast steps, with 10% noise."""
    by_id = {sid: g.set_index("timestamp")["value"].dropna() for sid, g in series.groupby("series_id")}
    rows, done = [], set()
    for t in tasks.itertuples():
        freq = {"h": "h", "D": "D"}.get(catalog.DOMAINS[t.domain]["freq"])
        for sid in json.loads(t.targets):
            if sid in done:
                continue
            done.add(sid)
            ctx = by_id[sid]
            last = ctx.index.max()
            stamps = (pd.date_range(last, periods=t.horizon + 1, freq=freq)[1:] if freq
                      else pd.bdate_range(last + pd.Timedelta(days=1), periods=t.horizon, tz="UTC"))
            season = max(int(t.season), 1)
            pattern = np.resize(ctx.to_numpy()[-season:], len(stamps))
            noise = rng.normal(0, 0.1 * (np.nanstd(ctx.to_numpy()[-4 * season:]) + 1e-9), len(stamps))
            rows += [{"series_id": sid, "timestamp": s, "value": v} for s, v in zip(stamps, pattern + noise)]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tierp")
    processed, results = tmp / "processed", tmp / "results"
    (processed / PERIOD).mkdir(parents=True)
    for f in ("series.parquet", "tasks.parquet", "ISSUE.json"):
        shutil.copy(DATA / f, processed / PERIOD / f)
    run = results / PERIOD / "P"
    run.mkdir(parents=True)
    for model_dir in RUN.iterdir():
        if model_dir.is_dir():
            (run / model_dir.name).symlink_to(model_dir, target_is_directory=True)
    tasks = pd.read_parquet(DATA / "tasks.parquet")
    tasks = tasks[~tasks.excluded.astype(bool)] if "excluded" in tasks else tasks
    out = _fake_outcomes(tasks, pd.read_parquet(DATA / "series.parquet"), np.random.default_rng(0))
    # one task loses its outcomes: it must be excluded at scoring, not scored on nothing
    gone = json.loads(tasks.iloc[0].targets)
    out = out[~out.series_id.isin(gone)]
    # a business-day series with only its first two window days published: also excluded (fewer than 80% observed)
    late = tasks[tasks.domain.isin(["fx", "rates"]) & tasks["mode"].eq("uni")].iloc[0]
    sid = json.loads(late.targets)[0]
    keep = out[out.series_id.eq(sid) & (out.timestamp >= late.score_from)].sort_values("timestamp").timestamp.iloc[:2]
    out = out[~out.series_id.eq(sid) | (out.timestamp < late.score_from) | out.timestamp.isin(keep)]
    out.to_parquet(processed / PERIOD / "outcomes.parquet", index=False)
    old = P.PROCESSED
    P.PROCESSED = processed
    try:
        df = P.score(ISSUE, results)
    finally:
        P.PROCESSED = old
    return df, tasks, run, (tasks.iloc[0].task_id, late.task_id)


def test_scoring_covers_every_task_and_model(scored):
    df, tasks, run, dropped_id = scored
    excluded = pd.read_csv(run / "excluded_at_scoring.csv")
    assert sorted(excluded.task_id) == sorted(dropped_id)
    assert set(df.task_id) == set(tasks.task_id) - set(dropped_id)
    assert set(df.model) == {p.name for p in RUN.iterdir() if p.is_dir()} | {"stat-ens"}
    s6_models = {"tfm3-uni", "stat-ens", "chronos2-uni", "tfm25", "naive"}  # S6 needs these on every task
    assert df.groupby("task_id").model.apply(lambda m: s6_models <= set(m)).all()
    assert np.isfinite(df.mase).all() and (df.mase > 0).all()
    assert np.isfinite(df.sql).all() and (df.sql > 0).all()
    assert df.coverage80.between(0, 1).all()
    # only window steps are scored: hourly tasks score 7 days = 168 steps
    hourly = df[df.domain.isin(["load", "price"])]
    assert (hourly.n_scored == 168).all()


def test_family_s6_on_two_pooled_issues(scored):
    df, *_ = scored
    second = df.assign(issue="2026-11-02", task_id=df.task_id.str.replace("P20261030", "P20261102"),
                       origin=df.origin + pd.Timedelta(days=3), mase=df.mase * np.exp(np.random.default_rng(1).normal(0, 0.05, len(df))),
                       sql=df.sql * 1.01)
    res = A.family_s6(pd.concat([df, second], ignore_index=True), reps=200)
    assert len(res) == 16
    assert {r["test"].split(":")[1] for r in res} == {"structured", "wiki"}
    assert all(0 < r["ratio"] < 10 and r["ci95"][0] <= r["ratio"] <= r["ci95"][1] for r in res)
    assert all("p_holm" in r and "resolved" in r for r in res)


def test_s6_entry_point_and_paper_table(scored, tmp_path, monkeypatch):
    import subprocess
    import sys

    from tfmstudy import paper

    df, _, run, _ = scored
    res = run.parent.parent  # <tmp>/results
    second = res / "prospective-2026-11-02" / "P"
    second.mkdir(parents=True)
    df.assign(issue="2026-11-02", task_id=df.task_id.str.replace("P20261030", "P20261102"),
              origin=df.origin + pd.Timedelta(days=3)).to_parquet(second / "metrics.parquet", index=False)
    subprocess.run([sys.executable, "-m", "tfmstudy.analysis", "--s6-only", "--reps", "200", "--results", str(res),
                    "--out", str(tmp_path)], check=True, cwd=ROOT, capture_output=True)
    s6 = json.loads((tmp_path / "prospective" / "confirmatory_s6.json").read_text())
    assert s6["issues"] == ["2026-10-30", "2026-11-02"] and len(s6["family_S6"]) == 16
    monkeypatch.setattr(paper, "GEN", tmp_path)
    paper.table_family_s6(s6)
    tex = (tmp_path / "table_family_s6.tex").read_text()
    assert tex.count("\\\\") == 16 + 2 + 1  # 16 tests, 2 group headers, 1 header row
    paper.table_family_s6(None)
    assert "not yet scored" in (tmp_path / "table_family_s6.tex").read_text()
