"""Streamlit eval dashboard — run: streamlit run evals/dashboard.py"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

RUNS_DIR = Path(__file__).parent / "runs"

st.set_page_config(page_title="ChordMap Evals", layout="wide")
st.title("ChordMap Eval Dashboard")

run_files = sorted(RUNS_DIR.glob("*.json"), reverse=True)
if not run_files:
    st.info("No eval runs yet. Run `python -m evals.run` to generate results.")
    st.stop()

runs = []
for f in run_files:
    with open(f) as fh:
        runs.append(json.load(fh))

st.subheader("Runs")
st.dataframe([
    {
        "run_id": r["run_id"],
        "suite": r["suite"],
        "system_version": r["system_version"],
        "timestamp": r["timestamp"],
        **r.get("aggregate", {}),
    }
    for r in runs
])

selected = st.selectbox("Inspect run", [r["run_id"] for r in runs])
run = next(r for r in runs if r["run_id"] == selected)

st.subheader("Aggregate scores")
st.json(run.get("aggregate", {}))

st.subheader("Per-example results")
st.dataframe(run.get("results", []))
