"""
sandbox/app.py
---------------
Minimal Streamlit app satisfying submission_spec.md Section 10.5: a hosted
sandbox that accepts a small candidate sample (<=100 candidates), runs the
ranking system end-to-end, and produces a ranked CSV, within the compute
budget (<=5 min on CPU).

This does NOT need to handle the full 100K pool -- that's verified separately
at Stage 3 against the GitHub repo. This is the "does it run at all" check.

Deploy as-is to Streamlit Community Cloud or HuggingFace Spaces: point the
app at this file, no secrets/config needed (everything runs offline).

Run locally with: streamlit run sandbox/app.py
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import behavioral_score as beh
import honeypot as hp
import reasoning as rsn
import structured_score as struct
from rank import JD_TEXT, score_candidate
from semantic_match import SemanticMatcher

st.set_page_config(page_title="Redrob Ranker Sandbox", layout="wide")

st.title("Redrob Ranker — Sandbox")
st.caption(
    "Hosted reproducibility check per submission_spec.md Section 10.5. "
    "Upload a small candidate sample (JSONL, ≤100 rows) and rank it against "
    "the Senior AI Engineer JD, fully offline."
)

with st.expander("Job description used for ranking", expanded=False):
    st.text(JD_TEXT.strip())

uploaded = st.file_uploader(
    "Upload a candidates.jsonl sample (one JSON object per line, ≤100 rows)",
    type=["jsonl", "json", "txt"],
)

use_bundled_sample = st.checkbox(
    "Use the bundled sample_candidates.json instead of uploading", value=not uploaded
)

candidates = None

if use_bundled_sample:
    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample_candidates.json"
    if sample_path.exists():
        with open(sample_path) as f:
            candidates = json.load(f)
        st.info(f"Loaded {len(candidates)} candidates from bundled sample_candidates.json")
    else:
        st.warning(
            "No bundled sample found at data/sample_candidates.json. "
            "Upload a file instead, or copy the organizer's sample_candidates.json there."
        )
elif uploaded is not None:
    raw = uploaded.read().decode("utf-8")
    try:
        # Try JSONL first (one object per line)
        candidates = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except json.JSONDecodeError:
        candidates = json.loads(raw)  # fall back to a single JSON array

if candidates:
    if len(candidates) > 100:
        st.warning(f"Sample has {len(candidates)} rows; using the first 100 only, per sandbox scope.")
        candidates = candidates[:100]

    if st.button("Rank candidates", type="primary"):
        t0 = time.time()
        with st.spinner("Building BM25 index and scoring..."):
            matcher = SemanticMatcher(candidates)
            semantic_scores = matcher.score_all(JD_TEXT)

            results = []
            for c in candidates:
                results.append(score_candidate(c, semantic_scores[c["candidate_id"]]))

            results.sort(key=lambda r: (-r["final_score"], r["candidate_id"]))

            rows = []
            by_id = {c["candidate_id"]: c for c in candidates}
            for i, r in enumerate(results):
                rank = i + 1
                candidate = by_id[r["candidate_id"]]
                reasoning_text = rsn.build_reasoning(
                    candidate,
                    r["structured_breakdown"],
                    r["behavioral_breakdown"],
                    r["semantic_score"],
                    r["final_score"],
                    rank,
                )
                rows.append({
                    "candidate_id": r["candidate_id"],
                    "rank": rank,
                    "score": round(r["final_score"], 4),
                    "reasoning": reasoning_text,
                    "honeypot_flagged": bool(r["honeypot_flags"]),
                })

        elapsed = time.time() - t0
        st.success(f"Ranked {len(rows)} candidates in {elapsed:.2f}s (compute budget: 5 min)")

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv_bytes = df.drop(columns=["honeypot_flagged"]).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download ranked CSV",
            data=csv_bytes,
            file_name="sandbox_ranked_sample.csv",
            mime="text/csv",
        )

        flagged = df[df["honeypot_flagged"]]
        if len(flagged):
            st.warning(f"{len(flagged)} candidate(s) in this sample tripped a honeypot consistency check.")
else:
    st.info("Upload a candidate sample or check the bundled-sample box above to get started.")

st.divider()
st.caption(
    "This sandbox runs the same scoring code as src/rank.py — no hosted LLM calls, "
    "no GPU, no network access required at ranking time."
)
