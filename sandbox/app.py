"""
sandbox/app.py
--------------
RedrobRanker — AI Hiring Co-Pilot
Hosted sandbox satisfying submission_spec.md Section 10.5:
accepts ≤100 candidate sample, runs the full ranking pipeline end-to-end,
produces a ranked CSV — completely offline, no GPU, no external APIs.

Deploy to Streamlit Community Cloud or HuggingFace Spaces as-is.
Run locally: streamlit run sandbox/app.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Bootstrap src/ on the Python path (required for all src/ imports)
# ---------------------------------------------------------------------------
SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import reasoning as rsn
from jd_intent import DEFAULT_JD as JD_TEXT
from rank import score_candidate
from semantic_match import SemanticMatcher

# ---------------------------------------------------------------------------
# Page config — must be called before any other st.* call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RedrobRanker — AI Hiring Co-Pilot",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — "case file" / hiring dossier aesthetic
#
# Token system:
#   paper     #E7E3D4   page background
#   panel     #DEDAC8   recessed panel background
#   ink       #23241F   primary text
#   ink-soft  #5B5C51   secondary text
#   navy      #26333F   structure / sidebar
#   navy-2    #35485A   sidebar hover / lighter structure
#   red       #A93226   flag / reject stamp
#   green     #3F6C4A   hire stamp
#   amber     #A9862F   rank / score accent
#   hairline  #C4BFA9   dividers, borders
#
# Type:
#   display   "Zilla Slab"      headers, hero title, stamps
#   mono      "IBM Plex Mono"   candidate IDs, ranks, scores
#   body      "IBM Plex Sans"   everything else
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Zilla+Slab:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    :root {
        --paper: #E7E3D4;
        --panel: #DEDAC8;
        --ink: #23241F;
        --ink-soft: #5B5C51;
        --navy: #26333F;
        --navy-2: #35485A;
        --red: #A93226;
        --green: #3F6C4A;
        --amber: #8C6D1F;
        --hairline: #C4BFA9;
    }

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        color: var(--ink);
    }

    .stApp { background: var(--paper); }

    h1, h2, h3, .df-title { font-family: 'Zilla Slab', serif; }

    code, .mono { font-family: 'IBM Plex Mono', monospace; }

    /* ---------------- Sidebar ---------------- */
    [data-testid="stSidebar"] {
        background: var(--navy);
        border-right: 1px solid #1b2530;
    }
    [data-testid="stSidebar"] * { color: #D9DEE3 !important; }
    [data-testid="stSidebar"] hr { border-color: #3d4e60 !important; }

    .file-tab {
        text-align: center;
        padding: 1.4rem 0 0.6rem;
    }
    .file-tab-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.14em;
        color: #8FA0AF;
        text-transform: uppercase;
    }
    .file-tab-title {
        font-family: 'Zilla Slab', serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #F2F0E6;
        margin-top: 0.25rem;
    }
    .file-tab-sub {
        font-size: 0.8rem;
        color: #92A2B0;
        margin-top: 0.1rem;
    }

    /* ---------------- Cover sheet (hero) ---------------- */
    .cover-sheet {
        background: var(--panel);
        border: 1px solid var(--hairline);
        border-radius: 2px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 1.4rem;
        position: relative;
    }
    .cover-sheet::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: repeating-linear-gradient(
            90deg, var(--navy) 0 14px, transparent 14px 20px
        );
    }
    .cover-classification {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.16em;
        color: var(--ink-soft);
        text-transform: uppercase;
        border-bottom: 1px solid var(--hairline);
        padding-bottom: 0.6rem;
        margin-bottom: 0.9rem;
    }
    .cover-title {
        font-family: 'Zilla Slab', serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: var(--ink);
        margin: 0;
        line-height: 1.15;
    }
    .cover-subtitle {
        font-size: 1rem;
        color: var(--ink-soft);
        margin-top: 0.3rem;
        max-width: 620px;
        line-height: 1.55;
    }
    .cover-fields {
        display: flex;
        gap: 2.4rem;
        margin-top: 1.2rem;
        padding-top: 1rem;
        border-top: 1px dashed var(--hairline);
        flex-wrap: wrap;
    }
    .cover-field-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.66rem;
        letter-spacing: 0.1em;
        color: var(--ink-soft);
        text-transform: uppercase;
    }
    .cover-field-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.86rem;
        color: var(--ink);
        margin-top: 0.15rem;
    }

    /* ---------------- Section headers ---------------- */
    .section-header {
        font-family: 'Zilla Slab', serif;
        font-size: 1.15rem;
        font-weight: 600;
        color: var(--ink);
        border-bottom: 2px solid var(--ink);
        padding-bottom: 5px;
        margin: 1.3rem 0 0.8rem;
        display: inline-block;
    }

    /* ---------------- Routing slip (pipeline) ---------------- */
    .routing-slip {
        border: 1px solid var(--hairline);
        background: var(--panel);
        padding: 0.4rem 1rem 0.6rem;
    }
    .routing-row {
        display: flex;
        align-items: baseline;
        gap: 0.7rem;
        padding: 0.5rem 0;
        border-bottom: 1px dashed var(--hairline);
    }
    .routing-row:last-child { border-bottom: none; }
    .routing-num {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: var(--amber);
        font-weight: 600;
        width: 1.6rem;
        flex-shrink: 0;
    }
    .routing-label {
        font-size: 0.9rem;
        color: var(--ink);
        font-weight: 500;
    }

    /* ---------------- Metric cards ---------------- */
    [data-testid="metric-container"] {
        background: var(--panel);
        border: 1px solid var(--hairline);
        border-radius: 2px;
        padding: 0.7rem 1rem;
    }
    [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace !important;
        color: var(--ink) !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--ink-soft) !important;
    }

    /* ---------------- Dataframe (ledger) ---------------- */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--hairline);
        border-radius: 0;
    }

    /* ---------------- Buttons ---------------- */
    .stButton > button[kind="primary"] {
        background: var(--navy);
        border: 1px solid var(--navy);
        border-radius: 2px;
        color: #F2F0E6;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-size: 0.82rem;
        padding: 0.6rem 1.4rem;
        width: 100%;
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--navy-2);
        border-color: var(--navy-2);
    }

    .stDownloadButton > button {
        width: 100%;
        background: var(--ink);
        color: #F2F0E6;
        border: 1px solid var(--ink);
        border-radius: 2px;
        padding: 0.6rem 1.2rem;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        font-size: 0.82rem;
    }
    .stDownloadButton > button:hover {
        background: #3a3b32;
        border-color: #3a3b32;
    }

    /* ---------------- Stamps (recommendation badges) ---------------- */
    .stamp {
        display: inline-block;
        font-family: 'Zilla Slab', serif;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.78rem;
        padding: 3px 10px;
        border: 2px solid currentColor;
        border-radius: 3px;
        transform: rotate(-2.5deg);
    }
    .stamp-strong-hire { color: var(--green); }
    .stamp-hire { color: var(--green); }
    .stamp-consider { color: var(--amber); }
    .stamp-reject { color: var(--red); }

    /* ---------------- Redaction bar (honeypot flag) ---------------- */
    .redaction {
        display: inline-block;
        background: var(--ink);
        color: var(--paper) !important;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 2px 8px;
    }
    .clean-mark {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--green);
        border-bottom: 1px solid var(--green);
        padding-bottom: 1px;
    }

    /* ---------------- Footer ---------------- */
    .footer-box {
        border-top: 1px solid var(--hairline);
        padding: 1rem 0 0.3rem;
        margin-top: 2rem;
        text-align: center;
    }
    .footer-text {
        font-family: 'IBM Plex Mono', monospace;
        color: var(--ink-soft);
        font-size: 0.72rem;
        letter-spacing: 0.03em;
        margin: 2px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# Sidebar
# ===========================================================================
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="file-tab">
                <div class="file-tab-label">Case File · Sandbox</div>
                <div class="file-tab-title">RedrobRanker</div>
                <div class="file-tab-sub">AI Hiring Co-Pilot</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("#### Sandbox Scope")
        st.markdown(
            """
            - Accepts up to **100 candidates**
            - Scored against the **Senior AI Engineer** JD
            - Runs **fully offline** — no APIs, no GPU
            - Completes inside the **5-minute** CPU budget
            """
        )
        st.divider()

        st.markdown("#### Assessment Method")
        st.markdown(
            """
            Each file passes through **semantic retrieval**,
            **structured fit scoring**, and a **behavioral trust
            multiplier**, then a **consistency check** demotes
            profiles that don't add up.
            """
        )
        st.divider()

        with st.expander("Job Description on File", expanded=False):
            st.text(JD_TEXT.strip())

        st.markdown(
            "<div style='color:#5f7180; font-family:\"IBM Plex Mono\",monospace; "
            "font-size:0.7rem; text-align:center; padding-top:1rem;'>"
            "ref. submission_spec.md §10.5</div>",
            unsafe_allow_html=True,
        )


# ===========================================================================
# Hero Section — "cover sheet"
# ===========================================================================
def render_hero() -> None:
    st.markdown(
        """
        <div class="cover-sheet">
            <div class="cover-classification">Internal · Candidate Assessment Sandbox</div>
            <div class="cover-title">RedrobRanker</div>
            <div class="cover-subtitle">
                Upload a candidate sample and receive a fully reasoned ranking —
                semantic match, structured fit, and behavioral consistency,
                assembled offline with no external calls.
            </div>
            <div class="cover-fields">
                <div>
                    <div class="cover-field-label">Requisition</div>
                    <div class="cover-field-value">Senior AI Engineer</div>
                </div>
                <div>
                    <div class="cover-field-label">Sample Limit</div>
                    <div class="cover-field-value">100 candidates</div>
                </div>
                <div>
                    <div class="cover-field-label">Compute Budget</div>
                    <div class="cover-field-value">5 min · CPU only</div>
                </div>
                <div>
                    <div class="cover-field-label">Network</div>
                    <div class="cover-field-value">offline</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# Upload / Input Section (left col) + Pipeline Diagram (right col)
# ===========================================================================
def render_input_section() -> Optional[list]:
    """Renders the upload UI and returns the candidate list, or None."""
    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        st.markdown('<div class="section-header">Intake</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload JSONL / JSON candidate sample (≤ 100 rows)",
            type=["jsonl", "json", "txt"],
            help="One JSON object per line (JSONL) or a single JSON array.",
            label_visibility="visible",
        )

        use_bundled = st.checkbox(
            "Use the bundled sample_candidates.json instead",
            value=(uploaded is None),
            help="Loads the pre-packaged demo sample from data/sample_candidates.json",
        )

        candidates: Optional[list] = None

        if use_bundled:
            sample_path = (
                Path(__file__).resolve().parent.parent / "data" / "sample_candidates.json"
            )
            if sample_path.exists():
                with open(sample_path) as f:
                    candidates = json.load(f)
                st.success(f"Loaded **{len(candidates)}** candidates from bundled sample.")
            else:
                st.warning(
                    "No bundled sample found at `data/sample_candidates.json`. "
                    "Upload a file above, or copy the organiser's sample there."
                )

        elif uploaded is not None:
            raw = uploaded.read().decode("utf-8")
            try:
                candidates = [
                    json.loads(line) for line in raw.splitlines() if line.strip()
                ]
            except json.JSONDecodeError:
                candidates = json.loads(raw)
            st.success(f"Uploaded **{len(candidates)}** candidates.")

        if candidates and len(candidates) > 100:
            st.warning(
                f"Sample has {len(candidates)} rows — using the first 100 only "
                "(sandbox scope)."
            )
            candidates = candidates[:100]

        rank_clicked = False
        if candidates:
            rank_clicked = st.button("Run Assessment", type="primary")
        else:
            st.info("Upload a sample or enable the bundled sample checkbox above.")

    with right_col:
        render_pipeline_diagram()

    return candidates, rank_clicked if candidates else (None, False)


def render_pipeline_diagram() -> None:
    st.markdown('<div class="section-header">Routing Slip</div>', unsafe_allow_html=True)
    steps = [
        "Job Description",
        "Intent Extraction",
        "Semantic Matching",
        "Structured Evaluation",
        "Behavioral Intelligence",
        "Consistency Check",
        "Final Ranking",
    ]
    rows = "".join(
        f'<div class="routing-row">'
        f'<div class="routing-num">{i+1:02d}</div>'
        f'<div class="routing-label">{label}</div>'
        f"</div>"
        for i, label in enumerate(steps)
    )
    st.markdown(f'<div class="routing-slip">{rows}</div>', unsafe_allow_html=True)


# ===========================================================================
# Recommendation label (presentation only — does not affect scoring)
# ===========================================================================
def get_recommendation(score: float) -> str:
    """Derives a hiring recommendation label from a final score.
    Thresholds are applied to the 0–1 score range after all pipeline
    multipliers; this is a display-only classification.
    """
    if score >= 0.90:
        return "Strong Hire"
    if score >= 0.80:
        return "Hire"
    if score >= 0.70:
        return "Consider"
    return "Reject"


STAMP_CLASS: dict[str, str] = {
    "Strong Hire": "stamp-strong-hire",
    "Hire": "stamp-hire",
    "Consider": "stamp-consider",
    "Reject": "stamp-reject",
}


def render_stamp(recommendation: str) -> str:
    cls = STAMP_CLASS.get(recommendation, "stamp-consider")
    return f'<span class="stamp {cls}">{recommendation}</span>'


# ===========================================================================
# Ranking Engine (pure backend — no logic changes)
# ===========================================================================
def run_ranking(candidates: list) -> tuple[pd.DataFrame, list, float]:
    """
    Executes the complete ranking pipeline and returns:
      - df            : ranked DataFrame ready for display
      - full_results  : list of raw score dicts (for detail cards)
      - elapsed       : wall-clock seconds
    """
    progress_bar = st.progress(0, text="Initialising …")
    t0 = time.time()

    # Step 1 — candidates already loaded
    progress_bar.progress(10, text="Loading candidates …")

    # Step 2 — build semantic index
    progress_bar.progress(20, text="Building semantic index …")
    matcher = SemanticMatcher(candidates)

    # Step 3 — semantic similarity
    progress_bar.progress(40, text="Computing semantic similarity …")
    semantic_scores = matcher.score_all(JD_TEXT)

    # Step 4 — structured + behavioral + honeypot per candidate
    progress_bar.progress(55, text="Computing structured scores …")
    raw_results = [
        score_candidate(c, semantic_scores[c["candidate_id"]]) for c in candidates
    ]

    progress_bar.progress(70, text="Computing behavioral scores …")
    # (behavioral scoring happens inside score_candidate — no duplicate work)

    # Step 5 — compute final score from components and sort
    # score_candidate() returns component scores; final_score is assembled here
    # using the same formula as rank.py's compute_final() (alpha=0.40, beta=0.60).
    progress_bar.progress(80, text="Computing final scores …")
    for r in raw_results:
        r["final_score"] = (
            (0.40 * r["semantic_score"] + 0.60 * r["structured_score"])
            * r["behavioral_mult"]
            * r["honeypot_mult"]
        )
    raw_results.sort(key=lambda r: (-r["final_score"], r["candidate_id"]))

    # Step 6 — reasoning
    progress_bar.progress(90, text="Generating reasoning …")
    by_id = {c["candidate_id"]: c for c in candidates}
    rows = []
    for i, r in enumerate(raw_results):
        # We temporarily set rank to i + 1, but we will recalculate it after the display-score sort
        candidate = by_id[r["candidate_id"]]
        reasoning_text = rsn.build_reasoning(
            candidate,
            r["structured_bd"],
            r["behavioral_bd"],
            r["semantic_score"],
            r["final_score"],
            i + 1,
        )
        final = round(r["final_score"], 4)
        rows.append(
            {
                "candidate_id": r["candidate_id"],
                "rank": i + 1,
                "score": final,
                "recommendation": get_recommendation(final),
                "semantic_score": round(r["semantic_score"], 4),
                "structured_score": round(r["structured_score"], 4),
                "behavioral_mult": round(r["behavioral_mult"], 4),
                "honeypot_flagged": bool(r["honeypot_flags"]),
                "reasoning": reasoning_text,
            }
        )

    rows.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    for i, row in enumerate(rows):
        row["rank"] = i + 1

    # Non-increasing guard
    for i in range(1, len(rows)):
        if rows[i]["score"] > rows[i - 1]["score"]:
            rows[i]["score"] = rows[i - 1]["score"]

    elapsed = time.time() - t0
    progress_bar.progress(100, text=f"Done — {len(rows)} candidates ranked in {elapsed:.2f}s")

    df = pd.DataFrame(rows)
    return df, raw_results, elapsed


# ===========================================================================
# KPI Metrics Row
# ===========================================================================
def render_kpi_metrics(df: pd.DataFrame, elapsed: float) -> None:
    st.markdown('<div class="section-header">Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    total = len(df)
    avg_score = df["score"].mean()
    top_score = df["score"].max()
    flagged = df["honeypot_flagged"].sum()

    with c1:
        st.metric("Candidates Ranked", total)
    with c2:
        st.metric("Average Score", f"{avg_score:.4f}")
    with c3:
        st.metric("Highest Score", f"{top_score:.4f}")
    with c4:
        st.metric(
            "Flags Raised",
            int(flagged),
            delta=f"{int(flagged)} flagged" if flagged else "All clear",
            delta_color="inverse" if flagged else "normal",
        )

    st.caption(f"Execution time: {elapsed:.2f}s  ·  Compute budget: 5 min")


# ===========================================================================
# Ranking Table
# ===========================================================================
def render_ranking_table(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-header">Ranked Results</div>', unsafe_allow_html=True)

    display_df = (
        df[["rank", "candidate_id", "score", "recommendation", "honeypot_flagged", "reasoning"]]
        .sort_values("rank")
        .rename(
            columns={
                "rank": "Rank",
                "candidate_id": "Candidate ID",
                "score": "Score",
                "recommendation": "Recommendation",
                "honeypot_flagged": "Flagged",
                "reasoning": "Reasoning",
            }
        )
        .reset_index(drop=True)
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Score": st.column_config.NumberColumn(format="%.4f", width="small"),
            "Recommendation": st.column_config.TextColumn(width="medium"),
            "Flagged": st.column_config.CheckboxColumn(width="small"),
            "Reasoning": st.column_config.TextColumn(width="large"),
        },
    )


# ===========================================================================
# Candidate Detail Expanders
# ===========================================================================
def render_candidate_details(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-header">Candidate Files</div>', unsafe_allow_html=True)
    st.caption("Top 3 candidates are expanded by default. Click any file to inspect it.")

    sorted_df = df.sort_values("rank").reset_index(drop=True)

    for _, row in sorted_df.iterrows():
        rank = int(row["rank"])
        flagged = row["honeypot_flagged"]
        rec = row.get("recommendation", "")
        flag_note = "  ·  FLAGGED" if flagged else ""
        label = f"#{rank:02d}  ·  {row['candidate_id']}  ·  {rec}  ·  {row['score']:.4f}{flag_note}"

        with st.expander(label, expanded=(rank <= 3)):
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.markdown("**Candidate ID**")
                st.code(row["candidate_id"], language=None)
                st.markdown(f"**Rank:** {rank:02d}")
                st.markdown("**Recommendation**")
                st.markdown(render_stamp(rec), unsafe_allow_html=True)

            with col_b:
                st.markdown("**Score Breakdown**")
                st.metric("Final Score", f"{row['score']:.4f}")
                if "semantic_score" in row and pd.notna(row["semantic_score"]):
                    st.metric("Semantic", f"{row['semantic_score']:.4f}")
                if "structured_score" in row and pd.notna(row["structured_score"]):
                    st.metric("Structured", f"{row['structured_score']:.4f}")
                if "behavioral_mult" in row and pd.notna(row["behavioral_mult"]):
                    st.metric("Behavioral ×", f"{row['behavioral_mult']:.4f}")

            with col_c:
                st.markdown("**Reasoning**")
                st.markdown(
                    f"<div style='font-size:0.88rem; color:var(--ink-soft); line-height:1.6;'>"
                    f"{row['reasoning']}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("**Consistency Check**")
                if flagged:
                    st.markdown(
                        '<span class="redaction">Flagged — review</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<span class="clean-mark">Clean</span>',
                        unsafe_allow_html=True,
                    )


# ===========================================================================
# Analytics Section
# ===========================================================================
def _build_histogram_df(scores: pd.Series, bins: int = 20) -> pd.DataFrame:
    """Helper: converts a score series into a labelled bucket count DataFrame."""
    min_s, max_s = scores.min(), scores.max()
    span = max_s - min_s if max_s != min_s else 1.0
    buckets = ((scores - min_s) / span * bins).astype(int).clip(0, bins - 1)
    counts = buckets.value_counts().sort_index()
    labels = {i: f"{min_s + (i / bins) * span:.3f}" for i in range(bins)}
    return pd.DataFrame(
        {"Score Range": [labels.get(i, str(i)) for i in counts.index], "Count": counts.values}
    ).set_index("Score Range")


def render_analytics(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-header">Analytics</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(
        ["Score Distribution", "Top 10 Candidates", "Recommendation Mix"]
    )

    with tab1:
        st.markdown("##### Score Distribution")
        st.bar_chart(_build_histogram_df(df["score"]), use_container_width=True, color="#26333F")
        st.caption("Each bar represents a score bucket across all ranked candidates.")

    with tab2:
        st.markdown("##### Top 10 Candidate Scores")
        top10 = (
            df.nsmallest(10, "rank")[["candidate_id", "score"]]
            .set_index("candidate_id")
        )
        st.bar_chart(top10, use_container_width=True, color="#26333F")
        st.caption("Scores of the top 10 ranked candidates.")

    with tab3:
        st.markdown("##### Recommendation Distribution")
        order = ["Strong Hire", "Hire", "Consider", "Reject"]
        rec_counts = (
            df["recommendation"]
            .value_counts()
            .reindex(order, fill_value=0)
            .reset_index()
        )
        rec_counts.columns = ["Recommendation", "Count"]
        rec_counts = rec_counts.set_index("Recommendation")
        st.bar_chart(rec_counts, use_container_width=True, color="#26333F")
        parts = [
            f"**{r}**: {int(rec_counts.loc[r, 'Count'])}"
            for r in order
            if int(rec_counts.loc[r, "Count"]) > 0
        ]
        st.caption("  ·  ".join(parts))


# ===========================================================================
# Honeypot Warning Block
# ===========================================================================
def render_honeypot_warning(df: pd.DataFrame) -> None:
    flagged = df[df["honeypot_flagged"]]
    if len(flagged) == 0:
        return
    st.warning(
        f"**{len(flagged)} candidate(s)** in this sample tripped a consistency "
        f"check and received a score penalty. Their IDs: "
        + ", ".join(f"`{cid}`" for cid in flagged["candidate_id"].tolist())
    )


# ===========================================================================
# Download Section
# ===========================================================================
def render_download(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)

    csv_bytes = (
        df[["candidate_id", "rank", "score", "reasoning"]]
        .sort_values("rank")
        .to_csv(index=False)
        .encode("utf-8")
    )

    col_dl, col_info = st.columns([1, 2])
    with col_dl:
        st.download_button(
            label="Download Ranked Results (CSV)",
            data=csv_bytes,
            file_name="sandbox_ranked_sample.csv",
            mime="text/csv",
            help="Downloads candidate_id, rank, score, reasoning — same schema as submission.csv",
        )
    with col_info:
        st.markdown(
            "<div style='color:var(--ink-soft); font-family:\"IBM Plex Mono\",monospace; "
            "font-size:0.8rem; padding-top:0.6rem;'>"
            "Columns: candidate_id · rank · score · reasoning<br>"
            "Matches the format validated by validate_submission.py"
            "</div>",
            unsafe_allow_html=True,
        )


# ===========================================================================
# Footer
# ===========================================================================
def render_footer() -> None:
    st.markdown(
        """
        <div class="footer-box">
            <div class="footer-text">POWERED BY THE REDROBRANKER OFFLINE RANKING PIPELINE</div>
            <div class="footer-text">RUNS ENTIRELY ON CPU · NO EXTERNAL APIS · NO NETWORK ACCESS REQUIRED</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# Main App Entrypoint
# ===========================================================================
def main() -> None:
    render_sidebar()
    render_hero()
    st.divider()

    # --- Input UI ---
    result = render_input_section()
    candidates, rank_clicked = result if isinstance(result, tuple) else (result, False)

    if not candidates:
        render_footer()
        return

    st.divider()

    # --- Run ranking when button is clicked ---
    if rank_clicked:
        with st.container():
            df, raw_results, elapsed = run_ranking(candidates)

        # Store in session state so results persist across reruns
        st.session_state["ranked_df"] = df
        st.session_state["elapsed"] = elapsed

    # --- Display results if available ---
    if "ranked_df" in st.session_state:
        df = st.session_state["ranked_df"]
        elapsed = st.session_state.get("elapsed", 0.0)

        st.success(
            f"Ranked **{len(df)} candidates** in **{elapsed:.2f}s** "
            f"(compute budget: 5 min)",
        )

        render_kpi_metrics(df, elapsed)
        st.divider()

        render_honeypot_warning(df)

        render_ranking_table(df)
        st.divider()

        render_analytics(df)
        st.divider()

        render_candidate_details(df)
        st.divider()

        render_download(df)

    render_footer()


main()