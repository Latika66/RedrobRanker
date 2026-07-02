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
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — subtle professional polish
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* System font stack — no external resources required */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    }

    /* Muted sidebar background */
    [data-testid="stSidebar"] { background: #0f1117; }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }

    /* Hero gradient banner */
    .hero-banner {
        background: linear-gradient(135deg, #1a1f36 0%, #232b4a 60%, #1e2d40 100%);
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #7c9cbf;
        margin-top: 0.3rem;
        margin-bottom: 1rem;
        font-weight: 400;
    }
    .hero-pill {
        display: inline-block;
        background: rgba(99, 149, 230, 0.15);
        border: 1px solid rgba(99, 149, 230, 0.35);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.78rem;
        color: #93b8f5;
        margin-right: 6px;
        margin-top: 4px;
        font-weight: 500;
    }

    /* Pipeline step card */
    .pipeline-step {
        background: #1a1f2e;
        border-left: 3px solid #3d64c8;
        border-radius: 6px;
        padding: 6px 14px;
        margin: 4px 0;
        font-size: 0.88rem;
        color: #c0d0e8;
        font-weight: 500;
    }
    .pipeline-arrow {
        text-align: center;
        font-size: 1rem;
        color: #3d64c8;
        line-height: 1;
        margin: 1px 0;
    }

    /* Section headers */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #e8eaf0;
        border-bottom: 2px solid #2d3561;
        padding-bottom: 6px;
        margin: 1.2rem 0 0.8rem;
    }

    /* Metric card overrides */
    [data-testid="metric-container"] {
        background: #151929;
        border: 1px solid #252d4a;
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

    /* Download button */
    .stDownloadButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2752c9, #3d64c8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #3060e0, #4a73d4);
        box-shadow: 0 4px 14px rgba(55, 100, 200, 0.4);
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2752c9, #3d64c8);
        border: none;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        padding: 0.55rem 1.4rem;
        font-size: 0.95rem;
        width: 100%;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #3060e0, #4a73d4);
        box-shadow: 0 4px 14px rgba(55, 100, 200, 0.4);
    }

    /* Footer */
    .footer-box {
        background: #0e1220;
        border: 1px solid #1e2540;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        text-align: center;
        margin-top: 2rem;
    }
    .footer-text {
        color: #586080;
        font-size: 0.8rem;
        margin: 2px 0;
    }

    /* Warning/flag badges */
    .flag-badge {
        background: rgba(220, 80, 60, 0.15);
        border: 1px solid rgba(220, 80, 60, 0.4);
        color: #f08070;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .ok-badge {
        background: rgba(50, 180, 100, 0.12);
        border: 1px solid rgba(50, 180, 100, 0.3);
        color: #60c080;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.78rem;
        font-weight: 600;
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
            <div style="text-align:center; padding: 1rem 0 0.5rem;">
                <div style="font-size:3rem; line-height:1;">🧠</div>
                <div style="font-size:1.3rem; font-weight:700; color:#fff; margin-top:0.4rem;">
                    RedrobRanker
                </div>
                <div style="font-size:0.82rem; color:#6a87b5; margin-top:0.2rem;">
                    AI Hiring Co-Pilot
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("#### 📋 Sandbox Info")
        st.markdown(
            """
            - Accepts **≤ 100 candidates**  
            - Scores against the **Senior AI Engineer JD**  
            - Runs **fully offline** — no APIs, no GPU  
            - Completes within the **5-minute CPU budget**
            """
        )
        st.divider()

        st.markdown("#### Ranking Pipeline")
        st.markdown(
            """
            The pipeline combines **semantic retrieval** with
            **structured fit scoring** and a **behavioral trust
            multiplier**, with a final **honeypot consistency check**
            to demote implausible profiles.
            """
        )
        st.divider()

        with st.expander("📄Job Description", expanded=False):
            st.text(JD_TEXT.strip())

        st.markdown(
            "<div style='color:#3a4565; font-size:0.75rem; text-align:center; "
            "padding-top:1rem;'>submission_spec.md §10.5</div>",
            unsafe_allow_html=True,
        )


# ===========================================================================
# Hero Section
# ===========================================================================
def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title"> RedrobRanker</div>
            <div class="hero-subtitle">AI Hiring Co-Pilot — Offline Candidate Ranking Engine</div>
            <div style="font-size:0.92rem; color:#8daacf; margin-bottom:1rem; max-width:680px; line-height:1.6;">
                Upload candidate profiles and receive explainable AI-powered rankings in seconds
                using an entirely offline ranking engine — no APIs, no GPU, no internet required.
            </div>
            <span class="hero-pill"> Semantic Matching</span>
            <span class="hero-pill"> Structured Evaluation</span>
            <span class="hero-pill"> Behavioral Intelligence</span>
            <span class="hero-pill"> Explainable AI</span>
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
        st.markdown('<div class="section-header">📂 Load Candidates</div>', unsafe_allow_html=True)

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
                st.success(
                    f"Loaded **{len(candidates)}** candidates from bundled sample.",
                    icon="📦",
                )
            else:
                st.warning(
                    "⚠️ No bundled sample found at `data/sample_candidates.json`. "
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
            st.success(f"Uploaded **{len(candidates)}** candidates.", icon="📄")

        if candidates and len(candidates) > 100:
            st.warning(
                f"⚠️ Sample has {len(candidates)} rows — using the first 100 only "
                "(sandbox scope)."
            )
            candidates = candidates[:100]

        rank_clicked = False
        if candidates:
            rank_clicked = st.button("🚀 Rank Candidates", type="primary")
        else:
            st.info("Upload a sample or enable the bundled sample checkbox above.")

    with right_col:
        render_pipeline_diagram()

    return candidates, rank_clicked if candidates else (None, False)


def render_pipeline_diagram() -> None:
    st.markdown('<div class="section-header">🔄 Ranking Pipeline</div>', unsafe_allow_html=True)
    steps = [
        ("Job Description"),
        ("Intent Extraction"),
        ("Semantic Matching"),
        ("Structured Evaluation"),
        ("Behavioral Intelligence"),
        ("Explainable AI"),
        ("Final Ranking"),
    ]
    for i, (icon, label) in enumerate(steps):
        st.markdown(
            f'<div class="pipeline-step">{icon} &nbsp; {label}</div>',
            unsafe_allow_html=True,
        )
        if i < len(steps) - 1:
            st.markdown(
                '<div class="pipeline-arrow">↓</div>', unsafe_allow_html=True
            )


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


RECOMMENDATION_EMOJI: dict[str, str] = {
    "Strong Hire": "🟢",
    "Hire": "🔵",
    "Consider": "🟡",
    "Reject": "🔴",
}


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
        if rows[i]["score"] > rows[i-1]["score"]:
            rows[i]["score"] = rows[i-1]["score"]

    elapsed = time.time() - t0
    progress_bar.progress(100, text=f"✅ Done — {len(rows)} candidates ranked in {elapsed:.2f}s")

    df = pd.DataFrame(rows)
    return df, raw_results, elapsed


# ===========================================================================
# KPI Metrics Row
# ===========================================================================
def render_kpi_metrics(df: pd.DataFrame, elapsed: float) -> None:
    st.markdown('<div class="section-header">📈 Summary Metrics</div>', unsafe_allow_html=True)
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
            "⚠️ Honeypot Flags",
            int(flagged),
            delta=f"{int(flagged)} flagged" if flagged else "All clear",
            delta_color="inverse" if flagged else "normal",
        )

    st.caption(f"⏱ Execution time: {elapsed:.2f}s  ·  Compute budget: 5 min")


# ===========================================================================
# Ranking Table
# ===========================================================================
def render_ranking_table(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-header">🏆 Ranked Results</div>', unsafe_allow_html=True)

    display_df = (
        df[["rank", "candidate_id", "score", "recommendation", "honeypot_flagged", "reasoning"]]
        .sort_values("rank")
        .rename(
            columns={
                "rank": "Rank",
                "candidate_id": "Candidate ID",
                "score": "Score",
                "recommendation": "Recommendation",
                "honeypot_flagged": "⚠️ Honeypot",
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
            "⚠️ Honeypot": st.column_config.CheckboxColumn(width="small"),
            "Reasoning": st.column_config.TextColumn(width="large"),
        },
    )


# ===========================================================================
# Candidate Detail Expanders
# ===========================================================================
def render_candidate_details(df: pd.DataFrame) -> None:
    st.markdown(
        '<div class="section-header"> Candidate Detail Cards</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Top 3 candidates are expanded by default. Click any card to inspect scores and reasoning."
    )

    sorted_df = df.sort_values("rank").reset_index(drop=True)

    for _, row in sorted_df.iterrows():
        rank = int(row["rank"])
        flagged = row["honeypot_flagged"]
        rec = row.get("recommendation", "")
        rec_emoji = RECOMMENDATION_EMOJI.get(rec, "")
        flag_icon = "  ⚠️" if flagged else ""
        label = (
            f"#{rank}  ·  {row['candidate_id']}  "
            f"·  {rec_emoji} {rec}  ·  Score {row['score']:.4f}{flag_icon}"
        )
        # Automatically expand the top 3 candidates
        with st.expander(label, expanded=(rank <= 3)):
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.markdown("**Candidate ID**")
                st.code(row["candidate_id"], language=None)
                st.markdown(f"**Rank:** {rank}")
                st.markdown(f"**Recommendation:** {rec_emoji} {rec}")

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
                    f"<div style='font-size:0.88rem; color:#b0bdd0; line-height:1.6;'>"
                    f"{row['reasoning']}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("**Honeypot Status**")
                if flagged:
                    st.markdown(
                        '<span class="flag-badge">⚠️ Flagged</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<span class="ok-badge">✅ Clean</span>',
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
    st.markdown('<div class="section-header">📉 Analytics</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(
        ["Score Distribution", "Top 10 Candidates", "Recommendation Mix"]
    )

    with tab1:
        st.markdown("##### Score Distribution Histogram")
        st.bar_chart(_build_histogram_df(df["score"]), use_container_width=True, color="#3d64c8")
        st.caption("Each bar represents a score bucket across all ranked candidates.")

    with tab2:
        st.markdown("##### Top 10 Candidate Scores")
        top10 = (
            df.nsmallest(10, "rank")[["candidate_id", "score"]]
            .set_index("candidate_id")
        )
        st.bar_chart(top10, use_container_width=True, color="#3d64c8")
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
        st.bar_chart(rec_counts, use_container_width=True, color="#3d64c8")
        # Summary caption using actual counts
        parts = [
            f"{RECOMMENDATION_EMOJI.get(r, '')} **{r}**: {int(rec_counts.loc[r, 'Count'])}"
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
        f"⚠️ **{len(flagged)} candidate(s)** in this sample tripped a honeypot "
        f"consistency check and received a strong score penalty. Their IDs: "
        + ", ".join(f"`{cid}`" for cid in flagged["candidate_id"].tolist())
    )


# ===========================================================================
# Download Section
# ===========================================================================
def render_download(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-header">📥 Download Results</div>', unsafe_allow_html=True)

    csv_bytes = (
        df[["candidate_id", "rank", "score", "reasoning"]]
        .sort_values("rank")
        .to_csv(index=False)
        .encode("utf-8")
    )

    col_dl, col_info = st.columns([1, 2])
    with col_dl:
        st.download_button(
            label="📥 Download Ranked Results (CSV)",
            data=csv_bytes,
            file_name="sandbox_ranked_sample.csv",
            mime="text/csv",
            help="Downloads candidate_id, rank, score, reasoning — same schema as submission.csv",
        )
    with col_info:
        st.markdown(
            "<div style='color:#6a87b5; font-size:0.88rem; padding-top:0.6rem;'>"
            "Columns: <code>candidate_id · rank · score · reasoning</code><br>"
            "Matches the format validated by <code>validate_submission.py</code>"
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
            <div class="footer-text"> Powered by the <strong>RedrobRanker</strong> offline ranking pipeline.</div>
            <div class="footer-text"> Runs entirely on CPU — no external APIs, no hosted LLMs, no network access required.</div>
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
            f"✅ Ranked **{len(df)} candidates** in **{elapsed:.2f}s** "
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
