"""
reasoning.py  —  Recruiter-Language Justification Generator
============================================================
Generates a 1-2 sentence recruiter brief for each ranked candidate.
Every claim is grounded in fields the scoring pipeline actually read —
no hallucination, no templates that repeat verbatim across rows.

Language style: what a senior technical recruiter would write in a
candidate brief, not what a scoring system would log. The goal is a
sentence that a hiring manager could read and immediately understand
why this person is (or isn't) worth a first call.
"""

from __future__ import annotations
from typing import Dict, List


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _join(items: List[str], limit: int = 3) -> str:
    items = items[:limit]
    if not items:     return ""
    if len(items) == 1: return items[0]
    if len(items) == 2: return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _domain_label(key: str) -> str:
    return {
        "embeddings_retrieval":  "embeddings & retrieval",
        "vector_search_infra":   "vector/hybrid search infra",
        "ranking_recsys":        "ranking/recsys",
        "eval_frameworks":       "evaluation frameworks (NDCG/MRR/A-B)",
        "llm_systems":           "LLM systems",
        "ml_production":         "production ML engineering",
        "python_engineering":    "Python",
    }.get(key, key.replace("_", " "))


# ---------------------------------------------------------------------------
# Build the justification
# ---------------------------------------------------------------------------
def build_reasoning(
    candidate: dict,
    structured_bd: dict,
    behavioral_bd: dict,
    semantic_score: float,
    final_score: float,
    rank: int,
) -> str:
    profile = candidate["profile"]
    sig     = candidate.get("redrob_signals", {})

    title   = profile.get("current_title", "Unknown Title")
    yoe     = profile.get("years_of_experience", 0)
    loc     = profile.get("location", "unknown location")

    # Demonstrated domains (career text, not just skill tags)
    demonstrated = [
        _domain_label(k)
        for k, v in structured_bd.get("skill_hits", {}).items()
        if v == "demonstrated"
    ]
    claimed_only = [
        _domain_label(k)
        for k, v in structured_bd.get("skill_hits", {}).items()
        if v == "claimed_only"
    ]
    dq_reasons   = structured_bd.get("disqualifier_reasons", [])
    applied_ai   = structured_bd.get("experience_detail", {}).get("applied_ai_yoe", 0)
    applied_ai   = min(applied_ai, yoe)  # never display more than total YOE

    resp_rate    = sig.get("recruiter_response_rate")
    notice_days  = sig.get("notice_period_days")
    open_work    = sig.get("open_to_work_flag", False)

    # ---- Sentence 1: who they are + core fit signal ----
    if demonstrated and len(demonstrated) >= 3:
        # Rotate phrasing shape on yoe decimal to avoid template repetition
        variants = [
            f"Experienced {title} ({yoe:.1f} yrs, {loc}) with hands-on production work across {_join(demonstrated)} — covers most of what this role needs.",
            f"{title} based in {loc} ({yoe:.1f} yrs total); career history shows real {_join(demonstrated)} work, not just listed keywords.",
            f"Strong {title} profile out of {loc} — {_join(demonstrated)} all appear in actual project descriptions, not just the skills section.",
            f"{yoe:.1f}-year {title} ({loc}) whose project history backs up {_join(demonstrated)} with demonstrated delivery.",
        ]
        s1 = variants[int(round(yoe * 10)) % len(variants)]

    elif demonstrated and len(demonstrated) in (1, 2):
        s1 = f"{title} in {loc} ({yoe:.1f} yrs) with solid {_join(demonstrated)} background in their work history."

    elif claimed_only:
        s1 = f"{title} in {loc} ({yoe:.1f} yrs) — lists {_join(claimed_only)} as skills but career descriptions don't clearly back that up with hands-on delivery."

    else:
        s1 = f"{title} in {loc} ({yoe:.1f} yrs) — limited overlap with the role's core embeddings/retrieval/ranking focus."

    # ---- Sentence 2: nuance, concerns, or availability ----
    clauses: List[str] = []

    if applied_ai >= 3 and demonstrated:
        clauses.append(f"~{applied_ai:.1f} of those years appear to be in applied AI/ML roles")

    if dq_reasons:
        clauses.append(f"key concern: {dq_reasons[0]}")

    if resp_rate is not None:
        if resp_rate >= 0.55:
            clauses.append(f"responsive ({resp_rate:.0%} recruiter response rate)")
        elif resp_rate <= 0.15:
            clauses.append(f"low recruiter responsiveness ({resp_rate:.0%}) — may be hard to reach")

    if open_work:
        clauses.append("actively open to new roles")

    if notice_days is not None and notice_days > 60:
        clauses.append(f"long notice period ({int(notice_days)} days)")

    if rank > 75 and not dq_reasons:
        clauses.append("a plausible but not top-tier fit given stronger profiles higher in the list")

    s2 = "; ".join(clauses)
    if s2:
        s2 = s2[0].upper() + s2[1:]
        if not s2.endswith("."):
            s2 += "."
        result = f"{s1} {s2}"
    else:
        result = s1

    # Hard length cap — keep it readable in a CSV cell
    if len(result) > 340:
        result = result[:337].rsplit(".", 1)[0] + "."

    return result
