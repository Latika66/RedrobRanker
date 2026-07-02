"""
structured_score.py  —  Structured Fit Scoring
===============================================
Scores each candidate against the parsed JD intent across five dimensions:

  1. Skill Coverage     — which JD skill groups are demonstrated vs claimed
  2. Experience Quality — YOE fit + applied-AI tenure estimate
  3. Career Growth      — progression, tenure stability, ownership signals
  4. Domain Match       — alignment of career roles with the JD domain
  5. Impact Signals     — outcome language in career descriptions

Each dimension is 0-1 and combined with learned-style weights.
Disqualifier penalties are applied multiplicatively on top.
"""

from __future__ import annotations
import re
from typing import Dict, List, Tuple

from jd_intent import JDIntent, INTENT, SkillGroup


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def _career_text(candidate: dict) -> str:
    parts = [
        candidate["profile"].get("headline", ""),
        candidate["profile"].get("summary", ""),
        candidate["profile"].get("current_title", ""),
    ]
    for j in candidate.get("career_history", []):
        parts.append(j.get("title", ""))
        parts.append(j.get("description", ""))
    return " ".join(parts).lower()


def _skills_text(candidate: dict) -> str:
    return " ".join(s["name"].lower() for s in candidate.get("skills", []))


def _all_text(candidate: dict) -> str:
    return _career_text(candidate) + " " + _skills_text(candidate)


# ---------------------------------------------------------------------------
# 1. Skill Coverage
# ---------------------------------------------------------------------------
def skill_coverage(candidate: dict, intent: JDIntent) -> Tuple[float, dict]:
    career = _career_text(candidate)
    skills = _skills_text(candidate)
    hits: dict = {}
    total_weight = sum(g.weight for g in intent.skill_groups)
    earned = 0.0

    for group in intent.skill_groups:
        in_career = any(kw in career for kw in group.keywords)
        in_skills  = any(kw in skills  for kw in group.keywords)

        if in_career:
            # Demonstrated in real work — full credit
            earned += group.weight
            hits[group.name] = "demonstrated"
        elif in_skills:
            # Listed as skill but not backed by career text — half credit
            earned += group.weight * 0.45
            hits[group.name] = "claimed_only"

    score = earned / total_weight if total_weight else 0.0
    return score, hits


# ---------------------------------------------------------------------------
# 2. Experience Quality
# ---------------------------------------------------------------------------
def experience_quality(candidate: dict, intent: JDIntent) -> Tuple[float, dict]:
    yoe     = candidate["profile"].get("years_of_experience", 0.0)
    ilo, ihi = intent.yoe_ideal_min, intent.yoe_ideal_max
    lo,  hi  = intent.yoe_min, intent.yoe_max

    # Band fit — smooth curve, never hard cliff
    if ilo <= yoe <= ihi:
        band_score = 1.0
    elif lo <= yoe <= hi:
        band_score = 0.85
    else:
        dist = (lo - yoe) if yoe < lo else (yoe - hi)
        band_score = max(0.35, 1.0 - dist / 3.0)

    # Applied-AI tenure estimate
    ai_kw = sum([g.keywords for g in intent.skill_groups if g.name in
                 ("embeddings_retrieval","vector_search_infra","ranking_recsys","llm_systems")], [])
    ai_months = 0
    for job in candidate.get("career_history", []):
        text = (job.get("title","") + " " + job.get("description","")).lower()
        if any(kw in text for kw in ai_kw) or any(k in text for k in ["machine learning","nlp","data scien","ml "]):
            ai_months += job.get("duration_months", 0)
    applied_ai_yoe = min(ai_months / 12.0, yoe)  # cap at total yoe

    ideal_applied = intent.yoe_ideal_min - 2  # ~4 years applied AI
    applied_score = min(1.0, applied_ai_yoe / max(ideal_applied, 1.0))

    score = 0.6 * band_score + 0.4 * applied_score
    return score, {"yoe": yoe, "band_score": round(band_score,3),
                   "applied_ai_yoe": round(applied_ai_yoe,1),
                   "applied_score": round(applied_score,3)}


# ---------------------------------------------------------------------------
# 3. Career Growth
# ---------------------------------------------------------------------------
_SENIORITY = {
    "intern": 0, "trainee": 0, "junior": 1, "associate": 1,
    "engineer": 2, "developer": 2, "analyst": 2,
    "senior": 3, "lead": 3, "specialist": 3,
    "staff": 4, "principal": 4, "architect": 4,
    "manager": 4, "director": 5, "vp": 6, "head": 5,
}

def _seniority(title: str) -> int:
    t = title.lower()
    best = 2  # default: mid-level
    for kw, lvl in _SENIORITY.items():
        if kw in t:
            best = max(best, lvl)
    return best

def career_growth(candidate: dict) -> Tuple[float, dict]:
    history = candidate.get("career_history", [])
    if not history:
        return 0.3, {"reason": "no career history"}

    # Tenure stability — penalise lots of short stints
    durations = [j.get("duration_months", 0) for j in history]
    short_stints = sum(1 for d in durations if d < 18)
    stability = 1.0 - min(1.0, short_stints / max(len(history), 1) * 0.7)

    # Seniority progression
    titles = [j.get("title","") for j in history]
    if len(titles) >= 2:
        lvl_start = _seniority(titles[-1])   # oldest role
        lvl_end   = _seniority(titles[0])    # most recent
        progression = min(1.0, max(0.0, (lvl_end - lvl_start + 1) / 4.0))
    else:
        progression = 0.5

    # Current role seniority
    current_seniority = _seniority(candidate["profile"].get("current_title","")) / 6.0

    score = 0.35 * stability + 0.35 * progression + 0.30 * current_seniority
    return score, {"stability": round(stability,3),
                   "progression": round(progression,3),
                   "current_seniority": round(current_seniority,3)}


# ---------------------------------------------------------------------------
# 4. Domain Match
# ---------------------------------------------------------------------------
_AI_TITLES = {
    "ai engineer", "ml engineer", "machine learning engineer",
    "applied scientist", "research engineer", "nlp engineer",
    "search engineer", "ranking engineer", "data scientist",
    "senior ai engineer", "staff ml engineer", "principal engineer",
    "software engineer", "backend engineer", "full stack engineer",
}
_PRODUCT_CO_SIGNALS = ["startup", "series", "saas", "platform", "product"]
_CONSULTING_FIRMS   = ["tcs", "infosys", "wipro", "accenture", "cognizant",
                        "capgemini", "hcl", "tech mahindra", "tata consultancy"]

def domain_match(candidate: dict) -> Tuple[float, dict]:
    title_lower = candidate["profile"].get("current_title","").lower()
    career      = _career_text(candidate)

    # Title alignment
    title_score = max(
        (1.0 if any(t in title_lower for t in _AI_TITLES) else 0.0),
        0.5 if any(kw in title_lower for kw in ["engineer","scientist","developer","analyst"]) else 0.0,
    )

    # Product company vs consulting
    history = candidate.get("career_history", [])
    companies = [j.get("company","").lower() for j in history]
    consulting_count = sum(1 for co in companies if any(f in co for f in _CONSULTING_FIRMS))
    product_signals  = sum(1 for s in _PRODUCT_CO_SIGNALS if s in career)
    product_score = 1.0 if (product_signals >= 2 or consulting_count == 0) else \
                    0.5 if consulting_count < len(companies) else 0.1

    score = 0.5 * title_score + 0.5 * product_score
    return score, {"title_score": round(title_score,3), "product_score": round(product_score,3)}


# ---------------------------------------------------------------------------
# 5. Impact Signals
# ---------------------------------------------------------------------------
def impact_signals(candidate: dict, intent: JDIntent) -> float:
    career = _career_text(candidate)
    count  = sum(1 for v in intent.impact_verbs if v in career)
    return min(1.0, count / 6.0)   # 6+ distinct impact phrases → full credit


# ---------------------------------------------------------------------------
# Disqualifier Penalties
# ---------------------------------------------------------------------------
def disqualifier_penalties(candidate: dict, intent: JDIntent) -> Tuple[float, List[str]]:
    career  = _career_text(candidate)
    skills  = _skills_text(candidate)
    history = candidate.get("career_history", [])
    reasons: List[str] = []
    multiplier = 1.0

    # 1. Pure research
    if (any(t in career for t in ["research scientist","research lab","academic","phd researcher"])
            and not any(t in career for t in ["production","deployed","shipped","live","real users"])):
        multiplier *= 0.45
        reasons.append("career appears research-only, no visible production deployment")

    # 2. LangChain-only recent AI
    current = next((j for j in history if j.get("is_current")), history[0] if history else None)
    has_langchain = "langchain" in skills or "langchain" in career
    pre_llm_ir = any(kw in career for kw in ["recommendation","search relevance","information retrieval","ranking","bm25"])
    if has_langchain and current and current.get("duration_months",999) < 14 and not pre_llm_ir:
        multiplier *= 0.35
        reasons.append("AI exposure looks limited to a recent short LangChain-only stint, no pre-LLM IR background")

    # 3. 18+ months off code
    lead_titles = ["architect","engineering manager","tech lead","head of","vp of","director of"]
    if current and any(t in current.get("title","").lower() for t in lead_titles):
        if current.get("duration_months",0) >= 18 and "hands-on" not in career and "code" not in career:
            multiplier *= 0.30
            reasons.append("current leadership role with 18+ months tenure suggests distance from hands-on coding")

    # 4. Consulting-only
    if history and all(any(f in j.get("company","").lower() for f in _CONSULTING_FIRMS) for j in history):
        multiplier *= 0.40
        reasons.append("entire career at IT-services/consulting firms with no product company experience")

    # 5. CV/speech/robotics only
    cv_terms  = ["computer vision","image classification","object detection","speech recognition","robotics","slam"]
    nlp_terms = ["nlp","retrieval","search","ranking","recommendation","embedding","text"]
    if any(t in career for t in cv_terms) and not any(t in career for t in nlp_terms):
        multiplier *= 0.25
        reasons.append("primary expertise is CV/speech/robotics with no visible NLP or IR exposure")

    # 6. Title-chasing
    if len(history) >= 4:
        short = sum(1 for j in history if j.get("duration_months",999) < 18)
        if short >= len(history) - 1:
            multiplier *= 0.82   # mild; title-chasing is real but weaker signal than the others
            reasons.append("pattern of short tenures (<18mo) across most roles")

    return round(multiplier, 3), reasons


# ---------------------------------------------------------------------------
# Nice-to-have bonus
# ---------------------------------------------------------------------------
def nice_to_have_bonus(candidate: dict, intent: JDIntent) -> float:
    blob = _all_text(candidate)
    hits = sum(1 for kw in intent.nice_to_have if kw in blob)
    return min(0.12, hits * 0.025)


# ---------------------------------------------------------------------------
# Location & Notice
# ---------------------------------------------------------------------------
def location_fit(candidate: dict, intent: JDIntent) -> float:
    profile  = candidate["profile"]
    sig      = candidate.get("redrob_signals", {})
    country  = profile.get("country","").lower()
    location = profile.get("location","").lower()
    if country != intent.preferred_country:
        return 0.35
    if any(city in location for city in intent.preferred_locations):
        return 1.0
    return 0.85 if sig.get("willing_to_relocate") else 0.60

def notice_fit(candidate: dict, intent: JDIntent) -> float:
    days = candidate.get("redrob_signals",{}).get("notice_period_days", 60)
    if days <= intent.notice_sweet_spot_days:
        return 1.0
    if days <= intent.notice_ceiling_days:
        span = intent.notice_ceiling_days - intent.notice_sweet_spot_days
        return 1.0 - 0.4 * (days - intent.notice_sweet_spot_days) / span
    return 0.50


# ---------------------------------------------------------------------------
# Top-level combiner
# ---------------------------------------------------------------------------
def structured_fit_score(candidate: dict, intent: JDIntent = INTENT) -> Tuple[float, dict]:
    """
    Returns (score_0_to_1, breakdown_dict).
    Weights:
      40% skill coverage   — the core "can they do the job" signal
      20% experience       — YOE fit + applied-AI tenure
      15% career growth    — stability, progression, current seniority
      10% domain match     — title + product vs consulting
       5% impact signals   — outcome language in descriptions
      10% location + notice fit
    """
    sc,  sc_d  = skill_coverage(candidate, intent)
    eq,  eq_d  = experience_quality(candidate, intent)
    cg,  cg_d  = career_growth(candidate)
    dm,  dm_d  = domain_match(candidate)
    imp        = impact_signals(candidate, intent)
    loc        = location_fit(candidate, intent)
    notice     = notice_fit(candidate, intent)
    dq_mult, dq_reasons = disqualifier_penalties(candidate, intent)
    bonus      = nice_to_have_bonus(candidate, intent)

    base = (
        0.40 * sc
        + 0.20 * eq
        + 0.15 * cg
        + 0.10 * dm
        + 0.05 * imp
        + 0.07 * loc
        + 0.03 * notice
    )
    base = min(1.0, base + bonus)
    final = base * dq_mult

    return final, {
        "skill_coverage": round(sc, 3),
        "skill_hits": sc_d,
        "experience_quality": round(eq, 3),
        "experience_detail": eq_d,
        "career_growth": round(cg, 3),
        "career_detail": cg_d,
        "domain_match": round(dm, 3),
        "domain_detail": dm_d,
        "impact_score": round(imp, 3),
        "location_fit": round(loc, 3),
        "notice_fit": round(notice, 3),
        "disqualifier_multiplier": dq_mult,
        "disqualifier_reasons": dq_reasons,
        "nice_to_have_bonus": round(bonus, 3),
    }
