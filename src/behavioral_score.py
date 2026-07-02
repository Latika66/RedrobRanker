"""
behavioral_score.py  —  Behavioral Trust Multiplier
====================================================
Converts the 23 redrob_signals fields into five sub-scores that together
form a multiplicative modifier in [0.50, 1.15]:

  Recruiter Engagement  — response rate × speed
  Reliability           — interview completion × offer acceptance
  Platform Trust        — completeness, email/phone/LinkedIn verification
  Availability          — open_to_work flag + recency of activity
  Market Consistency    — saves, views, search appearances from other recruiters

Why multiplicative, not additive:
  A perfect-skills candidate who hasn't logged in for 6 months and has a
  5% response rate is not actually hirable. This modifier compresses their
  final score without overriding skill quality — it can't make a weak
  candidate strong, but it can make an equally strong candidate rank lower
  when they're unreachable.
"""

from __future__ import annotations
from datetime import date
from typing import Tuple, Dict


def _parse_date(s):
    if not s:
        return None
    try:
        return date.fromisoformat(str(s)[:10])
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# 1. Recruiter Engagement
# ---------------------------------------------------------------------------
def _recruiter_engagement(sig: dict) -> float:
    rate  = sig.get("recruiter_response_rate", 0.0) or 0.0
    hours = sig.get("avg_response_time_hours", 999) or 999
    # Speed bonus: fast responders get a modest lift
    speed = 1.0 if hours <= 24 else (0.85 if hours <= 72 else (0.70 if hours <= 168 else 0.55))
    return min(1.0, rate) * 0.75 + speed * 0.25


# ---------------------------------------------------------------------------
# 2. Reliability
# ---------------------------------------------------------------------------
def _reliability(sig: dict) -> float:
    interview = sig.get("interview_completion_rate", 0.5) or 0.5
    offer     = sig.get("offer_acceptance_rate", -1)
    if offer is None or offer < 0:
        return interview
    return 0.65 * interview + 0.35 * offer


# ---------------------------------------------------------------------------
# 3. Platform Trust
# ---------------------------------------------------------------------------
def _platform_trust(sig: dict) -> float:
    completeness = (sig.get("profile_completeness_score", 50) or 50) / 100.0
    verified = sum([
        bool(sig.get("verified_email")),
        bool(sig.get("verified_phone")),
        bool(sig.get("linkedin_connected")),
    ]) / 3.0
    return 0.55 * completeness + 0.45 * verified


# ---------------------------------------------------------------------------
# 4. Availability (recency + open_to_work)
# ---------------------------------------------------------------------------
def _availability(sig: dict) -> float:
    last = _parse_date(sig.get("last_active_date"))
    today = date.today()
    if last:
        days = max(0, (today - last).days)
        if days <= 7:
            recency = 1.0
        elif days <= 30:
            recency = 0.90
        elif days <= 90:
            recency = 0.65
        elif days <= 180:
            recency = 0.35
        else:
            recency = 0.10    # 6+ months: the JD's own cited example
    else:
        recency = 0.30

    open_flag = 1.0 if sig.get("open_to_work_flag") else 0.50
    return 0.60 * recency + 0.40 * open_flag


# ---------------------------------------------------------------------------
# 5. Market Consistency
# ---------------------------------------------------------------------------
def _market_consistency(sig: dict) -> float:
    views       = sig.get("profile_views_received_30d", 0) or 0
    saved       = sig.get("saved_by_recruiters_30d", 0)   or 0
    appearances = sig.get("search_appearance_30d", 0)     or 0
    # Log-ish normalisation to handle long tail
    v = min(1.0, views       / 80.0)
    s = min(1.0, saved       / 15.0)
    a = min(1.0, appearances / 150.0)
    return 0.35 * v + 0.40 * s + 0.25 * a


# ---------------------------------------------------------------------------
# Combined modifier
# ---------------------------------------------------------------------------
def behavioral_modifier(candidate: dict) -> Tuple[float, Dict]:
    sig = candidate.get("redrob_signals", {})

    engagement  = _recruiter_engagement(sig)
    reliability = _reliability(sig)
    trust       = _platform_trust(sig)
    avail       = _availability(sig)
    market      = _market_consistency(sig)

    composite = (
        0.30 * engagement
        + 0.20 * avail
        + 0.20 * trust
        + 0.15 * reliability
        + 0.15 * market
    )
    # Map [0,1] → [0.50, 1.15]
    multiplier = 0.50 + composite * 0.65

    return round(multiplier, 3), {
        "recruiter_engagement": round(engagement, 3),
        "reliability":          round(reliability, 3),
        "platform_trust":       round(trust, 3),
        "availability":         round(avail, 3),
        "market_consistency":   round(market, 3),
        "behavioral_multiplier": round(multiplier, 3),
    }
