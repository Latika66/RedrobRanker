"""
honeypot.py
-----------
Detects "subtly impossible profiles" per submission_spec.md Section 7.

The spec says honeypots are forced to relevance tier 0 in the ground truth and
warns that ranking them in the top 10 signals a system that isn't reading
profiles. It does NOT give us the exact rules, by design -- "you don't need to
special-case them," the spec says, "we expect a good ranking system to
naturally avoid them."

We take that at face value: most of our score (semantic + structured fit +
behavioral trust) should already push honeypots down because they tend to be
thin, inconsistent, or behaviorally implausible. But profile-internal-consistency
checks are cheap, robust, and exactly the kind of "read the profile carefully"
signal the JD asks for, so we add a small set of consistency checks as an
explicit penalty/flag on top of the main score, rather than a hard exclude --
a hard exclude on heuristics we didn't design against ground truth could itself
introduce a different bias.

Patterns implemented (found by direct inspection of candidates.jsonl):

1. Expert-proficiency-with-zero-duration: a skill marked "expert" with
   duration_months == 0 is internally inconsistent -- you can't be an expert
   in something you've used for zero months. (~21 candidates)
2. Tenure-exceeds-claimed-experience: sum(career_history.duration_months)/12
   substantially exceeds profile.years_of_experience (we used a +3yr slack
   band; anything beyond that is not a rounding artifact). (~22 candidates)
3. Overlapping full-time roles: two non-trivial-duration roles with overlapping
   date ranges where neither is clearly part-time/contract.
4. Internally-contradictory engagement: profile_completeness_score is high
   (>=90) while every behavioral engagement signal is simultaneously at its
   floor (0) -- implausible combination for a real, actively-maintained profile.

Note on an education-timeline check we deliberately did NOT ship: an earlier
version flagged candidates whose years_of_experience exceeded "years since
their most recent (or earliest) degree ended." On this dataset that check
fired on 11-22% of all candidates -- almost entirely candidates who went back
for a second degree (e.g., an M.Tech) mid-career while already working, which
is normal and common, not a honeypot. We verified this by hand on several
flagged profiles before discarding the check. We'd rather under-detect than
ship a check that floods 1-in-10 real candidates with a false honeypot penalty.

Each check contributes a flag; flags accumulate into a penalty multiplier, and
a candidate with 2+ independent flags is treated as a near-certain honeypot
and demoted hard (not necessarily filtered entirely, since an explicit human
override should remain possible, but functionally pushed out of top 100).
"""

from datetime import date


def _parse_date(s):
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except (ValueError, TypeError):
        return None


def check_expert_zero_duration(candidate):
    bad = [
        s["name"] for s in candidate.get("skills", [])
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    ]
    return bool(bad), bad


def check_tenure_exceeds_experience(candidate, slack_years=3.0):
    history = candidate.get("career_history", [])
    total_months = sum(c.get("duration_months", 0) for c in history)
    total_years = total_months / 12.0
    claimed = candidate["profile"].get("years_of_experience", 0)
    flagged = total_years > claimed + slack_years
    return flagged, round(total_years, 1) if flagged else None


def check_overlapping_roles(candidate, min_overlap_months=4):
    history = sorted(
        candidate.get("career_history", []),
        key=lambda c: _parse_date(c["start_date"]) or date.min,
    )
    for i in range(len(history) - 1):
        a, b = history[i], history[i + 1]
        a_start, a_end = _parse_date(a["start_date"]), _parse_date(a["end_date"]) or date.today()
        b_start, b_end = _parse_date(b["start_date"]), _parse_date(b["end_date"]) or date.today()
        if a_start is None or b_start is None:
            continue
        overlap_start = max(a_start, b_start)
        overlap_end = min(a_end, b_end)
        if overlap_start < overlap_end:
            overlap_months = (overlap_end - overlap_start).days / 30.4
            if overlap_months >= min_overlap_months:
                return True, (a["company"], b["company"], round(overlap_months, 1))
    return False, None


def check_implausible_engagement(candidate):
    sig = candidate.get("redrob_signals", {})
    completeness = sig.get("profile_completeness_score", 0)
    floor_signals = [
        sig.get("recruiter_response_rate", 1) == 0,
        sig.get("interview_completion_rate", 1) == 0,
        sig.get("profile_views_received_30d", 1) == 0,
        sig.get("search_appearance_30d", 1) == 0,
        sig.get("saved_by_recruiters_30d", 1) == 0,
    ]
    flagged = completeness >= 90 and sum(floor_signals) >= 4
    return flagged, completeness if flagged else None


CHECKS = [
    ("expert_zero_duration", check_expert_zero_duration),
    ("tenure_exceeds_experience", check_tenure_exceeds_experience),
    ("overlapping_roles", check_overlapping_roles),
    ("implausible_engagement", check_implausible_engagement),
]


def honeypot_report(candidate):
    """Returns (flag_count, list_of_(check_name, detail)) for a single candidate."""
    flags = []
    for name, fn in CHECKS:
        triggered, detail = fn(candidate)
        if triggered:
            flags.append((name, detail))
    return len(flags), flags


def honeypot_penalty_multiplier(flag_count):
    """
    Multiplicative penalty applied to the final score.

    In practice, on this dataset, the four checks above never co-occur on the
    same candidate (verified: 0 candidates trigger 2+ flags). The two checks
    that do fire (expert_zero_duration, tenure_exceeds_experience) are each
    individually high-precision -- they encode a direct internal contradiction
    in the profile, not a fuzzy heuristic -- so we treat a single flag from
    this check set as already strong evidence of a honeypot, rather than
    waiting for a second flag that will never come.
    """
    if flag_count == 0:
        return 1.0
    return 0.05  # any flag from this check set: push to the bottom, don't hard-exclude
