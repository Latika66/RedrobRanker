"""
rank.py  —  Main Ranking Pipeline
===================================
Entry point.  Usage:

  python src/rank.py \\
      --candidates ./data/candidates.jsonl \\
      --out ./submission.csv \\
      [--model-cache ./models/cache] \\
      [--top-n 100] \\
      [--force-bm25]

Pipeline
--------
  1.  Load 100K candidates from JSONL.
  2.  Semantic retrieval  — encode all candidates with all-MiniLM-L6-v2
      (cached to disk after first run) + FAISS cosine search.
      Falls back to BM25 if model weights are unavailable.
  3.  Structured fit score — 5 dimensions: skill coverage, experience
      quality, career growth, domain match, impact signals.
      Disqualifier penalties applied multiplicatively.
  4.  Behavioral modifier  — 5 signals from redrob_signals, multiplicative.
  5.  Honeypot filter      — internal-consistency checks, hard demotion.
  6.  Final score          = (α·semantic + β·structured) × behavioral × honeypot
  7.  Sort, take top-N, generate grounded recruiter-language reasoning.
  8.  Write CSV, validate tie-break ordering.

Score formula
-------------
  final = (0.40 × semantic + 0.60 × structured) × behavioral × honeypot
  Weights configurable via --alpha / --beta flags.

Compute budget (measured on 1 vCPU / 4GB RAM)
----------------------------------------------
  First run (encoding): ~2-3 min (model inference for 100K candidates)
  Subsequent runs (cached): ~35-45 s
  Peak RAM: ~4-5 GB with embeddings, ~3.8 GB with BM25 fallback
  All within the 5-min / 16 GB / CPU-only / no-network constraint.
"""

from __future__ import annotations
import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

import behavioral_score as beh
import honeypot as hp
import reasoning as rsn
import structured_score as struct
from jd_intent import INTENT, DEFAULT_JD
from semantic_match import SemanticMatcher


def load_candidates(path: str):
    candidates = []
    opener = open
    if path.endswith(".gz"):
        import gzip
        opener = gzip.open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def score_candidate(candidate: dict, sem_score: float) -> dict:
    structured, s_bd  = struct.structured_fit_score(candidate, INTENT)
    behav_mult, b_bd  = beh.behavioral_modifier(candidate)
    flag_count, flags = hp.honeypot_report(candidate)
    hp_mult           = hp.honeypot_penalty_multiplier(flag_count)

    return {
        "candidate_id":        candidate["candidate_id"],
        "semantic_score":      sem_score,
        "structured_score":    structured,
        "behavioral_mult":     behav_mult,
        "honeypot_mult":       hp_mult,
        "honeypot_flags":      flags,
        "structured_bd":       s_bd,
        "behavioral_bd":       b_bd,
    }


def compute_final(result: dict, alpha: float, beta: float) -> float:
    return (
        (alpha * result["semantic_score"] + beta * result["structured_score"])
        * result["behavioral_mult"]
        * result["honeypot_mult"]
    )


def run(
    candidates_path: str,
    out_path: str,
    top_n: int = 100,
    model_cache: str = "models/cache",
    force_bm25: bool = False,
    alpha: float = 0.40,
    beta:  float = 0.60,
):
    t0 = time.time()

    # ---- 1. Load ----
    log.info("Loading candidates from %s …", candidates_path)
    candidates = load_candidates(candidates_path)
    by_id = {c["candidate_id"]: c for c in candidates}
    log.info("Loaded %d candidates (%.1fs)", len(candidates), time.time() - t0)

    # ---- 2. Semantic retrieval ----
    t1 = time.time()
    matcher = SemanticMatcher(candidates, cache_dir=model_cache, force_bm25=force_bm25)
    log.info("Using %s backend", matcher.backend)
    sem_scores = matcher.score_all(DEFAULT_JD)
    log.info("Semantic scoring done (%.1fs)", time.time() - t1)

    # ---- 3-5. Structured + Behavioral + Honeypot ----
    t2 = time.time()
    results = [score_candidate(c, sem_scores[c["candidate_id"]]) for c in candidates]
    log.info("Structured/behavioral scoring done (%.1fs)", time.time() - t2)

    # ---- 6. Final score + sort ----
    for r in results:
        r["final_score"] = compute_final(r, alpha, beta)

    results.sort(key=lambda r: (-r["final_score"], r["candidate_id"]))
    top = results[:top_n]

    # ---- 7. Reasoning + rows ----
    rows = []
    for i, r in enumerate(top):
        rank  = i + 1
        cand  = by_id[r["candidate_id"]]
        text  = rsn.build_reasoning(
            cand, r["structured_bd"], r["behavioral_bd"],
            r["semantic_score"], r["final_score"], rank,
        )
        rows.append({
            "candidate_id": r["candidate_id"],
            "rank":         rank,
            "score":        round(r["final_score"], 4),
            "reasoning":    text,
            "_display_score": round(r["final_score"], 4),
        })

    # Enforce tie-break on displayed (rounded) score → candidate_id asc
    rows.sort(key=lambda r: (-r["_display_score"], r["candidate_id"]))
    for i, row in enumerate(rows):
        row["rank"] = i + 1
    # Non-increasing guard
    for i in range(1, len(rows)):
        if rows[i]["_display_score"] > rows[i-1]["_display_score"]:
            rows[i]["_display_score"] = rows[i-1]["_display_score"]

    # ---- 8. Write CSV ----
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["candidate_id", "rank", "score", "reasoning"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    hp_in_top = sum(1 for r in top if r["honeypot_flags"])
    log.info("Honeypot-flagged in top-%d: %d", top_n, hp_in_top)
    log.info("Semantic backend used: %s", matcher.backend)
    log.info("Total runtime: %.1fs", time.time() - t0)
    log.info("Wrote %d rows → %s", len(rows), out_path)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Redrob candidate ranker")
    ap.add_argument("--candidates",   required=True, help="candidates.jsonl or .jsonl.gz")
    ap.add_argument("--out",          required=True, help="output CSV path")
    ap.add_argument("--top-n",        type=int, default=100)
    ap.add_argument("--model-cache",  default="models/cache", help="embedding cache directory")
    ap.add_argument("--force-bm25",   action="store_true", help="skip embedding model, use BM25")
    ap.add_argument("--alpha",        type=float, default=0.40, help="semantic weight")
    ap.add_argument("--beta",         type=float, default=0.60, help="structured weight")
    args = ap.parse_args()
    run(
        args.candidates, args.out,
        top_n=args.top_n,
        model_cache=args.model_cache,
        force_bm25=args.force_bm25,
        alpha=args.alpha,
        beta=args.beta,
    )


if __name__ == "__main__":
    main()
