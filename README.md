# Redrob Ranker

An AI-powered candidate ranking system for the Redrob Hackathon — Intelligent
Candidate Discovery & Ranking Challenge. Ranks 100,000 candidates against a
Senior AI Engineer job description using a hybrid embedding + structured scoring
pipeline. Fully offline, CPU-only, no hosted LLM calls at ranking time.

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the embedding model (one-time, ~90MB)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 3. Place candidates.jsonl in data/
#    (organizer-provided, ~480MB — not committed to this repo)

# 4. Run
python src/rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

# 5. Validate
python validate_submission.py ./submission.csv
```

**First run:** ~2-3 min (encoding 100K candidates). Embeddings are cached.
**Subsequent runs:** ~35-45 seconds.
**Peak RAM:** ~4-5 GB. **Compute budget:** 5 min / 16 GB / CPU-only ✓

> No embedding model downloaded? Add `--force-bm25` to fall back to BM25 scoring
> with no model required. Output format and validation are identical.

---

## Architecture

```
Job Description
      │
      ▼
┌─────────────┐
│  JD Parser  │  src/jd_intent.py
│             │  Parses any JD into structured intent:
│  skill_groups, disqualifiers,   │
│  yoe_band, locations, impact_verbs  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│     Semantic Retrieval              │  src/semantic_match.py
│                                     │
│  Candidate text (title + summary +  │
│  career descriptions × 3 + skills) │
│         ↓                           │
│  all-MiniLM-L6-v2 (CPU, cached)    │
│         ↓                           │
│  FAISS flat cosine index            │
│         ↓                           │
│  Similarity score per candidate     │
│  [Falls back to BM25 if needed]     │
└──────┬──────────────────────────────┘
       │  40% weight
       ▼
┌─────────────────────────────────────┐
│     Structured Fit Score            │  src/structured_score.py
│                                     │
│  1. Skill Coverage      (40%)       │
│     demonstrated > claimed only     │
│  2. Experience Quality  (20%)       │
│     YOE band + applied-AI tenure   │
│  3. Career Growth       (15%)       │
│     stability + progression         │
│  4. Domain Match        (10%)       │
│     title + product vs consulting   │
│  5. Impact Signals       (5%)       │
│     outcome language in descriptions│
│  6. Location + Notice   (10%)       │
│                                     │
│  × Disqualifier penalties (0.25-1×) │
└──────┬──────────────────────────────┘
       │  60% weight
       ▼
┌─────────────────────────────────────┐
│     Behavioral Modifier             │  src/behavioral_score.py
│                                     │
│  Recruiter Engagement   (30%)       │
│  Availability/Recency   (20%)       │
│  Platform Trust         (20%)       │
│  Reliability            (15%)       │
│  Market Consistency     (15%)       │
│                                     │
│  → Multiplier in [0.50, 1.15]      │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│     Honeypot Filter                 │  src/honeypot.py
│                                     │
│  Expert skill + 0 months used       │
│  Tenure >> claimed experience       │
│  Overlapping full-time roles        │
│  → Multiplier: 0.05 if flagged      │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│     Reasoning Engine                │  src/reasoning.py
│                                     │
│  Grounded in scoring breakdown:     │
│  never hallucinates skills or       │
│  companies not on the profile.      │
│  Varies sentence shape, not just    │
│  inserted values, to avoid template │
│  repetition across rows.            │
└──────┬──────────────────────────────┘
       │
       ▼
   Top-100 CSV
```

**Final score formula:**
```
final_score = (0.40 × semantic + 0.60 × structured)
              × behavioral_modifier
              × honeypot_multiplier
```

---

## Why this design

**Dense embeddings over BM25 for semantic matching:**
BM25 rewards exact keyword overlap — the same trap the JD warns about
(a Graphic Designer who lists "RAG, Pinecone, LangChain" as skills). A
sentence-transformer embedding understands that "built a real-time candidate
ranking pipeline" and "implemented search relevance for a job marketplace"
describe the same thing, even with zero shared keywords.

**Career text weighted 3× over skill tags in embedding input:**
Demonstrated work in project descriptions is a stronger signal than a
listed skill. We make this explicit in the text construction, not just
in the structured scorer.

**Structured score still at 60% weight:**
The JD has seven explicit disqualifiers and nuanced seniority/location
requirements that are hard for a general-purpose embedding model to capture
from similarity alone. The structured component encodes these directly.

**Behavioral as a multiplier, not additive:**
Skill quality and availability are orthogonal axes. A perfect-on-paper
candidate who's unreachable is not as valuable as a slightly weaker
candidate who responds in 24 hours. Treating behavioral signals as a
multiplier preserves this intuition without letting it dominate.

**Honeypot detection: precision over recall:**
We ship two checks we verified against the real dataset (expert + 0-month
skill usage; total tenure > reported experience by 3+ years). An earlier
education-timeline check was tested and discarded after it flagged 11-22%
of real candidates as false positives. Better to miss some honeypots
than to penalise legitimate candidates.

---

## Repository layout

```
src/
  rank.py              Main entry point + CLI
  jd_intent.py         JD parser → structured intent object
  semantic_match.py    Dense embeddings (FAISS) + BM25 fallback
  structured_score.py  5-dimension structured fit scorer
  behavioral_score.py  5-signal behavioral trust multiplier
  honeypot.py          Internal-consistency honeypot detection
  reasoning.py         Recruiter-language justification generator

models/
  cache/               Embedding cache (created on first run)

sandbox/
  app.py               Streamlit demo (deploy to Streamlit Cloud)
  requirements.txt

deck/
  redrob_methodology.pdf
  redrob_methodology.pptx

data/                  Place candidates.jsonl here (not committed, 480MB)
README.md
requirements.txt
submission.csv         Our ranked top-100 output
validate_submission.py Official organizer validator (unmodified)
submission_metadata.yaml
```

---

## CLI options

```bash
python src/rank.py \
  --candidates ./data/candidates.jsonl \   # input JSONL (or .jsonl.gz)
  --out        ./submission.csv \           # output CSV
  --top-n      100 \                        # candidates in output
  --model-cache ./models/cache \            # where to cache embeddings
  --force-bm25 \                            # skip embeddings, use BM25
  --alpha 0.40 \                            # semantic score weight
  --beta  0.60                              # structured score weight
```

---

## Honest limitations

- **Disqualifiers are rule-based:** detecting "consulting-only career" from
  company names misses firms not in our list. A semantic LLM judgment
  would be more robust but isn't allowed at ranking time.
- **~43 of ~80 honeypots explicitly caught:** the rest are naturally
  suppressed by low structured/semantic scores, but not explicitly flagged.
- **Embedding model is general-purpose:** all-MiniLM-L6-v2 wasn't trained
  on recruiting text. A domain-fine-tuned bi-encoder would perform better.
  This was the practical tradeoff for a 90MB model that runs on CPU.
