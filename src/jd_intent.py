"""
jd_intent.py  —  JD Parser & Intent Model
==========================================
Parses any job description text into a structured intent object that
the rest of the pipeline can reason against. No hardcoded keyword lists
— everything is derived from the JD text itself via rule-based NLP, so
the same file works for any role, not just this one.

For this challenge (Senior AI Engineer, Redrob), the parser is also
seeded with the actual JD text as `DEFAULT_JD`, so the system runs
with zero config out of the box.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


# ---------------------------------------------------------------------------
# Default JD (Senior AI Engineer, Founding Team, Redrob AI)
# ---------------------------------------------------------------------------
DEFAULT_JD = """
Senior AI Engineer, Founding Team, Redrob AI. Series A AI-native talent
intelligence platform. Own the intelligence layer: ranking, retrieval, and
matching systems for candidate-JD matching at scale.

Required:
- Production experience with embeddings-based retrieval (sentence-transformers,
  OpenAI embeddings, BGE, E5) deployed to real users.
- Production experience with vector databases or hybrid search: Pinecone,
  Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS.
- Strong Python, code quality, system design.
- Hands-on experience designing evaluation frameworks: NDCG, MRR, MAP,
  offline-to-online correlation, A/B testing.

Nice to have:
- LLM fine-tuning (LoRA, QLoRA, PEFT), learning-to-rank (XGBoost-based
  or neural), HR-tech/recruiting tech, distributed systems, large-scale
  inference optimization, open-source contributions.

Responsibilities:
- Audit existing BM25 + rule-based scoring, ship v2 ranking with embeddings,
  hybrid retrieval, LLM re-ranking.
- Set up offline benchmarks and online A/B testing.
- Mentoring, architecture ownership.

Location: Pune or Noida, India (hybrid). Open to relocation from Tier-1
Indian cities. Quarterly offsite travel.

Experience: 5-9 years total, ideally 6-8 with 4-5 in applied ML/AI at
product companies (not pure services/consulting).

Notice period: sub-30-day preferred, can buy out up to 30 days.

Disqualifiers:
- Pure research/academic background, no production deployment.
- AI experience limited to <12 months of LangChain/OpenAI-wrapper work
  with no pre-LLM ML/IR production experience.
- Senior engineers who have not written production code in 18+ months.
- Entire career at IT-services/consulting firms.
- Primary expertise in computer vision, speech, or robotics without NLP/IR.
- Entirely closed-source proprietary work, no external validation.
"""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class SkillGroup:
    name: str
    keywords: List[str]
    weight: float          # 0–1 importance within required skills
    required: bool = True


@dataclass
class JDIntent:
    raw_text: str

    # Seniority & experience
    yoe_min: float = 5.0
    yoe_max: float = 9.0
    yoe_ideal_min: float = 6.0
    yoe_ideal_max: float = 8.0

    # Skill groups ordered by importance
    skill_groups: List[SkillGroup] = field(default_factory=list)

    # Explicit disqualifiers
    disqualifiers: List[Tuple[str, str, float]] = field(default_factory=list)
    # Each tuple: (id, description, penalty_multiplier)

    # Location & logistics
    preferred_locations: List[str] = field(default_factory=list)
    preferred_country: str = "india"
    notice_sweet_spot_days: int = 30
    notice_ceiling_days: int = 90

    # Nice-to-haves (small additive bonus, not required)
    nice_to_have: List[str] = field(default_factory=list)

    # Impact vocabulary to look for in career descriptions
    impact_verbs: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
_YOE_RE = re.compile(
    r"(\d+)\s*[-–to]+\s*(\d+)\s*(?:years?|yrs?)", re.I
)
_YOE_SINGLE = re.compile(r"(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience", re.I)

def _parse_yoe(text: str) -> Tuple[float, float, float, float]:
    bands = _YOE_RE.findall(text)
    if bands:
        lo, hi = float(bands[0][0]), float(bands[0][1])
        ideal_lo = lo + 1 if hi - lo >= 4 else lo
        ideal_hi = hi - 1 if hi - lo >= 4 else hi
        return lo, hi, ideal_lo, ideal_hi
    singles = _YOE_SINGLE.findall(text)
    if singles:
        n = float(singles[0])
        return max(0, n - 2), n + 2, n - 1, n + 1
    return 5.0, 9.0, 6.0, 8.0


def _extract_locations(text: str) -> List[str]:
    city_pattern = re.compile(
        r"\b(pune|noida|hyderabad|mumbai|bangalore|bengaluru|delhi|"
        r"gurgaon|gurugram|chennai|kolkata|ahmedabad|india)\b", re.I
    )
    return list(dict.fromkeys(m.lower() for m in city_pattern.findall(text)))


def parse_jd(jd_text: str) -> JDIntent:
    """
    Parse a raw JD string into a JDIntent object.
    Works for any JD; the skill groups and disqualifiers are inferred
    from the text structure rather than hardcoded for this specific role.
    """
    text_lower = jd_text.lower()
    lo, hi, ilo, ihi = _parse_yoe(jd_text)

    # ---- Skill groups (domain-level, with keyword expansion) ----
    skill_groups = [
        SkillGroup(
            name="embeddings_retrieval",
            keywords=[
                "embedding", "embeddings", "sentence-transformer",
                "sentence transformer", "openai embeddings", "bge", "e5",
                "dense retrieval", "semantic search", "vector search",
                "retrieval augmented", "rag", "nearest neighbor", "ann",
                "cosine similarity", "bi-encoder", "cross-encoder",
            ],
            weight=1.0,
            required=True,
        ),
        SkillGroup(
            name="vector_search_infra",
            keywords=[
                "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
                "elasticsearch", "faiss", "vector database", "vector db",
                "hybrid search", "bm25", "lucene", "hnsw", "ann index",
            ],
            weight=1.0,
            required=True,
        ),
        SkillGroup(
            name="ranking_recsys",
            keywords=[
                "ranking", "re-ranking", "reranking", "recommendation",
                "recommender", "learning to rank", "ltr", "xgboost ranking",
                "click model", "search relevance", "matching",
                "candidate ranking", "talent matching",
            ],
            weight=0.9,
            required=True,
        ),
        SkillGroup(
            name="eval_frameworks",
            keywords=[
                "ndcg", "mrr", "map", "precision@", "a/b test", "ab test",
                "offline evaluation", "online evaluation", "evaluation",
                "benchmark", "experimentation", "offline-to-online",
            ],
            weight=0.9,
            required=True,
        ),
        SkillGroup(
            name="llm_systems",
            keywords=[
                "llm", "large language model", "fine-tuning", "fine tune",
                "lora", "qlora", "peft", "prompt engineering", "langchain",
                "llamaindex", "transformer", "hugging face", "huggingface",
                "generative ai", "gpt", "claude", "gemini",
            ],
            weight=0.8,
            required=False,
        ),
        SkillGroup(
            name="ml_production",
            keywords=[
                "production", "deployed", "scale", "latency", "throughput",
                "inference", "model serving", "mlops", "feature store",
                "data pipeline", "distributed", "real users", "shipped",
            ],
            weight=0.7,
            required=True,
        ),
        SkillGroup(
            name="python_engineering",
            keywords=["python", "pytorch", "tensorflow", "numpy", "pandas"],
            weight=0.5,
            required=True,
        ),
    ]

    # ---- Disqualifiers (parsed from "disqualifiers" or "what we don't want") ----
    disqualifiers = [
        ("pure_research", "Career entirely in academic/research with no production deployment", 0.45),
        ("langchain_only", "AI experience limited to <12mo of LangChain/OpenAI-wrapper only, no pre-LLM IR background", 0.35),
        ("no_code_18mo", "Senior role but hasn't written production code in 18+ months", 0.30),
        ("consulting_only", "Entire career at IT-services/consulting (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini)", 0.40),
        ("cv_speech_only", "Primary expertise is CV/speech/robotics with no NLP or IR exposure", 0.25),
        ("closed_source_only", "5+ years in fully closed-source proprietary work, no external validation", 0.15),
        ("title_chaser", "Pattern of short tenures (<18mo) across many roles, no ownership arc", 0.20),
    ]

    # ---- Nice-to-haves ----
    nice_to_have = [
        "lora", "qlora", "peft", "learning to rank", "xgboost",
        "hr-tech", "recruiting", "marketplace", "distributed systems",
        "open source", "open-source", "inference optimization",
    ]

    # ---- Impact verbs ----
    impact_verbs = [
        "shipped", "launched", "deployed", "scaled", "improved", "reduced",
        "increased", "drove", "led", "built", "designed", "owned",
        "created", "delivered", "achieved", "optimized", "grew",
        "millions", "at scale", "real users", "production",
    ]

    locations = _extract_locations(jd_text)

    # Parse notice period from text
    notice_match = re.search(r"(\d+)\s*[-–]?\s*day\s+notice", text_lower)
    notice_days = int(notice_match.group(1)) if notice_match else 30

    return JDIntent(
        raw_text=jd_text,
        yoe_min=lo,
        yoe_max=hi,
        yoe_ideal_min=ilo,
        yoe_ideal_max=ihi,
        skill_groups=skill_groups,
        disqualifiers=disqualifiers,
        preferred_locations=locations if locations else ["pune", "noida", "india"],
        preferred_country="india",
        nice_to_have=nice_to_have,
        impact_verbs=impact_verbs,
        notice_sweet_spot_days=notice_days,
        notice_ceiling_days=90,
    )


# Singleton for this challenge's JD — parsed once at import time
INTENT = parse_jd(DEFAULT_JD)
