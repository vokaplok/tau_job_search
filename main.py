import asyncio
import io
import re
from collections import Counter

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Reeds Jobs API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GREENHOUSE_BOARDS = [
    "riskified",
    "fireblocks",
    "pagayais",
    "gongio",
    "lightricks",
    "similarweb",
    "melio",
    "wizinc",
    "yotpo",
    "catonetworks",
]
GREENHOUSE_URL = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true"

# How many top-ranked jobs /rank returns.
RANK_LIMIT = 30


async def fetch_board(client: httpx.AsyncClient, token: str) -> list[dict]:
    """Fetch all jobs for a single Greenhouse board and tag them with the company."""
    response = await client.get(GREENHOUSE_URL.format(token=token))
    response.raise_for_status()
    data = response.json()
    jobs = []
    for job in data.get("jobs", []):
        location = job.get("location") or {}
        jobs.append(
            {
                "title": job.get("title"),
                "location": location.get("name"),
                "apply_url": job.get("absolute_url"),
                "company": token,
                # Plain-text-ish job body, used only for scoring (not returned).
                "content": _strip_html(job.get("content") or ""),
            }
        )
    return jobs


async def fetch_all_jobs() -> list[dict]:
    """Fetch jobs from all configured Greenhouse boards concurrently and combine them."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        results = await asyncio.gather(
            *(fetch_board(client, token) for token in GREENHOUSE_BOARDS),
            return_exceptions=True,
        )

    jobs: list[dict] = []
    for token, result in zip(GREENHOUSE_BOARDS, results):
        if isinstance(result, Exception):
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch jobs for board '{token}': {result}",
            )
        jobs.extend(result)
    return jobs


@app.get("/jobs")
async def get_jobs() -> dict:
    """Fetch and combine jobs from all configured Greenhouse boards."""
    jobs = await fetch_all_jobs()
    for job in jobs:
        job.pop("content", None)
    return {"count": len(jobs), "jobs": jobs}


# --------------------------------------------------------------------------- #
#  Ranking
# --------------------------------------------------------------------------- #

_HTML_TAG = re.compile(r"<[^>]+>")
_ENTITY = re.compile(r"&[a-zA-Z#0-9]+;")
_WORD = re.compile(r"[a-z0-9+#.]+")

STOPWORDS = {
    "the", "and", "for", "you", "your", "with", "our", "are", "will", "have",
    "this", "that", "from", "was", "were", "has", "had", "not", "but", "all",
    "can", "who", "job", "role", "work", "team", "teams", "company", "years",
    "experience", "including", "based", "part", "join", "looking", "ability",
    "strong", "new", "one", "per", "etc", "such", "into", "out", "about",
    "we", "a", "an", "in", "on", "of", "to", "is", "as", "at", "by", "or",
    "be", "it", "we're", "us", "their", "they", "them", "he", "she", "his",
    "her", "its", "also", "more", "most", "other", "over", "under", "within",
}


def _strip_html(text: str) -> str:
    text = _HTML_TAG.sub(" ", text)
    text = _ENTITY.sub(" ", text)
    return text


def tokenize(text: str) -> list[str]:
    return [
        w
        for w in _WORD.findall(text.lower())
        if len(w) > 1 and w not in STOPWORDS
    ]


def extract_pdf_text(data: bytes) -> str:
    """Best-effort text extraction from a PDF. Returns '' if it can't be read."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


def score_job(
    job: dict,
    role_tokens: set[str],
    cv_counts: Counter,
    cv_top: set[str],
) -> tuple[int, str]:
    """Score a single job 0–100 against the target role and CV, with a reason."""
    title = job.get("title") or ""
    title_tokens = set(tokenize(title))
    body_tokens = set(tokenize(job.get("content") or ""))
    haystack = title_tokens | body_tokens

    # 1) Role relevance — how many role keywords the posting covers.
    role_in_title = role_tokens & title_tokens
    role_in_body = (role_tokens & body_tokens) - role_in_title
    role_denom = max(len(role_tokens), 1)
    role_score = (len(role_in_title) * 1.0 + len(role_in_body) * 0.5) / role_denom
    role_score = min(role_score, 1.0)

    # 2) CV relevance — how much of your CV's vocabulary the posting shares.
    cv_hits = cv_top & haystack
    cv_score = len(cv_hits) / max(len(cv_top), 1) if cv_top else 0.0
    cv_score = min(cv_score, 1.0)

    if cv_top:
        combined = 0.65 * role_score + 0.35 * cv_score
    else:
        combined = role_score

    # Small floor so nothing shows a bare 0, scaled to 0–100.
    score = round(6 + combined * 94)
    score = max(0, min(100, score))

    # Build a human-readable reason.
    parts: list[str] = []
    if role_in_title:
        parts.append("matches your role in the title: " + ", ".join(sorted(role_in_title)))
    elif role_in_body:
        parts.append("mentions your role: " + ", ".join(sorted(list(role_in_body)[:4])))
    else:
        parts.append("limited overlap with your target role")

    if cv_hits:
        top_cv = sorted(cv_hits, key=lambda t: -cv_counts.get(t, 0))[:4]
        parts.append("overlaps with your CV on " + ", ".join(top_cv))

    reason = "This role " + "; ".join(parts) + "."
    return score, reason


@app.post("/rank")
async def rank(cv: UploadFile = File(...), role: str = Form(...)) -> dict:
    """Rank live Greenhouse jobs against an uploaded CV (PDF) and a target role."""
    cv_bytes = await cv.read()
    cv_text = extract_pdf_text(cv_bytes)

    role_tokens = set(tokenize(role))
    cv_counts = Counter(tokenize(cv_text))
    # Use the most frequent, meaningful CV terms as the CV "fingerprint".
    cv_top = {w for w, _ in cv_counts.most_common(40)}

    jobs = await fetch_all_jobs()

    ranked: list[dict] = []
    for job in jobs:
        score, reason = score_job(job, role_tokens, cv_counts, cv_top)
        ranked.append(
            {
                "title": job.get("title") or "Untitled role",
                "company": job.get("company") or "",
                "location": job.get("location") or "",
                "apply_url": job.get("apply_url") or "",
                "score": score,
                "reason": reason,
            }
        )

    ranked.sort(key=lambda j: j["score"], reverse=True)
    return {"jobs": ranked[:RANK_LIMIT]}
