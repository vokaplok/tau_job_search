import asyncio
import json
import os

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field

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

# Ranking configuration. Override the model with JOB_RANK_MODEL if needed. Jobs are
# scored in concurrent batches so a large board set doesn't turn into one enormous
# (and slow) model request.
RANK_MODEL = os.environ.get("JOB_RANK_MODEL", "gemini-2.5-flash")
RANK_BATCH_SIZE = 25
RANK_MAX_CONCURRENCY = 5

RANK_SYSTEM = (
    "You are a technical recruiter. You are given a candidate's CV and the role they "
    "are looking for, followed by a batch of job postings. Score how well each job fits "
    "this specific candidate and their target role.\n\n"
    "Scoring rubric (0-100):\n"
    "  90-100  Excellent fit: title/seniority and skills strongly match the target role and CV.\n"
    "  70-89   Good fit: clearly relevant, minor gaps in seniority, domain, or skills.\n"
    "  40-69   Partial fit: adjacent role or transferable skills, but notable mismatch.\n"
    "  10-39   Weak fit: different function or seniority; only loosely related.\n"
    "  0-9     No fit: unrelated to the candidate's role and background.\n\n"
    "Judge on role/title alignment, seniority, required skills, and domain relevance. "
    "Return one entry per job using the job's 'id'. Keep each reason to a single concise "
    "sentence (max ~20 words) explaining the score."
)


class Ranking(BaseModel):
    id: int
    score: int
    reason: str


class RankRequest(BaseModel):
    cv: str = Field(..., description="The candidate's CV / resume text.")
    role: str = Field(..., description="The role the candidate is looking for.")
    top_n: int | None = Field(
        default=None,
        ge=1,
        description="If set, only return the top N ranked jobs. Defaults to all jobs.",
    )


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


async def score_batch(
    client: genai.Client,
    semaphore: asyncio.Semaphore,
    cv: str,
    role: str,
    batch: list[tuple[int, dict]],
) -> dict[int, dict]:
    """Score a batch of (global_index, job) pairs. Returns {index: {score, reason}}."""
    payload = [
        {
            "id": index,
            "title": job.get("title"),
            "location": job.get("location"),
            "company": job.get("company"),
        }
        for index, job in batch
    ]
    prompt = (
        f"Target role: {role}\n\n"
        f"Candidate CV:\n{cv}\n\n"
        f"Job postings to score (JSON):\n{json.dumps(payload, ensure_ascii=False)}"
    )

    async with semaphore:
        response = await client.aio.models.generate_content(
            model=RANK_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=RANK_SYSTEM,
                temperature=0,
                # Structured output: guarantees valid, parseable JSON.
                response_mime_type="application/json",
                response_schema=list[Ranking],
            ),
        )

    rankings = json.loads(response.text)
    return {
        item["id"]: {"score": item["score"], "reason": item["reason"]}
        for item in rankings
    }


@app.get("/jobs")
async def get_jobs() -> dict:
    """Fetch jobs from all configured Greenhouse boards concurrently and combine them."""
    jobs = await fetch_all_jobs()
    return {"count": len(jobs), "jobs": jobs}


@app.post("/rank")
async def rank_jobs(request: RankRequest) -> dict:
    """Rank all fetched jobs by how well they fit the given CV and target role.

    Uses Gemini to score each job 0-100 with a short reason, then returns the jobs
    sorted best-fit first.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Ranking is unavailable: GEMINI_API_KEY is not set.",
        )

    jobs = await fetch_all_jobs()

    indexed = list(enumerate(jobs))
    batches = [
        indexed[i : i + RANK_BATCH_SIZE]
        for i in range(0, len(indexed), RANK_BATCH_SIZE)
    ]

    client = genai.Client(api_key=api_key)
    semaphore = asyncio.Semaphore(RANK_MAX_CONCURRENCY)
    try:
        batch_results = await asyncio.gather(
            *(
                score_batch(client, semaphore, request.cv, request.role, batch)
                for batch in batches
            )
        )
    except genai_errors.APIError as exc:
        raise HTTPException(status_code=502, detail=f"Ranking failed: {exc}") from exc

    scores: dict[int, dict] = {}
    for result in batch_results:
        scores.update(result)

    ranked = []
    for index, job in indexed:
        score_info = scores.get(index, {"score": 0, "reason": "Not scored."})
        ranked.append(
            {
                "title": job.get("title"),
                "company": job.get("company"),
                "location": job.get("location"),
                "apply_url": job.get("apply_url"),
                "score": score_info["score"],
                "reason": score_info["reason"],
            }
        )

    # Best fit first; stable within equal scores.
    ranked.sort(key=lambda job: job["score"], reverse=True)
    if request.top_n is not None:
        ranked = ranked[: request.top_n]

    return {"role": request.role, "count": len(ranked), "jobs": ranked}
