import asyncio
import json
import os

import anthropic
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# Ranking configuration. The model default matches the rest of our tooling; override
# with JOB_RANK_MODEL if needed. Jobs are scored in concurrent batches so a large board
# set doesn't turn into one enormous (and slow) model request.
RANK_MODEL = os.environ.get("JOB_RANK_MODEL", "claude-opus-4-8")
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

# Structured output: guarantees the model returns valid, parseable JSON.
RANK_SCHEMA = {
    "type": "object",
    "properties": {
        "rankings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "score": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "score", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["rankings"],
    "additionalProperties": False,
}


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
    client: anthropic.AsyncAnthropic,
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
    user_message = (
        f"Target role: {role}\n\n"
        f"Candidate CV:\n{cv}\n\n"
        f"Job postings to score (JSON):\n{json.dumps(payload, ensure_ascii=False)}"
    )

    async with semaphore:
        response = await client.messages.create(
            model=RANK_MODEL,
            max_tokens=8000,
            system=[
                {
                    "type": "text",
                    "text": RANK_SYSTEM,
                    # Cache the stable instructions across every batch of this request.
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_message}],
            output_config={"format": {"type": "json_schema", "schema": RANK_SCHEMA}},
        )

    # With output_config.format the first text block is guaranteed valid JSON.
    text = next(block.text for block in response.content if block.type == "text")
    rankings = json.loads(text)["rankings"]
    return {
        item["id"]: {"score": item["score"], "reason": item["reason"]}
        for item in rankings
    }


@app.get("/jobs")
async def get_jobs() -> dict:
    """Fetch jobs from all configured Greenhouse boards concurrently and combine them."""
    jobs = await fetch_all_jobs()
    return {"count": len(jobs), "jobs": jobs}


@app.post("/jobs/rank")
async def rank_jobs(request: RankRequest) -> dict:
    """Rank all fetched jobs by how well they fit the given CV and target role.

    Uses Claude to score each job 0-100 with a short reason, then returns the jobs
    sorted best-fit first.
    """
    jobs = await fetch_all_jobs()

    indexed = list(enumerate(jobs))
    batches = [
        indexed[i : i + RANK_BATCH_SIZE]
        for i in range(0, len(indexed), RANK_BATCH_SIZE)
    ]

    semaphore = asyncio.Semaphore(RANK_MAX_CONCURRENCY)
    try:
        async with anthropic.AsyncAnthropic() as client:
            batch_results = await asyncio.gather(
                *(
                    score_batch(client, semaphore, request.cv, request.role, batch)
                    for batch in batches
                )
            )
    except anthropic.AnthropicError as exc:
        raise HTTPException(status_code=502, detail=f"Ranking failed: {exc}") from exc
    except (TypeError, ValueError) as exc:
        # The SDK raises when it cannot resolve credentials (no API key configured).
        raise HTTPException(
            status_code=503,
            detail=(
                "Ranking is unavailable: no Anthropic credentials configured. "
                f"Set ANTHROPIC_API_KEY. ({exc})"
            ),
        ) from exc

    scores: dict[int, dict] = {}
    for result in batch_results:
        scores.update(result)

    ranked = []
    for index, job in indexed:
        score_info = scores.get(index, {"score": 0, "reason": "Not scored."})
        ranked.append(
            {
                **job,
                "score": score_info["score"],
                "reason": score_info["reason"],
            }
        )

    # Best fit first; stable within equal scores.
    ranked.sort(key=lambda job: job["score"], reverse=True)
    if request.top_n is not None:
        ranked = ranked[: request.top_n]

    return {"role": request.role, "count": len(ranked), "jobs": ranked}
