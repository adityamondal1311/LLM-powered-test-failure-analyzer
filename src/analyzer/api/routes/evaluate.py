from __future__ import annotations

import asyncio
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse

from analyzer.config import get_settings
from analyzer.eval.runner import run_eval
from analyzer.models.api import EvaluateRequest, EvaluateResponse

router = APIRouter(tags=["evaluate"])

# In-memory job store (sufficient for single-process dev/eval use)
_jobs: dict[str, dict[str, Any]] = {}


async def _run_job(job_id: str, dataset_path: str, limit: int | None, request: Request) -> None:
    _jobs[job_id]["status"] = "running"
    try:
        report = await run_eval(
            dataset_path=dataset_path,
            client=request.app.state.llm,
            confidence_threshold=get_settings().confidence_threshold,
            limit=limit,
        )
        _jobs[job_id].update(
            {
                "status": "completed",
                "precision": report.macro_precision,
                "recall": report.macro_recall,
                "f1": report.macro_f1,
                "accuracy": report.accuracy,
                "n_samples": report.n_total,
                "details": [r.model_dump() for r in report.case_results[:50]],
            }
        )
    except Exception as exc:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(exc)


@router.post("/evaluate", status_code=202, response_model=EvaluateResponse)
async def start_evaluate(
    body: EvaluateRequest,
    background_tasks: BackgroundTasks,
    request: Request,
) -> EvaluateResponse:
    settings = get_settings()
    dataset_path = body.dataset_path or settings.eval_dataset_path
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {"status": "queued"}
    background_tasks.add_task(_run_job, job_id, dataset_path, body.limit, request)
    return EvaluateResponse(job_id=job_id, status="queued")


@router.get("/evaluate/{job_id}", response_model=EvaluateResponse)
async def get_evaluate_result(job_id: str) -> EvaluateResponse:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return EvaluateResponse(job_id=job_id, **{k: v for k, v in job.items() if k != "error"})
