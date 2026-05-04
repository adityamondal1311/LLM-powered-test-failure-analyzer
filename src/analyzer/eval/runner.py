from __future__ import annotations

import json
from pathlib import Path

from analyzer.eval.metrics import compute_report
from analyzer.llm.client import LLMClient
from analyzer.models.eval import CaseResult, EvalDataset, EvalReport
from analyzer.pipeline.inference import run_inference
from analyzer.pipeline.ingestion import ingest_log
from analyzer.pipeline.scoring import score_result
from analyzer.pipeline.validation import validate_result


async def run_eval(
    dataset_path: str,
    client: LLMClient,
    confidence_threshold: float = 0.65,
    limit: int | None = None,
) -> EvalReport:
    """Run the evaluation pipeline over a labeled dataset and return a full report."""
    raw = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    dataset = EvalDataset.model_validate(raw)

    cases = dataset.cases
    if limit is not None:
        cases = cases[:limit]

    case_results: list[CaseResult] = []

    for case in cases:
        failure = await ingest_log(case.raw_log, test_id=case.id)
        inference = await run_inference(failure, client, confidence_threshold)
        validation = validate_result(inference, confidence_threshold)
        scored = score_result(validation)

        predicted = inference.hypothesis.category
        correct = predicted == case.ground_truth_category

        case_results.append(
            CaseResult(
                case_id=case.id,
                ground_truth=case.ground_truth_category,
                predicted=predicted,
                correct=correct,
                confidence=scored.final_confidence,
                fallback_used=inference.fallback_used,
                latency_ms=inference.latency_ms,
                difficulty=case.difficulty,
            )
        )

    return compute_report(case_results)
