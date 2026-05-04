from __future__ import annotations

from pydantic import BaseModel, Field

from analyzer.models.pipeline import FailureCategory


class LabeledCase(BaseModel):
    """A single labeled test failure case in the eval dataset."""

    id: str
    raw_log: str
    ground_truth_category: FailureCategory
    ground_truth_is_flaky: bool = False
    difficulty: str = Field(default="medium", pattern="^(easy|medium|hard)$")
    notes: str = ""


class EvalDataset(BaseModel):
    version: str
    description: str
    created: str
    cases: list[LabeledCase]


class CaseResult(BaseModel):
    """Eval result for a single case."""

    case_id: str
    ground_truth: FailureCategory
    predicted: FailureCategory
    correct: bool
    confidence: float
    fallback_used: bool
    latency_ms: float
    difficulty: str


class CategoryMetrics(BaseModel):
    category: FailureCategory
    precision: float
    recall: float
    f1: float
    support: int


class EvalReport(BaseModel):
    """Full report produced by the eval runner."""

    n_total: int
    n_correct: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_category: list[CategoryMetrics]
    case_results: list[CaseResult]
    avg_latency_ms: float
    fallback_rate: float
