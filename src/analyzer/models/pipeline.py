from __future__ import annotations

import time
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class FailureCategory(StrEnum):
    ASSERTION_ERROR = "assertion_error"
    IMPORT_ERROR = "import_error"
    FIXTURE_ERROR = "fixture_error"
    TIMEOUT = "timeout"
    ENVIRONMENT_ERROR = "environment_error"
    NETWORK_ERROR = "network_error"
    DATA_ERROR = "data_error"
    FLAKY = "flaky"
    UNKNOWN = "unknown"


class FallbackSource(StrEnum):
    LLM = "llm"
    HEURISTIC = "heuristic"
    CACHED = "cached"


class ParsedFailure(BaseModel):
    """Normalized pytest failure log — output of the ingestion stage."""

    test_id: str
    test_file: str
    test_function: str
    error_type: str
    error_message: str
    traceback_lines: list[str]
    duration_ms: float
    collected_at_utc: float = Field(default_factory=time.time)
    raw_log: str
    token_estimate: int = 0


class RootCauseHypothesis(BaseModel):
    """Schema-constrained LLM output — used as the tool-use input_schema."""

    category: FailureCategory
    summary: Annotated[str, Field(max_length=300)]
    explanation: Annotated[str, Field(max_length=1000)]
    fix_hint: Annotated[str, Field(max_length=500)]
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    is_flaky: bool

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 4)


class InferenceResult(BaseModel):
    """Output of the inference stage — before downstream validation."""

    parsed_failure: ParsedFailure
    hypothesis: RootCauseHypothesis
    model_id: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cache_hit_tokens: int
    fallback_used: bool
    fallback_source: FallbackSource


class ValidationResult(BaseModel):
    """Output of the validation stage."""

    inference: InferenceResult
    schema_valid: bool
    confidence_passed: bool
    issues: list[str] = []


class ScoredResult(BaseModel):
    """Output of the scoring stage."""

    validation: ValidationResult
    final_confidence: float
    rank_score: float
    actionable: bool
    routed_to: FallbackSource


class StoredRecord(BaseModel):
    """Output of the storage stage."""

    record_id: str
    scored_result: ScoredResult
    stored_at_utc: float
    db_path: str
