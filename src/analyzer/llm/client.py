from __future__ import annotations

import asyncio
import time
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from analyzer.config import Settings
from analyzer.llm.prompts import SYSTEM_PROMPT, build_user_message
from analyzer.models.pipeline import ParsedFailure, RootCauseHypothesis


class LLMClient:
    """Wraps AsyncAnthropic with prompt caching, tool-use structured output,
    exponential backoff retry, and hard per-request timeout.
    """

    def __init__(self, settings: Settings) -> None:
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model = settings.model_id
        self._timeout = settings.llm_timeout_ms / 1000.0
        self._max_retries = settings.llm_max_retries

    def _tool_definition(self) -> dict[str, Any]:
        schema = RootCauseHypothesis.model_json_schema()
        # Remove title from schema — Anthropic API doesn't need it
        schema.pop("title", None)
        return {
            "name": "classify_test_failure",
            "description": "Classify a pytest test failure and provide root cause analysis.",
            "input_schema": schema,
        }

    async def classify_failure(
        self, failure: ParsedFailure
    ) -> tuple[RootCauseHypothesis, dict[str, int]]:
        """Returns (hypothesis, usage_dict).

        usage_dict keys: input_tokens, output_tokens, cache_hit_tokens, cache_write_tokens
        Raises the last exception if all retries fail.
        """
        attempt = 0
        delay = 0.5
        last_exc: Exception | None = None

        while attempt < self._max_retries:
            try:
                return await asyncio.wait_for(
                    self._call_api(failure),
                    timeout=self._timeout,
                )
            except anthropic.RateLimitError as exc:
                last_exc = exc
                await asyncio.sleep(delay)
                delay = min(delay * 2, 8.0)
                attempt += 1
            except anthropic.InternalServerError as exc:
                last_exc = exc
                await asyncio.sleep(delay)
                delay = min(delay * 2, 8.0)
                attempt += 1
            except asyncio.TimeoutError as exc:
                last_exc = exc
                attempt += 1  # no sleep on timeout

        assert last_exc is not None
        raise last_exc

    async def _call_api(
        self, failure: ParsedFailure
    ) -> tuple[RootCauseHypothesis, dict[str, int]]:
        start = time.perf_counter()
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=512,
            # Frozen system prompt — eligible for prompt caching
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[self._tool_definition()],
            tool_choice={"type": "tool", "name": "classify_test_failure"},
            messages=[
                {
                    "role": "user",
                    "content": build_user_message(failure),
                }
            ],
        )
        elapsed = (time.perf_counter() - start) * 1000

        tool_block = next(
            (b for b in response.content if b.type == "tool_use"),
            None,
        )
        if tool_block is None:
            raise ValueError("No tool_use block in LLM response")

        hypothesis = RootCauseHypothesis.model_validate(tool_block.input)
        usage = response.usage

        return hypothesis, {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_hit_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
            "cache_write_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
            "latency_ms": int(elapsed),
        }
