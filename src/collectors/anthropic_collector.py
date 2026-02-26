"""
anthropic_collector.py
======================
Drop-in replacement for the Anthropic client with full OTel instrumentation.

Principal SRE Note:
    This wraps the Anthropic SDK so existing application code only needs
    a one-line change: replace `anthropic.Anthropic()` with `InstrumentedAnthropic()`.
    All telemetry is captured transparently.

Usage:
    # Before (uninstrumented)
    import anthropic
    client = anthropic.Anthropic()

    # After (fully instrumented)
    from src.collectors.anthropic_collector import InstrumentedAnthropic
    client = InstrumentedAnthropic()

    # Usage is identical
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}]
    )
"""

import logging
import time
from functools import wraps
from typing import Any, Optional

import anthropic
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from src.collectors.llm_instrumentor import LLMSpanAttributes

logger = logging.getLogger(__name__)

# Singleton SDK reference — set by calling setup_sdk()
_sdk = None


def setup_sdk(sdk) -> None:
    """Register the global SDK instance for this collector."""
    global _sdk
    _sdk = sdk


class InstrumentedMessages:
    """
    Wraps anthropic.resources.Messages to inject OTel spans around each call.
    Supports both synchronous and streaming completions.
    """

    def __init__(self, original_messages, sdk=None):
        self._messages = original_messages
        self._sdk = sdk or _sdk

    def create(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 1024,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **kwargs,
    ) -> Any:
        """
        Instrumented wrapper around client.messages.create().

        Captures:
        - Full request/response trace span
        - Input/output token counts
        - Estimated USD cost
        - Finish reason
        - Model used (may differ from requested for auto-routing)
        """
        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span("anthropic chat") as span:
            # Set request attributes
            span.set_attribute(LLMSpanAttributes.GEN_AI_SYSTEM, "anthropic")
            span.set_attribute(LLMSpanAttributes.GEN_AI_REQUEST_MODEL, model)
            span.set_attribute(LLMSpanAttributes.GEN_AI_REQUEST_MAX_TOKENS, max_tokens)
            span.set_attribute("gen_ai.request.message_count", len(messages))
            if temperature is not None:
                span.set_attribute(LLMSpanAttributes.GEN_AI_REQUEST_TEMPERATURE, temperature)

            start = time.perf_counter()
            try:
                call_kwargs = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    **kwargs,
                }
                if system:
                    call_kwargs["system"] = system
                if temperature is not None:
                    call_kwargs["temperature"] = temperature
                if stream:
                    call_kwargs["stream"] = stream

                response = self._messages.create(**call_kwargs)
                duration = time.perf_counter() - start

                if not stream:
                    self._record_response_attributes(span, response, duration)

                span.set_status(Status(StatusCode.OK))
                return response

            except anthropic.RateLimitError as exc:
                self._record_error(span, exc, "rate_limit")
                raise
            except anthropic.APITimeoutError as exc:
                self._record_error(span, exc, "timeout")
                raise
            except anthropic.APIConnectionError as exc:
                self._record_error(span, exc, "connection_error")
                raise
            except anthropic.APIStatusError as exc:
                self._record_error(span, exc, f"api_error_{exc.status_code}")
                raise

    def _record_response_attributes(
        self,
        span: trace.Span,
        response: anthropic.types.Message,
        duration: float,
    ) -> None:
        """Extract and record all relevant response attributes."""
        span.set_attribute(LLMSpanAttributes.GEN_AI_RESPONSE_MODEL, response.model)
        span.set_attribute(
            LLMSpanAttributes.GEN_AI_RESPONSE_FINISH_REASON,
            response.stop_reason or "unknown"
        )
        span.set_attribute(LLMSpanAttributes.GEN_AI_USAGE_INPUT_TOKENS, response.usage.input_tokens)
        span.set_attribute(LLMSpanAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, response.usage.output_tokens)
        span.set_attribute("llm.response.duration_seconds", round(duration, 4))
        span.set_attribute("llm.response.id", response.id)

        # Record metrics via SDK
        if self._sdk:
            self._sdk.record_token_usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                model=response.model,
                system="anthropic",
            )

        logger.info(
            "Anthropic API call completed",
            extra={
                "model": response.model,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "duration_seconds": round(duration, 4),
                "finish_reason": response.stop_reason,
            }
        )

    def _record_error(self, span: trace.Span, exc: Exception, error_type: str) -> None:
        """Record error details on the span."""
        span.set_status(Status(StatusCode.ERROR, str(exc)))
        span.record_exception(exc)
        span.set_attribute("error.type", error_type)
        logger.error(
            "Anthropic API call failed",
            extra={"error_type": error_type, "error": str(exc)},
            exc_info=True,
        )


class InstrumentedAnthropic:
    """
    Instrumented drop-in replacement for anthropic.Anthropic().

    Example:
        client = InstrumentedAnthropic()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": "What is SRE?"}]
        )
    """

    def __init__(self, sdk=None, **anthropic_kwargs):
        self._client = anthropic.Anthropic(**anthropic_kwargs)
        self.messages = InstrumentedMessages(self._client.messages, sdk=sdk)

    def __getattr__(self, name: str) -> Any:
        """Proxy any other attributes to the underlying client."""
        return getattr(self._client, name)
