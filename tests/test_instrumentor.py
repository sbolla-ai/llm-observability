"""
test_instrumentor.py
====================
Unit tests for the LLM Observability SDK core components.

Run with:
    pytest tests/ -v --cov=src --cov-report=term-missing
"""

import pytest
from unittest.mock import MagicMock, patch, call
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry import trace
import opentelemetry.sdk.metrics as sdk_metrics

from src.collectors.llm_instrumentor import LLMObservabilitySDK, LLMSpanAttributes, SDKConfig
from src.exporters.cost_estimator import CostEstimator


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def span_exporter():
    """In-memory span exporter for test assertions."""
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(span_exporter):
    """Test TracerProvider with in-memory exporter."""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    trace.set_tracer_provider(provider)
    return provider


@pytest.fixture
def mock_sdk():
    """Minimal SDK mock for collector tests."""
    sdk = MagicMock()
    sdk.record_token_usage = MagicMock()
    return sdk


# ─── SDKConfig Tests ──────────────────────────────────────────────────────────

class TestSDKConfig:
    def test_default_values(self):
        config = SDKConfig(service_name="test-service")
        assert config.service_name == "test-service"
        assert config.service_version == "1.0.0"
        assert config.environment == "development"
        assert config.otlp_endpoint == "http://localhost:4317"
        assert config.enable_cost_tracking is True
        assert config.token_budget_alert_threshold == 0.8

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("OTEL_SERVICE_NAME", "env-service")
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317")
        monkeypatch.setenv("LLM_TOKEN_BUDGET_THRESHOLD", "0.9")

        config = SDKConfig.from_env()
        assert config.service_name == "env-service"
        assert config.environment == "production"
        assert config.otlp_endpoint == "http://collector:4317"
        assert config.token_budget_alert_threshold == 0.9

    def test_from_env_defaults(self, monkeypatch):
        """Should use defaults when env vars are not set."""
        monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
        config = SDKConfig.from_env()
        assert config.service_name == "llm-app"


# ─── CostEstimator Tests ──────────────────────────────────────────────────────

class TestCostEstimator:
    def test_anthropic_sonnet_cost(self):
        # 1M input + 0 output = $3.00
        cost = CostEstimator.estimate("claude-sonnet-4-6", 1_000_000, 0)
        assert cost == pytest.approx(3.00, rel=1e-3)

    def test_anthropic_haiku_cost(self):
        cost = CostEstimator.estimate("claude-haiku-4-5-20251001", 1_000_000, 1_000_000)
        assert cost == pytest.approx(1.50, rel=1e-3)  # 0.25 + 1.25

    def test_openai_gpt4o_cost(self):
        cost = CostEstimator.estimate("gpt-4o", 100_000, 50_000)
        assert cost == pytest.approx(0.75, rel=1e-3)  # 0.25 + 0.50

    def test_zero_tokens(self):
        cost = CostEstimator.estimate("claude-sonnet-4-6", 0, 0)
        assert cost == 0.0

    def test_unknown_model_uses_fallback(self):
        cost_unknown = CostEstimator.estimate("unknown-model-xyz", 1000, 500)
        cost_fallback = CostEstimator.estimate("unknown", 1000, 500)
        assert cost_unknown == cost_fallback

    def test_prefix_matching(self):
        """Model variants should match base model pricing."""
        cost_base = CostEstimator.estimate("claude-sonnet-4-6", 1000, 1000)
        cost_variant = CostEstimator.estimate("claude-sonnet-4-6-20240229", 1000, 1000)
        assert cost_base == cost_variant

    def test_monthly_projection(self):
        monthly = CostEstimator.monthly_budget_usd(
            daily_requests=1000,
            avg_input_tokens=500,
            avg_output_tokens=200,
            model="claude-sonnet-4-6",
        )
        # 1000 req/day * 30 days * cost_per_request
        expected_per_req = CostEstimator.estimate("claude-sonnet-4-6", 500, 200)
        assert monthly == pytest.approx(expected_per_req * 1000 * 30, rel=1e-3)

    def test_get_all_models_excludes_unknown(self):
        models = CostEstimator.get_all_models()
        assert "unknown" not in models
        assert "claude-sonnet-4-6" in models
        assert "gpt-4o" in models


# ─── LLM Span Context Manager Tests ──────────────────────────────────────────

class TestLLMSpanContextManager:
    @pytest.fixture(autouse=True)
    def setup_sdk(self, tracer_provider):
        """Set up a real SDK using the test tracer provider."""
        self.sdk = LLMObservabilitySDK.__new__(LLMObservabilitySDK)
        self.sdk.config = SDKConfig(service_name="test")
        self.sdk._tracer = tracer_provider.get_tracer("test")
        self.sdk._initialized = True
        self.sdk._llm_request_duration = MagicMock()
        self.sdk._llm_error_counter = MagicMock()
        self.sdk._llm_active_requests = MagicMock()
        self.sdk._llm_token_counter = MagicMock()
        self.sdk._llm_cost_counter = MagicMock()

    def test_span_created_with_correct_attributes(self, span_exporter):
        with self.sdk.llm_span("chat", model="claude-sonnet-4-6", system="anthropic"):
            pass

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "anthropic chat"
        assert span.attributes[LLMSpanAttributes.GEN_AI_SYSTEM] == "anthropic"
        assert span.attributes[LLMSpanAttributes.GEN_AI_REQUEST_MODEL] == "claude-sonnet-4-6"

    def test_span_records_error_on_exception(self, span_exporter):
        with pytest.raises(ValueError):
            with self.sdk.llm_span("chat", model="gpt-4o", system="openai"):
                raise ValueError("LLM call failed")

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        from opentelemetry.trace import StatusCode
        assert span.status.status_code == StatusCode.ERROR

    def test_pipeline_stage_attribute(self, span_exporter):
        with self.sdk.llm_span("embed", model="text-embedding-3-small", system="openai", pipeline_stage="retrieval"):
            pass

        spans = span_exporter.get_finished_spans()
        assert spans[0].attributes[LLMSpanAttributes.LLM_PIPELINE_STAGE] == "retrieval"

    def test_active_requests_incremented_and_decremented(self):
        with self.sdk.llm_span("chat", model="test-model", system="test"):
            self.sdk._llm_active_requests.add.assert_called_once()

        # Should be called twice: once to add, once to subtract
        assert self.sdk._llm_active_requests.add.call_count == 2


# ─── Anthropic Collector Tests ────────────────────────────────────────────────

class TestAnthropicCollector:
    def test_instrumented_client_wraps_messages(self):
        """InstrumentedAnthropic should wrap messages attribute."""
        with patch("anthropic.Anthropic") as mock_anthropic_cls:
            from src.collectors.anthropic_collector import InstrumentedAnthropic, InstrumentedMessages
            client = InstrumentedAnthropic(api_key="test-key")
            assert isinstance(client.messages, InstrumentedMessages)

    def test_successful_response_records_attributes(self, span_exporter, tracer_provider):
        """Should record token usage and finish reason on success."""
        mock_response = MagicMock()
        mock_response.model = "claude-sonnet-4-6"
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.stop_reason = "end_turn"
        mock_response.id = "msg_test123"
        mock_response.content = [MagicMock(text="Test response")]

        with patch("anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.return_value = mock_response
            from src.collectors.anthropic_collector import InstrumentedAnthropic
            client = InstrumentedAnthropic(api_key="test")
            result = client.messages.create(
                model="claude-sonnet-4-6",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=100,
            )

        spans = span_exporter.get_finished_spans()
        assert any("anthropic" in s.name for s in spans)

        llm_spans = [s for s in spans if "anthropic" in s.name]
        assert len(llm_spans) > 0
        span = llm_spans[-1]
        assert span.attributes.get(LLMSpanAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 100
        assert span.attributes.get(LLMSpanAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 50


# ─── Integration-style Tests ──────────────────────────────────────────────────

class TestEndToEnd:
    def test_cost_flows_into_span(self, span_exporter, tracer_provider):
        """Token usage recorded via SDK should attach cost to the active span."""
        sdk = LLMObservabilitySDK.__new__(LLMObservabilitySDK)
        sdk.config = SDKConfig(service_name="test", enable_cost_tracking=True)
        sdk._tracer = tracer_provider.get_tracer("test")
        sdk._initialized = True
        sdk._llm_request_duration = MagicMock()
        sdk._llm_error_counter = MagicMock()
        sdk._llm_active_requests = MagicMock()
        sdk._llm_token_counter = MagicMock()
        sdk._llm_cost_counter = MagicMock()

        with sdk.llm_span("chat", model="claude-sonnet-4-6", system="anthropic") as span:
            sdk.record_token_usage(1000, 500, "claude-sonnet-4-6", "anthropic")

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert LLMSpanAttributes.LLM_ESTIMATED_COST_USD in spans[0].attributes
