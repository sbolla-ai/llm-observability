"""
cost_estimator.py
=================
Estimates LLM API costs based on token counts and model pricing.

Principal SRE Note:
    Cost visibility is critical for AI workloads. This module enables:
    1. Per-request cost attribution in traces
    2. Aggregate cost counters in Prometheus
    3. Budget utilization alerting

    Update pricing in PRICING_TABLE when vendors change rates.
    All prices are USD per 1,000,000 tokens (per-million pricing).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    input_per_million: float   # USD per 1M input tokens
    output_per_million: float  # USD per 1M output tokens


# Pricing table — update as vendors change rates
# Source: vendor pricing pages (verify before using in production billing)
PRICING_TABLE: dict[str, ModelPricing] = {
    # Anthropic Claude
    "claude-opus-4-6": ModelPricing(15.00, 75.00),
    "claude-sonnet-4-6": ModelPricing(3.00, 15.00),
    "claude-haiku-4-5-20251001": ModelPricing(0.25, 1.25),
    "claude-3-opus-20240229": ModelPricing(15.00, 75.00),
    "claude-3-5-sonnet-20241022": ModelPricing(3.00, 15.00),
    "claude-3-haiku-20240307": ModelPricing(0.25, 1.25),

    # OpenAI
    "gpt-4o": ModelPricing(2.50, 10.00),
    "gpt-4o-mini": ModelPricing(0.15, 0.60),
    "gpt-4-turbo": ModelPricing(10.00, 30.00),
    "gpt-3.5-turbo": ModelPricing(0.50, 1.50),

    # Meta / Open Source (via hosted APIs)
    "llama-3.1-70b-instruct": ModelPricing(0.88, 0.88),
    "llama-3.1-405b-instruct": ModelPricing(5.00, 5.00),

    # Google
    "gemini-1.5-pro": ModelPricing(3.50, 10.50),
    "gemini-1.5-flash": ModelPricing(0.075, 0.30),

    # Fallback for unknown models
    "unknown": ModelPricing(5.00, 15.00),
}


class CostEstimator:
    """
    Estimates LLM API call costs from token counts.

    Example:
        cost = CostEstimator.estimate("claude-sonnet-4-6", 1000, 500)
        # Returns 0.0105 (USD)
    """

    @staticmethod
    def estimate(model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost in USD for a single LLM API call.

        Args:
            model: Model identifier string (matched against PRICING_TABLE)
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens

        Returns:
            Estimated cost in USD (float, rounded to 8 decimal places)
        """
        pricing = CostEstimator._lookup_pricing(model)
        input_cost = (input_tokens / 1_000_000) * pricing.input_per_million
        output_cost = (output_tokens / 1_000_000) * pricing.output_per_million
        return round(input_cost + output_cost, 8)

    @staticmethod
    def _lookup_pricing(model: str) -> ModelPricing:
        """Find pricing by exact match, then prefix match, then fallback."""
        # Exact match
        if model in PRICING_TABLE:
            return PRICING_TABLE[model]

        # Prefix match (e.g., "claude-sonnet-4-6-20240229" → "claude-sonnet-4-6")
        for key in PRICING_TABLE:
            if model.startswith(key) or key.startswith(model.split("-")[0]):
                return PRICING_TABLE[key]

        # Fallback
        return PRICING_TABLE["unknown"]

    @staticmethod
    def monthly_budget_usd(
        daily_requests: int,
        avg_input_tokens: int,
        avg_output_tokens: int,
        model: str,
    ) -> float:
        """
        Project monthly cost for capacity planning.

        Example:
            budget = CostEstimator.monthly_budget_usd(
                daily_requests=10_000,
                avg_input_tokens=500,
                avg_output_tokens=200,
                model="claude-sonnet-4-6"
            )
        """
        cost_per_request = CostEstimator.estimate(model, avg_input_tokens, avg_output_tokens)
        return round(cost_per_request * daily_requests * 30, 2)

    @staticmethod
    def get_all_models() -> list[str]:
        """Return all supported model identifiers."""
        return [k for k in PRICING_TABLE if k != "unknown"]
