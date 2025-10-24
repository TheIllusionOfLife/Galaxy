"""Gemini 2.5 Flash Lite API client with rate limiting and cost tracking.

This module provides a client for the Gemini API optimized for the free tier
(15 RPM, 1000 RPD) with comprehensive cost tracking and error handling.
"""

import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured response from LLM API call.

    Attributes:
        code: Extracted Python code from response
        raw_response: Full text response from LLM
        tokens_used: Total tokens (prompt + completion)
        cost_usd: Estimated cost in USD
        model: Model name used
        success: Whether call succeeded
        error: Error message if failed
        generation_time_s: Time taken for API call
    """
    code: str
    raw_response: str
    tokens_used: int
    cost_usd: float
    model: str
    success: bool
    error: Optional[str] = None
    generation_time_s: float = 0.0


class RateLimiter:
    """Simple rate limiter for Gemini free tier (15 RPM).

    Enforces minimum interval between requests to stay within rate limits.
    """

    def __init__(self, requests_per_minute: int = 15):
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute
        """
        self.rpm = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute  # seconds between requests
        self.last_request_time = 0.0

    def wait_if_needed(self):
        """Block if we're going too fast to maintain rate limit."""
        now = time.time()
        time_since_last = now - self.last_request_time

        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self.last_request_time = time.time()


class GeminiClient:
    """Client for Gemini 2.5 Flash Lite API.

    Handles code generation with automatic retries, rate limiting,
    and cost tracking.
    """

    # Pricing per 1M tokens (as of 2025-10-24)
    INPUT_COST_PER_MTOK = 0.10   # $0.10/1M input tokens
    OUTPUT_COST_PER_MTOK = 0.40  # $0.40/1M output tokens

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash-lite",
        temperature: float = 0.8,
        max_output_tokens: int = 2000,
        enable_rate_limiting: bool = True
    ):
        """Initialize Gemini client.

        Args:
            api_key: Google AI API key
            model: Model name to use
            temperature: Sampling temperature (0.0-2.0)
            max_output_tokens: Maximum tokens in response
            enable_rate_limiting: Whether to enforce 15 RPM limit
        """
        genai.configure(api_key=api_key)

        # Safety settings: allow code generation
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        self.model = genai.GenerativeModel(
            model_name=model,
            safety_settings=safety_settings,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            }
        )

        self.model_name = model
        self.rate_limiter = RateLimiter() if enable_rate_limiting else None

    def generate_surrogate_code(
        self,
        prompt: str,
        retry_attempts: int = 3
    ) -> LLMResponse:
        """Generate surrogate model code with retry logic.

        Args:
            prompt: Prompt for code generation
            retry_attempts: Number of retry attempts on failure

        Returns:
            LLMResponse with generated code and metadata
        """
        if self.rate_limiter:
            self.rate_limiter.wait_if_needed()

        start_time = time.time()

        for attempt in range(retry_attempts):
            try:
                response = self.model.generate_content(prompt)

                # Extract code from response
                code = self._extract_code(response.text)

                # Calculate tokens and cost
                prompt_tokens = response.usage_metadata.prompt_token_count
                completion_tokens = response.usage_metadata.candidates_token_count
                total_tokens = prompt_tokens + completion_tokens

                cost = self._calculate_cost(prompt_tokens, completion_tokens)

                generation_time = time.time() - start_time

                logger.info(
                    f"Generated code: {total_tokens} tokens, "
                    f"${cost:.6f}, {generation_time:.2f}s"
                )

                return LLMResponse(
                    code=code,
                    raw_response=response.text,
                    tokens_used=total_tokens,
                    cost_usd=cost,
                    model=self.model_name,
                    success=True,
                    generation_time_s=generation_time
                )

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{retry_attempts} failed: {e}")
                if attempt < retry_attempts - 1:
                    sleep_time = 2 ** attempt  # Exponential backoff
                    logger.debug(f"Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    generation_time = time.time() - start_time
                    logger.error(f"All retry attempts failed: {e}")
                    return LLMResponse(
                        code="",
                        raw_response="",
                        tokens_used=0,
                        cost_usd=0.0,
                        model=self.model_name,
                        success=False,
                        error=str(e),
                        generation_time_s=generation_time
                    )

    def _extract_code(self, text: str) -> str:
        """Extract Python code from markdown code blocks or raw text.

        Args:
            text: Raw LLM response text

        Returns:
            Extracted Python code
        """
        # Try markdown code block with python language tag
        if "```python" in text:
            start = text.find("```python") + 9
            end = text.find("```", start)
            return text[start:end].strip()

        # Try generic markdown code block
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            return text[start:end].strip()

        # Assume raw code
        else:
            return text.strip()

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate API cost in USD.

        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens

        Returns:
            Total cost in USD
        """
        input_cost = (prompt_tokens / 1_000_000) * self.INPUT_COST_PER_MTOK
        output_cost = (completion_tokens / 1_000_000) * self.OUTPUT_COST_PER_MTOK
        return input_cost + output_cost


class CostTracker:
    """Track API costs across evolution run.

    Monitors total cost and individual API calls to enforce budgets
    and provide usage summaries.
    """

    def __init__(self, max_cost_usd: float = 1.0):
        """Initialize cost tracker.

        Args:
            max_cost_usd: Maximum allowed cost per run
        """
        self.max_cost_usd = max_cost_usd
        self.total_cost = 0.0
        self.calls: list[Dict[str, Any]] = []

    def add_call(
        self,
        response: LLMResponse,
        context: Optional[Dict[str, Any]] = None
    ):
        """Record an API call.

        Args:
            response: LLM response to record
            context: Optional context (generation, type, etc.)
        """
        self.total_cost += response.cost_usd
        self.calls.append({
            "tokens": response.tokens_used,
            "cost_usd": response.cost_usd,
            "success": response.success,
            "generation_time_s": response.generation_time_s,
            "error": response.error,
            "context": context or {}
        })

        logger.debug(
            f"Call recorded: ${response.cost_usd:.6f}, "
            f"total: ${self.total_cost:.6f}/{self.max_cost_usd:.2f}"
        )

    def check_budget_exceeded(self) -> bool:
        """Check if we've exceeded the budget.

        Returns:
            True if budget exceeded
        """
        exceeded = self.total_cost >= self.max_cost_usd
        if exceeded:
            logger.warning(
                f"Budget exceeded: ${self.total_cost:.4f} >= ${self.max_cost_usd:.2f}"
            )
        return exceeded

    def get_summary(self) -> Dict[str, Any]:
        """Get cost summary statistics.

        Returns:
            Dictionary with cost statistics
        """
        successful_calls = [c for c in self.calls if c["success"]]
        failed_calls = [c for c in self.calls if not c["success"]]

        return {
            "total_calls": len(self.calls),
            "successful_calls": len(successful_calls),
            "failed_calls": len(failed_calls),
            "total_cost_usd": self.total_cost,
            "avg_cost_per_call": (
                self.total_cost / len(self.calls) if self.calls else 0
            ),
            "total_tokens": sum(c["tokens"] for c in self.calls),
            "total_time_s": sum(c["generation_time_s"] for c in self.calls),
            "budget_remaining_usd": max(0, self.max_cost_usd - self.total_cost),
            "budget_used_percent": (
                (self.total_cost / self.max_cost_usd * 100) if self.max_cost_usd > 0 else 0
            )
        }
