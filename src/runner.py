"""runner.py: Dispatches clinical prompts to LLMs and records standardized responses."""

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from openai import OpenAI, OpenAIError


def run_prompt(
    prompt: str,
    case_id: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    client: Optional[OpenAI] = None,
) -> Dict[str, Any]:
    """Execute a single model request with explicit retry controls and status tracking.

    Strict boundary: This function accepts only prompt text and case_id.
    Clinician-only benchmarks (ground_truth, evaluation) must never be passed.
    """
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    timestamp = datetime.now(timezone.utc).isoformat()

    request_settings = {
        "model": model,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }

    base_record: Dict[str, Any] = {
        "run_id": run_id,
        "case_id": case_id,
        "timestamp": timestamp,
        "model": model,
        "request_settings": request_settings,
        "prompt": prompt,
        "raw_response_text": None,
        "api_status": {
            "status": "pending",
            "finish_reason": None,
            "error_type": None,
            "error_message": None,
        },
    }

    # Initialize client with 0 automatic retries for transparent evaluation
    if client is None:
        key = (
            api_key
            or os.environ.get("CLINICAL_LLM_EVALUATOR_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not key:
            base_record["api_status"]["status"] = "api_error"
            base_record["api_status"]["error_type"] = "ConfigurationError"
            base_record["api_status"]["error_message"] = (
                "Neither CLINICAL_LLM_EVALUATOR_API_KEY nor OPENAI_API_KEY "
                "was found in environment or arguments."
            )
            return base_record
        client = OpenAI(api_key=key, max_retries=0)

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
    except OpenAIError as exc:
        base_record["api_status"]["status"] = "api_error"
        base_record["api_status"]["error_type"] = type(exc).__name__
        base_record["api_status"]["error_message"] = str(exc)
        return base_record
    except Exception as exc:
        base_record["api_status"]["status"] = "api_error"
        base_record["api_status"]["error_type"] = "UnexpectedException"
        base_record["api_status"]["error_message"] = str(exc)
        return base_record

    choice = response.choices[0]
    message = choice.message

    # Detect API-level model refusals
    if getattr(message, "refusal", None):
        base_record["raw_response_text"] = message.refusal
        base_record["api_status"]["status"] = "refusal"
        base_record["api_status"]["finish_reason"] = choice.finish_reason
        base_record["api_status"]["error_message"] = "Model generated a formal refusal."
        return base_record

    raw_text = message.content
    base_record["raw_response_text"] = raw_text
    finish_reason = choice.finish_reason
    base_record["api_status"]["finish_reason"] = finish_reason

    # Validate output completeness
    if finish_reason != "stop":
        base_record["api_status"]["status"] = "incomplete_generation"
        base_record["api_status"]["error_message"] = (
            f"Generation stopped prematurely with finish_reason: '{finish_reason}'."
        )
        return base_record

    if not raw_text or not raw_text.strip():
        base_record["api_status"]["status"] = "empty_response"
        base_record["api_status"]["error_message"] = "Model returned empty content."
        return base_record

    base_record["api_status"]["status"] = "success"
    return base_record