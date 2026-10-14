"""parser.py: Validates JSON schemas and parses LLM consult-style headache recommendations."""

import json
from typing import Any, Dict, List, Optional, Tuple

ALLOWED_ROOT_KEYS = {
    "primary_diagnosis",
    "differential",
    "recommended_action",
    "reasoning",
}
ALLOWED_ACTION_KEYS = {"workup", "treatment"}
ALLOWED_TREATMENT_KEYS = {"immediate_supportive", "conditional_on_workup"}


def _validate_nonempty_string(val: Any, field_name: str) -> Optional[str]:
    """Check that value is a string and contains non-whitespace characters."""
    if not isinstance(val, str):
        return f"'{field_name}' must be a string, got {type(val).__name__}."
    if not val.strip():
        return f"'{field_name}' must be a nonempty string."
    return None


def _validate_action_list(items: Any, field_name: str) -> Optional[str]:
    """Ensure action lists contain at least one nonempty string."""
    if not isinstance(items, list):
        return f"{field_name} must be a list, got {type(items).__name__}."
    if len(items) == 0:
        return f"{field_name} must contain at least one nonempty string."
    for idx, item in enumerate(items):
        if not isinstance(item, str) or not item.strip():
            return f"{field_name}[{idx}] must be a nonempty string."
    return None


def _validate_treatment_structure(treatment: Any) -> Tuple[bool, Optional[str]]:
    """Validate treatment structure (structured dict with immediate and conditional keys)."""
    if isinstance(treatment, dict):
        keys = set(treatment.keys())
        if keys != ALLOWED_TREATMENT_KEYS:
            missing = sorted(ALLOWED_TREATMENT_KEYS - keys)
            unexpected = sorted(keys - ALLOWED_TREATMENT_KEYS)
            errs = []
            if missing:
                errs.append(f"missing keys: {missing}")
            if unexpected:
                errs.append(f"unexpected keys: {unexpected}")
            return False, "recommended_action.treatment schema mismatch: " + "; ".join(errs) + "."

        for field in ("immediate_supportive", "conditional_on_workup"):
            err = _validate_action_list(treatment[field], f"recommended_action.treatment.{field}")
            if err:
                return False, err
        return True, None

    # Fallback compatibility for flat lists if needed
    if isinstance(treatment, list):
        err = _validate_action_list(treatment, "recommended_action.treatment")
        if err:
            return False, err
        return True, None

    return False, f"'recommended_action.treatment' must be an object (or list), got {type(treatment).__name__}."


def _validate_schema(data: Any) -> Tuple[bool, Optional[str]]:
    """Validate structure strictly against the clinical consult schema."""
    if not isinstance(data, dict):
        return False, f"Top-level response must be a JSON object, got {type(data).__name__}."

    actual_keys = set(data.keys())
    if actual_keys != ALLOWED_ROOT_KEYS:
        missing = sorted(ALLOWED_ROOT_KEYS - actual_keys)
        unexpected = sorted(actual_keys - ALLOWED_ROOT_KEYS)
        errs = []
        if missing:
            errs.append(f"missing required fields: {missing}")
        if unexpected:
            errs.append(f"unexpected extra fields: {unexpected}")
        return False, "Schema mismatch: " + "; ".join(errs) + "."

    # primary_diagnosis
    err = _validate_nonempty_string(data["primary_diagnosis"], "primary_diagnosis")
    if err:
        return False, err

    # reasoning
    err = _validate_nonempty_string(data["reasoning"], "reasoning")
    if err:
        return False, err

    # differential
    diff = data["differential"]
    if not isinstance(diff, list):
        return False, f"'differential' must be a list, got {type(diff).__name__}."
    for idx, item in enumerate(diff):
        if not isinstance(item, str) or not item.strip():
            return False, f"'differential[{idx}]' must be a nonempty string."

    # recommended_action
    rec_action = data["recommended_action"]
    if not isinstance(rec_action, dict):
        return False, f"'recommended_action' must be a JSON object, got {type(rec_action).__name__}."

    action_keys = set(rec_action.keys())
    if action_keys != ALLOWED_ACTION_KEYS:
        missing = sorted(ALLOWED_ACTION_KEYS - action_keys)
        unexpected = sorted(action_keys - ALLOWED_ACTION_KEYS)
        errs = []
        if missing:
            errs.append(f"missing keys: {missing}")
        if unexpected:
            errs.append(f"unexpected keys: {unexpected}")
        return False, "recommended_action schema mismatch: " + "; ".join(errs) + "."

    # recommended_action.workup
    err = _validate_action_list(rec_action["workup"], "recommended_action.workup")
    if err:
        return False, err

    # recommended_action.treatment
    is_valid, err = _validate_treatment_structure(rec_action["treatment"])
    if not is_valid:
        return False, err

    return True, None


def parse_response(runner_record: Dict[str, Any]) -> Dict[str, Any]:
    """Parse runner output record into a structured evaluation payload."""
    case_id = runner_record.get("case_id")
    run_id = runner_record.get("run_id")
    raw_text = runner_record.get("raw_response_text")
    api_status = runner_record.get("api_status", {})
    runner_stat = api_status.get("status")

    parsed_envelope: Dict[str, Any] = {
        "run_id": run_id,
        "case_id": case_id,
        "model": runner_record.get("model"),
        "timestamp": runner_record.get("timestamp"),
        "status": None,
        "failure_reason": None,
        "error_message": None,
        "raw_response_text": raw_text,
        "parsed_response": None,
    }

    if runner_stat != "success":
        parsed_envelope["status"] = "failed"
        parsed_envelope["failure_reason"] = runner_stat
        parsed_envelope["error_message"] = api_status.get("error_message")
        return parsed_envelope

    try:
        data = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError) as exc:
        parsed_envelope["status"] = "failed"
        parsed_envelope["failure_reason"] = "invalid_json"
        parsed_envelope["error_message"] = f"Failed to decode JSON: {str(exc)}"
        return parsed_envelope

    is_valid, validation_err = _validate_schema(data)
    if not is_valid:
        parsed_envelope["status"] = "failed"
        parsed_envelope["failure_reason"] = "schema_validation_error"
        parsed_envelope["error_message"] = validation_err
        return parsed_envelope

    parsed_envelope["status"] = "success"
    parsed_envelope["parsed_response"] = data
    return parsed_envelope