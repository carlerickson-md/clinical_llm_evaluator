"""test_phase2.py: Validates runner and parser functionality using mocks."""

import json
from unittest.mock import MagicMock
import pytest
from openai import OpenAIError

from src.runner import run_prompt
from src.parser import parse_response, _validate_schema


def _make_mock_client(content: str, finish_reason: str = "stop", refusal: str = None):
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.finish_reason = finish_reason
    mock_choice.message.content = content
    mock_choice.message.refusal = refusal
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def valid_response_json():
    return json.dumps({
        "primary_diagnosis": "Cluster Headache",
        "differential": ["Migraine without aura", "Trigeminal Neuralgia"],
        "recommended_action": {
            "workup": ["Brain MRI with and without contrast to rule out structural pathology."],
            "treatment": ["100% Oxygen via non-rebreather mask at 12 L/min for acute abortive therapy."]
        },
        "reasoning": "Unilateral severe orbital pain associated with ipsilateral autonomic features fits cluster headache."
    })


def test_valid_response(valid_response_json):
    client = _make_mock_client(valid_response_json)
    runner_rec = run_prompt("Clinical prompt text", case_id="HA_001", client=client)
    assert runner_rec["api_status"]["status"] == "success"

    result = parse_response(runner_rec)
    assert result["status"] == "success"
    assert result["parsed_response"]["primary_diagnosis"] == "Cluster Headache"
    assert len(result["parsed_response"]["recommended_action"]["workup"]) == 1


def test_malformed_json():
    client = _make_mock_client("{\"primary_diagnosis\": \"Migraine\", unclosed...")
    runner_rec = run_prompt("Prompt", case_id="HA_001", client=client)
    result = parse_response(runner_rec)
    
    assert result["status"] == "failed"
    assert result["failure_reason"] == "invalid_json"


def test_missing_fields():
    # Omitting 'differential'
    bad_data = {
        "primary_diagnosis": "Migraine",
        "recommended_action": {"workup": ["None"], "treatment": ["Sumatriptan"]},
        "reasoning": "Classic migraine presentation."
    }
    client = _make_mock_client(json.dumps(bad_data))
    runner_rec = run_prompt("Prompt", case_id="HA_001", client=client)
    result = parse_response(runner_rec)

    assert result["status"] == "failed"
    assert result["failure_reason"] == "schema_validation_error"
    assert "missing required fields" in result["error_message"]


def test_wrong_types():
    # primary_diagnosis as an integer
    bad_data = {
        "primary_diagnosis": 12345,
        "differential": [],
        "recommended_action": {"workup": ["MRI"], "treatment": ["Rest"]},
        "reasoning": "Presentation is clear."
    }
    client = _make_mock_client(json.dumps(bad_data))
    runner_rec = run_prompt("Prompt", case_id="HA_001", client=client)
    result = parse_response(runner_rec)

    assert result["status"] == "failed"
    assert result["failure_reason"] == "schema_validation_error"
    assert "'primary_diagnosis' must be a string" in result["error_message"]


def test_empty_action_lists():
    # Empty workup list
    bad_data = {
        "primary_diagnosis": "Tension-type Headache",
        "differential": [],
        "recommended_action": {
            "workup": [],
            "treatment": ["NSAIDs"]
        },
        "reasoning": "Bilateral pressing headache without red flags."
    }
    client = _make_mock_client(json.dumps(bad_data))
    runner_rec = run_prompt("Prompt", case_id="HA_001", client=client)
    result = parse_response(runner_rec)

    assert result["status"] == "failed"
    assert result["failure_reason"] == "schema_validation_error"
    assert "recommended_action.workup must contain at least one nonempty string" in result["error_message"]


def test_model_refusal():
    client = _make_mock_client(content="", finish_reason="stop", refusal="I cannot give medical advice.")
    runner_rec = run_prompt("Prompt", case_id="HA_001", client=client)
    assert runner_rec["api_status"]["status"] == "refusal"

    result = parse_response(runner_rec)
    assert result["status"] == "failed"
    assert result["failure_reason"] == "refusal"


def test_incomplete_generation():
    client = _make_mock_client(content="{\"primary_diag", finish_reason="length")
    runner_rec = run_prompt("Prompt", case_id="HA_001", client=client)
    assert runner_rec["api_status"]["status"] == "incomplete_generation"

    result = parse_response(runner_rec)
    assert result["status"] == "failed"
    assert result["failure_reason"] == "incomplete_generation"


def test_api_failure():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = OpenAIError("Service Unavailable (503)")
    runner_rec = run_prompt("Prompt", case_id="HA_001", client=mock_client)
    assert runner_rec["api_status"]["status"] == "api_error"

    result = parse_response(runner_rec)
    assert result["status"] == "failed"
    assert result["failure_reason"] == "api_error"


def test_leakage_boundary():
    """Verify ground truth or benchmark fields never reach parser logic."""
    full_case_json = {
        "case_id": "HA_001",
        "case": {"presentation": {}},
        "ground_truth": {"primary_diagnosis": "Cluster Headache"},
        "evaluation": {"acceptable_diagnoses": ["Cluster Headache"]}
    }

    # Runner only takes prompt string + case_id
    runner_rec = run_prompt(
        prompt="Safe Prompt",
        case_id=full_case_json["case_id"],
        client=_make_mock_client("{\"primary_diagnosis\": \"Test\"}")
    )
    
    # Assert neither ground_truth nor evaluation exist in runner_rec
    assert "ground_truth" not in runner_rec
    assert "evaluation" not in runner_rec

    # Parser receives only runner_rec
    parsed = parse_response(runner_rec)
    assert "ground_truth" not in parsed
    assert "evaluation" not in parsed