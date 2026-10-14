"""test_prompt_builder.py: Verification tests using synthetic case data."""

import json
import subprocess
import sys
from pathlib import Path
from src.prompt_builder import build_headache_prompt

# SYNTHETIC TEST RECORD 1: Populated values with explicit negative statements
SYNTHETIC_POPULATED_RECORD = {
    "case_id": "SYNTHETIC_001",
    "schema_version": "0.1",
    "domain": "headache",
    "case": {
        "demographics": {"age": 42, "sex": "Female"},
        "presentation": {
            "chief_complaint": "Severe unilateral throbbing headache",
            "history_of_present_illness": "Onset 6 hours ago with photophobia. No aura reported."
        },
        "history": {
            "medical_history": ["Migraine without aura", "No history of hypertension"],
            "medications": ["No daily medications", "Sumatriptan 50mg PRN"],
            "relevant_context": ["Mother had similar episodic headaches"]
        },
        "exam": {
            "vital_signs": {"BP": "122/78", "HR": 72, "Temp": "98.6 F"},
            "general": "Alert, in moderate distress from ambient light.",
            "neurologic": "Cranial nerves II-XII intact. No focal motor deficits."
        },
        "studies": {
            "laboratory": [],
            "imaging": ["Non-contrast head CT: Unremarkable"],
            "other": []
        }
    },
    "ground_truth": {
        "primary_diagnosis": "SENTINEL_GROUND_TRUTH_MIGRAINE_DIAGNOSIS",
        "acceptable_differential": ["SENTINEL_CLUSTER_HEADACHE"],
        "dangerous_misses": ["SENTINEL_SUBARACHNOID_HEMORRHAGE"],
        "key_evidence": ["Unilateral throbbing", "Photophobia"],
        "distractors": []
    },
    "evaluation": {
        "required_actions": ["SENTINEL_ADMINISTER_TRIPTAN"],
        "contraindicated_actions": ["SENTINEL_DO_NOT_TAP_SPINE"],
        "acceptable_actions": []
    },
    "metadata": {
        "difficulty": "moderate",
        "tags": ["migraine", "synthetic"]
    }
}

# SYNTHETIC TEST RECORD 2: Minimal case with null, blank, and empty values
SYNTHETIC_EMPTY_RECORD = {
    "case_id": "SYNTHETIC_002",
    "schema_version": "0.1",
    "domain": "headache",
    "case": {
        "demographics": {"age": None, "sex": ""},
        "presentation": {"chief_complaint": "", "history_of_present_illness": ""},
        "history": {
            "medical_history": [],
            "medications": [],
            "relevant_context": []
        },
        "exam": {
            "vital_signs": {},
            "general": "",
            "neurologic": ""
        },
        "studies": {
            "laboratory": [],
            "imaging": [],
            "other": []
        }
    },
    "ground_truth": {"primary_diagnosis": "", "acceptable_differential": []},
    "evaluation": {"required_actions": []},
    "metadata": {"difficulty": "unspecified"}
}


def test_populated_preserves_facts_and_negatives():
    prompt = build_headache_prompt(SYNTHETIC_POPULATED_RECORD["case"])
    
    # Preserves explicit facts
    assert "Age: 42" in prompt
    assert "Sex: Female" in prompt
    assert "Severe unilateral throbbing headache" in prompt
    
    # Preserves explicit clinical negatives without altering them
    assert "No aura reported." in prompt
    assert "No history of hypertension" in prompt
    assert "No daily medications" in prompt
    
    # Empty lists in studies become "Not provided."
    assert "Laboratory: Not provided." in prompt
    assert "Other Studies: Not provided." in prompt
    print("[PASS] Populated case preserves facts and explicit negatives.")


def test_empty_fields_render_not_provided():
    prompt = build_headache_prompt(SYNTHETIC_EMPTY_RECORD["case"])
    
    assert "Age: Not provided." in prompt
    assert "Sex: Not provided." in prompt
    assert "Chief Complaint: Not provided." in prompt
    assert "History of Present Illness: Not provided." in prompt
    assert "Past Medical History: Not provided." in prompt
    assert "Current Medications: Not provided." in prompt
    assert "Vital Signs: Not provided." in prompt
    assert "General Exam: Not provided." in prompt
    assert "Neurologic Exam: Not provided." in prompt
    assert "Laboratory: Not provided." in prompt
    
    # Ensure missing values did not invent 'normal' or 'negative'
    clinical_segment = prompt.split("TASK INSTRUCTIONS:")[0]
    assert "normal" not in clinical_segment.lower()
    assert "negative" not in clinical_segment.lower()
    print("[PASS] Empty values render strictly as 'Not provided.' without invented negatives.")


def test_full_record_leakage_rejection():
    try:
        build_headache_prompt(SYNTHETIC_POPULATED_RECORD)
    except ValueError as e:
        assert "Input boundary violation" in str(e)
        print("[PASS] Accidental submission of full record correctly raises ValueError.")
        return
    raise AssertionError("Failed to reject full record containing metadata/ground truth.")


def test_sentinel_isolation_via_cli(tmp_path: Path):
    test_json_file = tmp_path / "synthetic_case_sentinel.json"
    test_json_file.write_text(json.dumps(SYNTHETIC_POPULATED_RECORD), encoding="utf-8")

    # Run preview_prompt.py as a subprocess
    result = subprocess.run(
        [sys.executable, "preview_prompt.py", str(test_json_file)],
        capture_output=True,
        text=True,
        check=True
    )
    cli_output = result.stdout

    # Verify no sentinel strings from ground_truth or evaluation leaked
    sentinels = [
        "SENTINEL_GROUND_TRUTH_MIGRAINE_DIAGNOSIS",
        "SENTINEL_CLUSTER_HEADACHE",
        "SENTINEL_SUBARACHNOID_HEMORRHAGE",
        "SENTINEL_ADMINISTER_TRIPTAN",
        "SENTINEL_DO_NOT_TAP_SPINE",
    ]
    for sentinel in sentinels:
        assert sentinel not in cli_output, f"Leakage detected: {sentinel} found in output!"

    # Verify case ID and metadata keys do not appear
    assert "SYNTHETIC_001" not in cli_output
    assert "difficulty" not in cli_output
    print("[PASS] CLI runner strictly prevents ground_truth and evaluation sentinel leakage.")


if __name__ == "__main__":
    import tempfile
    test_populated_preserves_facts_and_negatives()
    test_empty_fields_render_not_provided()
    test_full_record_leakage_rejection()
    with tempfile.TemporaryDirectory() as temp_dir:
        test_sentinel_isolation_via_cli(Path(temp_dir))
    print("\nAll verification checks passed successfully.")