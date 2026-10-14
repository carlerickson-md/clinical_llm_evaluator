"""src/prompt_builder.py: Formats clinical headache case data into an LLM prompt."""

from typing import Any

__all__ = ["build_headache_prompt"]

LEAKAGE_KEYS = {
    "case_id",
    "schema_version",
    "domain",
    "ground_truth",
    "evaluation",
    "metadata",
}

EXPECTED_SECTIONS = {"demographics", "presentation", "history", "exam", "studies"}

FIXED_TASK_INSTRUCTIONS = """Assess this patient's headache presentation and provide a concise clinical assessment for the referring primary care or urgent care clinician.

Use the supplied patient information and your medical knowledge. Do not invent patient findings or assume that unreported findings are normal or absent.

State the most likely diagnosis, important alternative diagnoses, and appropriate next steps in workup and treatment. Include urgency or disposition in the recommended actions when relevant. Express uncertainty when warranted.

MANDATORY TREATMENT SEQUENCING:
1. immediate_supportive: Safe initial measures prior to completing workup (e.g., quiet room, IV access/hydration, antiemetics, blood pressure stabilization). If acute life-threatening pathology is suspected, do NOT administer definitive primary headache therapies (e.g., triptans, vasoconstrictors) here.
2. conditional_on_workup: Therapies that require diagnostic exclusion or confirmation before administration. Explicitly state the condition (e.g., "If non-contrast head CT and LP exclude subarachnoid hemorrhage, consider sumatriptan 6 mg SubQ"). If no treatment is conditional on workup, state that explicitly.

Provide a brief explanation, preferably one sentence, citing the case findings that support your assessment.

Return only a JSON object matching the structure below, with no Markdown fences or additional text."""

REQUIRED_MODEL_RESPONSE_STRUCTURE = """{
  "primary_diagnosis": "string",
  "differential": ["string"],
  "recommended_action": {
    "workup": ["string"],
    "treatment": {
      "immediate_supportive": ["string"],
      "conditional_on_workup": ["string"]
    }
  },
  "reasoning": "string"
}

Format clarification:
- primary_diagnosis and reasoning must be nonempty strings.
- differential must be a list of strings; it may be empty if no important alternatives are identified.
- workup, immediate_supportive, and conditional_on_workup must each contain at least one nonempty statement."""


def _validate_case_input(case: dict) -> None:
    if not isinstance(case, dict):
        raise TypeError(f"Expected dict for 'case', received {type(case).__name__}")
    leaked = LEAKAGE_KEYS.intersection(case.keys())
    if leaked:
        raise ValueError(
            f"Input boundary violation: Outer metadata/ground-truth fields detected: {sorted(leaked)}. "
            "Pass only record['case'] to build_headache_prompt."
        )
    for section in EXPECTED_SECTIONS:
        if section not in case:
            raise KeyError(f"Missing required case section: '{section}'")
        if not isinstance(case[section], dict):
            raise TypeError(f"Section '{section}' must be a dict, got {type(case[section]).__name__}")


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


def _format_value(value: Any) -> str:
    if _is_empty(value):
        return "Not provided."
    return str(value).strip()


def _format_list(items: Any) -> str:
    if not isinstance(items, list):
        raise TypeError(f"Expected list, received {type(items).__name__}")
    if _is_empty(items):
        return "Not provided."
    return "\n" + "\n".join(f"  - {str(item).strip()}" for item in items)


def _format_dict(dict_val: Any) -> str:
    if not isinstance(dict_val, dict):
        raise TypeError(f"Expected dict, received {type(dict_val).__name__}")
    if _is_empty(dict_val):
        return "Not provided."
    return "\n" + "\n".join(
        f"  - {k}: {'Not provided.' if _is_empty(dict_val[k]) else str(dict_val[k]).strip()}"
        for k in sorted(dict_val.keys())
    )


def build_headache_prompt(case: dict) -> str:
    """Build a deterministic LLM prompt from an isolated clinical case dictionary."""
    _validate_case_input(case)

    clinical_data = f"""PATIENT CLINICAL DATA:

Demographics:
- Age: {_format_value(case['demographics'].get('age'))}
- Sex: {_format_value(case['demographics'].get('sex'))}

Presentation:
- Chief Complaint: {_format_value(case['presentation'].get('chief_complaint'))}
- History of Present Illness: {_format_value(case['presentation'].get('history_of_present_illness'))}

Medical History:
- Past Medical History: {_format_list(case['history'].get('medical_history', []))}
- Current Medications: {_format_list(case['history'].get('medications', []))}
- Relevant Context: {_format_list(case['history'].get('relevant_context', []))}

Physical Examination:
- Vital Signs: {_format_dict(case['exam'].get('vital_signs', {}))}
- General Exam: {_format_value(case['exam'].get('general'))}
- Neurologic Exam: {_format_value(case['exam'].get('neurologic'))}

Diagnostic Studies:
- Laboratory: {_format_list(case['studies'].get('laboratory', []))}
- Imaging: {_format_list(case['studies'].get('imaging', []))}
- Other Studies: {_format_list(case['studies'].get('other', []))}"""

    return f"""{clinical_data}

TASK INSTRUCTIONS:
{FIXED_TASK_INSTRUCTIONS}

RESPONSE STRUCTURE:
{REQUIRED_MODEL_RESPONSE_STRUCTURE}"""