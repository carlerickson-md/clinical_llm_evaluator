"""src/summary_report.py: Aggregates evaluation records into descriptive clinical evaluation reports."""

import json
from pathlib import Path
from typing import Any, Dict, List


def aggregate_evaluations(eval_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Deduplicate records by run_id
    unique_runs: Dict[str, Dict[str, Any]] = {}
    for rec in eval_records:
        r_id = rec.get("run_id")
        if r_id:
            unique_runs[r_id] = rec

    records = list(unique_runs.values())
    total_responses = len(records)
    unique_cases = len({r.get("case_id") for r in records if r.get("case_id")})

    upstream_failures = 0
    parser_failures = 0
    usable_responses = 0

    diagnosis_tallies: Dict[str, int] = {
        "correct": 0,
        "acceptable_but_imprecise": 0,
        "incorrect": 0,
        "needs_review": 0,
        "not_evaluated": 0,
    }

    pending_review_total = 0
    completed_review_total = 0
    dangerous_misses_total = 0
    unsafe_actions_total = 0

    safety_status_counts: Dict[str, int] = {
        "no_identified_safety_issue": 0,
        "potentially_unsafe": 0,
        "clearly_unsafe": 0,
        "pending_review": 0,
        "not_evaluated": 0,
    }

    for r in records:
        usability = r.get("dimensions", {}).get("response_usability", {})
        if not usability.get("usable", False):
            reason = usability.get("failure_reason")
            if reason in ("api_error", "timeout", "refusal", "empty_response", "incomplete_generation"):
                upstream_failures += 1
            else:
                parser_failures += 1
        else:
            usable_responses += 1

        diag_stat = r.get("dimensions", {}).get("leading_diagnosis", {}).get("status", "not_evaluated")
        diagnosis_tallies[diag_stat] = diagnosis_tallies.get(diag_stat, 0) + 1

        num_pending = len(r.get("pending_reviews", []))
        num_completed = len(r.get("clinician_reviews", {}))
        pending_review_total += num_pending
        completed_review_total += num_completed

        safety = r.get("safety", {})
        dangerous_misses_total += safety.get("dangerous_misses_count", 0)
        unsafe_actions_total += safety.get("unsafe_actions_count", 0)

        s_stat = safety.get("overall_safety_status", "pending_review")
        safety_status_counts[s_stat] = safety_status_counts.get(s_stat, 0) + 1

    return {
        "total_responses_attempted": total_responses,
        "unique_cases_evaluated": unique_cases,
        "upstream_api_failures": upstream_failures,
        "parser_validation_failures": parser_failures,
        "usable_responses": usable_responses,
        "diagnosis_tallies": diagnosis_tallies,
        "pending_review_items": pending_review_total,
        "completed_review_items": completed_review_total,
        "dangerous_misses_count": dangerous_misses_total,
        "unsafe_actions_count": unsafe_actions_total,
        "safety_status_breakdown": safety_status_counts,
    }


def generate_markdown_report(summary: Dict[str, Any]) -> str:
    tot = summary["total_responses_attempted"]
    usable = summary["usable_responses"]

    return f"""# Phase 3 Clinical Evaluation Summary Report

## 1. Pipeline Usability & Execution Metrics
- **Total Responses Attempted**: {tot}
- **Unique Cases Evaluated**: {summary['unique_cases_evaluated']}
- **Upstream API Failures**: {summary['upstream_api_failures']}/{tot}
- **Parser Validation Failures**: {summary['parser_validation_failures']}/{tot}
- **Structurally Usable Responses**: {usable}/{tot} ({(usable/tot*100 if tot else 0):.1f}%)

## 2. Leading Diagnosis Performance (Usable Denominator: {usable})
- **Correct (Exact/Approved Synonym)**: {summary['diagnosis_tallies']['correct']}/{usable}
- **Acceptable but Imprecise**: {summary['diagnosis_tallies']['acceptable_but_imprecise']}/{usable}
- **Incorrect**: {summary['diagnosis_tallies']['incorrect']}/{usable}
- **Pending Clinician Review**: {summary['diagnosis_tallies']['needs_review']}/{usable}

## 3. Safety & Clinician Review Metrics
- **Pending Review Items Remaining**: {summary['pending_review_items']}
- **Completed Clinician Reviews**: {summary['completed_review_items']}
- **Identified Dangerous Misses**: {summary['dangerous_misses_count']}
- **Identified Unsafe Actions**: {summary['unsafe_actions_count']}

## 4. Overall Clinician-Audited Safety Classification
- **No Identified Safety Issue**: {summary['safety_status_breakdown']['no_identified_safety_issue']}/{tot}
- **Potentially Unsafe**: {summary['safety_status_breakdown']['potentially_unsafe']}/{tot}
- **Clearly Unsafe**: {summary['safety_status_breakdown']['clearly_unsafe']}/{tot}
- **Pending Clinical Review**: {summary['safety_status_breakdown']['pending_review']}/{tot}
- **Not Evaluated (Unusable / Upstream Failure)**: {summary['safety_status_breakdown']['not_evaluated']}/{tot}

> *Note: Metrics reflect deterministic benchmarks and recorded clinician reviews. Small sample counts are intended for safety validation and auditability, not population-level statistical inference.*
"""