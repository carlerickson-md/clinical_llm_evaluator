"""scripts/test_single_case.py: Runs and validates a single headache case through the full pipeline."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.prompt_builder import build_headache_prompt
from src.runner import run_prompt
from src.parser import parse_response
from src.evaluator import evaluate_response


def test_single_case(case_filename: str = "HA_001.json", model: str = "gpt-4o-mini"):
    case_path = PROJECT_ROOT / "cases" / case_filename
    if not case_path.exists():
        print(f"Error: Case file not found at {case_path}")
        return

    key = os.environ.get("CLINICAL_LLM_EVALUATOR_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("Error: Project API key is not configured.")
        return

    print(f"--- Loading: {case_filename} ---")
    with open(case_path, "r", encoding="utf-8") as f:
        case_data = json.load(f)

    case_stem = case_path.stem
    case_id = case_data.get("case_id", case_stem)

    # 1. Prompt Builder (strictly isolated clinical dict)
    print("1. Building prompt...")
    prompt = build_headache_prompt(case_data["case"])

    # 2. LLM Runner
    print(f"2. Calling {model} via runner...")
    runner_record = run_prompt(prompt=prompt, case_id=case_id, model=model)
    print(f"   API Status: {runner_record['api_status']['status']}")

    # 3. Parser & Schema Validation
    print("3. Parsing and validating output...")
    parsed_result = parse_response(runner_record)
    print(f"   Parser Status: {parsed_result['status']}")

    # 4. Phase 3 Evaluator
    print("4. Evaluating response against clinical benchmarks...")
    eval_result = evaluate_response(parsed_result, case_data)

    # 5. Save Artifacts
    runs_dir = PROJECT_ROOT / "outputs" / "runs"
    evals_dir = PROJECT_ROOT / "outputs" / "evaluations"
    runs_dir.mkdir(parents=True, exist_ok=True)
    evals_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_file = runs_dir / f"{case_stem}_{timestamp_str}_run.json"
    eval_file = evals_dir / f"{case_stem}_{timestamp_str}_eval.json"

    with open(run_file, "w", encoding="utf-8") as f:
        json.dump(parsed_result, f, indent=2)
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2)

    print(f"\n[Artifacts Saved]")
    print(f" - Run envelope: {run_file}")
    print(f" - Evaluation:   {eval_file}")

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY:")
    print("=" * 50)
    print(f"Primary Diagnosis Status: {eval_result['dimensions']['leading_diagnosis'].get('status')}")
    print(f"Mandatory Risks Missed:   {eval_result['safety']['dangerous_misses_count']}")
    print(f"Secondary Breadth Hits:   {eval_result['diagnostic_breadth'].get('secondary_considerations_found', 0)}/"
          f"{eval_result['diagnostic_breadth'].get('secondary_considerations_checked', 0)}")
    print(f"Items Pending Review:     {eval_result['safety']['review_pending_count']}")
    print(f"Safety Classification:    {eval_result['safety']['overall_safety_status']}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "HA_001.json"
    test_single_case(target)