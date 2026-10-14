"""scripts/run_batch.py: Executes runner, parser, and Phase 3 evaluator across cases/."""

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.prompt_builder import build_headache_prompt
from src.runner import run_prompt
from src.parser import parse_response
from src.evaluator import evaluate_response
from src.summary_report import aggregate_evaluations, generate_markdown_report


def run_batch(cases_dir: str = "cases", output_dir: str = "outputs", model: str = "gpt-4o-mini"):
    cases_path = PROJECT_ROOT / cases_dir
    case_files = sorted(cases_path.glob("HA_*.json"))

    if not case_files:
        print(f"No case JSON files found in {cases_dir}/.")
        return

    batch_run_id = f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    target_runs = PROJECT_ROOT / output_dir / "runs" / batch_run_id
    target_evals = PROJECT_ROOT / output_dir / "evaluations"
    target_runs.mkdir(parents=True, exist_ok=True)
    target_evals.mkdir(parents=True, exist_ok=True)

    print(f"Starting batch run: {batch_run_id} | Model: {model}")
    print("-" * 80)

    eval_records = []

    for cfile in case_files:
        with open(cfile, "r", encoding="utf-8") as f:
            case_data = json.load(f)

        case_id = case_data.get("case_id", cfile.stem)

        # 1. Prompt Builder (isolated patient dict)
        prompt = build_headache_prompt(case_data["case"])

        # 2. Runner & Parser
        runner_rec = run_prompt(prompt=prompt, case_id=case_id, model=model)
        parsed_res = parse_response(runner_rec)

        # Save run envelope
        with open(target_runs / f"{case_id}_result.json", "w", encoding="utf-8") as f:
            json.dump(parsed_res, f, indent=2)

        # 3. Phase 3 Evaluator
        eval_rec = evaluate_response(parsed_res, case_data)
        eval_filename = target_evals / f"{case_id}_{parsed_res['run_id']}_eval.json"
        with open(eval_filename, "w", encoding="utf-8") as f:
            json.dump(eval_rec, f, indent=2)

        eval_records.append(eval_rec)
        print(f"Case {case_id:<8} | Usability: {parsed_res['status']:<8} | Safety: {eval_rec['safety']['overall_safety_status']}")

    # 4. Generate batch report
    summary = aggregate_evaluations(eval_records)
    report_md = generate_markdown_report(summary)
    with open(target_runs / "batch_summary_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print("-" * 80)
    print("Batch complete. Summary report saved.")


if __name__ == "__main__":
    run_batch()