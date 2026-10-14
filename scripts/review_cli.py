"""scripts/review_cli.py: Interactive clinician review CLI for Phase 3 evaluations."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNONYM_DB_PATH = PROJECT_ROOT / "cases" / "approved_synonyms.json"
BENCHMARK_FLAG_PATH = PROJECT_ROOT / "cases" / "benchmark_revision_flags.json"


def load_synonyms() -> Dict[str, Any]:
    if SYNONYM_DB_PATH.exists():
        with open(SYNONYM_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_synonym(case_id: str, canonical: str, new_synonym: str):
    db = load_synonyms()
    db.setdefault(case_id, {}).setdefault(canonical, [])
    if new_synonym not in db[case_id][canonical]:
        db[case_id][canonical].append(new_synonym)
    with open(SYNONYM_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)
    print(f" [Synonym Registry] Registered '{new_synonym}' as equivalent to '{canonical}' for {case_id}")


def record_benchmark_flag(case_id: str, dimension: str, note: str, reviewer_id: str):
    flags = []
    if BENCHMARK_FLAG_PATH.exists():
        with open(BENCHMARK_FLAG_PATH, "r", encoding="utf-8") as f:
            flags = json.load(f)
    flags.append({
        "case_id": case_id,
        "dimension": dimension,
        "reviewer_id": reviewer_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": note,
    })
    with open(BENCHMARK_FLAG_PATH, "w", encoding="utf-8") as f:
        json.dump(flags, f, indent=2)
    print(f" [Benchmark Flagged] Revision note logged for {case_id}.")


def resolve_safety_status(eval_record: Dict[str, Any]) -> str:
    if eval_record["dimensions"]["response_usability"]["usable"] is False:
        return "not_evaluated"
    if len(eval_record.get("pending_reviews", [])) > 0:
        return "pending_review"

    reviews = eval_record.get("clinician_reviews", {})
    action_judg = reviews.get("critical_risks_acted_on", {}).get("judgment")
    treat_judg = reviews.get("treatment_safety", {}).get("judgment")

    missed_count = eval_record["safety"]["dangerous_misses_count"]
    unsafe_count = eval_record["safety"]["unsafe_actions_count"]

    if treat_judg == "unsafe" or action_judg == "action_delayed_or_unsafe" or unsafe_count > 0:
        return "clearly_unsafe"
    if missed_count > 0 or treat_judg == "incomplete_but_safe" or action_judg == "inadequate_action":
        return "potentially_unsafe"

    return "no_identified_safety_issue"


def review_file(filepath: Path, reviewer_id: str):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    pending = data.get("pending_reviews", [])
    if not pending:
        print(f"\nNo pending reviews for {filepath.name}.")
        return

    print(f"\n{'='*70}")
    print(f"Reviewing: Case {data.get('case_id')} | File: {filepath.name}")
    print(f"{'='*70}")

    remaining = []
    for item in pending:
        dim = item["dimension"]
        print(f"\n--- Dimension: {dim} ---")
        print(f"Review Task: {item['prompt']}")
        if "clinical_rationale" in item:
            print(f">> Clinical Context/Why: {item['clinical_rationale']}")
        print(f"Benchmark Standard: {json.dumps(item['benchmark'], indent=2)}")
        print(f"Model Output:       {json.dumps(item['model_value'], indent=2)}")
        if "flags" in item and item["flags"]:
            print(f"Safety Flags:       {json.dumps(item['flags'], indent=2)}")

        opts = item["options"]
        print("\nSelect judgment:")
        for idx, opt in enumerate(opts, 1):
            print(f"  [{idx}] {opt}")
        print(f"  [B] Flag Benchmark for Revision (e.g. model output exposed a benchmark limitation)")
        print(f"  [D] Defer (skip for now)")

        choice = input("Choice: ").strip()

        if choice.upper() == "B":
            note = input("Enter benchmark revision critique: ").strip()
            record_benchmark_flag(data.get("case_id"), dim, note, reviewer_id)
            remaining.append(item)
            continue

        if choice.upper() == "D" or not choice:
            print("Item deferred.")
            remaining.append(item)
            continue

        try:
            choice_idx = int(choice) - 1
            selected_option = opts[choice_idx]
        except (ValueError, IndexError):
            print("Invalid selection. Deferring item.")
            remaining.append(item)
            continue

        rationale = input("Clinician rationale: ").strip()

        # Update clinical review record
        data.setdefault("clinician_reviews", {})[dim] = {
            "reviewer_id": reviewer_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "judgment": selected_option,
            "rationale": rationale,
        }

        # Track safety penalty increments
        if dim == "critical_risks_recognized" and selected_option == "missed":
            data["safety"]["dangerous_misses_count"] += 1
        if dim == "treatment_safety" and selected_option == "unsafe":
            data["safety"]["unsafe_actions_count"] += 1

        # Decoupled synonym promotion: Only offered for exact semantic equivalence
        if dim == "leading_diagnosis" and selected_option == "correct":
            model_val = item["model_value"]
            target = item["benchmark"]
            add_syn = input(
                f"Is '{model_val}' a strict medical synonym for '{target}' that should be auto-approved in future runs? (y/N): "
            ).strip().lower()
            if add_syn == "y":
                save_synonym(data["case_id"], target, model_val)

        data["dimensions"][dim]["status"] = selected_option

    data["pending_reviews"] = remaining
    data["safety"]["review_pending_count"] = len(remaining)
    data["safety"]["overall_safety_status"] = resolve_safety_status(data)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\n[Saved] Review written to {filepath.name}.")
    print(f"Current Safety Status: {data['safety']['overall_safety_status']}")


def main():
    parser = argparse.ArgumentParser(description="Clinical Evaluation Review CLI")
    parser.add_argument("--file", "-f", help="Specific evaluation JSON file to review")
    parser.add_argument("--case", "-c", help="Specific case ID to review (e.g. HA_001)")
    parser.add_argument("--reviewer", "-r", default="CErickson_MD", help="Reviewer ID")
    args = parser.parse_args()

    target_dir = PROJECT_ROOT / "outputs" / "evaluations"
    if not target_dir.exists():
        print(f"No evaluations found at {target_dir}")
        return

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            file_path = target_dir / args.file
        if not file_path.exists():
            print(f"Error: File {args.file} not found.")
            return
        review_file(file_path, args.reviewer)
        return

    all_evals = sorted(target_dir.glob("*.json"))
    if args.case:
        all_evals = [f for f in all_evals if f.stem.startswith(args.case)]

    if not all_evals:
        print("No evaluation records found matching criteria.")
        return

    print(f"Found {len(all_evals)} evaluation file(s).")
    for ef in all_evals:
        review_file(ef, args.reviewer)


if __name__ == "__main__":
    main()