"""preview_prompt.py: CLI runner to extract case data and output prompt text."""

import json
import sys
from pathlib import Path
from src.prompt_builder import build_headache_prompt


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python preview_prompt.py <path_to_case_json>", file=sys.stderr)
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with file_path.open("r", encoding="utf-8") as f:
            record = json.load(f)
    except json.JSONDecodeError as err:
        print(f"Error: Invalid JSON file ({err})", file=sys.stderr)
        sys.exit(1)

    if not isinstance(record, dict) or "case" not in record:
        print("Error: Input JSON must be an object containing a 'case' key.", file=sys.stderr)
        sys.exit(1)

    # Strict isolation boundary: pass only record["case"]
    case_data = record["case"]

    try:
        prompt = build_headache_prompt(case_data)
        print(prompt)
    except Exception as err:
        print(f"Error generating prompt: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()