# Clinical LLM Evaluator

An evaluation pipeline for assessing clinical reasoning in LLM headache consultations.

## Project Structure

- `cases/`: Clinical cases formatted in the authoritative JSON schema.
- `outputs/`: Timestamped execution payloads and batch results.
- `src/`: Core evaluation, prompt generation, runner, and parser logic.
- `scripts/`: Local developer tools, single-case test utilities, and batch CLI runners.
- `tests/`: Automated unit and integration tests with synthetic data and mock API responses.

## Configuration

Set your project API key as a persistent user environment variable before executing model runs:

```powershell
[Environment]::SetEnvironmentVariable("CLINICAL_LLM_EVALUATOR_API_KEY", "your-openai-api-key", "User")
```

## Architecture & Boundary Enforcement

### Phase 1: Case Ingestion to LLM Prompt
- **Data Isolation:** `build_headache_prompt` accepts only the isolated clinical dictionary (`case`), rejecting root containers containing metadata, ground-truth diagnoses, or evaluation criteria.
- **Deterministic Rendering:** Formats unstructured history, vitals, exam findings, and diagnostic studies into a standardized clinical summary with explicit schema constraints.

### Phase 2: Runner & Parser
- **Strict Boundary Enforcement:** The runner receives only the model-facing prompt string and `case_id`. Clinician-only benchmarks (`ground_truth`, `evaluation`) are strictly excluded.
- **Explicit Failure Classification:** The runner distinguishes network failures (`api_error`), safety refusals (`refusal`), and truncated responses (`incomplete_generation`), disabling SDK automatic retries for transparent evaluation.
- **Strict Schema Parsing:** The parser decodes JSON without speculative heuristics or Markdown-fence stripping, validating:
  - Required top-level keys: `primary_diagnosis`, `differential`, `recommended_action`, `reasoning`.
  - Non-empty string requirements for primary diagnosis and reasoning.
  - Structured treatment schema dividing actions into immediate_supportive (safe pre-workup care) and conditional_on_workup (therapies gated by diagnostic findings).
- **Audit Preservation:** Malformed or failed responses record a structured failure status alongside the raw text for downstream evaluator skipping.

### Phase 3: Clinical Evaluation & Safety Scoring (Clinician-in-the-Loop v0.1)
- **Usability Gating:** Upstream runner and parser failures bypass clinical scoring and are marked `not_evaluated`, distinguishing infrastructural/schema defects from clinical errors.
- **Diagnosis & Negation Guards:** Exact boundary matching and case-specific synonym mappings prevent false passes on negated strings (e.g., "SAH unlikely").
- **Tiered Risk Benchmarks:**
  - **Mandatory Rule-Outs (`dangerous_misses`):** Defining failure points (e.g., Subarachnoid Hemorrhage). Omission or negation constitutes an immediate safety failure.
  - **Secondary Considerations (`secondary_considerations`):** Diagnostic breadth criteria (e.g., RCVS, CVT) that award credit for diagnostic scope without penalizing safety if omitted.
  - **Clinical Rationales (`benchmarks_rationale`):** Provides the contextual "why" for benchmark standards directly to reviewers.
- **Treatment Safety Verification:** Automatic detection of premature contraindicated prescribing in `immediate_supportive`, while accepting safely gated treatments in `conditional_on_workup`.
- **Decoupled Clinician Review CLI:** Interactive review interface that isolates clinical judgment from global synonym promotion, allows flagging benchmark flaws for revision, and preserves audit trails across interruptions.

## Usage

### 1. Preview a Case Prompt
Inspect the formatted prompt built from an isolated clinical case dictionary:
```powershell
python -m scripts.preview_prompt cases\HA_001.json
```

### 2. Test a Single Case
Run an isolated case through the runner and parser pipeline, inspect prompt and outputs in the console, and save a timestamped payload to `outputs/`:
```powershell
python scripts/test_single_case.py HA_001.json
```

### 3. Run Batch Evaluation
Execute sequentially across all case files in `cases/` without stopping on individual failures:
```powershell
python scripts/run_batch.py
```

### 4. Interactive Clinician Review CLI
Audit pending reviews for a single case or an entire batch, inspect clinical rationales, record reviewer notes, or flag benchmarks for revision:

```powershell
# Review all pending evaluations
python scripts/review_cli.py

# Review evaluations for a specific case
python scripts/review_cli.py --case HA_001

# Review a specific evaluation artifact
python scripts/review_cli.py --file outputs/evaluations/HA_001_run123_eval.json
```

### 5. Run Automated Test Suite
Run local unit and schema boundary tests with synthetic mock clients (no network calls):
```powershell
pytest tests/ -v
```

## Development Roadmap

### Phase 1: Case Ingestion & Prompting
- [x] Authoritative case schema validation
- [x] Strict data isolation / boundary enforcement
- [x] Deterministic prompt generation

### Phase 2: Model Runner & Parser
- [x] Robust JSON response extraction from LLM completions
- [x] Graceful fallback handling for non-JSON or malformed outputs
- [x] Batch execution script across `cases/`

### Phase 3: Clinical Evaluation & Scoring
- [x] Primary diagnosis exact/semantic match scoring
- [x] Penalty scoring for dangerous missed diagnoses
- [x] Contraindicated action flag detection

### Phase 4: Two-Tier Automated Grading Architecture (Future)
- [ ] Secondary LLM Judge API Integration: Deploy a rubric-guided evaluator LLM call to assess ambiguous diagnostic synonyms, conditional workup logic, and reasoning explanations.
- [ ] Confidence-Calibrated Escalation: Automatically approve high-confidence LLM judge evaluations; route low-confidence, borderline, or contradictory assessments to the clinician review queue.
- [ ] Inter-Rater & Concordance Metrics: Track alignment statistics ($\kappa$ / concordance rates) between the LLM Judge and the expert clinician reviewer.
- [ ] Automated Benchmark Optimization Suggestions: Surface data-driven suggestions for benchmark updates based on recurring model insights and reviewer critiques.

## AI Assistance & Transparency

- **Tooling & Co-Design:** Development followed a human-in-the-loop, guided co-design workflow with LLM-based assistants, collaborating interactively on architectural decisions, boilerplate implementation, and synthetic unit-test authoring.
- **Human Oversight:** All clinical logic, schema design, leakage boundaries, and programmatic checks were independently defined, reviewed, and validated by the maintainer.
- **Data Privacy:** Synthetic data only was used for testing and prompting verification during development; no protected health information (PHI) or private case records were shared with external developer tooling.