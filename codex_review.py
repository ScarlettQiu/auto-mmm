"""Multi-Model Code Review Agent — runs MMM scripts through GPT-4o and Claude in parallel.

Usage:
    python codex_review.py                        # review all model scripts with all available LLMs
    python codex_review.py run_models.py          # review a specific file
    python codex_review.py --round 5              # tag output with round number
    python codex_review.py --models openai        # only OpenAI
    python codex_review.py --models anthropic     # only Claude

Writes:
    rounds/R{N:02d}_codex_review.md

Requires (at least one):
    OPENAI_API_KEY in .env or environment
    ANTHROPIC_API_KEY in .env or environment
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path


DEFAULT_FILES = [
    "run_models.py",
    "prepare.py",
    "compare.py",
    "models/ridge_mmm.py",
    "models/pymc_mmm.py",
    "models/lightweight_mmm.py",
]

SYSTEM_PROMPT = """You are a senior data scientist and Python engineer specialising in
Marketing Mix Modeling (MMM). Review the pipeline code for:
- Correctness: bugs, off-by-one errors, wrong formula implementations
- Statistical validity: adstock/Hill saturation applied correctly, train/test leakage,
  MAPE/R² computed on the right split
- Code quality: silent failures, missing error handling at model boundaries, type issues
- MMM-specific risks: negative ROI values accepted silently, contributions not summing
  to 100%, holdout contamination

Be concise. Flag real issues with file + line reference where possible.
Use severity labels: [CRITICAL], [WARNING], [INFO].
End with a one-line verdict: REVIEW_PASS or REVIEW_FAIL: <reason>."""


# ---------------------------------------------------------------------------
# Key loading
# ---------------------------------------------------------------------------

def load_env_key(key_name: str) -> str:
    value = os.environ.get(key_name, "")
    if not value:
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith(f"{key_name}=") and not line.startswith("#"):
                    value = line.split("=", 1)[1].strip()
                    break
    return value


# ---------------------------------------------------------------------------
# Per-model reviewers
# ---------------------------------------------------------------------------

def review_openai(file_blocks: str, round_num: int) -> tuple[str, str]:
    """Returns (model_label, review_text)."""
    try:
        from openai import OpenAI
    except ImportError:
        return "GPT-4o", "REVIEW_SKIPPED — `openai` package not installed. Run `pip install openai`."

    key = load_env_key("OPENAI_API_KEY")
    if not key:
        return "GPT-4o", "REVIEW_SKIPPED — no OPENAI_API_KEY found."

    client = OpenAI(api_key=key)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Round {round_num} — review these files:\n\n{file_blocks}"},
            ],
            temperature=0.2,
        )
        return "GPT-4o", response.choices[0].message.content
    except Exception as e:
        return "GPT-4o", f"REVIEW_SKIPPED — OpenAI error: {e}"


def review_anthropic(file_blocks: str, round_num: int) -> tuple[str, str]:
    """Returns (model_label, review_text)."""
    try:
        import anthropic
    except ImportError:
        return "Claude", "REVIEW_SKIPPED — `anthropic` package not installed. Run `pip install anthropic`."

    key = load_env_key("ANTHROPIC_API_KEY")
    if not key:
        return "Claude", "REVIEW_SKIPPED — no ANTHROPIC_API_KEY found."

    client = anthropic.Anthropic(api_key=key)
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Round {round_num} — review these files:\n\n{file_blocks}"},
            ],
        )
        return "Claude", response.content[0].text
    except Exception as e:
        return "Claude", f"REVIEW_SKIPPED — Anthropic error: {e}"


# ---------------------------------------------------------------------------
# Agreement summary
# ---------------------------------------------------------------------------

def extract_verdict(text: str) -> str:
    for line in reversed(text.splitlines()):
        # Strip markdown bold/italic and leading labels like "**Verdict:**"
        clean = line.strip().lstrip("*").strip()
        if ":" in clean:
            clean = clean.split(":", 1)[-1].strip()
        for prefix in ("REVIEW_PASS", "REVIEW_FAIL", "REVIEW_SKIPPED"):
            if prefix in line:
                # Return just from the prefix onward
                idx = line.index(prefix)
                return line[idx:].strip()
    return "REVIEW_SKIPPED"


def build_agreement_summary(results: dict[str, str]) -> str:
    """Compare verdicts across models and highlight common findings."""
    verdicts = {model: extract_verdict(text) for model, text in results.items()}

    lines = ["## Model Agreement Summary\n"]
    lines.append("| Model | Verdict |")
    lines.append("|---|---|")
    for model, verdict in verdicts.items():
        lines.append(f"| {model} | {verdict} |")

    active = {m: v for m, v in verdicts.items() if not v.startswith("REVIEW_SKIPPED")}
    if len(active) >= 2:
        all_pass = all(v.startswith("REVIEW_PASS") for v in active.values())
        any_fail = any(v.startswith("REVIEW_FAIL") for v in active.values())
        if all_pass:
            lines.append("\n**Both models agree: PASS** — code looks correct.")
        elif any_fail:
            failing = [m for m, v in active.items() if v.startswith("REVIEW_FAIL")]
            lines.append(f"\n**Disagreement or shared concern** — {', '.join(failing)} flagged issues. Review findings below.")
    elif len(active) == 1:
        lines.append(f"\n_Only one model ran ({list(active.keys())[0]}). Add the other API key for cross-model comparison._")
    else:
        lines.append("\n_No models ran — add OPENAI_API_KEY or ANTHROPIC_API_KEY to .env._")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-model code review for MMM pipeline")
    parser.add_argument("files", nargs="*", help="Files to review (default: all model scripts)")
    parser.add_argument("--round", type=int, default=1, help="Round number for output file")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["openai", "anthropic"],
        default=["openai", "anthropic"],
        help="Which LLMs to use (default: both)",
    )
    args = parser.parse_args()

    files = [Path(f) for f in args.files] if args.files else [Path(f) for f in DEFAULT_FILES]
    round_num = args.round

    rounds_dir = Path("rounds")
    rounds_dir.mkdir(exist_ok=True)
    out_path = rounds_dir / f"R{round_num:02d}_codex_review.md"

    print(f"\nMulti-Model Code Review — Round {round_num}")
    print(f"Files:  {[str(f) for f in files]}")
    print(f"Models: {args.models}\n")

    # Build shared file content block
    file_blocks_parts = []
    for f in files:
        if f.exists():
            content = f.read_text()
            file_blocks_parts.append(f"### {f}\n```python\n{content}\n```")
        else:
            file_blocks_parts.append(f"### {f}\n_(file not found — skipped)_")
    file_blocks = "\n\n".join(file_blocks_parts)

    # Map model name → reviewer function
    reviewer_map = {
        "openai":    lambda: review_openai(file_blocks, round_num),
        "anthropic": lambda: review_anthropic(file_blocks, round_num),
    }

    # Run models sequentially (avoids threading issues with HTTP clients)
    results: dict[str, str] = {}
    for m in args.models:
        try:
            model_label, review_text = reviewer_map[m]()
        except Exception as e:
            model_label = m
            review_text = f"REVIEW_SKIPPED — unexpected error: {e}"
        results[model_label] = review_text
        verdict = extract_verdict(review_text)
        print(f"  [{model_label}] {verdict}")

    # Build report
    agreement = build_agreement_summary(results)

    sections = [
        f"# Round {round_num} — Multi-Model Code Review",
        f"**Reviewed at:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Files reviewed:** {', '.join(str(f) for f in files)}",
        f"**Models used:** {', '.join(results.keys())}",
        "---",
        agreement,
        "---",
    ]

    for model_label, review_text in results.items():
        sections.append(f"## {model_label} Review\n\n{review_text}")
        sections.append("---")

    report = "\n\n".join(sections)
    out_path.write_text(report)
    print(f"\nReport saved → {out_path}")

    # Overall verdict for orchestrator
    verdicts = [extract_verdict(t) for t in results.values() if not extract_verdict(t).startswith("REVIEW_SKIPPED")]
    if not verdicts:
        overall = "REVIEW_SKIPPED"
    elif any(v.startswith("REVIEW_FAIL") for v in verdicts):
        overall = "REVIEW_FAIL: see rounds/" + out_path.name
    else:
        overall = "REVIEW_PASS"

    print(f"\n{overall}")


if __name__ == "__main__":
    main()
