#!/usr/bin/env python3
"""
Medusa Benchmark Runner
=======================

This script evaluates a **single Medusa model** on two commonly‑used academic
benchmarks:

    • HumanEval  — functional code generation (pass@1)
    • CNN / DailyMail — abstractive summarisation (ROUGE‑1/2/L)

Key design choices
------------------
* **One‑stop entry‑point**:  `python test.py --task humaneval` or `--task cnndm`
* **GPU friendly**:  the Medusa checkpoint is loaded **once** and reused.
* **Storage hygiene**: all intermediate / output artefacts are saved under
  `./outputs/` (created automatically).
* **Self‑documenting**: major sections are delimited by loud banner comments
  to make the control‑flow crystal‑clear.

Directory layout
----------------
project_root/
├── ckpt/medusa‑7b/           # your HF‑format checkpoint (unchanged)
├── outputs/                  # auto‑generated: metrics JSON, rouge cache, …
└── test.py                   # THIS script
"""
# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────
import argparse
import json
import subprocess
import tempfile
from pathlib import Path
import re

import evaluate
import torch
from datasets import load_dataset
from tqdm import tqdm
from fastchat.model import get_conversation_template
from models import get_model_class

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='medusa',
                    help='Name of the model wrapper to use (must be registered in models/__init__.py)')
parser.add_argument('--task', choices=['humaneval', 'cnndm'], default='humaneval', help='Benchmark to run')
parser.add_argument('--sample_limit', type=int, default=None, help='Limit number of examples for quick debugging')
parser.add_argument('--n_samples', type=int, default=1, help='Completions per HumanEval problem')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for CNN/DM summarisation')
parser.add_argument('--output_dir', type=str, default='outputs',
                    help='Directory to save metrics and temporary files')
parser.add_argument('--save_gens', action='store_true',
                    help='Save every generated solution for HumanEval to files')
parser.add_argument('--model_ckpt', type=str,
                    default='/home/neil/sd/Medusa/ckpt/medusa-7b',
                    help='Path to the Medusa checkpoint (HF format)')
parser.add_argument('--use_template', action='store_true',
                    help='Wrap each instruction with a FastChat conversation template')
parser.add_argument('--template_model_id', type=str, default='vicuna-7b',
                    help='Model id passed to get_conversation_template (ignored if --use_template is not set)')



# ──────────────────────────────────────────────────────────────────────────────
# 1. Low‑level Generation Helper
#    Wraps `model.medusa_generate` so other tasks can call it like a black box.
# ──────────────────────────────────────────────────────────────────────────────

def build_prompt(instruction: str, model_id: str) -> str:
    """
    Wrap a single-turn instruction with FastChat conversation template.
    """
    conv = get_conversation_template(model_id)
    conv.system_message = (
    "You are a helpful coding assistant. "
    "When I ask you to write code, reply with **ONLY** valid Python code, "
    "wrapped in ```python ``` fences, no explanation.")
    conv.append_message(conv.roles[0], instruction)  # user
    conv.append_message(conv.roles[1], None)         # assistant placeholder
    return conv.get_prompt()

# ──────────────────────────────────────────────────────────────────────────────
# 2. HumanEval (pass@1)
# ──────────────────────────────────────────────────────────────────────────────
def run_unit_tests(code: str, unit_tests: str, args) -> bool:
    """Run HumanEval unit tests in an isolated tmp dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        code_path = Path(tmpdir) / "solution.py"
        code_path.write_text(code, encoding="utf-8")
        test_path = Path(tmpdir) / "test.py"
        test_path.write_text(unit_tests, encoding="utf-8")
        import sys
        import pdb; pdb.set_trace()  # Debugging breakpoint, remove in production
        try:
            result = subprocess.run(
                [sys.executable, str(test_path)],
                cwd=tmpdir,                  # Ensure test.py can import solution.py
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10
            )
            if getattr(args, "show_test_output", False):
                print("----- Unit test output -----")
                print(result.stdout)
                print("----- End output -----")
            return result.returncode == 0

        except Exception as e:
            if getattr(args, "show_test_output", False):
                print("Exception while running tests:", e)
            return False

def evaluate_humaneval(model, args):
    # NOTE: HumanEval test split ~164 problems, each contains prompt + unit tests
    ds = load_dataset("openai_humaneval", split="test")
    if args.sample_limit:
        ds = ds.select(range(args.sample_limit))

    samples = []
    for item in tqdm(ds, desc="HumanEval Tasks"):
        prompt_stub, unit_tests = item["prompt"], item["test"]
        if args.use_template:
            prompt = build_prompt(prompt_stub, args.template_model_id)
        else:
            prompt = prompt_stub

        completion = model.generate(
            prompt,
            max_steps=getattr(args, "max_new_tokens", 256)
        )

        code_only = completion
        matches = re.findall(r"```(?:python)?\s*([\s\S]*?)(?:```|$)", completion, re.I)

        for frag in reversed(matches):           # 从最后一个往前找
            frag = frag.strip()
            if frag:                             # 跳过空代码块
                code_only = frag
                break
        # 避免模型把行号、Assistant: 也吐出来
        code_only = re.sub(r"^In \[\d+\]:\s*", "", code_only, flags=re.M)
        # Optionally save full solution file for manual inspection
        if args.save_gens:
            sol_dir = Path(args.output_dir) / "humaneval_solutions"
            sol_dir.mkdir(parents=True, exist_ok=True)
            (sol_dir / f"{item['task_id']}.py").write_text(prompt_stub + code_only, encoding="utf-8")

        samples.append({"task_id": item["task_id"], "completion": code_only})


    # ── Dump all completions to JSONL (official HumanEval format) ──
    samples_path = Path(args.output_dir) / "humaneval_samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as f:
        for s in samples:
            json.dump(s, f, ensure_ascii=False)
            f.write("\n")
    print(f"🔄 Completions saved to {samples_path.resolve()}")

    return {"num_samples": len(samples), "samples_file": str(samples_path)}

# ──────────────────────────────────────────────────────────────────────────────
# 3. CNN / DailyMail (ROUGE‑1/2/L)
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_cnndm(model, args):
    ds = load_dataset("cnn_dailymail", "3.0.0", split="test")
    # ~11k examples; we optionally slice via --sample_limit for quick smoke tests
    if args.sample_limit:
        ds = ds.select(range(args.sample_limit))
    rouge = evaluate.load("rouge")
    preds, refs = [], []
    for i in tqdm(range(0, len(ds), args.batch_size), desc="CNN/DM Examples"):
        batch = ds[i:i + args.batch_size]
        for art, ref in zip(batch["article"], batch["highlights"]):
            instruction = f"Summarize the following article:\n\n{art}\n\nSummary:"
            if args.use_template:
                prompt = build_prompt(instruction, args.template_model_id)
            else:
                prompt = instruction
            summary = model.generate(prompt, max_steps=256)
            preds.append(summary)
            refs.append(ref)
    # SacreROUGE via 🤗 evaluate → returns dict with rouge1/2/L (F1 by default)
    return rouge.compute(predictions=preds, references=refs, use_stemmer=True)

def main(args):
    """
    Main entrypoint for Medusa evaluation.
    """
    # ╭───────────────────────── Model Loading ─────────────────────────╮
    ModelClass = get_model_class(args.model)
    model = ModelClass(args.model_ckpt, **vars(args))
    # ╰──────────────────────────────────────────────────────────────────╯

    # Ensure output directory exists: e.g. outputs/medusa-7b/
    # Build a descriptive output directory name including model, temperature, and checkpoint
    temp_str = f"T{args.temperature}" if getattr(args, "temperature", None) is not None else ""
    model_name = f"{args.model}{temp_str}_{Path(args.model_ckpt).name}"
    output_dir = Path(args.output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    # Propagate the expanded path so helper functions write under the same directory
    args.output_dir = str(output_dir)

    if args.task == 'humaneval':
        metrics = evaluate_humaneval(model, args)
        outfile = output_dir / 'humaneval_metrics.json'
    elif args.task == 'cnndm':
        metrics = evaluate_cnndm(model, args)
        outfile = output_dir / 'cnndm_metrics.json'
    else:  # Should never happen thanks to argparser
        raise RuntimeError(f"Unsupported task: {args.task}")

    # Save & print
    outfile.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\n✅  Results saved to {outfile.resolve()}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    # Parse once to discover which --model the user selected
    _args, remaining = parser.parse_known_args()
    ModelClass = get_model_class(_args.model)
    # Inject model‑specific CLI flags
    ModelClass.add_cli_args(parser)
    # Now parse all arguments
    args = parser.parse_args()
    main(args)