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
import time

import torch
from datasets import load_dataset
from tqdm import tqdm
from fastchat.model import get_conversation_template
from models import get_model_class

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='medusa',
                    help='Name of the model wrapper to use (must be registered in models/__init__.py)')
parser.add_argument('--dataset_name', type=str,
                    default='data-is-better-together/10k_prompts_ranked',
                    help='Dataset on HuggingFace Hub (default: data-is-better-together/10k_prompts_ranked)')
parser.add_argument('--dataset_split', type=str, default='train',
                    help='Split name to load from the dataset (default: train)')
parser.add_argument('--sample_limit', type=int, default=200,
                    help='Limit number of examples to run (default: 200)')
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
    conv.append_message(conv.roles[0], instruction)  # user
    conv.append_message(conv.roles[1], None)         # assistant placeholder
    return conv.get_prompt()

# ──────────────────────────────────────────────────────────────────────────────
# New: Single‑dataset evaluation (no validation)
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_dataset(model, args):
    """
    Generate completions for every example in the specified HuggingFace dataset
    and record execution time. Assumes each item contains a 'prompt' field; if not,
    falls back to 'text' or string conversion.
    """
    ds = load_dataset(args.dataset_name, split=args.dataset_split)
    if args.sample_limit:
        ds = ds.select(range(args.sample_limit))

    start_ts = time.time()
    completions = []

    for item in tqdm(ds, desc=f"Dataset: {args.dataset_name}"):
        prompt_raw = (item.get("prompt")
                      or item.get("text")
                      or str(item))
        prompt = build_prompt(prompt_raw, args.template_model_id) if args.use_template else prompt_raw
        completion = model.generate(prompt, max_steps=getattr(args, "max_new_tokens", 256))
        completions.append({"id": item.get("id", len(completions)),
                            "prompt": prompt_raw,
                            "completion": completion})

    total_sec = time.time() - start_ts
    metrics = {
        "num_samples": len(completions),
        "total_time_sec": total_sec,
        "avg_time_per_sample_sec": total_sec / len(completions) if completions else None
    }

    # Save completions to JSONL under the output directory
    samples_path = Path(args.output_dir) / f"{args.dataset_name}_completions.jsonl"
    # Ensure the directory for this dataset exists
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    with samples_path.open("w", encoding="utf-8") as f:
        for c in completions:
            json.dump(c, f, ensure_ascii=False)
            f.write("\n")
    metrics["samples_file"] = str(samples_path)

    print(f"🔄 Completions saved to {samples_path.resolve()}")
    return metrics

def main(args):
    """
    Main entrypoint for Medusa evaluation.
    """
    # ╭───────────────────────── Model Loading ─────────────────────────╮
    ModelClass = get_model_class(args.model)
    model = ModelClass(args.model_ckpt, **vars(args))
    # ╰──────────────────────────────────────────────────────────────────╯

    # Build a descriptive output directory name including model, temperature, and checkpoint
    temp_str = f"T{args.temperature}" if getattr(args, "temperature", None) is not None else ""
    model_name = f"{args.model}{temp_str}_{Path(args.model_ckpt).name}"
    output_dir = Path(args.output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    # Propagate the expanded path so helper functions write under the same directory
    args.output_dir = str(output_dir)

    metrics = evaluate_dataset(model, args)
    outfile = output_dir / f"{args.dataset_name}_metrics.json"

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