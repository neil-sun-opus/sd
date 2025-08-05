#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare local MMLU data (cais/mmlu 'all') in /home/neil/sd/

- Loads from disk if present; else downloads.
- Exports zero-shot and few-shot prompt CSVs.
"""

import os
import argparse
import pandas as pd
from datasets import load_dataset, load_from_disk

IDX2LETTER = ["A", "B", "C", "D"]

def build_zero_shot_prompts(ds_test):
    rows = []
    for i, ex in enumerate(ds_test):
        q = ex["question"].strip()
        ch = ex["choices"]
        # ex["answer"] is int index
        ans = IDX2LETTER[int(ex["answer"])]
        subj = ex.get("subject", "unknown")
        prompt = (
            f"{q}\n"
            f"A. {ch[0]}\n"
            f"B. {ch[1]}\n"
            f"C. {ch[2]}\n"
            f"D. {ch[3]}\n"
            "Answer:"
        )
        rows.append({"qid": i, "subject": subj, "prompt_text": prompt, "gold": ans})
    return pd.DataFrame(rows)

def build_fewshot_prompts(ds_dev, ds_test, num_shots=5):
    # build subject -> list of dev examples
    dev_map = {}
    for ex in ds_dev:
        subj = ex.get("subject", "unknown")
        dev_map.setdefault(subj, []).append(ex)

    rows = []
    for i, ex in enumerate(ds_test):
        subj = ex.get("subject", "unknown")
        shots = dev_map.get(subj, [])[:num_shots]
        parts = []
        # prepend few-shot exemplars (with answers)
        for s in shots:
            parts.append(
                f"Q: {s['question'].strip()}\n"
                f"A. {s['choices'][0]}\n"
                f"B. {s['choices'][1]}\n"
                f"C. {s['choices'][2]}\n"
                f"D. {s['choices'][3]}\n"
                f"Answer: {IDX2LETTER[int(s['answer'])]}\n"
            )
        # test item (no answer)
        parts.append(
            f"Q: {ex['question'].strip()}\n"
            f"A. {ex['choices'][0]}\n"
            f"B. {ex['choices'][1]}\n"
            f"C. {ex['choices'][2]}\n"
            f"D. {ex['choices'][3]}\n"
            "Answer:"
        )
        prompt = "\n".join(parts)
        gold = IDX2LETTER[int(ex["answer"])]
        rows.append({"qid": i, "subject": subj, "prompt_text": prompt, "gold": gold})
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/neil/sd", help="Base directory to store datasets & CSVs.")
    parser.add_argument("--force_redownload", action="store_true")
    parser.add_argument("--fewshot", type=int, default=5, help="How many dev examples to prepend per subject (<=5). Use 0 to skip few-shot CSV.")
    args = parser.parse_args()

    root = os.path.abspath(os.path.expanduser(args.root))
    os.makedirs(root, exist_ok=True)
    save_dir = os.path.join(root, "mmlu_all")

    if os.path.isdir(save_dir) and not args.force_redownload:
        print(f"[info] Found existing MMLU at {save_dir}; loading from disk.")
        ds_all = load_from_disk(save_dir)
    else:
        print("[info] Downloading cais/mmlu (all)...")
        ds_all = load_dataset("cais/mmlu", "all")
        ds_all.save_to_disk(save_dir)
        print(f"[info] Saved MMLU to {save_dir}")

    # sanity
    print(f"[info] test size = {len(ds_all['test'])}, dev size = {len(ds_all['dev'])}")

    # zero-shot
    df_zero = build_zero_shot_prompts(ds_all["test"])
    zero_path = os.path.join(root, "mmlu_test_prompts.csv")
    df_zero.to_csv(zero_path, index=False)
    print(f"[done] zero-shot CSV: {zero_path}")

    # few-shot
    if args.fewshot > 0:
        df_fs = build_fewshot_prompts(ds_all["dev"], ds_all["test"], num_shots=args.fewshot)
        fs_path = os.path.join(root, f"mmlu_test_prompts_{args.fewshot}shot.csv")
        df_fs.to_csv(fs_path, index=False)
        print(f"[done] {args.fewshot}-shot CSV: {fs_path}")
    else:
        print("[info] Skipped few-shot CSV (fewshot=0).")

if __name__ == "__main__":
    main()