#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run MMLU evaluation on a Medusa model using EleutherAI's lm-evaluation-harness.

Usage (example):
    python run_mmlu_with_medusa.py \
        --model-path /home/neil/checkpoints/medusa-7b \
        --batch-size 8 \
        --shots 5 \
        --output results/eval_medusa_mmlu.json
"""
# ---- monkeypatch: strip trust_remote_code from datasets.load_dataset ----
import datasets as _ds
if hasattr(_ds, "load_dataset"):
    _orig_load_dataset = _ds.load_dataset
    def _load_dataset_no_trust(*args, **kwargs):
        kwargs.pop("trust_remote_code", None)  # drop it if present
        return _orig_load_dataset(*args, **kwargs)
    _ds.load_dataset = _load_dataset_no_trust
# ------------------------------------------------------------------------

import argparse
import torch
import math
from typing import List, Tuple, Optional, Union, Iterable, Dict, Any
import json
import os
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


# harness
from lm_eval import simple_evaluate
from lm_eval.api.model import LM

# your Medusa
from medusa.model.medusa_model import MedusaModel


# -----------------------------
# Utility: move tensors to device
# -----------------------------
def _to_device(t, device):
    """Move tensor to device (no-op for non-tensors)."""
    if isinstance(t, torch.Tensor):
        return t.to(device)
    return t


# ============================================================
# Wrapper: adapt MedusaModel to lm-evaluation-harness LM API
# ============================================================
class MedusaLM(LM):
    """
    A lightweight adapter that exposes a MedusaModel (which wraps a base HF causal LM)
    to lm-evaluation-harness.

    NOTE:
    - For loglikelihood scoring (used by MMLU), we bypass speculative decoding and
      directly score with the *base* model's logits to ensure correctness.
    - Generation uses the base model's generate(), unless --use-medusa-generate is enabled.
    """

    def __init__(
        self,
        medusa_model: MedusaModel,
        tokenizer,
        max_batch_size: int = 8,
        use_medusa_generate: bool = False,
    ):
        self.medusa = medusa_model
        self.tokenizer = tokenizer
        self._max_batch_size = max_batch_size
        self.use_medusa_generate = use_medusa_generate

        # choose a device (Medusa loads with device_map="auto"; pick first param device)
        self.device = next(self.medusa.parameters()).device

        # convenience: underlying base causal LM (HF AutoModelForCausalLM-like)
        self.base_model = getattr(self.medusa, "base_model", self.medusa)
        # lm-evaluation-harness expects these attributes for distributed eval
        # (see lm_eval/api/model.py). We run single-process, so rank=0, world_size=1.
        self._rank = 0
        self._world_size = 1
        logging.info(f"MedusaLM initialized on device {self.device} (batch_size={self._max_batch_size}, use_medusa_generate={self.use_medusa_generate})")

    # ---- required properties ----
    @property
    def eot_token_id(self) -> int:
        # harness uses this in some generation loops; fall back to tokenizer.eos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        # try model.config.max_position_embeddings; else fallback
        return getattr(self.base_model.config, "max_position_embeddings", 2048)

    @property
    def max_gen_toks(self) -> int:
        return 512  # safe default; override via generate_until args if needed

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    # ---- tokenization helpers ----
    def tok_encode(self, s: str) -> List[int]:
        return self.tokenizer.encode(s, add_special_tokens=False)

    def tok_decode(self, toks: List[int]) -> str:
        return self.tokenizer.decode(toks, skip_special_tokens=True)

    # ======================================================
    # loglikelihood: core for MMLU (multiple-choice scoring)
    # ======================================================
    def loglikelihood(
        self, requests: Iterable[Tuple[str, str]]
    ) -> List[Tuple[float, bool]]:
        """
        Each request: (context, continuation)
        Return list of (logprob, is_greedy) where
          - logprob = sum log p(token | context + prev continuation tokens)
          - is_greedy = whether continuation tokens are greedy argmax (optional)
        """
        reqs = list(requests)
        logging.debug(f"loglikelihood(): received {len(reqs)} requests")
        results = []

        # batch by similar length for efficiency (simple version: process sequentially)
        # You can optimize later.
        for context, continuation in reqs:
            # Tokenize
            ctx_ids = self.tok_encode(context)
            cont_ids = self.tok_encode(continuation)

            # Truncate if too long
            # (Harness normally handles but we guard)
            if len(ctx_ids) + len(cont_ids) > self.max_length:
                ctx_ids = ctx_ids[-(self.max_length - len(cont_ids)) :]

            input_ids = torch.tensor([ctx_ids + cont_ids], device=self.device)

            with torch.no_grad():
                outputs = self.base_model(input_ids=input_ids)
                # logits shape [1, seq, vocab]
                logits = outputs.logits

            # We want logprobs of continuation tokens *conditioned on prefix*
            # Shift so that logits for position t predict token t
            # (standard causal LM: logits[:, :-1] predict input_ids[:, 1:])
            logits = logits[:, :-1, :]
            labels = input_ids[:, 1:]

            # Identify which positions correspond to continuation
            ctx_len = len(ctx_ids)
            # positions in labels index 0..seq-2; continuation starts at ctx_len-1? careful:
            # Example: input = [ctx..., cont...]
            # labels exclude first token; label positions align with input_ids[1:]
            # Continuation tokens start at global index len(ctx_ids) in input_ids.
            # In labels-space: continuation positions start at (len(ctx_ids) - 1)
            start = ctx_len - 1
            if start < 0:
                start = 0

            cont_positions = range(start, start + len(cont_ids))

            # gather logprobs
            logprobs = torch.log_softmax(logits[0], dim=-1)
            total_logprob = 0.0
            greedy = True

            for i, token_id in zip(cont_positions, cont_ids):
                token_logprobs = logprobs[i]
                total_logprob += token_logprobs[token_id].item()
                pred_token = torch.argmax(token_logprobs).item()
                if pred_token != token_id:
                    greedy = False

            results.append((float(total_logprob), greedy))

        return results

    # ======================================================
    # loglikelihood_rolling: optional; can proxy to base impl
    # ======================================================
    def loglikelihood_rolling(self, requests: Iterable[str]) -> List[float]:
        """
        For perplexity-like metrics; simple per-token NLL over the string.
        """
        reqs = list(requests)
        logging.debug(f"loglikelihood_rolling(): received {len(reqs)} requests")
        res = []
        for text in reqs:
            ids = self.tok_encode(text)
            if len(ids) < 2:
                res.append(0.0)
                continue

            input_ids = torch.tensor([ids], device=self.device)
            with torch.no_grad():
                logits = self.base_model(input_ids=input_ids).logits
            logits = logits[:, :-1, :]
            labels = input_ids[:, 1:]
            logprobs = torch.log_softmax(logits, dim=-1)
            token_logprobs = logprobs.gather(
                2, labels.unsqueeze(-1)
            ).squeeze(-1)  # [1, seq-1]
            res.append(float(token_logprobs.sum().item()))
        return res

    # ======================================================
    # generate_until: minimal impl for harness generation tasks
    # ======================================================
    def generate_until(
        self,
        requests: Iterable[Tuple[str, Optional[List[str]]]],
    ) -> List[str]:
        """
        Each request: (context, until_list)
        Return generated text (string) per item, stopping when any string in until_list occurs
        or when max_gen_toks reached.
        """
        reqs = list(requests)
        logging.debug(f"generate_until(): received {len(reqs)} requests")
        outputs = []
        for context, until in reqs:
            ctx_ids = self.tok_encode(context)
            input_ids = torch.tensor([ctx_ids], device=self.device)

            if self.use_medusa_generate and hasattr(self.medusa, "medusa_generate"):
                # use Medusa speculative decoding
                gen_text = self._generate_with_medusa(input_ids, until)
            else:
                gen_text = self._generate_with_base(input_ids, until)
            outputs.append(gen_text)
        return outputs

    def _generate_with_base(self, input_ids, until):
        max_new_tokens = self.max_gen_toks
        generate_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # deterministic for eval
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        with torch.no_grad():
            out_ids = self.base_model.generate(**generate_kwargs)
        gen_ids = out_ids[0][input_ids.shape[1] :]
        text = self.tok_decode(gen_ids.tolist())

        if until:
            text = _truncate_at_any(text, until)
        logging.debug(f"_generate_with_base: generated {len(gen_ids)} tokens, {len(text)} chars")
        return text

    def _generate_with_medusa(self, input_ids, until):
        """
        Call Medusa speculative decoding; assumes signature like used in CLI.
        """
        with torch.no_grad():
            stream = self.medusa.medusa_generate(
                input_ids,
                temperature=0.0,
                max_steps=self.max_gen_toks,
            )
            # CLI's medusa_generate returns an iterator of text chunks; here we collect
            if isinstance(stream, str):
                text = stream
            else:
                text = "".join(list(stream))
        if until:
            text = _truncate_at_any(text, until)
        logging.debug(f"_generate_with_medusa: generated {len(text)} chars")
        return text


def _truncate_at_any(text: str, stops: Optional[List[str]]) -> str:
    if not stops:
        return text
    idxs = [text.find(s) for s in stops if s in text]
    if not idxs:
        return text
    cut = min(i for i in idxs if i >= 0)
    return text[:cut]


# ============================================================
# main
# ============================================================
def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="Medusa model name or local path (same as CLI --model).")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Eval batch size passed to harness.")
    parser.add_argument("--shots", type=int, default=5,
                        help="Few-shot setting for MMLU (0 = zero-shot).")
    parser.add_argument("--output", type=str, default=None,
                        help="Where to write results JSON.")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--use-medusa-generate", action="store_true",
                        help="Use medusa speculative decoding for generation tasks.")
    args = parser.parse_args()
    logging.info(f"Args: {args}")

    try:
        logging.info("Loading Medusa model...")
        medusa_model = MedusaModel.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
        )
        tokenizer = medusa_model.get_tokenizer()
        logging.info("Model loaded. Building LM wrapper...")

        lm = MedusaLM(
            medusa_model=medusa_model,
            tokenizer=tokenizer,
            max_batch_size=args.batch_size,
            use_medusa_generate=args.use_medusa_generate,
        )

        logging.info("Starting evaluation with lm-evaluation-harness...")
        results = simple_evaluate(
            model=lm,
            tasks=["mmlu"],
            batch_size=args.batch_size,
            num_fewshot=args.shots,
        )
        logging.info("Evaluation complete.")
        print(json.dumps(results, indent=2, ensure_ascii=False))
        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logging.info(f"Results written to {args.output}")
    except Exception as e:
        logging.exception("Evaluation failed with an exception.")
        raise


if __name__ == "__main__":
    main()