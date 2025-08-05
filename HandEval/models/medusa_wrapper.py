# models/vicuna_wrapper.py
from . import register

from medusa.model.medusa_model import MedusaModel
import torch
from .base import BaseLM

def m_generate(model, tokenizer, prompt: str, *, temperature: float = 0.0, max_steps: int = 20000, sampling: str = "typical", top_p: float | None = None) -> str:
    """
    One‑shot wrapper around model.medusa_generate that returns the full text.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.base_model.device)
    gen_iter = model.medusa_generate(
        input_ids,
        temperature=temperature,
        max_steps=max_steps,
        sampling=sampling,
        top_p=top_p,
        fast=False,
    )
    out_parts, prev_len = [], 0
    for chunk in gen_iter:
        full = chunk["text"]
        out_parts.append(full[prev_len:])
        prev_len = len(full)
    return "".join(out_parts).strip()

@register("medusa")
class MedusaLM(BaseLM):
    @classmethod
    def add_cli_args(cls, parser):
        g = parser.add_argument_group("Medusa")
        g.add_argument('--revision', type=str, default='main')
        g.add_argument('--temperature', type=float, default=0.7,
                       help='Default sampling temperature for Medusa generation')
        g.add_argument('--top_p', type=float, default=None,
                       help='Top‑p value for nucleus sampling (ignored for other strategies)')
        

    def __init__(self, ckpt, revision='main', temperature=0.7, sampling='typical', top_p=None, **_):
        super().__init__(ckpt)

        self.model = MedusaModel.from_pretrained(
            ckpt,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        self.tokenizer = self.model.get_tokenizer()
        self.default_temperature = temperature
        self.top_p = top_p
        if temperature == 0.0:
            self.sampling = "greedy"
        else:
            self.sampling = 'nucleus' if top_p is not None else 'typical'
        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompt, temperature=None, max_steps=256, top_p=None):
        if temperature is None:
            temperature = self.default_temperature
        if top_p is None:
            top_p = self.top_p
        return m_generate(
            self.model,
            self.tokenizer,
            prompt,
            temperature=temperature,
            max_steps=max_steps,
            sampling=self.sampling,
            top_p=top_p
        )