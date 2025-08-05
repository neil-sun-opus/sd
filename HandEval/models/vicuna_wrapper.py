# models/vicuna_wrapper.py
from importlib import import_module
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from . import register          # 从 models/__init__.py 引入注册器
from .base import BaseLM        # 统一接口

def hf_generate(model, tokenizer, prompt: str,
                *, temperature: float = 0.7,
                max_steps: int = 256,
                sampling: str = "typical",
                top_p: float | None = None) -> str:
    """
    通用 HuggingFace 文本生成辅助。
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = sampling != "greedy"
    gen_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_steps,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
    )
    if sampling == "nucleus":
        gen_kwargs["top_p"] = 1.0 if top_p is None else top_p
    # 其他 sampling 策略可按需扩展

    output = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(output[0], skip_special_tokens=True)


@register("vicuna")           # 注册名需与 CLI --model 一致
class VicunaLM(BaseLM):
    @classmethod
    def add_cli_args(cls, parser):
        g = parser.add_argument_group("Vicuna")
        g.add_argument('--revision', type=str, default='main')
        g.add_argument('--temperature', type=float, default=0.7,
                       help='Default temperature')
        g.add_argument('--sampling', type=str, default='greedy',
                       choices=['greedy', 'typical', 'nucleus'],
                       help='Sampling strategy')
        g.add_argument('--top_p', type=float, default=None,
                       help='top-p for nucleus')
        g.add_argument('--dtype', type=str, default='float16',
                       choices=['float16', 'bfloat16', 'float32'],
                       help='Model dtype')

    def __init__(self, ckpt, *,
                 revision='main',
                 temperature=0.7,
                 sampling='greedy',
                 top_p=None,
                 dtype='float16',
                 **_):
        super().__init__(ckpt)
        dtype_map = dict(float16=torch.float16,
                         bfloat16=torch.bfloat16,
                         float32=torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            ckpt,
            revision=revision,
            torch_dtype=dtype_map[dtype],
            device_map='auto',
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        # 默认推理超参
        self.default_temperature = temperature
        self.sampling = sampling
        self.top_p = top_p

    @torch.inference_mode()
    def generate(self, prompt, *,
                 temperature=None,
                 max_steps=256,
                 top_p=None):
        if temperature is None:
            temperature = self.default_temperature
        if top_p is None:
            top_p = self.top_p
        return hf_generate(
            self.model,
            self.tokenizer,
            prompt,
            temperature=temperature,
            max_steps=max_steps,
            sampling=self.sampling,
            top_p=top_p,
        )