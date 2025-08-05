# models/eagle_wrapper.py
from importlib import import_module
import torch
from eagle.model.ea_model import EaModel

from . import register          # 从 models/__init__.py 引入注册器
from .base import BaseLM        # 统一接口


@register("eagle")              # 注册名需与 CLI --model 一致
class EagleLM(BaseLM):
    @classmethod
    def add_cli_args(cls, parser):
        g = parser.add_argument_group("Eagle")
        g.add_argument('--ea_ckpt', type=str, required=True,
                       help='Path to the EAGLE adapter checkpoint')
        g.add_argument('--temperature', type=float, default=0.7,
                       help='Default temperature')
        g.add_argument('--dtype', type=str, default='float16',
                       choices=['float16', 'bfloat16', 'float32'],
                       help='Model dtype')
        g.add_argument('--top_p', type=float, default=None,
                       help='Top‑p value for nucleus sampling (ignored for other strategies)')
        

    def __init__(self, ckpt, *,
                 ea_ckpt,
                 temperature=0.7,
                 dtype='float16',
                 **kwargs):
        """
        Args:
            ckpt (str): Path to the base Vicuna model.
            ea_ckpt (str): Path to the EAGLE adapter checkpoint.
        """
        super().__init__(ckpt)
        dtype_map = dict(float16=torch.float16,
                         bfloat16=torch.bfloat16,
                         float32=torch.float32)

        # Load EAGLE‑adapted model
        print(f"Loading EAGLE model from {ea_ckpt} with dtype {dtype}")
        print(f"Base model: {ckpt}")
        self.model = EaModel.from_pretrained(
            base_model_path=ckpt,
            ea_model_path=ea_ckpt,
            torch_dtype=dtype_map[dtype],
            device_map='auto',
            use_eagle3= True
        )
        self.tokenizer = self.model.tokenizer
        
        # 默认推理超参
        self.default_temperature = temperature
        self.default_top_p = kwargs.get('top_p', None)

    @torch.inference_mode()
    def generate(self, prompt: str, *,
                 temperature: float | None = None,
                 top_p: float | None = None,
                 max_steps: int = 256):
        """
        EAGLE‑style text generation.

        Args:
            prompt (str): Input prompt string.
            temperature (float | None): Sampling temperature.
            max_steps (int): Maximum new tokens to generate.

        Returns:
            str: The generated completion.
        """
        if temperature is None:
            temperature = self.default_temperature

        if top_p is None:
            top_p = self.default_top_p

        # Tokenize and move to the same device as the base model
        inputs = self.tokenizer(
            prompt, return_tensors="pt"
        ).to(self.model.base_model.device)

        # EAGLE generation
        output_ids = self.model.eagenerate(
            inputs.input_ids,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_steps,
            max_length=8192, # EAGLE supports longer sequences，
            log=False
        )

        # Strip the prompt part and decode
        orig_len = inputs.input_ids.shape[1]
        return self.tokenizer.decode(
            output_ids[0][orig_len:], skip_special_tokens=True
        )