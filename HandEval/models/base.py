from abc import ABC, abstractmethod
from pathlib import Path

class BaseLM(ABC):
    """
    统一的大语言模型接口。所有具体模型 wrapper 必须实现下列 3 个方法。
    """

    def __init__(self, ckpt: str | Path, **kwargs):
        self.ckpt = Path(ckpt)

    @classmethod
    @abstractmethod
    def add_cli_args(cls, parser):
        """
        向顶层 ArgumentParser 注入与该模型相关的专属参数。
        例如 Vicuna 需要 --revision, Eagle 需要 --use_flash 等。
        """

    @abstractmethod
    def generate(self, prompt: str, *, temperature=0.0, max_steps=256) -> str:
        """
        核心推理 API，返回完整文本。内部自行处理 batch / tokenizer / device。
        """

    def cleanup(self):
        """
        可选：显式释放显存，或保存 KV 缓存。
        """