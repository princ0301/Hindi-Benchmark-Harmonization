import modal

from src.models.base import ModelClient


class RemoteGPUClient(ModelClient):
    def __init__(self, hf_model_id: str, load_in_4bit: bool = False, gpu: str | None = None, prompt_format: str = ""):
        self._hf_model_id = hf_model_id
        gpu_cls = modal.Cls.from_name("hindi-benchmark-gpu", "GPUModel")
        if gpu:
            gpu_cls = gpu_cls.with_options(gpu=gpu)
        self._instance = gpu_cls(model_id=hf_model_id, load_in_4bit=load_in_4bit, prompt_format=prompt_format)

    @property
    def model_id(self) -> str:
        return self._hf_model_id

    def generate(self, prompt: str, temperature: float = 0.0, max_new_tokens: int = 512) -> str:
        return self._instance.generate.remote(prompt, temperature, max_new_tokens)

    def generate_batch(self, prompts: list[str], temperature: float = 0.0, max_new_tokens: int = 512) -> list[str]:
        return self._instance.generate_batch.remote(prompts, temperature, max_new_tokens)