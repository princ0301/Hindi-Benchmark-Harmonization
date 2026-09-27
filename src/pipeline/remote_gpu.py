import modal

app = modal.App("hindi-benchmark-gpu")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "transformers>=4.44",
        "accelerate>=0.33",
        "bitsandbytes>=0.43",
        "torch",
        "sentencepiece",
    )
)

weights_volume = modal.Volume.from_name("hindi-benchmark-weights", create_if_missing=True)

CACHE_PATH = "/cache"


@app.cls(
    image=image,
    gpu="A10G",
    volumes={CACHE_PATH: weights_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=600,
    scaledown_window=120,
)
class GPUModel:
    model_id: str = modal.parameter()
    load_in_4bit: bool = modal.parameter(default=False)
    prompt_format: str = modal.parameter(default="")

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=CACHE_PATH)

        quantization_config = None
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            cache_dir=CACHE_PATH,
            dtype=torch.bfloat16,
            device_map="cuda",
            quantization_config=quantization_config,
        )

    def _render_prompt(self, prompt: str) -> str:
        if self.prompt_format:
            return self.prompt_format.format(prompt=prompt)
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    @modal.method()
    def generate(self, prompt: str, temperature: float = 0.0, max_new_tokens: int = 512) -> str:
        import torch

        text = self._render_prompt(prompt)
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    @modal.method()
    def generate_batch(self, prompts: list[str], temperature: float = 0.0, max_new_tokens: int = 512) -> list[str]:
        return [self.generate.local(p, temperature, max_new_tokens) for p in prompts]


@app.local_entrypoint()
def main():
    model = GPUModel(model_id="google/gemma-2-9b-it")
    result = model.generate.remote("भारत की राजधानी क्या है?")
    print(result)