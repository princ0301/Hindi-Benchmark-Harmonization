import os

from tenacity import retry, stop_after_attempt, wait_exponential

from src.models.base import ModelClient
from src.pipeline.rate_limiter import RateLimiter

RETRY_CONFIG = dict(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=60))


class GeminiClient(ModelClient):
    def __init__(self, model_name: str = "gemini-3.6-flash"):
        from google import genai

        self._model_name = model_name
        self._client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    @property
    def model_id(self) -> str:
        return self._model_name

    @retry(**RETRY_CONFIG)
    def generate(self, prompt: str, temperature: float = 0.0, max_new_tokens: int = 512) -> str:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_new_tokens,
            ),
        )
        return response.text


class GroqClient(ModelClient):
    def __init__(self, model_name: str = "openai/gpt-oss-120b", reasoning_effort: str | None = "low"):
        from groq import Groq

        self._model_name = model_name
        self._reasoning_effort = reasoning_effort
        self._client = Groq(api_key=os.environ["GROQ_API_KEY"])

    @property
    def model_id(self) -> str:
        return self._model_name

    @retry(**RETRY_CONFIG)
    def generate(self, prompt: str, temperature: float = 0.0, max_new_tokens: int = 512) -> str:
        kwargs = dict(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        if self._reasoning_effort is not None:
            kwargs["reasoning_effort"] = self._reasoning_effort

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


class OpenRouterClient(ModelClient):
    _rate_limiter = RateLimiter(max_calls=15, period_seconds=60)

    def __init__(self, model_name: str):
        from openai import OpenAI

        self._model_name = model_name
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    @property
    def model_id(self) -> str:
        return self._model_name

    @retry(**RETRY_CONFIG)
    def generate(self, prompt: str, temperature: float = 0.0, max_new_tokens: int = 512) -> str:
        self._rate_limiter.acquire()
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        return response.choices[0].message.content


class NvidiaNimClient(ModelClient):
    def __init__(self, model_name: str = "google/gemma-4-31b-it", enable_thinking: bool = False):
        from openai import OpenAI

        self._model_name = model_name
        self._enable_thinking = enable_thinking
        self._client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ["NVIDIA_API_KEY"],
        )

    @property
    def model_id(self) -> str:
        return self._model_name

    @retry(**RETRY_CONFIG)
    def generate(self, prompt: str, temperature: float = 0.0, max_new_tokens: int = 512) -> str:
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_new_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": self._enable_thinking}},
        )
        return response.choices[0].message.content


class TogetherClient(ModelClient):
    def __init__(self, model_name: str):
        from together import Together

        self._model_name = model_name
        self._client = Together(api_key=os.environ["TOGETHER_API_KEY"])

    @property
    def model_id(self) -> str:
        return self._model_name

    @retry(**RETRY_CONFIG)
    def generate(self, prompt: str, temperature: float = 0.0, max_new_tokens: int = 512) -> str:
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        return response.choices[0].message.content


class OllamaClient(ModelClient):
    def __init__(self, model_name: str = "gemma4:31b-cloud"):
        from openai import OpenAI

        self._model_name = model_name
        self._client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    @property
    def model_id(self) -> str:
        return self._model_name

    @retry(**RETRY_CONFIG)
    def generate(self, prompt: str, temperature: float = 0.0, max_new_tokens: int = 512) -> str:
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        return response.choices[0].message.content