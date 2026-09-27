from dotenv import load_dotenv

from src.models.api_clients import GeminiClient, GroqClient, OpenRouterClient
from src.models.local_clients import RemoteGPUClient

load_dotenv()

PROMPT = "भारत की राजधानी क्या है?"


def main():
    clients = [
        GeminiClient(),
        GroqClient(),
        OpenRouterClient(model_name="deepseek/deepseek-chat"),
        RemoteGPUClient(hf_model_id="meta-llama/Llama-3.1-8B-Instruct"),
        RemoteGPUClient(hf_model_id="google/gemma-2-9b-it"),
        RemoteGPUClient(
            hf_model_id="ai4bharat/Airavata",
            prompt_format="<|user|>\n{prompt}\n<|assistant|>\n",
        ),
    ]

    for client in clients:
        print(client.model_id)
        print(client.generate(PROMPT))
        print()


if __name__ == "__main__":
    main()