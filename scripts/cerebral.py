# !pip install cerebras-cloud-sdk

import os

from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras

load_dotenv()


def get_cerebras_client():
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "CEREBRAS_API_KEY is not set. Add it to your environment or .env file before running this script."
        )
    return Cerebras(api_key=api_key)


if __name__ == "__main__":
    client = get_cerebras_client()

    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": "Why is fast inference important?"}],
        model="gemma-4-31b",
        max_completion_tokens=1024,
        temperature=0.2,
        top_p=1,
        stream=False,
        reasoning_effort="medium",
    )

    print(completion.choices[0].message.content)