import os

from dotenv import load_dotenv
from together import Together

load_dotenv()


def get_together_client():
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TOGETHER_API_KEY is not set. Add it to your environment or .env file before running this script."
        )
    return Together(api_key=api_key)


def main():
    client = get_together_client()

    response = client.chat.completions.create(
        model="google/gemma-4-31B-it",
        messages=[
            {
                "role": "user",
                "content": "What are some fun things to do in New York?",
            }
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()