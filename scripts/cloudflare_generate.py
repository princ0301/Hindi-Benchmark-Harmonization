import os
import requests

ACCOUNT_ID = os.environ["CLOUDFLARE_ACCOUNT_ID"]
API_TOKEN = os.environ["CLOUDFLARE_API_TOKEN"]

url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run"

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json",
}

payload = {
    "model": "google/gemini-3.7-flash",
    "input": {
        "contents": [
            {
                "parts": [
                    {
                        "text": "What are the three laws of thermodynamics?"
                    }
                ],
                "role": "user",
            }
        ]
    },
}

response = requests.post(
    url,
    headers=headers,
    json=payload,
)

response.raise_for_status()

result = response.json()

print(result)