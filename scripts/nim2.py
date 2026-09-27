from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

completion = client.chat.completions.create(
  model="openai/gpt-oss-120b",
  messages=[{"role":"user","content":"Which number is larger, 9.11 or 9.8?"}],
  temperature=1,
  top_p=1,
  max_tokens=4096,
  stream=False
)

reasoning = getattr(completion.choices[0].message, "reasoning_content", None)
if reasoning:
  print(reasoning)
print(completion.choices[0].message.content)