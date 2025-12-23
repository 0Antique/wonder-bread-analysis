import os
import requests

url = "https://api.siliconflow.cn/v1/chat/completions"
api_key = "sk-lvnlqzekyxpdzybsdyzrigwoqnydcvpuefprddmcmqokhbvv"

if not api_key:
    raise SystemExit("Missing env SILICONFLOW_API_KEY")

payload = {
    "model": "Qwen/Qwen3-8B",
    "messages": [
        {"role": "user", "content": "who are you?"}
    ],
    "stream": False,
    "max_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.9,
}

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

resp = requests.post(url, json=payload, headers=headers, timeout=(10,600))
print("status:", resp.status_code)
print(resp.text)
resp.raise_for_status()

data = resp.json()
print("\nassistant:\n", data["choices"][0]["message"]["content"])