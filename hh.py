from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

client = InferenceClient(
    provider="hf-inference",
    api_key=os.getenv("HF_TOKEN")
)

response = client.chat_completion(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    messages=[
        {"role": "user", "content": "What is cancer?"}
    ]
)

print(response.choices[0].message.content)