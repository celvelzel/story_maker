import os

from openai import OpenAI


BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "sk-local-test")
MODEL_NAME = "qwen-local"


client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def test_standard_completion() -> None:
    print("=== Standard completion ===")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Say hello in one short sentence."},
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content or ""
    print(content)
    print()


def test_streaming_completion() -> None:
    print("=== Streaming completion ===")
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Count from 1 to 3, separated by spaces."},
        ],
        temperature=0.2,
        stream=True,
    )

    print("Streaming output:", end=" ", flush=True)
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            print(delta, end="", flush=True)
    print()
    print()


if __name__ == "__main__":
    test_standard_completion()
    test_streaming_completion()
