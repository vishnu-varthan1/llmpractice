
import json
import time
import urllib.request

MODEL = "qwen2.5-coder:3b"
API_URL = "http://127.0.0.1:11434/api/chat"

PROMPT = (
    "Explain how context windows affect local LLM memory use, "
    "prompt processing, and response latency. Give practical examples."
)

def run_test(context_size):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False,
        "options": {
            "num_ctx": context_size,
            "num_predict": 120,
            "temperature": 0,
        },
    }

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()

    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.loads(response.read().decode("utf-8"))

    wall_time = time.perf_counter() - start

    output_tokens = result.get("eval_count", 0)
    generation_ns = result.get("eval_duration", 0)
    generation_seconds = generation_ns / 1e9

    speed = (
        output_tokens / generation_seconds
        if generation_seconds > 0 else 0
    )

    print(f"\nContext setting: {context_size}")
    print(f"Input tokens: {result.get('prompt_eval_count', 0)}")
    print(f"Output tokens: {output_tokens}")
    print(f"Generation speed: {speed:.2f} tokens/sec")
    print(f"Wall time: {wall_time:.2f} seconds")

if __name__ == "__main__":
    # Warm-up to reduce first-load bias.
    run_test(2048)

    # Compare identical prompts at two context settings.
    run_test(2048)
    run_test(4096)

    print("\nCheck memory and processor placement with: ollama ps")
