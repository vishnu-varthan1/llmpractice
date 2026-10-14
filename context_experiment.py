
import json
import time
import statistics
import urllib.request
import urllib.error

# Your installed local Ollama model
MODEL = "qwen2.5-coder:3b"

# Ollama local API
API_URL = "http://127.0.0.1:11434/api/chat"

# Experiment prompts
PROMPTS = {
    "Short prompt": (
        "Explain tokens in simple English."
    ),
    "Long prompt": (
        "Explain tokens, context windows, RAM usage, and inference speed. "
        "Describe how they are connected, give practical examples, and "
        "explain how an engineer can measure them when running a local "
        "large language model. Keep the explanation clear and structured. "
    ) * 8,
}


def run_test(label, prompt, context_size=2048):
    """Send a prompt to Ollama and measure model performance."""

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
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

    start_time = time.perf_counter()

    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.loads(response.read().decode("utf-8"))

    wall_seconds = time.perf_counter() - start_time

    # Metrics returned by Ollama
    input_tokens = result.get("prompt_eval_count", 0)
    output_tokens = result.get("eval_count", 0)

    prompt_duration_ns = result.get("prompt_eval_duration", 0)
    generation_duration_ns = result.get("eval_duration", 0)
    load_duration_ns = result.get("load_duration", 0)

    # Convert nanoseconds to seconds
    prompt_seconds = prompt_duration_ns / 1_000_000_000
    generation_seconds = generation_duration_ns / 1_000_000_000
    load_seconds = load_duration_ns / 1_000_000_000

    # Avoid division by zero
    prompt_speed = (
        input_tokens / prompt_seconds
        if prompt_seconds > 0 else 0
    )

    generation_speed = (
        output_tokens / generation_seconds
        if generation_seconds > 0 else 0
    )

    print(f"\n{'=' * 50}")
    print(f"TEST: {label}")
    print(f"{'=' * 50}")

    print(f"Model                 : {MODEL}")
    print(f"Context setting       : {context_size} tokens")
    print(f"Input tokens          : {input_tokens}")
    print(f"Output tokens         : {output_tokens}")
    print(f"Prompt processing     : {prompt_speed:.2f} tokens/sec")
    print(f"Output generation     : {generation_speed:.2f} tokens/sec")
    print(f"Model load time       : {load_seconds:.2f} seconds")
    print(f"Total wall time       : {wall_seconds:.2f} seconds")

    print("\nResponse preview:")
    print(result.get("message", {}).get("content", "")[:300])

    return {
        "label": label,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "prompt_speed": prompt_speed,
        "generation_speed": generation_speed,
        "wall_seconds": wall_seconds,
        "load_seconds": load_seconds,
    }


def main():
    print("LOCAL LLM PERFORMANCE EXPERIMENT")
    print(f"Model: {MODEL}")
    print("Checking connection to Ollama...")

    # Confirm that the Ollama API is reachable
    with urllib.request.urlopen(
        "http://127.0.0.1:11434/api/tags",
        timeout=10
    ) as response:
        available_models = json.loads(
            response.read().decode("utf-8")
        )

    model_names = [
        model["name"] for model in available_models.get("models", [])
    ]

    if MODEL not in model_names:
        raise RuntimeError(
            f"Model '{MODEL}' not found. "
            f"Installed models: {model_names}"
        )

    print("Ollama connection successful.")
    print("Installed model verified.")

    # Warm up the model before collecting measurements
    print("\nWarming up model...")
    run_test(
        "Warm-up",
        "Reply with one short sentence.",
        context_size=2048,
    )

    # Run the actual experiments
    results = []

    for label, prompt in PROMPTS.items():
        results.append(
            run_test(
                label,
                prompt,
                context_size=2048,
            )
        )

    # Display comparison
    print("\n\n" + "=" * 65)
    print("BENCHMARK SUMMARY")
    print("=" * 65)

    for item in results:
        print(
            f"{item['label']}: "
            f"input={item['input_tokens']} tokens | "
            f"output={item['output_tokens']} tokens | "
            f"generation={item['generation_speed']:.2f} tokens/sec | "
            f"wall={item['wall_seconds']:.2f} sec"
        )

    speeds = [
        item["generation_speed"]
        for item in results
        if item["generation_speed"] > 0
    ]

    if speeds:
        print(
            f"\nAverage generation speed: "
            f"{statistics.mean(speeds):.2f} tokens/sec"
        )

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    try:
        main()

    except urllib.error.URLError as error:
        print("\nERROR: Cannot connect to Ollama.")
        print("Check that Ollama is running.")
        print(f"Details: {error}")

    except Exception as error:
        print(f"\nERROR: {type(error).__name__}: {error}")
