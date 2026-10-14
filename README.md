\# Practical LLM Lab



Hands-on experiments with local Large Language Models using Python and Ollama.



\## Experiments



\- `context\_experiment.py` — Measures prompt tokens, output tokens, inference speed, and request latency.

\- `context\_comparison.py` — Compares performance with 2,048-token and 4,096-token context settings.



\## Technology Stack



\- Python 3.11

\- Ollama

\- Qwen 2.5 Coder 3B

\- NVIDIA RTX 2050



\## Setup



Install Python and Ollama, then download the model:



```powershell

ollama pull qwen2.5-coder:3b

```



Start Ollama and run the experiments:



```powershell

python context\_experiment.py

python context\_comparison.py

```



\## Notes



Benchmark results depend on hardware, model loading, context settings, and runtime conditions. Repeat tests before drawing conclusions.

