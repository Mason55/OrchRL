# Trajectory Example

Prerequisites: a reachable vLLM endpoint, the OrchRL Search MAS dependencies installed, and the retrieval service used by `examples/mas_app/search` running.

Run from the repository root:

```bash
python examples/trajectory/run_trajectory_example.py \
  --vllm-url http://127.0.0.1:8000 \
  --model Qwen3-4B-Instruct-2507 \
  --mas-dir examples/mas_app/search \
  --config examples/mas_app/search/configs/search_mas_example.yaml \
  --expected-answer "wilhelm conrad rontgen"
```
