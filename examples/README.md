# Streaming Evaluations

This directory houses runnable scripts for comparing Adaptive Dual-Memory Streaming (ADMS) against the baseline StreamingLLM cache.

## Quick benchmark

Use the evaluation driver to measure perplexity and throughput while sweeping the new hybrid retention knobs:

```powershell
python examples/eval_adms_vs_streaming.py `
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 `
  --dataset_name wikitext `
  --task wikitext-2-raw-v1 `
  --split test `
  --num_samples 8 `
  --start_size 4 `
  --recent_size 2000 `
  --compressed_budget 192 `
  --importance_ratio 0.5 `
  --min_importance_tokens 8 `
  --importance_metric value_norm `
  --concat_stream `
  --coverage_segments 4 `
  --coverage_priority 0.35 `
  --output_dir outputs/adms_vs_streaming/tinyllama
```

The command above:

- Loads a lightweight TinyLlama checkpoint that is easy to run on a single GPU.
- Streams eight test samples from WikiText-2 (raw) while concatenating them into one long sequence, which stresses the middle cache.
- Keeps 50% of the compressed budget for high-importance exact tokens (with a floor of eight) and fills the remainder using low-rank compression.
- Writes JSON and text summaries into `outputs/adms_vs_streaming/tinyllama`.

Adjust `--compressed_budget`, `--importance_ratio`, and `--importance_metric` to explore the perplexity/speed trade-off. For very long streams, you may also increase `--compression_interval` to reduce SVD frequency.

## Key arguments

| Flag | Purpose |
| ---- | ------- |
| `--start_size` | Size of the attention sink tier shared by both methods. |
| `--recent_size` | Length of the exact recent window. Larger values increase latency but reduce perplexity. |
| `--compressed_budget` | Maximum number of proxy tokens kept by ADMS per layer/head. |
| `--importance_ratio` | Fraction of the budget reserved for the highest-scoring middle tokens that are kept exactly. |
| `--min_importance_tokens` | Minimum number of exact tokens when `importance_ratio` is non-zero. |
| `--importance_metric` | Score used to rank middle tokens (`value_norm`, `key_norm`, or `mixed`). |
| `--compression_interval` | Compress every N new tokens to throttle SVD cost. |
| `--svd_max_tokens` | Cap on columns used in low-rank SVD; helps control runtime on long streams. |
| `--coverage_segments` | Number of segments used to enforce coverage across the middle timeline. |
| `--coverage_priority` | Fraction of the compressed budget reserved for per-segment picks (0 disables coverage). |
| `--concat_stream` | If set, joins multiple samples into one long sequence before evaluation. |

## Outputs

Each run produces:

- `summary.json` – machine-readable metrics and the full set of arguments.
- `readable.txt` – plain-text recap of the configuration and results.
- Console logs highlighting online perplexity estimates and the final comparison table.

## Tips

- Enable `--enable_pos_shift` to match the positional adjustment used by the StreamingLLM baseline scripts when the model requires it (e.g., LLaMA variants).
- `--num_eval_tokens` can be set to cap the total evaluation length without truncating individual samples.
- For vector-quantized compression, switch to `--compressor_type vq` and tune `--num_clusters` to the desired fidelity.

## Experiment log

### 2025-09-30 · Long-stream TinyLlama benchmark

```
python examples/eval_adms_vs_streaming.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name wikitext \
  --task wikitext-2-raw-v1 \
  --split test \
  --num_samples 32 \
  --concat_stream \
  --start_size 4 \
  --recent_size 1024 \
  --compressed_budget 192 \
  --importance_ratio 0.5 \
  --min_importance_tokens 8 \
  --importance_metric value_norm \
  --output_dir outputs/adms_vs_streaming/tinyllama_long
```

**Outcome**

| Method         | PPL   | tok/s | Time (s) | Tokens |
|---------------|-------|-------|----------|--------|
| ADMS           | 56.75 | 39.6  | 84.9     | 3360   |
| StreamingLLM   | 119.06| 44.2  | 75.9     | 3360   |

**Why it succeeded**

- Concatenating 32 samples pushed the stream to 3360 tokens, exceeding the `start_size + recent_size` boundary so a true “middle” region formed.
- The hybrid retention scheme reserved half of the compressed budget for the highest-value middle tokens, preserving critical facts exactly before applying low-rank compression to the remainder.
- StreamingLLM had to drop all middle tokens once they slid past the 1024-token recent window, doubling perplexity, while ADMS retained and reused them at a modest throughput cost (≈10% slower).
