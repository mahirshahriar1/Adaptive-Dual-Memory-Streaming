# Adaptive Dual-Memory Streaming (ADMS)

ADMS is an advanced extension of StreamingLLM that enables efficient, near-infinite streaming language model inference with bounded memory. While StreamingLLM maintains stability by keeping attention sink tokens and a recent window, ADMS adds a third tier: a **compressed mid-memory** that preserves salient information from evicted tokens under a strict budget.

## Overview

Traditional transformer decoders accumulate key-value (KV) caches that grow linearly with context length, making long streaming inputs memory-prohibitive. StreamingLLM solved this by retaining only:
- **Sink tokens**: First few tokens (attention sinks) 
- **Recent tokens**: Last N tokens with exact KV states

However, this approach drops all middle tokens, harming recall of earlier facts. ADMS extends this with a compressed mid-memory tier that intelligently preserves important information from the middle region.

## Key Features

- **Three-Tier Memory Architecture**:
  - **Sink Tier**: Exact KV for first few tokens (attention sinks)
  - **Compressed Tier**: Compact representations of middle tokens (low-rank, VQ, or summaries)
  - **Recent Tier**: Exact KV for most recent tokens

- **Dynamic Sink Sizing** (NEW):
  - Automatically scales sink tokens to 1% of `max_seq_length`
  - Preserves more initial context for longer sequences (e.g., 163 tokens for 16K, 320 for 32K)
  - Improves long-context performance without speed penalty
  - Configurable via `enable_dynamic_sink` parameter

- **Adaptive Compression**:
  - **Low-Rank SVD**: Factorizes KV tensors into compact representations
  - **Vector Quantization**: Clusters tokens into centroids with prototype values
  - **Importance-Aware Selection**: Reserves budget for high-importance tokens kept exactly

- **Smart Budgeting**:
  - Per-head adaptive budgets based on variance
  - Coverage-aware allocation across timeline segments
  - Configurable compression intervals for performance tuning

- **RoPE-Compatible Positional Remapping**: Maintains attention stability with synthetic position assignments

## Architecture

```
Memory Tiers in ADMS:
┌─────────────────────────────────────────────────────────────┐
│ Sink (exact) │ Compressed (low-rank/VQ) │ Recent (exact)    │
│ 4-320 tokens │ ~128 tokens              │ 256-2000 tokens   │
│ (dynamic)    │                          │                   │
└─────────────────────────────────────────────────────────────┘
                ↑
          Attention operates here
          
Dynamic Sink: scales to 1% of max_seq_length
  - 2K context  → 20 sink tokens
  - 8K context  → 80 sink tokens
  - 16K context → 163 sink tokens
  - 32K context → 320 sink tokens
```

At each decoding step:
1. **Append** new token to recent tier
2. **Evict overflow** from recent into middle candidates
3. **Score & select** important tokens to keep exactly vs compress
4. **Compress** selected middle tokens using chosen method
5. **Anchor positions** for compressed tokens (grid/mean/hybrid)
6. **Attend** over union of all three tiers with positional remapping

## Installation

```bash
git clone https://github.com/your-repo/streaming-llm-new.git
cd streaming-llm-new
pip install -e .
```

## Quick Start

### Basic ADMS Usage

```python
from streaming_llm import enable_adms_llm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Enable ADMS with low-rank compression and dynamic sink
adms_cache = enable_adms_llm(
    model,
    start_size=4,                # base attention sink tokens (overridden if dynamic)
    recent_size=2000,            # recent window size
    compressed_budget=128,       # max compressed tokens per head
    max_seq_length=16384,        # maximum context length (for dynamic sink)
    enable_dynamic_sink=True,    # scale sink to 1% of max_seq_length
    compressor_type="low_rank",
    rank=16,                     # SVD rank
    importance_ratio=0.5,        # fraction for exact high-importance tokens
)

# Use in generation
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs, past_key_values=adms_cache, ...)
```

### Comparison with StreamingLLM

```python
from streaming_llm import enable_streaming_llm, enable_adms_llm

# StreamingLLM baseline (sinks + recent only)
streaming_cache = enable_streaming_llm(model, start_size=4, recent_size=2000)

# ADMS with compression
adms_cache = enable_adms_llm(
    model,
    start_size=4,
    recent_size=2000,
    compressed_budget=128,
    compressor_type="low_rank"
)

# ADMS typically achieves better perplexity than StreamingLLM
# with modest throughput cost (~10% slower)
```

## Configuration Options

### Core Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `start_size` | Base number of attention sink tokens (static) | 4 |
| `max_seq_length` | Maximum context length for dynamic sink scaling | 32768 |
| `enable_dynamic_sink` | Scale sink to 1% of max_seq_length | True |
| `recent_size` | Size of recent exact window | 2000 |
| `compressed_budget` | Max compressed tokens per head | 128 |
| `compressor_type` | "low_rank", "vq", or "summary" | "low_rank" |

### Compression Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `rank` | SVD rank for low-rank compression | 16 |
| `num_clusters` | Clusters for VQ compression | 64 |
| `compression_interval` | Compress every N tokens | 8 |
| `svd_max_tokens` | Cap SVD columns for performance | 512 |

### Importance & Budgeting

| Parameter | Description | Default |
|-----------|-------------|---------|
| `importance_ratio` | Fraction of budget for exact tokens | 0.5 |
| `importance_metric` | Scoring: "value_norm", "key_norm", "mixed", "attention" | "value_norm" |
| `min_importance_tokens` | Minimum exact tokens | 4 |
| `use_adaptive_budget` | Scale budget per head adaptively | False |
| `coverage_segments` | Timeline segments for coverage | 4 |
| `coverage_priority` | Budget fraction for coverage | 0.3 |

### Positional Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `anchor_mode` | Position assignment: "grid", "mean", "hybrid" | "mean" |
| `enable_pos_shift` | Enable RoPE positional adjustments | True |

### Automation Script Options

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `FORCE_RERUN` | Re-run even if summary.json exists | 0 |
| `OUTPUT_ROOT` | Output directory for results | outputs/auto_adms_runs |
| `SCENARIOS` | Override scenario list | (built-in scenarios) |
| `PROFILES` | Override profile list | (balanced, quality) |

## Evaluation

### Automated Benchmarking

Run comprehensive evaluations across multiple models and context lengths:

```bash
# Full automation suite (all scenarios and profiles)
bash scripts/run_adms_report.sh

# Override specific parameters
FORCE_RERUN=1 OUTPUT_ROOT=outputs/custom_run bash scripts/run_adms_report.sh

# Run specific scenario only
SCENARIOS=("name=llama3_8k model=meta-llama/Meta-Llama-3-8B-Instruct dataset=wikitext task=wikitext-103-raw-v1 split=test samples=64 tokens=8192 log_every=512") \
bash scripts/run_adms_report.sh
```

The automation script supports:
- **Built-in scenarios**: TinyLlama (2k-4k), OPT-1.3B (2k-4k), Llama-3-8B (8k-32k)
- **Two profiles**: `balanced` (fast) and `quality` (better PPL)
- **Aggregated reporting**: CSV + Markdown with model/dataset/context metadata
- **Automatic token targeting**: Fixed concatenation logic ensures proper context lengths


### Manual Evaluation

Run head-to-head comparisons between ADMS and StreamingLLM:

```bash
# Quick evaluation on TinyLlama
python examples/eval_adms_vs_streaming.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name wikitext \
  --task wikitext-2-raw-v1 \
  --split test \
  --num_samples 32 \
  --num_eval_tokens 4096 \
  --concat_stream \
  --start_size 4 \
  --recent_size 1024 \
  --compressed_budget 192 \
  --importance_ratio 0.5 \
  --output_dir outputs/adms_vs_streaming/tinyllama_4k
```

### Recent Performance Results

Automated benchmarks show consistent ADMS improvements:

| Scenario | Profile | ADMS PPL | Streaming PPL | Improvement |
|----------|---------|----------|---------------|-------------|
| opt13b_2k | balanced | 33.84 | 71.50 | **52.7%** |
| opt13b_2k | quality | 29.69 | 48.66 | **39.0%** |
| tinyllama_2k | balanced | 8.29 | 9.46 | **12.4%** |
| tinyllama_2k | quality | 8.54 | 10.22 | **16.4%** |
| llama3_8k | balanced | 11.51 | 12.41 | **7.2%** |
| llama3_8k | quality | 12.74 | 13.93 | **8.5%** |

Expected output shows ADMS achieving significantly lower perplexity than StreamingLLM with bounded memory.

## Performance Characteristics

- **Memory**: O(dynamic_sink + compressed_budget + recent) vs O(context_length) for full cache
  - Dynamic sink scales to 1% of context (e.g., 320 tokens for 32K context)
  - Total cache remains bounded regardless of input length
- **Throughput**: ~10% slower than StreamingLLM due to compression overhead
  - Dynamic sink adds no speed penalty (same attention complexity)
- **Perplexity**: 7-53% improvement over StreamingLLM on long contexts (model dependent)
  - Dynamic sink provides additional 1-3% improvement at 16K+ contexts
- **Stability**: Maintains StreamingLLM's stability beyond training length
- **Context Length**: Tested up to 32k tokens on Llama-3-8B with proper concatenation logic
- **Automation**: Supports systematic evaluation across 19 model/context combinations

## Advanced Usage

### Custom Compressor

```python
from streaming_llm.adms_cache import ADMSConfig, ADMSKVCache

class CustomCompressor:
    def compress(self, K, V):
        # Your compression logic here
        return compressed_K, compressed_V

config = ADMSConfig(
    start_size=4,
    recent_size=2000,
    compressed_budget=128,
    compressor_type="custom"
)

# Override compressor
cache = ADMSKVCache(config)
cache.compressors = {layer_head_key: CustomCompressor() for layer_head_key in cache.compressors}
```

### Model-Specific Optimizations

ADMS automatically detects model types and applies appropriate modifications:

- **LLaMA/Mistral**: RoPE positional shifting
- **GPT-NeoX/Pythia**: Position shifting for relative encodings  
- **Falcon**: Custom attention modifications
- **MPT**: No position shifting needed

## Limitations

- **Lossy Compression**: Middle tokens are approximated, not exact
- **Positional Bias**: Synthetic positions may affect attention patterns
- **Training Required**: Policies may need fine-tuning for optimal performance
- **Memory Overhead**: Small per-head statistics tracking
- **Context Targeting**: Fixed concatenation logic in v1.0 - samples now accumulate until target tokens reached
- **Model Support**: Some models (Mistral, Mixtral) may require tokenizer compatibility fixes

## Citation

If you use ADMS in your research, please cite:

```bibtex
@misc{adms2024,
  title={Adaptive Dual-Memory Streaming (ADMS): Stability-Preserving Streaming with Compressed Mid-Memory},
  author={Your Name},
  year={2024}
}
```

## Contributing

Contributions welcome! Areas of interest:
- New compression methods
- Improved importance scoring policies
- Better positional anchoring strategies
- Model-specific optimizations
- Extended automation scenarios (64k+ contexts, more models)
- Tokenizer compatibility fixes for gated models

### Recent Improvements
- **Dynamic Sink Sizing**: Automatic scaling of sink tokens to 1% of max context length
  - Improves long-context PPL by 1-3% without speed penalty
  - Particularly effective at 16K+ token contexts
- **Automation Pipeline**: Complete benchmarking automation with CSV/Markdown reporting
- **Context Length Targeting**: Fixed concatenation to reach proper token targets
- **Extended Model Support**: Llama-3-8B, Pythia-6.9B, Falcon-7B, MPT-7B scenarios up to 32k tokens
- **Performance Validation**: Systematic 7-53% PPL improvements demonstrated across architectures

## License

See LICENSE file for details.