# ADM++ Quick Start Guide

This guide shows how to use the newly implemented ADM++ features.

## Basic Usage

### Enable All ADM++ Features (Recommended)

```python
from streaming_llm import enable_adms_llm
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Full ADM++ configuration
adms_cache = enable_adms_llm(
    model,
    start_size=4,
    recent_size=2000,
    compressed_budget=128,
    
    # Dual-Fidelity Mid Memory
    enable_dual_fidelity=True,
    sketch_budget=32,              # Additional sketch tokens
    sketch_reduction="mean",        # mean, sum, or first
    importance_ratio=0.5,          # Precision bank allocation
    
    # Residual Replay Engine
    enable_residual_replay=True,
    replay_budget=16,              # Max replayed tokens
    energy_replay_threshold=0.88,  # Trigger threshold
    
    # RoPE Alignment Calibration
    enable_position_calibration=True,
    calibration_window=512,        # Anchor window size
    calibration_regularization=0.1,
    
    # Adaptive Budget Controller
    enable_adaptive_controller=True,
    controller_gain=0.35,          # EMA smoothing
    controller_energy_floor=0.8,   # Expand budget below this
    controller_energy_ceiling=0.97,# Shrink budget above this
    controller_group_size=2,       # Heads per group
    
    # Base settings
    compressor_type="low_rank",
    rank=16,
)

# Use in generation
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs, past_key_values=adms_cache, max_new_tokens=100)
```

## Feature-Specific Configurations

### 1. Dual-Fidelity Only (Better Coverage)

Focus on preserving more middle tokens via sketching:

```python
adms_cache = enable_adms_llm(
    model,
    start_size=4,
    recent_size=2000,
    compressed_budget=128,
    
    # Enable dual-fidelity
    enable_dual_fidelity=True,
    sketch_budget=48,              # Higher sketch budget
    sketch_reduction="mean",        # Best for general use
    importance_ratio=0.4,          # More room for sketches
    
    # Disable other features
    enable_residual_replay=False,
    enable_position_calibration=False,
    enable_adaptive_controller=False,
)
```

**When to use**: Long contexts where you want maximum coverage of middle tokens.

### 2. Replay Engine Only (Prevent Drift)

Focus on preventing compression artifacts:

```python
adms_cache = enable_adms_llm(
    model,
    start_size=4,
    recent_size=2000,
    compressed_budget=128,
    
    # Enable replay
    enable_residual_replay=True,
    replay_budget=24,              # Higher replay budget
    energy_replay_threshold=0.85,  # More aggressive triggering
    
    # Disable other features
    enable_dual_fidelity=False,
    sketch_budget=0,
    enable_position_calibration=False,
    enable_adaptive_controller=False,
)
```

**When to use**: Tasks sensitive to compression errors (e.g., reasoning, math).

### 3. Calibration Only (Fix Positional Drift)

Focus on position alignment:

```python
adms_cache = enable_adms_llm(
    model,
    start_size=4,
    recent_size=2000,
    compressed_budget=128,
    
    # Enable calibration
    enable_position_calibration=True,
    calibration_window=1024,       # Larger window for stability
    calibration_regularization=0.05, # Less regularization
    
    # Disable other features
    enable_dual_fidelity=False,
    sketch_budget=0,
    enable_residual_replay=False,
    enable_adaptive_controller=False,
)
```

**When to use**: Very long contexts (32k+) where position drift is an issue.

### 4. Controller Only (Adaptive Budgets)

Focus on optimal budget allocation:

```python
adms_cache = enable_adms_llm(
    model,
    start_size=4,
    recent_size=2000,
    compressed_budget=128,
    
    # Enable controller
    enable_adaptive_controller=True,
    controller_gain=0.4,           # More responsive
    controller_energy_floor=0.75,  # More aggressive expansion
    controller_energy_ceiling=0.98,# Conservative shrinking
    controller_group_size=1,       # Per-head control
    
    # Disable other features
    enable_dual_fidelity=False,
    sketch_budget=0,
    enable_residual_replay=False,
    enable_position_calibration=False,
)
```

**When to use**: Variable-difficulty tasks where different heads need different budgets.

## Ablation Studies

### Baseline: Pure ADMS (No ADM++)

```python
adms_cache = enable_adms_llm(
    model,
    start_size=4,
    recent_size=2000,
    compressed_budget=128,
    
    # Disable all ADM++ features
    enable_dual_fidelity=False,
    sketch_budget=0,
    enable_residual_replay=False,
    replay_budget=0,
    enable_position_calibration=False,
    enable_adaptive_controller=False,
)
```

### Progressive Enabling

```python
# Level 1: Base ADMS
config_l1 = {
    "enable_dual_fidelity": False,
    "enable_residual_replay": False,
    "enable_position_calibration": False,
    "enable_adaptive_controller": False,
}

# Level 2: + Dual-Fidelity
config_l2 = {
    **config_l1,
    "enable_dual_fidelity": True,
    "sketch_budget": 32,
}

# Level 3: + Replay
config_l3 = {
    **config_l2,
    "enable_residual_replay": True,
    "replay_budget": 16,
}

# Level 4: + Calibration
config_l4 = {
    **config_l3,
    "enable_position_calibration": True,
}

# Level 5: Full ADM++ (+ Controller)
config_l5 = {
    **config_l4,
    "enable_adaptive_controller": True,
}
```

## Command-Line Usage

### Using eval_adms_vs_streaming.py

```bash
# Full ADM++ (default)
python examples/eval_adms_vs_streaming.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name wikitext \
  --task wikitext-2-raw-v1 \
  --split test \
  --num_samples 32 \
  --num_eval_tokens 4096 \
  --start_size 4 \
  --recent_size 1024 \
  --compressed_budget 192

# Disable specific features
python examples/eval_adms_vs_streaming.py \
  ... \
  --disable_dual_fidelity \
  --disable_residual_replay \
  --disable_position_calibration \
  --disable_adaptive_controller \
  --sketch_budget 0 \
  --replay_budget 0

# Custom feature mix
python examples/eval_adms_vs_streaming.py \
  ... \
  --disable_residual_replay \
  --disable_position_calibration \
  --sketch_budget 64 \
  --energy_replay_threshold 0.85
```

## Monitoring Performance

### Interpreting Stats Output

```
[ADMS Stats @ 2048] Avg per head: exact=45.2, compressed=67.8, dropped=112.0
[ADMS Dual-Fidelity @ 2048] Sketches: 234, Promotions: 18
[ADMS Replay @ 2048] Triggers: 12, Tokens: 156
[ADMS Calibration @ 2048] Total calibrations: 89
[ADMS Controller @ 2048] Budget adjustments: 145
[ADMS Cache @ 2048] Actual size: 1156, Expected max: 1216, Ratio: 56.45%
```

**What to look for**:
- **Sketches**: Should increase with longer contexts
- **Promotions**: Non-zero means high-energy sketches are being recovered
- **Replay Triggers**: Low count = good compression quality
- **Calibrations**: Should increase steadily (one per compression)
- **Budget Adjustments**: Active controller (varies per head)
- **Cache Ratio**: Should stay well below 100% of sequence length

## Tuning Guidelines

### Sketch Budget
- **Small contexts (2-4k)**: 16-32 tokens
- **Medium contexts (8-16k)**: 32-48 tokens  
- **Large contexts (32k+)**: 48-64 tokens

### Replay Budget
- **High accuracy tasks**: 16-24 tokens
- **Balanced tasks**: 8-16 tokens
- **Speed-critical tasks**: 4-8 tokens

### Energy Threshold
- **Conservative** (fewer replays): 0.90-0.95
- **Balanced**: 0.85-0.90
- **Aggressive** (more replays): 0.80-0.85

### Controller Gain
- **Stable** (slow adaptation): 0.2-0.3
- **Balanced**: 0.3-0.4
- **Responsive** (fast adaptation): 0.4-0.5

## Common Issues

### Cache Growing Too Large

**Problem**: Cache size exceeds expected max

**Solutions**:
1. Reduce sketch_budget
2. Reduce replay_budget
3. Enable controller with lower energy_ceiling
4. Reduce importance_ratio

### Poor Perplexity with ADM++

**Problem**: ADM++ performs worse than base ADMS

**Solutions**:
1. Check if replay_budget is too small (should be ≥8)
2. Increase sketch_budget for better coverage
3. Try sketch_reduction="first" for stability
4. Disable calibration if model doesn't use RoPE

### High Memory Usage

**Problem**: Too much memory overhead from ADM++ features

**Solutions**:
1. Use controller_group_size=4 or higher
2. Reduce calibration_window to 256
3. Lower replay storage (affects replay_budget * 4)
4. Disable dual_fidelity if not needed

### Slow Performance

**Problem**: ADM++ is significantly slower

**Solutions**:
1. Disable residual_replay (SVD is expensive)
2. Disable position_calibration (linear solve overhead)
3. Use sketch_reduction="first" (cheapest)
4. Increase compression_interval to compress less frequently

## Example: Full Pipeline

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from streaming_llm import enable_adms_llm

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Enable ADM++
cache = enable_adms_llm(
    model,
    start_size=4,
    recent_size=2000,
    compressed_budget=128,
    enable_dual_fidelity=True,
    sketch_budget=32,
    enable_residual_replay=True,
    replay_budget=16,
    enable_position_calibration=True,
    enable_adaptive_controller=True,
)

# Long context generation
prompt = "Your very long prompt here..." * 100  # Simulate long context
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate with ADM++
outputs = model.generate(
    **inputs,
    past_key_values=cache,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
)

print(tokenizer.decode(outputs[0]))

# Check stats
print("\nFinal Stats:")
for key, value in cache.stats.items():
    print(f"  {key}: {value}")
```

## Next Steps

1. **Experiment**: Try different feature combinations for your task
2. **Benchmark**: Compare perplexity with/without ADM++ features
3. **Profile**: Monitor memory and speed with your workload
4. **Tune**: Adjust budgets and thresholds based on results
5. **Report**: Share findings to help improve defaults

For detailed implementation info, see `ADM_PLUS_PLUS_IMPLEMENTATION.md`.
