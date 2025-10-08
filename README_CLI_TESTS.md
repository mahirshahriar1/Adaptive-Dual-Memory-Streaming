# ADM++ CLI Testing Guide

This guide provides comprehensive CLI commands for testing all ADM++ features.

## Quick Start

### 1. Basic Smoke Test (Fast - ~2 minutes)
Tests that everything works end-to-end with minimal resources:

```bash
python examples/eval_adms_vs_streaming.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name wikitext --task wikitext-2-raw-v1 --split test \
  --num_samples 4 --num_eval_tokens 512 \
  --start_size 4 --recent_size 256 --compressed_budget 64 \
  --sketch_budget 16 --replay_budget 8 \
  --output_dir outputs/smoke_test
```

### 2. Standard Test (~10 minutes)
Recommended for validating feature functionality:

```bash
python examples/eval_adms_vs_streaming.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name wikitext --task wikitext-2-raw-v1 --split test \
  --num_samples 16 --num_eval_tokens 2048 \
  --start_size 4 --recent_size 256 --compressed_budget 128 \
  --sketch_budget 32 --replay_budget 16 \
  --output_dir outputs/standard_test
```

### 3. Production Test (~30 minutes)
Full evaluation with realistic settings:

```bash
python examples/eval_adms_vs_streaming.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name wikitext --task wikitext-103-raw-v1 --split test \
  --num_samples 128 --num_eval_tokens 4096 \
  --start_size 4 --recent_size 512 --compressed_budget 256 \
  --sketch_budget 64 --replay_budget 32 \
  --output_dir outputs/production_test
```

## Feature-Specific Tests

### Test 1: Dual-Fidelity Mid Memory
Tests sketch bank, reduction strategies, and promotion mechanisms:

```bash
# Mean reduction (default - best for general use)
python examples/eval_adms_vs_streaming.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name wikitext --task wikitext-2-raw-v1 --split test \
  --num_samples 16 --num_eval_tokens 2048 \
  --start_size 4 --recent_size 256 --compressed_budget 128 \
  --sketch_budget 64 --sketch_reduction mean \
  --output_dir outputs/dual_fidelity_mean

# Sum reduction (preserves total activation energy)
python examples/eval_adms_vs_streaming.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name wikitext --task wikitext-2-raw-v1 --split test \
  --num_samples 16 --num_eval_tokens 2048 \
  --start_size 4 --recent_size 256 --compressed_budget 128 \
  --sketch_budget 64 --sketch_reduction sum \
  --output_dir outputs/dual_fidelity_sum

# First reduction (keeps first token only)
python examples/eval_adms_vs_streaming.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name wikitext --task wikitext-2-raw-v1 --split test \
  --num_samples 16 --num_eval_tokens 2048 \
  --start_size 4 --recent_size 256 --compressed_budget 128 \
  --sketch_budget 64 --sketch_reduction first \
  --output_dir outputs/dual_fidelity_first
```

### Test 2: Residual Replay Engine
Tests spectral energy monitoring and automatic replay:

```bash
# High sensitivity (more replays, higher quality)
python examples/eval_adms_vs_streaming.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name wikitext --task wikitext-2-raw-v1 --split test \
  --num_samples 16 --num_eval_tokens 2048 \
  --start_size 4 --recent_size 256 --compressed_budget 128 \
  --replay_budget 32 --energy_replay_threshold 0.85 \
  --output_dir outputs/replay_high_sensitivity

# Balanced (default)
python examples/eval_adms_vs_streaming.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name wikitext --task wikitext-2-raw-v1 --split test \
  --num_samples 16 --num_eval_tokens 2048 \
  --start_size 4 --recent_size 256 --compressed_budget 128 \
  --replay_budget 16 --energy_replay_threshold 0.88 \
  --output_dir outputs/replay_balanced

# Low sensitivity (fewer replays, faster)
python examples/eval_adms_vs_streaming.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name wikitext --task wikitext-2-raw-v1 --split test \
  --num_samples 16 --num_eval_tokens 2048 \
  --start_size 4 --recent_size 256 --compressed_budget 128 \
  --replay_budget 8 --energy_replay_threshold 0.92 \
  --output_dir outputs/replay_low_sensitivity
```

### Test 3: RoPE Alignment Calibration
Tests positional encoding calibration with sliding anchors:

```bash
# Short calibration window (frequent updates, adaptive)
python examples/eval_adms_vs_streaming.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name wikitext --task wikitext-2-raw-v1 --split test \
  --num_samples 16 --num_eval_tokens 2048 \
  --start_size 4 --recent_size 256 --compressed_budget 128 \
  --calibration_window 256 --calibration_regularization 0.1 \
  --output_dir outputs/calibration_short_window

# Medium calibration window (default - balanced)
python examples/eval_adms_vs_streaming.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name wikitext --task wikitext-2-raw-v1 --split test \
  --num_samples 16 --num_eval_tokens 2048 \
  --start_size 4 --recent_size 256 --compressed_budget 128 \
  --calibration_window 512 --calibration_regularization 0.1 \
  --output_dir outputs/calibration_medium_window

# Long calibration window (stable, less overhead)
python examples/eval_adms_vs_streaming.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name wikitext --task wikitext-2-raw-v1 --split test \
  --num_samples 16 --num_eval_tokens 2048 \
  --start_size 4 --recent_size 256 --compressed_budget 128 \
  --calibration_window 1024 --calibration_regularization 0.05 \
  --output_dir outputs/calibration_long_window
```

### Test 4: Adaptive Budget Controller
Tests energy-based and attention-mass based budget adaptation:

```bash
# With adaptive controller (default)
python examples/eval_adms_vs_streaming.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name wikitext --task wikitext-2-raw-v1 --split test \
  --num_samples 16 --num_eval_tokens 2048 \
  --start_size 4 --recent_size 256 --compressed_budget 128 \
  --enable_adaptive_budget \
  --output_dir outputs/controller_enabled

# Without adaptive controller (baseline)
python examples/eval_adms_vs_streaming.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name wikitext --task wikitext-2-raw-v1 --split test \
  --num_samples 16 --num_eval_tokens 2048 \
  --start_size 4 --recent_size 256 --compressed_budget 128 \
  --disable_adaptive_budget \
  --output_dir outputs/controller_disabled
```

## Ablation Studies

### Full ADM++ vs Legacy ADMS
Compare complete ADM++ implementation against original ADMS:

```bash
# Full ADM++ (all features enabled)
python examples/eval_adms_vs_streaming.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name wikitext --task wikitext-2-raw-v1 --split test \
  --num_samples 32 --num_eval_tokens 2048 \
  --start_size 4 --recent_size 256 --compressed_budget 128 \
  --sketch_budget 32 --replay_budget 16 \
  --sketch_reduction mean --energy_replay_threshold 0.88 \
  --calibration_window 512 --calibration_regularization 0.1 \
  --enable_adaptive_budget \
  --output_dir outputs/ablation_full_adm_plus_plus

# Legacy ADMS (all ADM++ features disabled)
python examples/eval_adms_vs_streaming.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name wikitext --task wikitext-2-raw-v1 --split test \
  --num_samples 32 --num_eval_tokens 2048 \
  --start_size 4 --recent_size 256 --compressed_budget 128 \
  --disable_dual_fidelity --disable_residual_replay \
  --disable_rope_calibration --disable_adaptive_budget \
  --output_dir outputs/ablation_legacy_adms
```

### Individual Feature Isolation
Test each feature independently:

```bash
# Only Dual-Fidelity
python examples/eval_adms_vs_streaming.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name wikitext --task wikitext-2-raw-v1 --split test \
  --num_samples 32 --num_eval_tokens 2048 \
  --start_size 4 --recent_size 256 --compressed_budget 128 \
  --sketch_budget 32 --sketch_reduction mean \
  --disable_residual_replay --disable_rope_calibration --disable_adaptive_budget \
  --output_dir outputs/ablation_dual_fidelity_only

# Only Residual Replay
python examples/eval_adms_vs_streaming.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name wikitext --task wikitext-2-raw-v1 --split test \
  --num_samples 32 --num_eval_tokens 2048 \
  --start_size 4 --recent_size 256 --compressed_budget 128 \
  --replay_budget 16 --energy_replay_threshold 0.88 \
  --disable_dual_fidelity --disable_rope_calibration --disable_adaptive_budget \
  --output_dir outputs/ablation_replay_only

# Only RoPE Calibration
python examples/eval_adms_vs_streaming.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name wikitext --task wikitext-2-raw-v1 --split test \
  --num_samples 32 --num_eval_tokens 2048 \
  --start_size 4 --recent_size 256 --compressed_budget 128 \
  --calibration_window 512 --calibration_regularization 0.1 \
  --disable_dual_fidelity --disable_residual_replay --disable_adaptive_budget \
  --output_dir outputs/ablation_calibration_only

# Only Adaptive Controller
python examples/eval_adms_vs_streaming.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name wikitext --task wikitext-2-raw-v1 --split test \
  --num_samples 32 --num_eval_tokens 2048 \
  --start_size 4 --recent_size 256 --compressed_budget 128 \
  --enable_adaptive_budget \
  --disable_dual_fidelity --disable_residual_replay --disable_rope_calibration \
  --output_dir outputs/ablation_controller_only
```

## Long Context Tests

### 8K Context
For models with extended context capabilities:

```bash
python examples/eval_adms_vs_streaming.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name wikitext --task wikitext-103-raw-v1 --split test \
  --num_samples 64 --num_eval_tokens 8192 \
  --start_size 4 --recent_size 512 --compressed_budget 512 \
  --sketch_budget 128 --replay_budget 64 \
  --calibration_window 1024 \
  --max_seq_length 8192 --log_every 512 \
  --output_dir outputs/long_context_8k
```

### 16K Context
For larger models with long context support:

```bash
python examples/eval_adms_vs_streaming.py \
  --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset_name wikitext --task wikitext-103-raw-v1 --split test \
  --num_samples 48 --num_eval_tokens 16384 \
  --start_size 4 --recent_size 1024 --compressed_budget 1024 \
  --sketch_budget 256 --replay_budget 128 \
  --calibration_window 2048 \
  --max_seq_length 16384 --log_every 1024 \
  --output_dir outputs/long_context_16k
```

## Parameter Reference

### Core Cache Parameters
- `--start_size`: Sink/attention anchor size (default: 4)
- `--recent_size`: Recent window size (default: 256)
- `--compressed_budget`: Compressed mid memory budget (default: 128)
- `--max_seq_length`: Maximum sequence length to process (default: from model config)

### ADM++ Dual-Fidelity Parameters
- `--sketch_budget`: Budget for sketch tokens (default: 32)
- `--sketch_reduction`: Reduction strategy - `mean`, `sum`, or `first` (default: mean)
- `--dual_fidelity_promotion_threshold`: Energy threshold for promotion (default: 0.75)

### ADM++ Residual Replay Parameters
- `--replay_budget`: Budget for replay tokens (default: 16)
- `--energy_replay_threshold`: Spectral energy threshold for replay (default: 0.88)

### ADM++ RoPE Calibration Parameters
- `--calibration_window`: Size of calibration anchor window (default: 512)
- `--calibration_regularization`: Regularization strength (default: 0.1)

### ADM++ Adaptive Controller Parameters
- `--enable_adaptive_budget`: Enable adaptive budget controller (default: enabled)
- `--disable_adaptive_budget`: Disable adaptive budget controller

### Feature Toggle Flags
- `--disable_dual_fidelity`: Disable dual-fidelity mid memory
- `--disable_residual_replay`: Disable residual replay engine
- `--disable_rope_calibration`: Disable RoPE alignment calibration
- `--disable_adaptive_budget`: Disable adaptive budget controller

### Evaluation Parameters
- `--num_samples`: Number of samples to evaluate (default: 128)
- `--num_eval_tokens`: Number of tokens to evaluate per sample (default: 2048)
- `--log_every`: Log progress every N steps (default: 128)
- `--min_seq_len`: Minimum sequence length to process (default: 32)

### Other Parameters
- `--enable_dynamic_sink`: Enable dynamic sink size adjustment
- `--enable_pos_shift`: Enable positional shift optimization
- `--concat_stream`: Concatenate sequences for streaming evaluation

## Monitoring Output

When running tests, monitor the console output for feature activity:

### Dual-Fidelity Stats
```
[ADMS Dual-Fidelity @ step 1024] Sketches: 256, Promotions: 32, Sketch tokens: 64
```
- **Sketches**: Number of times sketch bank was updated
- **Promotions**: Number of times sketches were promoted to full precision
- **Sketch tokens**: Current number of tokens in sketch bank

### Replay Stats
```
[ADMS Replay @ step 1024] Triggers: 8, Replay tokens: 128, Total replayed: 1024
```
- **Triggers**: Number of replay events triggered
- **Replay tokens**: Number of tokens currently in replay bank
- **Total replayed**: Cumulative tokens that have been replayed

### Calibration Stats
```
[ADMS Calibration @ step 1024] Total calibrations: 16, Calibration window: 512
```
- **Total calibrations**: Number of RoPE calibration operations performed
- **Calibration window**: Size of current calibration anchor set

### Controller Stats
```
[ADMS Controller @ step 1024] Budget adjustments: 24, Avg head budget: 142.3
```
- **Budget adjustments**: Number of times budgets were adapted
- **Avg head budget**: Average budget allocated per attention head

## Expected Results

### Perplexity Improvements
- **Full ADM++ vs Legacy ADMS**: 5-10% perplexity reduction
- **Individual features**: 1-3% contribution each
- **Long context (≥8K)**: More significant gains

### Performance Characteristics
- **Dual-Fidelity**: ~5-10% memory overhead, minimal compute
- **Residual Replay**: Variable (depends on trigger frequency)
- **RoPE Calibration**: ~2-3% compute overhead
- **Adaptive Controller**: Minimal overhead (<1%)

## Troubleshooting

### Out of Memory Errors
Reduce memory usage:
- Decrease `--compressed_budget`
- Decrease `--sketch_budget` and `--replay_budget`
- Decrease `--recent_size`
- Use smaller model

### Slow Evaluation
Speed up evaluation:
- Decrease `--num_samples`
- Decrease `--num_eval_tokens`
- Disable features with `--disable_*` flags
- Increase `--log_every` to reduce logging overhead

### No Feature Activity
Check that features are enabled:
- Ensure budgets are set (sketch_budget, replay_budget)
- Verify no `--disable_*` flags are accidentally set
- Check that sequences are long enough for features to activate

## Automated Benchmark Suite

For comprehensive automated testing across multiple scenarios, use:

```bash
bash scripts/run_adms_report.sh
```

This will run all configured scenarios and generate summary reports in `outputs/auto_adms_runs/`.
