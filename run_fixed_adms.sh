#!/bin/bash

# ADMS Fixed Configuration - Addresses long-context collapse
# Key changes:
#   1. MUCH larger budget (2048 tokens for 15k context)
#   2. Higher importance_ratio (0.9 - mostly exact)
#   3. Rare compression (interval=128)
#   4. NO coverage priority (disable complex logic)

echo "========================================"
echo "  ADMS Fixed Configuration"
echo "========================================"
echo ""
echo "Running with conservative settings..."
echo "  - Budget: 2048 (13% of 15k context)"
echo "  - Ratio: 0.9 (90% exact, 10% compressed)"
echo "  - Interval: 128 (compress rarely)"
echo "  - Coverage: disabled"
echo ""

python examples/eval_adms_vs_streaming.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name wikitext \
  --task wikitext-103-raw-v1 \
  --split test \
  --num_samples 256 \
  --concat_stream \
  --start_size 4 \
  --recent_size 256 \
  --compressed_budget 2048 \
  --importance_ratio 0.9 \
  --importance_metric value_norm \
  --compression_interval 128 \
  --svd_max_tokens 256 \
  --coverage_priority 0.0 \
  --output_dir outputs/fixed_adms

echo ""
echo "Expected: PPL ~80-120, Speed ~35 tok/s"
