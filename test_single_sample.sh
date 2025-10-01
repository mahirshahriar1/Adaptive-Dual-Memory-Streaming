#!/bin/bash

# Quick diagnostic: run ADMS on a single sample with full output
# Use this to verify ADMS is actually running and producing logs

echo "========================================"
echo "  ADMS Single Sample Diagnostic"
echo "========================================"
echo ""
echo "Running ADMS on multiple samples to get enough tokens..."
echo "Look for these indicators:"
echo "  1. 'ADMS Config: start=4, recent=256, budget=512'"
echo "  2. '[ADMS Stats @ XXX]' during generation"
echo "  3. '[ADMS Cache @ XXX]' showing cache size"
echo ""
echo "----------------------------------------"

python examples/eval_adms_vs_streaming.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_name wikitext \
  --task wikitext-103-raw-v1 \
  --split test \
  --num_samples 100 \
  --concat_stream \
  --start_size 4 \
  --recent_size 256 \
  --compressed_budget 512 \
  --importance_ratio 0.8 \
  --compression_interval 32 \
  --min_seq_len 512 \
  --min_middle_size_for_compress 8 \
  --output_dir "outputs/diagnostic"

echo ""
echo "========================================"
echo "  Check output above for ADMS logs"
echo "========================================"
