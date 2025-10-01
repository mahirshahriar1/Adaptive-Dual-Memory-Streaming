#!/bin/bash

# Test ADMS cache behavior at different sequence lengths
# Verifies that cache size stays bounded

echo "========================================"
echo "  ADMS Cache Size Verification"
echo "========================================"
echo ""

for samples in 128 192 256 512; do
    echo "----------------------------------------"
    echo "Testing with $samples samples (expecting ~$((samples * 50)) tokens)..."
    echo "----------------------------------------"
    
    python examples/eval_adms_vs_streaming.py \
      --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
      --dataset_name wikitext \
      --task wikitext-103-raw-v1 \
      --split test \
      --num_samples $samples \
      --concat_stream \
      --start_size 4 \
      --recent_size 256 \
      --compressed_budget 512 \
      --importance_ratio 0.8 \
      --compression_interval 32 \
      --output_dir "outputs/cache_test/$samples" \
      2>&1 | grep -E "(ADMS|Loading|Evaluating|PPL)"
    
    echo ""
done

echo "========================================"
echo "  Summary"
echo "========================================"
echo "Check for [ADMS Cache] logs above"
echo "Cache size should stay < 800 tokens"
echo "If no ADMS logs appear, check full output:"
echo "  python examples/eval_adms_vs_streaming.py --num_samples 8 ..."
echo "========================================"
