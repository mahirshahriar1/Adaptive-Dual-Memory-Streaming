#!/bin/bash
# ADMS Quick Comparison (Fast version for testing)
# Uses only 32 samples for rapid iteration

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "      ADMS Quick Comparison (32 samples)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Configuration array (name, budget, ratio, interval)
declare -a CONFIGS=(
    "Exact-Only|512|1.0|32"
    "Balanced|384|0.7|64"
    "Light|256|0.9|16"
)

for config in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME BUDGET RATIO INTERVAL <<< "$config"
    
    echo ""
    echo "▶ Running: ${NAME}..."
    
    python examples/eval_adms_vs_streaming.py \
        --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --dataset_name wikitext --task wikitext-103-v1 --split validation \
        --num_samples 32 \
        --concat_stream \
        --start_size 4 --recent_size 256 \
        --compressed_budget ${BUDGET} \
        --importance_ratio ${RATIO} \
        --compression_interval ${INTERVAL} \
        --svd_max_tokens 256 \
        --output_dir "outputs/quick/${NAME}"
    
    SUMMARY_PATH="outputs/quick/${NAME}/summary.json"
    if [ -f "${SUMMARY_PATH}" ]; then
        PPL=$(python3 -c "import json; print(round(json.load(open('${SUMMARY_PATH}'))['results']['ADMS']['ppl'], 1))")
        SPEED=$(python3 -c "import json; print(round(json.load(open('${SUMMARY_PATH}'))['results']['ADMS']['tokens_per_sec'], 1))")
        STREAM_PPL=$(python3 -c "import json; print(round(json.load(open('${SUMMARY_PATH}'))['results']['StreamingLLM']['ppl'], 1))")
        IMPROVEMENT=$(python3 -c "print(round((${STREAM_PPL} - ${PPL}) / ${STREAM_PPL} * 100, 1))")
        
        echo "  ✓ PPL: ${PPL} | Speed: ${SPEED} tok/s | Improvement: +${IMPROVEMENT}%"
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  For full evaluation (128 samples), run:"
echo "    ./run_comparison_suite.sh"
echo "═══════════════════════════════════════════════════════════"
echo ""
