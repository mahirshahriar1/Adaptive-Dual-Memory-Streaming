#!/bin/bash
# ADMS Performance Comparison Suite (Bash version)
# Runs three configurations: Exact-only, Reduced Frequency, Smaller Budget

set -e

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "      ADMS Performance Comparison Suite"
echo "═══════════════════════════════════════════════════════════"
echo ""

MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET="wikitext"
TASK="wikitext-103-v1"
SPLIT="validation"
NUM_SAMPLES=128
START_SIZE=4
RECENT_SIZE=256

# Configuration array (name, budget, ratio, interval, svd_max, description)
declare -a CONFIGS=(
    "Exact-Only|512|1.0|32|256|Fastest: No SVD, pure importance selection"
    "Reduced-Frequency|512|0.7|64|256|Balanced: Light compression, infrequent updates"
    "Smaller-Budget|256|0.8|16|512|Memory-efficient: Lower budget, mostly exact"
)

# Array to store results
declare -a RESULTS=()

for config in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME BUDGET RATIO INTERVAL SVD_MAX DESC <<< "$config"
    OUTPUT_DIR="outputs/comparison/${NAME}"
    
    echo ""
    echo "───────────────────────────────────────────────────────────"
    echo "  Configuration: ${NAME}"
    echo "───────────────────────────────────────────────────────────"
    echo "  Description: ${DESC}"
    echo ""
    echo "  Parameters:"
    echo "    - compressed_budget:    ${BUDGET}"
    echo "    - importance_ratio:     ${RATIO}"
    echo "    - compression_interval: ${INTERVAL}"
    echo "    - svd_max_tokens:       ${SVD_MAX}"
    echo ""
    
    START_TIME=$(date +%s)
    
    # Run evaluation
    python examples/eval_adms_vs_streaming.py \
        --model_name_or_path "${MODEL}" \
        --dataset_name "${DATASET}" \
        --task "${TASK}" \
        --split "${SPLIT}" \
        --num_samples ${NUM_SAMPLES} \
        --concat_stream \
        --start_size ${START_SIZE} \
        --recent_size ${RECENT_SIZE} \
        --compressed_budget ${BUDGET} \
        --importance_ratio ${RATIO} \
        --importance_metric value_norm \
        --compression_interval ${INTERVAL} \
        --svd_max_tokens ${SVD_MAX} \
        --output_dir "${OUTPUT_DIR}"
    
    END_TIME=$(date +%s)
    TOTAL_TIME=$((END_TIME - START_TIME))
    
    # Parse results from summary.json
    SUMMARY_PATH="${OUTPUT_DIR}/summary.json"
    if [ -f "${SUMMARY_PATH}" ]; then
        ADMS_PPL=$(python3 -c "import json; print(round(json.load(open('${SUMMARY_PATH}'))['results']['ADMS']['ppl'], 2))")
        ADMS_SPEED=$(python3 -c "import json; print(round(json.load(open('${SUMMARY_PATH}'))['results']['ADMS']['tokens_per_sec'], 1))")
        STREAM_PPL=$(python3 -c "import json; print(round(json.load(open('${SUMMARY_PATH}'))['results']['StreamingLLM']['ppl'], 2))")
        STREAM_SPEED=$(python3 -c "import json; print(round(json.load(open('${SUMMARY_PATH}'))['results']['StreamingLLM']['tokens_per_sec'], 1))")
        TOKENS=$(python3 -c "import json; print(json.load(open('${SUMMARY_PATH}'))['results']['ADMS']['tokens'])")
        
        IMPROVEMENT=$(python3 -c "print(round((${STREAM_PPL} - ${ADMS_PPL}) / ${STREAM_PPL} * 100, 1))")
        PENALTY=$(python3 -c "print(round((${STREAM_SPEED} - ${ADMS_SPEED}) / ${STREAM_SPEED} * 100, 1))")
        
        RESULTS+=("${NAME}|${ADMS_PPL}|${ADMS_SPEED}|${IMPROVEMENT}|${PENALTY}|${TOTAL_TIME}")
        
        echo ""
        echo "  ✓ Completed in ${TOTAL_TIME}s"
        echo "    ADMS PPL: ${ADMS_PPL} | Speed: ${ADMS_SPEED} tok/s"
        echo "    Improvement: ${IMPROVEMENT}% | Speed penalty: ${PENALTY}%"
    else
        echo ""
        echo "  ✗ Failed - no summary.json found"
    fi
done

# Generate comparison report
echo ""
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "      COMPARISON REPORT"
echo "═══════════════════════════════════════════════════════════"
echo ""

if [ ${#RESULTS[@]} -gt 0 ]; then
    echo "Results Summary:"
    echo ""
    printf "%-25s %-12s %-10s %-15s %-12s\n" "Configuration" "ADMS PPL" "tok/s" "vs Streaming" "Speed Cost"
    printf "%-25s %-12s %-10s %-15s %-12s\n" "-------------" "--------" "-----" "------------" "----------"
    
    for result in "${RESULTS[@]}"; do
        IFS='|' read -r NAME PPL SPEED IMPROVEMENT PENALTY TIME <<< "$result"
        printf "%-25s %-12s %-10s %-15s %-12s\n" "${NAME}" "${PPL}" "${SPEED}" "+${IMPROVEMENT}%" "${PENALTY}%"
    done
    
    echo ""
    echo "Recommendations:"
    echo ""
    
    # Find best PPL
    BEST_PPL_NAME=""
    BEST_PPL_VAL=999999
    for result in "${RESULTS[@]}"; do
        IFS='|' read -r NAME PPL SPEED IMPROVEMENT PENALTY TIME <<< "$result"
        if (( $(echo "$PPL < $BEST_PPL_VAL" | bc -l) )); then
            BEST_PPL_VAL=$PPL
            BEST_PPL_NAME=$NAME
            BEST_PPL_IMPROVEMENT=$IMPROVEMENT
        fi
    done
    
    # Find best speed
    BEST_SPEED_NAME=""
    BEST_SPEED_VAL=0
    for result in "${RESULTS[@]}"; do
        IFS='|' read -r NAME PPL SPEED IMPROVEMENT PENALTY TIME <<< "$result"
        if (( $(echo "$SPEED > $BEST_SPEED_VAL" | bc -l) )); then
            BEST_SPEED_VAL=$SPEED
            BEST_SPEED_NAME=$NAME
            BEST_SPEED_PENALTY=$PENALTY
        fi
    done
    
    echo "  🏆 Best Quality:     ${BEST_PPL_NAME}"
    echo "     PPL: ${BEST_PPL_VAL} (+${BEST_PPL_IMPROVEMENT}% better than StreamingLLM)"
    echo ""
    echo "  ⚡ Best Speed:       ${BEST_SPEED_NAME}"
    echo "     Speed: ${BEST_SPEED_VAL} tok/s (${BEST_SPEED_PENALTY}% penalty)"
    echo ""
    
    # Save detailed report
    REPORT_PATH="outputs/comparison/comparison_report.txt"
    mkdir -p "$(dirname "${REPORT_PATH}")"
    
    cat > "${REPORT_PATH}" <<EOF
ADMS Performance Comparison Report
Generated: $(date)
═══════════════════════════════════════════════════════════

Test Configuration:
  Model: ${MODEL}
  Dataset: ${DATASET} (${TASK}, ${SPLIT})
  Samples: ${NUM_SAMPLES}
  Context: start=${START_SIZE}, recent=${RECENT_SIZE}

Results Summary:

Configuration              ADMS PPL    tok/s      vs Streaming    Speed Cost
------------------------------------------------------------------------------
EOF
    
    for result in "${RESULTS[@]}"; do
        IFS='|' read -r NAME PPL SPEED IMPROVEMENT PENALTY TIME <<< "$result"
        printf "%-25s %-12s %-10s %-15s %-12s\n" "${NAME}" "${PPL}" "${SPEED}" "+${IMPROVEMENT}%" "${PENALTY}%" >> "${REPORT_PATH}"
    done
    
    cat >> "${REPORT_PATH}" <<EOF

Recommendations:
  Best Quality:    ${BEST_PPL_NAME} (PPL: ${BEST_PPL_VAL})
  Best Speed:      ${BEST_SPEED_NAME} (${BEST_SPEED_VAL} tok/s)

Key Insights:
  - Exact-only (ratio=1.0) typically offers best quality/speed balance
  - Higher compression_interval reduces overhead but may miss updates
  - Smaller budgets reduce memory but may hurt long-range retention

For production: Use the balanced configuration
For research:   Use ${BEST_PPL_NAME} configuration

═══════════════════════════════════════════════════════════
EOF
    
    echo "  📄 Detailed report saved to: ${REPORT_PATH}"
    echo ""
    
    # Create CSV export
    CSV_PATH="outputs/comparison/results.csv"
    echo "Configuration,ADMS_PPL,Speed_tok_per_sec,Improvement_pct,SpeedPenalty_pct,Runtime_sec" > "${CSV_PATH}"
    for result in "${RESULTS[@]}"; do
        IFS='|' read -r NAME PPL SPEED IMPROVEMENT PENALTY TIME <<< "$result"
        echo "${NAME},${PPL},${SPEED},${IMPROVEMENT},${PENALTY},${TIME}" >> "${CSV_PATH}"
    done
    
    echo "  📊 CSV data exported to: ${CSV_PATH}"
    
else
    echo "  ✗ No results to report"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "      Comparison suite completed!"
echo "═══════════════════════════════════════════════════════════"
echo ""
