#!/usr/bin/env bash
set -euo pipefail

# Automated ADMS vs StreamingLLM evaluation runner
# -------------------------------------------------
# This script executes a set of evaluation sweeps using
# examples/eval_adms_vs_streaming.py and aggregates the
# resulting metrics into a concise CSV + Markdown report.
#
# You can override defaults via environment variables:
#   MODEL_NAME            (default TinyLlama/TinyLlama-1.1B-Chat-v1.0)
#   DATASET_NAME          (default wikitext)
#   TASK_NAME             (default wikitext-103-raw-v1)
#   SPLIT_NAME            (default test)
#   NUM_SAMPLES           (default 128)
#   NUM_EVAL_TOKENS       (default 2048)
#   OUTPUT_ROOT           (default outputs/auto_adms_runs)
#   LOG_EVERY             (default 128)
#   MIN_SEQ_LEN           (default 32)
#   FORCE_RERUN=1         (re-run even if summary.json already exists)

MODEL_NAME=${MODEL_NAME:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}
DATASET_NAME=${DATASET_NAME:-wikitext}
TASK_NAME=${TASK_NAME:-wikitext-103-raw-v1}
SPLIT_NAME=${SPLIT_NAME:-test}
NUM_SAMPLES=${NUM_SAMPLES:-128}
NUM_EVAL_TOKENS=${NUM_EVAL_TOKENS:-2048}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/auto_adms_runs}
LOG_EVERY=${LOG_EVERY:-128}
MIN_SEQ_LEN=${MIN_SEQ_LEN:-32}
FORCE_RERUN=${FORCE_RERUN:-0}

mkdir -p "${OUTPUT_ROOT}"

# Define evaluation profiles (balanced + quality-focused + ADM++ profiles)
# and model/dataset scenarios. Each entry is a space-separated
# string of key=value pairs. Feel free to extend both lists.
# NOTE: Long-context (≥8k) scenarios below assume you have enough VRAM
# and have accepted any gated model licenses on Hugging Face.
PROFILES=(
    # Legacy ADMS profiles (all ADM++ features disabled)
    "name=legacy_balanced start=4 recent=256 budget=128 ratio=0.6 interval=64 rank=8 svd=256 pos_shift=1 dynamic_sink=1 sketch_budget=0 replay_budget=0 disable_dual_fidelity=1 disable_replay=1 disable_calibration=1 disable_controller=1"
    "name=legacy_quality start=4 recent=128 budget=128 ratio=0.75 interval=64 rank=8 svd=256 pos_shift=1 dynamic_sink=1 sketch_budget=0 replay_budget=0 disable_dual_fidelity=1 disable_replay=1 disable_calibration=1 disable_controller=1"
    
    # ADM++ profiles (all features enabled with different budgets)
    "name=adm_plus_plus_small start=4 recent=256 budget=128 ratio=0.6 interval=64 rank=8 svd=256 pos_shift=1 dynamic_sink=1 sketch_budget=32 replay_budget=16 sketch_reduction=mean energy_threshold=0.88 calib_window=512 calib_reg=0.1"
    "name=adm_plus_plus_balanced start=4 recent=256 budget=128 ratio=0.6 interval=64 rank=8 svd=256 pos_shift=1 dynamic_sink=1 sketch_budget=64 replay_budget=32 sketch_reduction=mean energy_threshold=0.88 calib_window=512 calib_reg=0.1"
    "name=adm_plus_plus_quality start=4 recent=128 budget=128 ratio=0.75 interval=64 rank=8 svd=256 pos_shift=1 dynamic_sink=1 sketch_budget=128 replay_budget=64 sketch_reduction=mean energy_threshold=0.85 calib_window=1024 calib_reg=0.05"
    
    # Feature ablation profiles
    "name=dual_fidelity_only start=4 recent=256 budget=128 ratio=0.6 interval=64 rank=8 svd=256 pos_shift=1 dynamic_sink=1 sketch_budget=64 replay_budget=0 sketch_reduction=mean disable_replay=1 disable_calibration=1 disable_controller=1"
    "name=replay_only start=4 recent=256 budget=128 ratio=0.6 interval=64 rank=8 svd=256 pos_shift=1 dynamic_sink=1 sketch_budget=0 replay_budget=32 energy_threshold=0.88 disable_dual_fidelity=1 disable_calibration=1 disable_controller=1"
    "name=calibration_only start=4 recent=256 budget=128 ratio=0.6 interval=64 rank=8 svd=256 pos_shift=1 dynamic_sink=1 sketch_budget=0 replay_budget=0 calib_window=512 calib_reg=0.1 disable_dual_fidelity=1 disable_replay=1 disable_controller=1"
    "name=controller_only start=4 recent=256 budget=128 ratio=0.6 interval=64 rank=8 svd=256 pos_shift=1 dynamic_sink=1 sketch_budget=0 replay_budget=0 disable_dual_fidelity=1 disable_replay=1 disable_calibration=1"
)

SCENARIOS=(
    # Small models (1-2B) - Fast experiments
    "name=tinyllama_2k model=TinyLlama/TinyLlama-1.1B-Chat-v1.0 dataset=wikitext task=wikitext-103-raw-v1 split=test samples=128 tokens=2048 max_len=2048"
    "name=tinyllama_4k model=TinyLlama/TinyLlama-1.1B-Chat-v1.0 dataset=wikitext task=wikitext-103-raw-v1 split=test samples=96 tokens=4096 log_every=256 max_len=4096"
    "name=opt13b_2k model=facebook/opt-1.3b dataset=wikitext task=wikitext-2-raw-v1 split=test samples=128 tokens=2048 max_len=2048"
    "name=opt13b_4k model=facebook/opt-1.3b dataset=wikitext task=wikitext-2-raw-v1 split=test samples=96 tokens=4096 log_every=256 max_len=4096"
    
    # Medium models (7B) from StreamingLLM paper
    "name=pythia69b_4k model=EleutherAI/pythia-6.9b dataset=wikitext task=wikitext-103-raw-v1 split=test samples=96 tokens=4096 log_every=256 max_len=4096"
    "name=pythia69b_8k model=EleutherAI/pythia-6.9b dataset=wikitext task=wikitext-103-raw-v1 split=test samples=64 tokens=8192 log_every=512 max_len=8192"
    "name=falcon7b_4k model=tiiuae/falcon-7b dataset=wikitext task=wikitext-103-raw-v1 split=test samples=96 tokens=4096 log_every=256 max_len=4096"
    "name=falcon7b_8k model=tiiuae/falcon-7b dataset=wikitext task=wikitext-103-raw-v1 split=test samples=64 tokens=8192 log_every=512 max_len=8192"
    "name=mpt7b_4k model=mosaicml/mpt-7b dataset=wikitext task=wikitext-103-raw-v1 split=test samples=96 tokens=4096 log_every=256 max_len=4096"
    "name=mpt7b_8k model=mosaicml/mpt-7b dataset=wikitext task=wikitext-103-raw-v1 split=test samples=64 tokens=8192 log_every=512 max_len=8192"
    
    # Large models (8B) - Long context
    "name=llama3_8k model=meta-llama/Meta-Llama-3-8B-Instruct dataset=wikitext task=wikitext-103-raw-v1 split=test samples=64 tokens=8192 log_every=512 max_len=8192"
    "name=llama3_16k model=meta-llama/Meta-Llama-3-8B-Instruct dataset=wikitext task=wikitext-103-raw-v1 split=test samples=48 tokens=16384 log_every=1024 max_len=16384"
    "name=llama3_32k model=meta-llama/Meta-Llama-3-8B-Instruct dataset=wikitext task=wikitext-103-raw-v1 split=test samples=32 tokens=32768 log_every=2048 max_len=32768"
    
    # Modern alternatives (optional - comment out if too slow)
    # "name=mistral7b_16k model=mistralai/Mistral-7B-v0.1 dataset=wikitext task=wikitext-103-raw-v1 split=test samples=48 tokens=16384 log_every=1024 max_len=16384"
    # "name=mistral7b_32k model=mistralai/Mistral-7B-v0.1 dataset=wikitext task=wikitext-103-raw-v1 split=test samples=32 tokens=32768 log_every=2048 max_len=32768"
)


parse_kv_pairs() {
    local input=$1
    local -n out_ref=$2
    out_ref=()
    for kv in ${input}; do
        local key=${kv%%=*}
        local val=${kv#*=}
        out_ref["${key}"]=${val}
    done
}

run_combination() {
    local scenario_str=$1
    local profile_str=$2

    declare -A scenario cfg
    parse_kv_pairs "${scenario_str}" scenario
    parse_kv_pairs "${profile_str}" cfg

    local scenario_name=${scenario[name]:-default}
    local profile_name=${cfg[name]}

    local model_name=${scenario[model]:-${MODEL_NAME}}
    local dataset_name=${scenario[dataset]:-${DATASET_NAME}}
    local task_name=${scenario[task]:-${TASK_NAME}}
    local split_name=${scenario[split]:-${SPLIT_NAME}}
    local num_samples=${scenario[samples]:-${NUM_SAMPLES}}
    local num_eval_tokens=${scenario[tokens]:-${NUM_EVAL_TOKENS}}
    local log_every_value=${scenario[log_every]:-${LOG_EVERY}}
    local min_seq_len_value=${scenario[min_seq_len]:-${MIN_SEQ_LEN}}
    local max_seq_length=${scenario[max_len]:-${num_eval_tokens}}
    local recent_size=${scenario[recent]:-${cfg[recent]}}
    local compressed_budget=${scenario[budget]:-${cfg[budget]}}

    local out_dir="${OUTPUT_ROOT}/${scenario_name}/${profile_name}"
    mkdir -p "${out_dir}"

    if [[ ${FORCE_RERUN} -ne 0 ]]; then
        rm -f "${out_dir}/summary.json" "${out_dir}/readable.txt"
    fi

    if [[ -f "${out_dir}/summary.json" ]]; then
        echo "[SKIP] ${scenario_name}/${profile_name}: summary.json already exists. Set FORCE_RERUN=1 to re-run."
        return
    fi

    echo "\n=== Running scenario: ${scenario_name} | profile: ${profile_name} ==="
    echo "    Recent size: ${recent_size}, Compressed budget: ${compressed_budget}"

    local num_eval_tokens_args=()
    if [[ -n "${num_eval_tokens}" && "${num_eval_tokens}" != "none" ]]; then
        num_eval_tokens_args=(--num_eval_tokens "${num_eval_tokens}")
    fi

    local dynamic_sink_flag=""
    if [[ "${cfg[dynamic_sink]:-0}" == "1" ]]; then
        dynamic_sink_flag="--enable_dynamic_sink"
    fi

    # Build ADM++ feature flags
    local sketch_budget_args=()
    if [[ -n "${cfg[sketch_budget]:-}" && "${cfg[sketch_budget]}" != "0" ]]; then
        sketch_budget_args=(--sketch_budget "${cfg[sketch_budget]}")
        if [[ -n "${cfg[sketch_reduction]:-}" ]]; then
            sketch_budget_args+=(--sketch_reduction "${cfg[sketch_reduction]}")
        fi
    fi

    local replay_budget_args=()
    if [[ -n "${cfg[replay_budget]:-}" && "${cfg[replay_budget]}" != "0" ]]; then
        replay_budget_args=(--replay_budget "${cfg[replay_budget]}")
        if [[ -n "${cfg[energy_threshold]:-}" ]]; then
            replay_budget_args+=(--energy_replay_threshold "${cfg[energy_threshold]}")
        fi
    fi

    local calibration_args=()
    if [[ -n "${cfg[calib_window]:-}" ]]; then
        calibration_args=(--calibration_window "${cfg[calib_window]}")
        if [[ -n "${cfg[calib_reg]:-}" ]]; then
            calibration_args+=(--calibration_regularization "${cfg[calib_reg]}")
        fi
    fi

    # Build disable flags
    local disable_flags=""
    if [[ "${cfg[disable_dual_fidelity]:-0}" == "1" ]]; then
        disable_flags="${disable_flags} --disable_dual_fidelity"
    fi
    if [[ "${cfg[disable_replay]:-0}" == "1" ]]; then
        disable_flags="${disable_flags} --disable_residual_replay"
    fi
    if [[ "${cfg[disable_calibration]:-0}" == "1" ]]; then
        disable_flags="${disable_flags} --disable_position_calibration"
    fi
    if [[ "${cfg[disable_controller]:-0}" == "1" ]]; then
        disable_flags="${disable_flags} --disable_adaptive_controller"
    fi

    python examples/eval_adms_vs_streaming.py \
        --model_name_or_path "${model_name}" \
        --dataset_name "${dataset_name}" \
        --task "${task_name}" \
        --split "${split_name}" \
        --num_samples "${num_samples}" \
        "${num_eval_tokens_args[@]}" \
        --start_size "${cfg[start]}" \
        --recent_size "${recent_size}" \
        --compressed_budget "${compressed_budget}" \
        --importance_ratio "${cfg[ratio]}" \
        --compression_interval "${cfg[interval]}" \
        --min_middle_size_for_compress 64 \
        --rank "${cfg[rank]}" \
        --svd_max_tokens "${cfg[svd]}" \
        --max_seq_length "${max_seq_length}" \
        ${dynamic_sink_flag} \
        "${sketch_budget_args[@]}" \
        "${replay_budget_args[@]}" \
        "${calibration_args[@]}" \
        ${disable_flags} \
        --log_every "${log_every_value}" \
        --min_seq_len "${min_seq_len_value}" \
        --output_dir "${out_dir}" \
        ${cfg[pos_shift]:+--enable_pos_shift} \
        --concat_stream
}

for scenario in "${SCENARIOS[@]}"; do
    for profile in "${PROFILES[@]}"; do
        run_combination "${scenario}" "${profile}"
        echo "=== Completed: ${scenario} | ${profile} ===\n"
        sleep 2
    done
done

# Aggregate results into CSV and Markdown summary
SUMMARY_CSV="${OUTPUT_ROOT}/summary.csv"
SUMMARY_MD="${OUTPUT_ROOT}/summary.md"

python - "${OUTPUT_ROOT}" "${SUMMARY_CSV}" "${SUMMARY_MD}" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
out_csv = Path(sys.argv[2])
out_md = Path(sys.argv[3])

rows = []
for summary_path in sorted(root.glob("*/*/summary.json")):
    profile = summary_path.parent.name
    scenario = summary_path.parent.parent.name
    data = json.loads(summary_path.read_text())
    args = data.get("args", {})
    adms = data["results"]["ADMS"]
    streaming = data["results"]["StreamingLLM"]
    ppl_gain = float('nan')
    if math.isfinite(streaming["ppl"]) and streaming["ppl"] > 0:
        ppl_gain = 100.0 * (streaming["ppl"] - adms["ppl"]) / streaming["ppl"]
    context_tokens = args.get("num_eval_tokens")
    if context_tokens in (None, "None"):
        context_tokens = ""
    num_samples = args.get("num_samples")
    if num_samples in (None, "None"):
        num_samples = ""
    rows.append({
        "scenario": scenario,
        "profile": profile,
        "model_name": args.get("model_name_or_path", ""),
        "dataset_name": args.get("dataset_name", ""),
        "task_name": args.get("task", ""),
        "split_name": args.get("split", ""),
        "num_eval_tokens": context_tokens,
        "num_samples": num_samples,
        "adms_ppl": adms["ppl"],
        "streaming_ppl": streaming["ppl"],
        "ppl_gain_pct": ppl_gain,
        "adms_tok_per_sec": adms["tokens_per_sec"],
        "streaming_tok_per_sec": streaming["tokens_per_sec"],
        "adms_time_sec": adms["time_sec"],
        "streaming_time_sec": streaming["time_sec"],
        "adms_tokens": adms["tokens"],
        "streaming_tokens": streaming["tokens"],
    })

if not rows:
    print("No summary.json files found. Run configurations first.", file=sys.stderr)
    sys.exit(1)

rows.sort(key=lambda r: (r["scenario"], r["profile"]))

header = [
    "scenario",
    "profile",
    "model_name",
    "dataset_name",
    "task_name",
    "split_name",
    "num_eval_tokens",
    "num_samples",
    "adms_ppl",
    "streaming_ppl",
    "ppl_gain_pct",
    "adms_tok_per_sec",
    "streaming_tok_per_sec",
    "adms_time_sec",
    "streaming_time_sec",
    "adms_tokens",
    "streaming_tokens",
]

with out_csv.open("w", encoding="utf-8") as f:
    f.write(",".join(header) + "\n")
    for row in rows:
        csv_vals = []
        for key in header:
            val = row[key]
            if isinstance(val, float):
                if math.isnan(val):
                    csv_vals.append("")
                else:
                    csv_vals.append(f"{val:.4f}")
            else:
                csv_vals.append(str(val))
        f.write(",".join(csv_vals) + "\n")

with out_md.open("w", encoding="utf-8") as f:
    f.write("# ADMS vs StreamingLLM Summary\n\n")
    f.write("| Scenario | Profile | Model | Dataset | Task | Split | Context Tokens | Samples | ADMS PPL | Streaming PPL | Δ PPL (%) | ADMS tok/s | Streaming tok/s |\n")
    f.write("|----------|---------|-------|---------|------|-------|----------------|---------|----------|----------------|-----------|------------|-----------------|\n")
    for row in rows:
        gain = "n/a" if math.isnan(row["ppl_gain_pct"]) else f"{row['ppl_gain_pct']:.1f}%"
        f.write(
            f"| {row['scenario']} "
            f"| {row['profile']} "
            f"| {row['model_name']} "
            f"| {row['dataset_name']} "
            f"| {row['task_name']} "
            f"| {row['split_name']} "
            f"| {row['num_eval_tokens'] or 'n/a'} "
            f"| {row['num_samples'] or 'n/a'} "
            f"| {row['adms_ppl']:.4f} "
            f"| {row['streaming_ppl']:.4f} "
            f"| {gain} "
            f"| {row['adms_tok_per_sec']:.2f} "
            f"| {row['streaming_tok_per_sec']:.2f} |\n"
        )
    f.write("\nGenerated from run_adms_report.sh\n")

print(f"Wrote {out_csv}")
print(f"Wrote {out_md}")
PY

cat "${SUMMARY_MD}"
