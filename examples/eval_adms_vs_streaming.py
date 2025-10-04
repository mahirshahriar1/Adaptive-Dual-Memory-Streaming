"""
ADMS vs StreamingLLM evaluation script

This script runs a head-to-head comparison between:
  - ADMS (start + compressed middle + recent)
  - StreamingLLM (start + recent only)

It reports:
  - Perplexity (lower is better)
  - Tokens/sec (higher is better)
  - Elapsed time

Notes
  - Vanilla (no eviction) is not included here to keep focus on the two bounded-memory methods.
  - For fairness, both methods use the same start_size and recent_size.
  - ADMS additionally uses a compressed_budget for the middle; StreamingLLM drops the middle.
"""

import argparse
import json
import os
import time
from typing import Optional, Tuple

import torch
from datasets import load_dataset
from torch.nn import CrossEntropyLoss

from streaming_llm.utils import load as load_model_and_tokenizer
from streaming_llm.kv_cache import StartRecentKVCache
from streaming_llm.enable_adms_llm import enable_adms_llm


def detect_seq_dims(model) -> Tuple[int, int]:
    """Return (k_seq_dim, v_seq_dim) based on model type.

    Shapes differ across HF model families; match the project's conventions.
    """
    mt = model.config.model_type.lower()
    if "llama" in mt:
        return 2, 2
    if "mpt" in mt:
        return 3, 2
    if "pythia" in mt or "gpt_neox" in mt:
        return 2, 2
    if "falcon" in mt:
        return 1, 1
    # Default fallback
    return 2, 2


def enable_pos_shift_for_model(model):
    """Enable positional shift modifications matching existing scripts."""
    mt = model.config.model_type.lower()
    try:
        if "llama" in mt:
            from streaming_llm.pos_shift.modify_llama import enable_llama_pos_shift_attention

            enable_llama_pos_shift_attention(model)
        elif "falcon" in mt:
            from streaming_llm.pos_shift.modify_falcon import enable_falcon_pos_shift_attention

            enable_falcon_pos_shift_attention(model)
        elif "gpt_neox" in mt or "pythia" in mt:
            from streaming_llm.pos_shift.modify_gpt_neox import enable_gpt_neox_pos_shift_attention

            enable_gpt_neox_pos_shift_attention(model)
        elif "mpt" in mt:
            # MPT doesn't need pos shift
            pass
    except Exception as e:
        print(f"Warning: failed to enable pos shift for {mt}: {e}")


def evaluate_method(
    method_name: str,
    model,
    tokenizer,
    dataset,
    kv_cache: Optional[object],
    num_samples: int,
    num_eval_tokens: Optional[int],
    log_every: int = 256,
    text_field: str = "text",
    concat_stream: bool = False,
    concat_limit: Optional[int] = None,
    min_seq_len: int = 32,
    sep: str = "\n\n",
):
    """Evaluate one method and return ppl, tokens/sec, elapsed_secs.

    This streams token-by-token over the first `num_samples` examples, accumulating NLLs.
    """
    device = next(model.parameters()).device
    loss_fn = CrossEntropyLoss(reduction="none")

    nlls = []
    past_key_values = None
    num_tokens = 0

    start_time = time.time()

    # Prepare one or many sequences depending on concat_stream
    sequences = []
    if concat_stream:
        # Keep concatenating until we reach num_eval_tokens (if specified) or exhaust dataset
        # Note: num_samples parameter is IGNORED in concat_stream mode - we pull as many
        # samples as needed to reach the token target
        max_items = min(len(dataset), concat_limit or len(dataset))
        texts = []
        empty_count = 0
        target_tokens = num_eval_tokens if num_eval_tokens is not None else float('inf')
        
        # Accumulate texts until we have enough tokens
        for i in range(max_items):
            item = dataset[i]
            if text_field not in item:
                raise KeyError(f"Dataset does not have field '{text_field}'. Available keys: {list(item.keys())}")
            t = item[text_field]
            if isinstance(t, str) and len(t.strip()) > 0:
                texts.append(t.strip())
            else:
                empty_count += 1
            
            # Check periodically (every 10 samples) to avoid excessive tokenization overhead
            if len(texts) > 0 and len(texts) % 10 == 0:
                big_text = sep.join(texts)
                enc = tokenizer(big_text, return_tensors="pt", add_special_tokens=False)
                current_tokens = enc.input_ids.size(1)
                if current_tokens >= target_tokens:
                    print(f"{method_name}: Reached target of {target_tokens} tokens after {len(texts)} samples (actual: {current_tokens} tokens)")
                    break
        
        print(f"{method_name}: Scanned {empty_count + len(texts)} items, found {len(texts)} non-empty (skipped {empty_count} empty)")
        
        if len(texts) == 0:
            print(f"{method_name} WARNING: No non-empty texts found in first {max_items} items. Check dataset.")
            return float("inf"), 0.0, 0.0, 0
        
        big_text = sep.join(texts)
        enc = tokenizer(big_text, return_tensors="pt")
        actual_tokens = enc.input_ids.size(1)
        print(f"{method_name}: Concatenated {len(texts)} samples into {actual_tokens} tokens (target was {num_eval_tokens or 'unlimited'})")
        sequences.append(enc.input_ids.to(device))
    else:
        # Iterate samples; skip empty or too-short items
        for sample_idx in range(min(num_samples, len(dataset))):
            item = dataset[sample_idx]
            if text_field not in item:
                raise KeyError(f"Dataset does not have field '{text_field}'. Available keys: {list(item.keys())}")
            text = item[text_field]
            if not isinstance(text, str) or len(text.strip()) == 0:
                continue
            encodings = tokenizer(text, return_tensors="pt")
            sequences.append(encodings.input_ids.to(device))

    # Evaluate over prepared sequences
    for input_ids in sequences:
        seq_len = input_ids.size(1)
        if seq_len < min_seq_len:
            # Skip very short sequences that won't exercise compression
            continue
        # Iterate token-by-token
        for t in range(seq_len - 1):
            with torch.no_grad():
                out = model(
                    input_ids[:, t : t + 1],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = out.logits.view(-1, model.config.vocab_size)
                past_key_values = out.past_key_values
                labels = input_ids[:, t + 1 : t + 2].to(logits.device).view(-1)
                nll = loss_fn(logits, labels)
                if kv_cache is not None:
                    past_key_values = kv_cache(past_key_values)

            nlls.append(nll)
            num_tokens += 1

            if log_every and (num_tokens % log_every == 0):
                ppl_now = torch.exp(torch.stack(nlls).mean()).item()
                print(f"{method_name} | tokens={num_tokens} ppl={ppl_now:.3f}")

            if num_eval_tokens is not None and num_tokens >= num_eval_tokens:
                break

        if num_eval_tokens is not None and num_tokens >= num_eval_tokens:
            break

    elapsed = time.time() - start_time
    ppl = float("inf") if len(nlls) == 0 else torch.exp(torch.stack(nlls).mean()).item()
    tok_per_sec = 0.0 if elapsed <= 0 else num_tokens / elapsed

    if num_tokens == 0:
        print(f"{method_name} WARNING: processed 0 tokens. Check dataset split/field or increase num_samples.")
    print(f"{method_name} Results: PPL={ppl:.4f}, Tokens/sec={tok_per_sec:.2f}, Time={elapsed:.1f}s, Tokens={num_tokens}")
    return ppl, tok_per_sec, elapsed, num_tokens


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ADMS vs StreamingLLM")
    # Model/Data
    p.add_argument("--model_name_or_path", type=str, required=True, help="HF model path or local path")
    p.add_argument("--dataset_name", type=str, default="wikitext")
    p.add_argument("--task", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--split", type=str, default="test", choices=["validation", "test"]) 
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--num_eval_tokens", type=int, default=None, help="Stop after this many tokens (across samples)")
    p.add_argument("--output_dir", type=str, default="outputs/adms_vs_streaming")

    # Shared cache settings
    p.add_argument("--start_size", type=int, default=4)
    p.add_argument("--recent_size", type=int, default=2000)
    p.add_argument("--enable_pos_shift", action="store_true", help="Enable positional-shift modifications for both methods")

    # ADMS-specific
    p.add_argument("--compressed_budget", type=int, default=128)
    p.add_argument("--compressor_type", type=str, default="low_rank", choices=["low_rank", "vq"]) 
    p.add_argument("--rank", type=int, default=16, help="Low-rank rank")
    p.add_argument("--num_clusters", type=int, default=64, help="VQ clusters")
    p.add_argument("--anchor_mode", type=str, default="mean", choices=["mean", "grid", "hybrid"], help="Anchor positions for compressed tokens")
    p.add_argument("--max_seq_length", type=int, default=32768, help="Maximum expected sequence length for dynamic sink sizing")
    p.add_argument("--enable_dynamic_sink", action="store_true", help="Enable dynamic sink size scaling (1%% of max_seq_length)")
    p.add_argument("--compression_interval", type=int, default=8)
    p.add_argument("--svd_max_tokens", type=int, default=512)
    p.add_argument("--min_middle_size_for_compress", type=int, default=64)
    p.add_argument("--importance_ratio", type=float, default=0.5, help="Fraction of compressed budget kept as exact top tokens")
    p.add_argument("--min_importance_tokens", type=int, default=4, help="Minimum exact tokens when importance ratio > 0")
    p.add_argument("--importance_metric", type=str, default="value_norm", choices=["value_norm", "key_norm", "mixed", "attention"], help="Scoring metric for importance-aware selection")
    p.add_argument("--use_adaptive_budget", action="store_true", help="Dynamically scale budget per head")
    p.add_argument("--attention_window", type=int, default=128, help="Recent tokens for attention-based importance")
    p.add_argument("--attention_blend", type=float, default=0.7, help="Blend factor between attention sim and value norm (0=value,1=attention)")
    p.add_argument("--no_importance_normalize", action="store_true", help="Disable normalization of importance scores")
    p.add_argument("--adaptive_budget_cap", type=float, default=2.5, help="Upper multiplier for adaptive budgeting")
    p.add_argument("--adaptive_budget_floor", type=float, default=0.5, help="Lower multiplier for adaptive budgeting")
    p.add_argument("--adaptive_variance_smoothing", type=float, default=0.1, help="EMA smoothing factor for variance-based scaling")
    p.add_argument("--compression_middle_threshold", type=int, default=256, help="Force compression when the middle exceeds this length")
    p.add_argument("--coverage_segments", type=int, default=4, help="Split middle region into this many segments for coverage-aware retention")
    p.add_argument("--coverage_priority", type=float, default=0.3, help="Fraction of budget reserved for per-segment coverage (0 disables)")

    # ADM++ feature toggles
    p.add_argument("--sketch_budget", type=int, default=32, help="Sketch-tier tokens per head when dual fidelity is enabled")
    p.add_argument("--replay_budget", type=int, default=16, help="Replay budget per head when residual replay is enabled")
    p.add_argument("--controller_gain", type=float, default=0.35)
    p.add_argument("--controller_energy_floor", type=float, default=0.8)
    p.add_argument("--controller_energy_ceiling", type=float, default=0.97)
    p.add_argument("--controller_group_size", type=int, default=2)

    p.add_argument("--enable_dual_fidelity", dest="enable_dual_fidelity", action="store_true", help="Enable sketch-tier dual fidelity (default)")
    p.add_argument("--disable_dual_fidelity", dest="enable_dual_fidelity", action="store_false", help="Disable sketch-tier dual fidelity")
    p.set_defaults(enable_dual_fidelity=True)

    p.add_argument("--enable_residual_replay", dest="enable_residual_replay", action="store_true", help="Enable residual replay (default)")
    p.add_argument("--disable_residual_replay", dest="enable_residual_replay", action="store_false", help="Disable residual replay")
    p.set_defaults(enable_residual_replay=True)

    p.add_argument("--enable_position_calibration", dest="enable_position_calibration", action="store_true", help="Enable energy-aware position calibration (default)")
    p.add_argument("--disable_position_calibration", dest="enable_position_calibration", action="store_false", help="Disable position calibration")
    p.set_defaults(enable_position_calibration=True)

    p.add_argument("--enable_adaptive_controller", dest="enable_adaptive_controller", action="store_true", help="Enable adaptive budget controller (default)")
    p.add_argument("--disable_adaptive_controller", dest="enable_adaptive_controller", action="store_false", help="Disable adaptive budget controller")
    p.set_defaults(enable_adaptive_controller=True)

    # Misc
    p.add_argument("--log_every", type=int, default=256)
    p.add_argument("--text_field", type=str, default="text", help="Field name in dataset item containing input text")
    p.add_argument("--concat_stream", action="store_true", help="Concatenate multiple samples into one long stream")
    p.add_argument("--concat_limit", type=int, default=None, help="Max items to pull for concatenation (default: all)")
    p.add_argument("--min_seq_len", type=int, default=32, help="Skip sequences shorter than this many tokens")
    p.add_argument("--sep", type=str, default="\n\n", help="Separator used when concatenating samples")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dataset {args.dataset_name}/{args.task}:{args.split}")
    data = load_dataset(args.dataset_name, args.task, split=args.split)

    print(f"Loading model {args.model_name_or_path}")
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)

    if args.enable_pos_shift:
        enable_pos_shift_for_model(model)

    k_seq_dim, v_seq_dim = detect_seq_dims(model)

    # 1) ADMS setup
    print("\nSetting up ADMS...")
    adms_cache = enable_adms_llm(
        model,
        start_size=args.start_size,
        recent_size=args.recent_size,
        compressed_budget=args.compressed_budget,
        compressor_type=args.compressor_type,
        anchor_mode=args.anchor_mode,
        rank=args.rank,
        num_clusters=args.num_clusters,
        enable_pos_shift=False,  # already applied above if requested
        max_seq_length=args.max_seq_length,
        enable_dynamic_sink=args.enable_dynamic_sink,
        compression_interval=args.compression_interval,
        svd_max_tokens=args.svd_max_tokens,
        min_middle_size_for_compress=args.min_middle_size_for_compress,
        importance_ratio=args.importance_ratio,
        min_importance_tokens=args.min_importance_tokens,
        importance_metric=args.importance_metric,
        use_adaptive_budget=args.use_adaptive_budget,
        attention_window=args.attention_window,
        attention_blend=args.attention_blend,
        importance_normalize=not args.no_importance_normalize,
        adaptive_budget_cap=args.adaptive_budget_cap,
        adaptive_budget_floor=args.adaptive_budget_floor,
        adaptive_variance_smoothing=args.adaptive_variance_smoothing,
        compression_middle_threshold=args.compression_middle_threshold,
        coverage_segments=args.coverage_segments,
        coverage_priority=args.coverage_priority,
        enable_dual_fidelity=args.enable_dual_fidelity,
        sketch_budget=args.sketch_budget,
        enable_residual_replay=args.enable_residual_replay,
        replay_budget=args.replay_budget,
        enable_position_calibration=args.enable_position_calibration,
        enable_adaptive_controller=args.enable_adaptive_controller,
        controller_gain=args.controller_gain,
        controller_energy_floor=args.controller_energy_floor,
        controller_energy_ceiling=args.controller_energy_ceiling,
        controller_group_size=args.controller_group_size,
    )

    adms_ppl, adms_speed, adms_time, adms_tokens = evaluate_method(
        "ADMS", model, tokenizer, data, adms_cache, args.num_samples, args.num_eval_tokens, args.log_every, args.text_field,
        concat_stream=args.concat_stream, concat_limit=args.concat_limit, min_seq_len=args.min_seq_len, sep=args.sep
    )

    # 2) StreamingLLM setup
    print("\nSetting up StreamingLLM (Start+Recent)...")
    streaming_cache = StartRecentKVCache(
        start_size=args.start_size,
        recent_size=args.recent_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )

    streaming_ppl, streaming_speed, streaming_time, streaming_tokens = evaluate_method(
        "StreamingLLM", model, tokenizer, data, streaming_cache, args.num_samples, args.num_eval_tokens, args.log_every, args.text_field,
        concat_stream=args.concat_stream, concat_limit=args.concat_limit, min_seq_len=args.min_seq_len, sep=args.sep
    )

    # Summary
    print("\nSummary (lower PPL is better):")
    print(f"ADMS         : PPL={adms_ppl:.4f}  tok/s={adms_speed:.1f}  time={adms_time:.1f}s  tokens={adms_tokens}")
    print(f"StreamingLLM : PPL={streaming_ppl:.4f}  tok/s={streaming_speed:.1f}  time={streaming_time:.1f}s  tokens={streaming_tokens}")

    if adms_ppl != float("inf") and streaming_ppl != float("inf"):
        if adms_ppl < streaming_ppl:
            improvement = ((streaming_ppl - adms_ppl) / streaming_ppl) * 100
            print(f"\n✓ ADMS better PPL than StreamingLLM by {improvement:.1f}%")
        else:
            decline = ((adms_ppl - streaming_ppl) / streaming_ppl) * 100
            print(f"\n! ADMS worse PPL than StreamingLLM by {decline:.1f}% (consider increasing compressed_budget/rank)")

    # Save
    out = {
        "args": vars(args),
        "results": {
            "ADMS": {
                "ppl": adms_ppl,
                "tokens_per_sec": adms_speed,
                "time_sec": adms_time,
                "tokens": adms_tokens,
            },
            "StreamingLLM": {
                "ppl": streaming_ppl,
                "tokens_per_sec": streaming_speed,
                "time_sec": streaming_time,
                "tokens": streaming_tokens,
            },
        },
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(args.output_dir, "readable.txt"), "w") as f:
        f.write("ADMS vs StreamingLLM Evaluation\n")
        f.write(f"Model: {args.model_name_or_path}\n")
        f.write(f"Data: {args.dataset_name}/{args.task}:{args.split}\n")
        f.write(f"Samples: {args.num_samples}  Tokens cap: {args.num_eval_tokens}\n")
        f.write(f"start={args.start_size} recent={args.recent_size} pos_shift={args.enable_pos_shift}\n")
        f.write(
            "ADMS: "
            f"budget={args.compressed_budget} compressor={args.compressor_type} "
            f"rank={args.rank} clusters={args.num_clusters} "
            f"importance_ratio={args.importance_ratio} min_exact={args.min_importance_tokens} "
            f"importance_metric={args.importance_metric}\n"
        )
        f.write(f"ADMS         : PPL={adms_ppl:.4f}  tok/s={adms_speed:.1f}  time={adms_time:.1f}s  tokens={adms_tokens}\n")
        f.write(f"StreamingLLM : PPL={streaming_ppl:.4f}  tok/s={streaming_speed:.1f}  time={streaming_time:.1f}s  tokens={streaming_tokens}\n")


if __name__ == "__main__":
    main()
