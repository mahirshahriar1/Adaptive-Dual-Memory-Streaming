"""
Evaluation script for ADMS (Adaptive Dual-Memory Streaming) LLM.
Optimized to avoid unnecessary compression and reduce per-step overhead.
"""

import os
import argparse
from typing import Optional

import torch
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from streaming_llm.enable_adms_llm import enable_adms_llm
from streaming_llm.utils import load


device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_adms_args():
    parser = argparse.ArgumentParser(description="Evaluate ADMS performance")

    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)

    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--task", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--split", type=str, default="test", choices=["validation", "test"])
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--num_eval_tokens", type=int, default=None)

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="outputs/adms_eval")

    # ADMS arguments
    parser.add_argument("--enable_adms", action="store_true")
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=2000)
    parser.add_argument("--compressed_budget", type=int, default=128)
    parser.add_argument("--compressor_type", type=str, default="low_rank", choices=["low_rank", "vq"])
    parser.add_argument("--anchor_mode", type=str, default="mean", choices=["grid", "mean", "hybrid"])
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--num_clusters", type=int, default=64)
    parser.add_argument("--enable_pos_shift", action="store_true")
    parser.add_argument("--compression_interval", type=int, default=8, help="Call ADMS every N tokens after threshold")

    # Comparison arguments
    parser.add_argument("--compare_baselines", action="store_true")

    return parser.parse_args()


def evaluate_perplexity(model, tokenizer, data, kv_cache: Optional[object], args, output_prefix: str = ""):
    nlls = []
    loss_fn = CrossEntropyLoss(reduction="none")

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f"{output_prefix}log.txt")
    with open(log_file, "w") as log_f:
        num_eval_tokens = 0
        print(f"Evaluating {output_prefix.replace('_', ' ').title()}...")

        for sample_idx, text in enumerate(data["text"][: args.num_samples]):
            print(f"\n=== Processing sample {sample_idx + 1}/{args.num_samples} ===")

            encodings = tokenizer(text, return_tensors="pt")
            print(f"First 10 tokens: {encodings.input_ids[:, :10]}")

            seq_len = encodings.input_ids.size(1)
            print(f"Sequence length: {seq_len}")

            if seq_len <= 1:
                print("Skipping sequence that's too short")
                continue

            # Reset per sample to avoid cross-sample accumulation
            past_key_values = None
            pbar = tqdm(range(0, seq_len - 1), desc=f"Sample {sample_idx + 1}")

            with torch.inference_mode():
                for idx in pbar:
                    input_ids = encodings.input_ids[:, idx : idx + 1].to(device)

                    outputs = model(
                        input_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    logits = outputs.logits.view(-1, model.config.vocab_size)
                    past_key_values = outputs.past_key_values

                    label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
                    neg_log_likelihood = loss_fn(logits, label)

                    # Apply KV cache compression: only after threshold and every N steps
                    if (
                        kv_cache is not None
                        and (idx + 1) > (args.start_size + args.recent_size)
                        and ((idx + 1) % max(1, args.compression_interval) == 0)
                    ):
                        past_key_values = kv_cache(past_key_values)

                    nlls.append(neg_log_likelihood)
                    current_ppl = torch.exp(neg_log_likelihood).item()

                    pbar.set_description(
                        f"Sample {sample_idx + 1} | NLL: {neg_log_likelihood.item():.2f}, "
                        f"PPL: {current_ppl:.2f} | Total tokens: {num_eval_tokens}"
                    )

                    print(neg_log_likelihood.item(), file=log_f, flush=True)
                    num_eval_tokens += 1

                    if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
                        break

            if nlls:
                current_overall_ppl = torch.exp(torch.stack(nlls).mean()).item()
                print(f"Completed sample {sample_idx + 1}. Current overall PPL: {current_overall_ppl:.2f}")

            if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
                break

    if len(nlls) == 0:
        print(f"No tokens evaluated for {output_prefix}")
        ppl = float("inf")
    else:
        ppl = torch.exp(torch.stack(nlls).mean()).item()

    print(f"{output_prefix.replace('_', ' ').title()} Final Perplexity: {ppl:.4f}")

    ppl_file = os.path.join(args.output_dir, f"{output_prefix}ppl.txt")
    with open(ppl_file, "w") as f:
        f.write(f"{ppl}\n")

    return ppl


def main():
    args = parse_adms_args()

    print("Loading dataset...")
    data = load_dataset(args.dataset_name, args.task, split=args.split)

    print("Loading model...")
    model, tokenizer = load(args.model_name_or_path)
    model.eval()

    results = {}

    if args.enable_adms:
        print("\n" + "=" * 50)
        print("EVALUATING WITH ADMS")
        print("=" * 50)

        adms_cache = enable_adms_llm(
            model,
            start_size=args.start_size,
            recent_size=args.recent_size,
            compressed_budget=args.compressed_budget,
            compressor_type=args.compressor_type,
            anchor_mode=args.anchor_mode,
            rank=args.rank,
            num_clusters=args.num_clusters,
            enable_pos_shift=args.enable_pos_shift,
        )

        adms_ppl = evaluate_perplexity(model, tokenizer, data, adms_cache, args, "adms_")
        results["ADMS"] = adms_ppl

    if args.compare_baselines:
        print("\n" + "=" * 50)
        print("EVALUATING WITH STREAMINGLLM BASELINE")
        print("=" * 50)

        from streaming_llm.enable_streaming_llm import enable_streaming_llm

        streaming_cache = enable_streaming_llm(
            model, start_size=args.start_size, recent_size=args.recent_size
        )

        streaming_ppl = evaluate_perplexity(
            model, tokenizer, data, streaming_cache, args, "streaming_"
        )
        results["StreamingLLM"] = streaming_ppl

        print("\n" + "=" * 50)
        print("EVALUATING VANILLA (NO CACHE)")
        print("=" * 50)

        vanilla_args = argparse.Namespace(**vars(args))
        if vanilla_args.num_eval_tokens is None:
            vanilla_args.num_eval_tokens = 1000

        vanilla_ppl = evaluate_perplexity(
            model, tokenizer, data, None, vanilla_args, "vanilla_"
        )
        results["Vanilla"] = vanilla_ppl

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)

    for method, ppl in results.items():
        print(f"{method:15s}: {ppl:.4f}")

    summary_file = os.path.join(args.output_dir, "summary.txt")
    with open(summary_file, "w") as f:
        f.write("ADMS Evaluation Summary\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Model: {args.model_name_or_path}\n")
        f.write(f"Dataset: {args.dataset_name}-{args.task}\n")
        f.write(f"Samples: {args.num_samples}\n")
        if args.num_eval_tokens:
            f.write(f"Max tokens: {args.num_eval_tokens}\n")
        f.write("\nResults:\n")
        for method, ppl in results.items():
            f.write(f"{method:15s}: {ppl:.4f}\n")

        if args.enable_adms:
            f.write("\nADMS Configuration:\n")
            f.write(f"Start size: {args.start_size}\n")
            f.write(f"Recent size: {args.recent_size}\n")
            f.write(f"Compressed budget: {args.compressed_budget}\n")
            f.write(f"Compressor: {args.compressor_type}\n")
            f.write(f"Anchor mode: {args.anchor_mode}\n")
            if args.compressor_type == "low_rank":
                f.write(f"Rank: {args.rank}\n")
            elif args.compressor_type == "vq":
                f.write(f"Clusters: {args.num_clusters}\n")
            f.write(f"Position shift: {args.enable_pos_shift}\n")

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

"""
Evaluation script for ADMS (Adaptive Dual-Memory Streaming) LLM.
Based on eval_long_ppl.py but with ADMS-specific configurations.
"""

import torch
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import os

from streaming_llm.adms_cache import ADMSKVCache, ADMSConfig
from streaming_llm.enable_adms_llm import enable_adms_llm
from streaming_llm.utils import parse_args, load

device = "cuda"

def parse_adms_args():
    """Parse arguments with ADMS-specific options"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate ADMS performance")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, required=True,
                       help="Model name or path")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--task", type=str, default="wikitext-2-raw-v1") 
    parser.add_argument("--split", type=str, default="test", choices=["validation", "test"])
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--num_eval_tokens", type=int, default=None)
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="outputs/adms_eval")
    
    # ADMS arguments
    parser.add_argument("--enable_adms", action="store_true", 
                       help="Enable ADMS compression")
    parser.add_argument("--start_size", type=int, default=4,
                       help="Number of sink tokens")
    parser.add_argument("--recent_size", type=int, default=2000,
                    """
                    Evaluation script for ADMS (Adaptive Dual-Memory Streaming) LLM.
                    Optimized to avoid unnecessary compression and reduce per-step overhead.
                    """

                    import os
                    import torch
                    from datasets import load_dataset
                    from torch.nn import CrossEntropyLoss
                    from tqdm import tqdm

                    from streaming_llm.enable_adms_llm import enable_adms_llm
                    from streaming_llm.utils import load

                    device = "cuda" if torch.cuda.is_available() else "cpu"


                    def parse_adms_args():
                        import argparse

                        parser = argparse.ArgumentParser(description="Evaluate ADMS performance")

                        # Model arguments
                        parser.add_argument("--model_name_or_path", type=str, required=True)
                        parser.add_argument("--revision", type=str, default="main")
                        parser.add_argument("--tokenizer_name_or_path", type=str, default=None)

                        # Dataset arguments
                        parser.add_argument("--dataset_name", type=str, default="wikitext")
                        parser.add_argument("--task", type=str, default="wikitext-2-raw-v1")
                        parser.add_argument("--split", type=str, default="test", choices=["validation", "test"])
                        parser.add_argument("--num_samples", type=int, default=10)
                        parser.add_argument("--num_eval_tokens", type=int, default=None)

                        # Output arguments
                        parser.add_argument("--output_dir", type=str, default="outputs/adms_eval")

                        # ADMS arguments
                        parser.add_argument("--enable_adms", action="store_true")
                        parser.add_argument("--start_size", type=int, default=4)
                        parser.add_argument("--recent_size", type=int, default=2000)
                        parser.add_argument("--compressed_budget", type=int, default=128)
                        parser.add_argument("--compressor_type", type=str, default="low_rank", choices=["low_rank", "vq"])
                        parser.add_argument("--anchor_mode", type=str, default="mean", choices=["grid", "mean", "hybrid"])
                        parser.add_argument("--rank", type=int, default=16)
                        parser.add_argument("--num_clusters", type=int, default=64)
                        parser.add_argument("--enable_pos_shift", action="store_true")
                        parser.add_argument("--compression_interval", type=int, default=8, help="Call ADMS every N tokens after threshold")

                        # Comparison arguments
                        parser.add_argument("--compare_baselines", action="store_true")

                        return parser.parse_args()


                    def evaluate_perplexity(model, tokenizer, data, kv_cache, args, output_prefix=""):
                        nlls = []
                        loss_fn = CrossEntropyLoss(reduction="none")

                        os.makedirs(args.output_dir, exist_ok=True)
                        log_file = os.path.join(args.output_dir, f"{output_prefix}log.txt")
                        log_f = open(log_file, "w")

                        num_eval_tokens = 0
                        print(f"Evaluating {output_prefix.replace('_', ' ').title()}...")

                        for sample_idx, text in enumerate(data["text"][: args.num_samples]):
                            print(f"\n=== Processing sample {sample_idx + 1}/{args.num_samples} ===")

                            encodings = tokenizer(text, return_tensors="pt")
                            print(f"First 10 tokens: {encodings.input_ids[:, :10]}")

                            seq_len = encodings.input_ids.size(1)
                            print(f"Sequence length: {seq_len}")

                            if seq_len <= 1:
                                print("Skipping sequence that's too short")
                                continue

                            # Reset per sample to avoid cross-sample accumulation
                            past_key_values = None
                            pbar = tqdm(range(0, seq_len - 1), desc=f"Sample {sample_idx + 1}")

                            with torch.inference_mode():
                                for idx in pbar:
                                    input_ids = encodings.input_ids[:, idx : idx + 1].to(device)

                                    outputs = model(
                                        input_ids,
                                        past_key_values=past_key_values,
                                        use_cache=True,
                                    )
                                    logits = outputs.logits.view(-1, model.config.vocab_size)
                                    past_key_values = outputs.past_key_values

                                    label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
                                    neg_log_likelihood = loss_fn(logits, label)

                                    # Apply KV cache compression: only after threshold and every N steps
                                    if (
                                        kv_cache is not None
                                        and (idx + 1) > (args.start_size + args.recent_size)
                                        and ((idx + 1) % max(1, args.compression_interval) == 0)
                                    ):
                                        past_key_values = kv_cache(past_key_values)

                                    nlls.append(neg_log_likelihood)
                                    current_ppl = torch.exp(neg_log_likelihood).item()

                                    pbar.set_description(
                                        f"Sample {sample_idx + 1} | NLL: {neg_log_likelihood.item():.2f}, "
                                        f"PPL: {current_ppl:.2f} | Total tokens: {num_eval_tokens}"
                                    )

                                    print(neg_log_likelihood.item(), file=log_f, flush=True)
                                    num_eval_tokens += 1

                                    if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
                                        break

                            if nlls:
                                current_overall_ppl = torch.exp(torch.stack(nlls).mean()).item()
                                print(f"Completed sample {sample_idx + 1}. Current overall PPL: {current_overall_ppl:.2f}")

                            if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
                                break

                        log_f.close()

                        if len(nlls) == 0:
                            print(f"No tokens evaluated for {output_prefix}")
                            ppl = float("inf")
                        else:
                            ppl = torch.exp(torch.stack(nlls).mean()).item()

                        print(f"{output_prefix.replace('_', ' ').title()} Final Perplexity: {ppl:.4f}")

                        ppl_file = os.path.join(args.output_dir, f"{output_prefix}ppl.txt")
                        with open(ppl_file, "w") as f:
                            f.write(f"{ppl}\n")

                        return ppl


                    def main():
                        args = parse_adms_args()

                        print("Loading dataset...")
                        data = load_dataset(args.dataset_name, args.task, split=args.split)

                        print("Loading model...")
                        model, tokenizer = load(args.model_name_or_path)
                        model.eval()

                        results = {}

                        if args.enable_adms:
                            print("\n" + "=" * 50)
                            print("EVALUATING WITH ADMS")
                            print("=" * 50)

                            adms_cache = enable_adms_llm(
                                model,
                                start_size=args.start_size,
                                recent_size=args.recent_size,
                                compressed_budget=args.compressed_budget,
                                compressor_type=args.compressor_type,
                                anchor_mode=args.anchor_mode,
                                rank=args.rank,
                                num_clusters=args.num_clusters,
                                enable_pos_shift=args.enable_pos_shift,
                            )

                            adms_ppl = evaluate_perplexity(model, tokenizer, data, adms_cache, args, "adms_")
                            results["ADMS"] = adms_ppl

                        if args.compare_baselines:
                            print("\n" + "=" * 50)
                            print("EVALUATING WITH STREAMINGLLM BASELINE")
                            print("=" * 50)

                            from streaming_llm.enable_streaming_llm import enable_streaming_llm

                            streaming_cache = enable_streaming_llm(
                                model, start_size=args.start_size, recent_size=args.recent_size
                            )

                            streaming_ppl = evaluate_perplexity(
                                model, tokenizer, data, streaming_cache, args, "streaming_"
                            )
                            results["StreamingLLM"] = streaming_ppl

                            print("\n" + "=" * 50)
                            print("EVALUATING VANILLA (NO CACHE)")
                            print("=" * 50)

                            vanilla_args = argparse.Namespace(**vars(args))
                            if vanilla_args.num_eval_tokens is None:
                                vanilla_args.num_eval_tokens = 1000

                            vanilla_ppl = evaluate_perplexity(
                                model, tokenizer, data, None, vanilla_args, "vanilla_"
                            )
                            results["Vanilla"] = vanilla_ppl

                        print("\n" + "=" * 50)
                        print("EVALUATION SUMMARY")
                        print("=" * 50)

                        for method, ppl in results.items():
                            print(f"{method:15s}: {ppl:.4f}")

                        summary_file = os.path.join(args.output_dir, "summary.txt")
                        with open(summary_file, "w") as f:
                            f.write("ADMS Evaluation Summary\n")
                            f.write("=" * 30 + "\n\n")
                            f.write(f"Model: {args.model_name_or_path}\n")
                            f.write(f"Dataset: {args.dataset_name}-{args.task}\n")
                            f.write(f"Samples: {args.num_samples}\n")
                            if args.num_eval_tokens:
                                f.write(f"Max tokens: {args.num_eval_tokens}\n")
                            f.write("\nResults:\n")
                            for method, ppl in results.items():
                                f.write(f"{method:15s}: {ppl:.4f}\n")

                            if args.enable_adms:
                                f.write("\nADMS Configuration:\n")
                                f.write(f"Start size: {args.start_size}\n")
                                f.write(f"Recent size: {args.recent_size}\n")
                                f.write(f"Compressed budget: {args.compressed_budget}\n")
                                f.write(f"Compressor: {args.compressor_type}\n")
                                f.write(f"Anchor mode: {args.anchor_mode}\n")
                                if args.compressor_type == "low_rank":
                                    f.write(f"Rank: {args.rank}\n")
                                elif args.compressor_type == "vq":
                                    f.write(f"Clusters: {args.num_clusters}\n")
                                f.write(f"Position shift: {args.enable_pos_shift}\n")

                        print(f"\nResults saved to: {args.output_dir}")


                    if __name__ == "__main__":
                        import argparse
                        main()