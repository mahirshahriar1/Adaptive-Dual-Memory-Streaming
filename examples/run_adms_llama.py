"""
Interactive ADMS (Adaptive Dual-Memory Streaming) demo script.
Based on run_streaming_llama.py but with ADMS compression.
"""

import torch
import argparse
import json
import os
from tqdm import tqdm

from streaming_llm.enable_adms_llm import enable_adms_llm
from streaming_llm.utils import load


@torch.no_grad()
def adms_greedy_generate(model, tokenizer, input_ids, past_key_values, kv_cache, max_gen_len):
    """Greedy generation with ADMS compression"""
    outputs = []
    
    for _ in range(max_gen_len):
        with torch.no_grad():
            model_output = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = model_output.past_key_values
            pred_token_idx = model_output.logits[0, -1:].argmax(dim=-1).unsqueeze(0)
            
            # Apply ADMS compression
            if kv_cache is not None:
                past_key_values = kv_cache(past_key_values)
            
            outputs.append(pred_token_idx.item())
            input_ids = pred_token_idx
            
            # Check for EOS token
            if pred_token_idx.item() == tokenizer.eos_token_id:
                break
    
    return outputs


@torch.no_grad()
def adms_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=1000, verbose=True):
    """Run inference with ADMS on multiple prompts"""
    
    past_key_values = None
    
    for idx, prompt in enumerate(prompts):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing prompt {idx + 1}/{len(prompts)}")
            print(f"{'='*60}")
            print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        
        # Tokenize prompt
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        
        if verbose:
            print(f"Prompt length: {input_ids.shape[1]} tokens")
        
        # Generate response
        if verbose:
            print("Generating response...")
        
        output_ids = adms_greedy_generate(
            model, tokenizer, input_ids, past_key_values, kv_cache, max_gen_len
        )
        
        # Decode response
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        if verbose:
            print(f"Response length: {len(output_ids)} tokens")
            print(f"Response: {output_text}")
            
            # Print memory usage info if available
            if kv_cache is not None:
                seq_len = input_ids.shape[1] + len(output_ids)
                estimated_cache_size = (kv_cache.start_size + 
                                      kv_cache.compressed_budget + 
                                      kv_cache.recent_size)
                compression_ratio = estimated_cache_size / max(seq_len, 1)
                print(f"Sequence length: {seq_len}")
                print(f"Estimated cache size: {estimated_cache_size}")
                print(f"Compression ratio: {compression_ratio:.3f}")
        
        # Update past key values for next prompt
        # In practice, you might want to reset for each conversation turn
        # past_key_values = None  # Uncomment to reset context between prompts


def parse_adms_demo_args():
    """Parse command line arguments for ADMS demo"""
    parser = argparse.ArgumentParser(description="ADMS Interactive Demo")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, 
                       default="lmsys/vicuna-13b-v1.3",
                       help="Model name or path")
    parser.add_argument("--data_root", type=str, default="data/",
                       help="Directory containing test data")
    
    # ADMS arguments
    parser.add_argument("--enable_adms", action="store_true",
                       help="Enable ADMS compression")
    parser.add_argument("--start_size", type=int, default=4,
                       help="Number of sink tokens")
    parser.add_argument("--recent_size", type=int, default=2000,
                       help="Size of recent window")
    parser.add_argument("--compressed_budget", type=int, default=128,
                       help="Maximum compressed tokens")
    parser.add_argument("--compressor_type", type=str, default="low_rank",
                       choices=["low_rank", "vq"], help="Type of compressor")
    parser.add_argument("--anchor_mode", type=str, default="mean",
                       choices=["grid", "mean", "hybrid"], help="Anchoring mode")
    parser.add_argument("--rank", type=int, default=16,
                       help="Rank for low-rank compression")
    parser.add_argument("--num_clusters", type=int, default=64,
                       help="Clusters for VQ compression")
    parser.add_argument("--enable_pos_shift", action="store_true",
                       help="Enable positional shifting")
    
    # Generation arguments
    parser.add_argument("--max_gen_len", type=int, default=1000,
                       help="Maximum generation length")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive mode")
    parser.add_argument("--prompts_file", type=str, default=None,
                       help="JSON file with prompts to test")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Verbose output")
    
    return parser.parse_args()


def load_test_prompts(data_root, prompts_file=None):
    """Load test prompts from file or use defaults"""
    
    if prompts_file and os.path.exists(prompts_file):
        with open(prompts_file, 'r') as f:
            data = json.load(f)
            return data.get('prompts', [])
    
    # Try to load MT-Bench data
    mt_bench_file = os.path.join(data_root, "mt_bench.jsonl")
    
    if os.path.exists(mt_bench_file):
        prompts = []
        with open(mt_bench_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                prompts.extend(data.get("turns", []))
        return prompts[:5]  # Limit to first 5 for demo
    
    # Default prompts if no data files found
    default_prompts = [
        "Tell me about the benefits of renewable energy sources.",
        "Explain the concept of artificial intelligence in simple terms.",
        "What are the main challenges facing modern education systems?",
        "Describe the process of photosynthesis in plants.",
        "How do social media platforms influence modern communication?"
    ]
    
    return default_prompts


def interactive_mode(model, tokenizer, kv_cache, args):
    """Run ADMS in interactive mode"""
    print("\n" + "="*60)
    print("ADMS INTERACTIVE MODE")
    print("="*60)
    print("Type 'quit' or 'exit' to end the session")
    print("Type 'reset' to clear conversation history")
    print("Type 'stats' to show compression statistics")
    print("="*60)
    
    past_key_values = None
    conversation_length = 0
    
    while True:
        try:
            user_input = input("\nUser: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                break
            elif user_input.lower() == 'reset':
                past_key_values = None
                conversation_length = 0
                print("Conversation history cleared.")
                continue
            elif user_input.lower() == 'stats':
                if kv_cache is not None:
                    estimated_cache = (kv_cache.start_size + 
                                     kv_cache.compressed_budget + 
                                     kv_cache.recent_size)
                    print(f"Conversation length: {conversation_length} tokens")
                    print(f"Estimated cache size: {estimated_cache}")
                    print(f"Compression ratio: {estimated_cache / max(conversation_length, 1):.3f}")
                else:
                    print("ADMS not enabled")
                continue
            elif not user_input:
                continue
            
            # Generate response
            print("Assistant: ", end="", flush=True)
            
            input_ids = tokenizer(user_input, return_tensors="pt").input_ids.to(model.device)
            conversation_length += input_ids.shape[1]
            
            output_ids = adms_greedy_generate(
                model, tokenizer, input_ids, past_key_values, kv_cache, args.max_gen_len
            )
            
            response = tokenizer.decode(output_ids, skip_special_tokens=True)
            print(response)
            
            conversation_length += len(output_ids)
            
        except KeyboardInterrupt:
            print("\n\nSession interrupted by user.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


def main():
    args = parse_adms_demo_args()
    
    print("Loading model...")
    model, tokenizer = load(args.model_name_or_path)
    
    # Setup ADMS if enabled
    kv_cache = None
    if args.enable_adms:
        print("\nEnabling ADMS...")
        kv_cache = enable_adms_llm(
            model,
            start_size=args.start_size,
            recent_size=args.recent_size,
            compressed_budget=args.compressed_budget,
            compressor_type=args.compressor_type,
            anchor_mode=args.anchor_mode,
            rank=args.rank,
            num_clusters=args.num_clusters,
            enable_pos_shift=args.enable_pos_shift
        )
    else:
        print("\nRunning without ADMS compression")
    
    if args.interactive:
        # Interactive mode
        interactive_mode(model, tokenizer, kv_cache, args)
    else:
        # Batch evaluation mode
        print("\nLoading test prompts...")
        prompts = load_test_prompts(args.data_root, args.prompts_file)
        
        if not prompts:
            print("No prompts found. Use --interactive mode or provide --prompts_file")
            return
        
        print(f"Running ADMS demo on {len(prompts)} prompts...")
        
        adms_inference(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            kv_cache=kv_cache,
            max_gen_len=args.max_gen_len,
            verbose=args.verbose
        )
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()