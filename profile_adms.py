"""
ADMS Profiler - Identifies bottlenecks in compression pipeline
"""

import torch
import time
from streaming_llm.adms_cache import LowRankCompressor, ADMSConfig

def profile_compression(seq_len: int, rank: int, max_tokens: int = None):
    """Profile SVD compression time"""
    compressor = LowRankCompressor(rank=rank, max_tokens=max_tokens)
    
    # Create dummy data
    K = torch.randn(64, seq_len, device='cuda' if torch.cuda.is_available() else 'cpu')
    V = torch.randn(64, seq_len, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Warmup
    for _ in range(3):
        _ = compressor.compress(K, V)
    
    # Time it
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    
    for _ in range(10):
        K_comp, V_comp = compressor.compress(K, V)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start
    
    avg_time = elapsed / 10
    return avg_time, K_comp.shape[1]

def main():
    print("=" * 60)
    print("ADMS Compression Profiler")
    print("=" * 60)
    
    configs = [
        # (seq_len, rank, max_tokens, label)
        (1000, 16, None, "Small context, no subsample"),
        (1000, 16, 256, "Small context, subsample=256"),
        (5000, 16, None, "Medium context, no subsample"),
        (5000, 16, 256, "Medium context, subsample=256"),
        (10000, 16, None, "Large context, no subsample"),
        (10000, 16, 256, "Large context, subsample=256"),
        (15000, 16, None, "XL context, no subsample"),
        (15000, 16, 256, "XL context, subsample=256"),
    ]
    
    print(f"\n{'Configuration':<35} {'Time (ms)':<12} {'Output cols':<12} {'Speedup':<10}")
    print("-" * 70)
    
    baseline_time = None
    for seq_len, rank, max_tokens, label in configs:
        try:
            avg_time, out_cols = profile_compression(seq_len, rank, max_tokens)
            time_ms = avg_time * 1000
            
            if baseline_time is None:
                baseline_time = avg_time
                speedup = "baseline"
            else:
                speedup = f"{baseline_time / avg_time:.2f}x"
            
            print(f"{label:<35} {time_ms:>10.2f}ms {out_cols:>10} {speedup:>10}")
        except Exception as e:
            print(f"{label:<35} FAILED: {str(e)[:30]}")
    
    print("\n" + "=" * 60)
    print("Recommendations:")
    print("  - For sequences > 5000: Use max_tokens=256-512")
    print("  - For sequences > 10000: Set compression_interval=32-64")
    print("  - For sequences > 15000: Consider importance_ratio=1.0 (skip SVD)")
    print("=" * 60)

if __name__ == "__main__":
    main()
