#!/usr/bin/env python3
"""
Quick unit test for ADMS cache size bug
Simulates cache calls without running full model inference
"""
import torch
from streaming_llm.adms_cache import ADMSKVCache, ADMSConfig

print("=" * 60)
print("ADMS CACHE UNIT TEST")
print("=" * 60)

# Create config matching your test parameters
config = ADMSConfig(
    start_size=4,
    recent_size=256,
    compressed_budget=512,
    compressor_type="low_rank",
    rank=16,
    compression_interval=32,
    svd_max_tokens=512,
    min_middle_size_for_compress=8,  # Low threshold for fast testing
    importance_ratio=0.8,
    compression_middle_threshold=256,
)

cache = ADMSKVCache(config)

# Simulate cache growth
num_layers = 2
num_heads = 4
d_k = 64
d_v = 64

print("\nSimulating token-by-token cache growth...")
print("-" * 60)

for seq_len in [260, 268, 300, 350, 400, 500, 600, 700, 772, 800, 1000, 1500, 2000]:
    # Create fake past_key_values
    past_kv = []
    for layer in range(num_layers):
        k = torch.randn(1, num_heads, seq_len, d_k)
        v = torch.randn(1, num_heads, seq_len, d_v)
        past_kv.append((k, v))
    
    # Apply cache
    result = cache(past_kv)
    
    # Check result size
    result_size = result[0][0].shape[2]
    expected_max = config.start_size + config.compressed_budget + config.recent_size
    
    status = "✓ OK" if result_size <= expected_max else "✗ BUG!"
    print(f"seq_len={seq_len:4d} → cache_size={result_size:4d} (max={expected_max}) {status}")
    
    # Show stats at key milestones
    if seq_len in [512, 1000, 2000] and cache.stats["total_compressions"] > 0:
        avg_exact = cache.stats["total_exact_kept"] / cache.stats["total_compressions"]
        avg_comp = cache.stats["total_compressed_kept"] / cache.stats["total_compressions"]
        avg_dropped = cache.stats["total_middle_dropped"] / cache.stats["total_compressions"]
        print(f"  Stats: exact={avg_exact:.1f}, compressed={avg_comp:.1f}, dropped={avg_dropped:.1f}")

print("-" * 60)
print("\nTest complete!")
print(f"Total compressions: {cache.stats['total_compressions']}")

if cache.stats['total_compressions'] == 0:
    print("❌ FAIL: No compressions ran!")
else:
    print("✓ Compression ran successfully")
