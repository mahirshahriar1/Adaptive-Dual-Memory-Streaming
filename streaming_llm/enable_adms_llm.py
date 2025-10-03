"""
ADMS (Adaptive Dual-Memory Streaming) enabler functions.
Similar to enable_streaming_llm.py but with compression support.
"""

from .adms_cache import ADMSKVCache, ADMSConfig
from .pos_shift.modify_llama import enable_llama_pos_shift_attention
from .pos_shift.modify_falcon import enable_falcon_pos_shift_attention
from .pos_shift.modify_gpt_neox import enable_gpt_neox_pos_shift_attention


def enable_adms_llm(
    model,
    start_size=4,
    recent_size=2000,
    compressed_budget=128,
    compressor_type="low_rank",
    anchor_mode="mean",
    k_seq_dim=None,
    v_seq_dim=None,
    rank=16,
    num_clusters=64,
    enable_pos_shift=True,
    max_seq_length=32768,
    enable_dynamic_sink=True,
    # Performance knobs
    compression_interval: int = 8,
    svd_max_tokens: int = 512,
    min_middle_size_for_compress: int = 64,
    importance_ratio: float = 0.5,
    min_importance_tokens: int = 4,
    importance_metric: str = "value_norm",
    use_adaptive_budget: bool = True,
    attention_window: int = 128,
    attention_blend: float = 0.7,
    importance_normalize: bool = True,
    adaptive_budget_cap: float = 2.5,
    adaptive_budget_floor: float = 0.5,
    adaptive_variance_smoothing: float = 0.1,
    compression_middle_threshold: int = 256,
    coverage_segments: int = 4,
    coverage_priority: float = 0.3,
):
    """
    Enable ADMS (Adaptive Dual-Memory Streaming) for a language model.
    
    Args:
        model: The transformer model to enable ADMS for
        start_size: Number of sink tokens to keep (default: 4, base/minimum)
        recent_size: Size of recent window (default: 2000)  
        compressed_budget: Maximum compressed tokens per layer/head (default: 128)
        compressor_type: Type of compressor - "low_rank" or "vq" (default: "low_rank")
        anchor_mode: Position anchoring strategy - "grid", "mean", or "hybrid" (default: "mean")
        k_seq_dim: Key sequence dimension (auto-detected if None)
        v_seq_dim: Value sequence dimension (auto-detected if None)
        rank: Rank for low-rank compression (default: 16)
        num_clusters: Number of clusters for VQ compression (default: 64)
        enable_pos_shift: Whether to enable positional shifting (default: True)
        max_seq_length: Maximum expected sequence length for dynamic sink sizing (default: 32768)
        enable_dynamic_sink: Scale sink size proportionally with max_seq_length (default: True)
        importance_ratio: Fraction of compressed budget reserved for exact top tokens (default: 0.5)
        min_importance_tokens: Minimum number of exact tokens if ratio > 0 (default: 4)
        importance_metric: Scoring metric for importance-aware selection (default: "value_norm")
        use_adaptive_budget: Dynamically scale budget per head based on middle size (default: True)
        attention_window: Recent tokens used to compute attention-based importance (default: 128)
    attention_blend: Weight for attention vs value norm when ``importance_metric="attention"``
    importance_normalize: Whether to normalize importance scores to [0,1]
    adaptive_budget_cap: Maximum multiplier applied during adaptive budgeting
    adaptive_budget_floor: Minimum multiplier applied during adaptive budgeting
    adaptive_variance_smoothing: EMA smoothing factor for variance-based scaling
    compression_middle_threshold: Trigger compression sooner once middle exceeds this length
        
    Returns:
        ADMSKVCache: The ADMS KV cache object
    """
    
    # Auto-detect sequence dimensions based on model type
    model_type = model.config.model_type.lower()
    
    if k_seq_dim is None or v_seq_dim is None:
        if "llama" in model_type:
            k_seq_dim = v_seq_dim = 2
        elif "mpt" in model_type:
            v_seq_dim = 2
            k_seq_dim = 3
        elif "pythia" in model_type or "gpt_neox" in model_type:
            k_seq_dim = v_seq_dim = 2
        elif "falcon" in model_type:
            v_seq_dim = 1
            k_seq_dim = 1
        else:
            # Default fallback
            k_seq_dim = v_seq_dim = 2
            print(f"Warning: Unknown model type {model_type}, using default k_seq_dim=v_seq_dim=2")
    
    # Enable positional shifting if requested
    if enable_pos_shift:
        if "llama" in model_type:
            enable_llama_pos_shift_attention(model)
        elif "falcon" in model_type:
            enable_falcon_pos_shift_attention(model)
        elif "gpt_neox" in model_type or "pythia" in model_type:
            enable_gpt_neox_pos_shift_attention(model)
        elif "mpt" in model_type:
            pass  # MPT doesn't need pos shift
        else:
            print(f"Warning: Position shift not implemented for {model_type}")
    
    # Create ADMS configuration
    config = ADMSConfig(
        start_size=start_size,
        recent_size=recent_size,
        compressed_budget=compressed_budget,
        compressor_type=compressor_type,
        anchor_mode=anchor_mode,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
        rank=rank,
        num_clusters=num_clusters,
        max_seq_length=max_seq_length,
        enable_dynamic_sink=enable_dynamic_sink,
        compression_interval=compression_interval,
        svd_max_tokens=svd_max_tokens,
        min_middle_size_for_compress=min_middle_size_for_compress,
        importance_ratio=importance_ratio,
        min_importance_tokens=min_importance_tokens,
        importance_metric=importance_metric,
        use_adaptive_budget=use_adaptive_budget,
        attention_window=attention_window,
        attention_blend=attention_blend,
        importance_normalize=importance_normalize,
        adaptive_budget_cap=adaptive_budget_cap,
        adaptive_budget_floor=adaptive_budget_floor,
        adaptive_variance_smoothing=adaptive_variance_smoothing,
        compression_middle_threshold=compression_middle_threshold,
        coverage_segments=coverage_segments,
        coverage_priority=coverage_priority,
    )
    
    # Create and return ADMS KV cache
    adms_cache = ADMSKVCache(config)
    
    print(f"ADMS enabled for {model_type} model:")
    print(f"  - Sink tokens (base): {start_size}")
    if enable_dynamic_sink:
        dynamic_size = max(start_size, int(0.01 * max_seq_length))
        print(f"  - Sink tokens (dynamic): {dynamic_size} (for {max_seq_length} context)")
    print(f"  - Recent window: {recent_size}")
    print(f"  - Compressed budget: {compressed_budget}")
    print(f"  - Compressor: {compressor_type}")
    print(f"  - Anchor mode: {anchor_mode}")
    if compressor_type == "low_rank":
        print(f"  - Rank: {rank}")
    elif compressor_type == "vq":
        print(f"  - Clusters: {num_clusters}")
    print(f"  - Position shift: {enable_pos_shift}")
    print("  - Performance knobs:")
    print(f"    * compression_interval: {compression_interval}")
    print(f"    * compression_middle_threshold: {compression_middle_threshold}")
    print(f"    * svd_max_tokens: {svd_max_tokens}")
    print(f"    * min_middle_size_for_compress: {min_middle_size_for_compress}")
    print(f"    * coverage_segments: {coverage_segments}")
    print(f"    * coverage_priority: {coverage_priority}")
    
    return adms_cache


# Convenience functions for different compressor types

def enable_adms_low_rank(model, start_size=4, recent_size=2000, compressed_budget=128, 
                        rank=16, **kwargs):
    """Enable ADMS with low-rank compression"""
    return enable_adms_llm(
        model=model,
        start_size=start_size,
        recent_size=recent_size,
        compressed_budget=compressed_budget,
        compressor_type="low_rank",
        rank=rank,
        **kwargs
    )


def enable_adms_vq(model, start_size=4, recent_size=2000, compressed_budget=128, 
                  num_clusters=64, **kwargs):
    """Enable ADMS with vector quantization compression"""
    return enable_adms_llm(
        model=model,
        start_size=start_size,
        recent_size=recent_size,
        compressed_budget=compressed_budget,
        compressor_type="vq",
        num_clusters=num_clusters,
        **kwargs
    )


# Backward compatibility alias
enable_streaming_adms = enable_adms_llm