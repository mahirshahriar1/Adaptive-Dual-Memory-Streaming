"""
StreamingLLM: Efficient Streaming Language Models with Attention Sinks
Extended with ADMS: Adaptive Dual-Memory Streaming
"""

from .enable_streaming_llm import enable_streaming_llm
from .kv_cache import StartRecentKVCache

# ADMS extensions
from .enable_adms_llm import (
    enable_adms_llm, 
    enable_adms_low_rank, 
    enable_adms_vq,
    enable_streaming_adms  # backward compatibility
)
from .adms_cache import ADMSKVCache, ADMSConfig, LowRankCompressor, VQCompressor

__version__ = "1.0.0"

__all__ = [
    # Original StreamingLLM
    "enable_streaming_llm",
    "StartRecentKVCache",
    
    # ADMS extensions
    "enable_adms_llm",
    "enable_adms_low_rank", 
    "enable_adms_vq",
    "enable_streaming_adms",
    "ADMSKVCache",
    "ADMSConfig",
    "LowRankCompressor",
    "VQCompressor",
]