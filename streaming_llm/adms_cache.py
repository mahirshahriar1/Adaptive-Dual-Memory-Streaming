"""
Adaptive Dual-Memory Streaming (ADMS) KV Cache implementation.
Extends StreamingLLM with a compressed mid-memory tier.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass


@dataclass
class ADMSConfig:
    """Configuration for ADMS KV Cache"""
    start_size: int = 4  # sink tokens (base/minimum)
    recent_size: int = 2000  # recent window
    compressed_budget: int = 128  # max compressed tokens
    compressor_type: str = "low_rank"  # "low_rank", "vq", "summary"
    anchor_mode: str = "mean"  # "grid", "mean", "hybrid"
    k_seq_dim: int = 2
    v_seq_dim: int = 2
    rank: int = 16  # for low-rank compression
    num_clusters: int = 64  # for VQ compression
    compression_ratio: float = 0.25  # compression ratio (0.25 = 4:1)
    max_seq_length: int = 32768  # maximum expected sequence length
    enable_dynamic_sink: bool = True  # scale sink size with context length
    # Performance knobs
    compression_interval: int = 8  # only run compression every N new tokens
    compression_middle_threshold: int = 256  # force compression once middle exceeds this many tokens
    svd_max_tokens: int = 512  # cap columns used in SVD to this many tokens per head
    min_middle_size_for_compress: int = 64  # skip compression if middle < this
    importance_ratio: float = 0.5  # fraction of budget reserved for exact top tokens
    min_importance_tokens: int = 4  # minimum number of exact tokens if ratio > 0
    importance_metric: str = "value_norm"  # scoring: value_norm, key_norm, mixed, attention
    use_adaptive_budget: bool = False  # dynamically adjust budget per head (DISABLED by default - prevents cache explosion)
    attention_window: int = 128  # recent tokens to compute importance from
    attention_blend: float = 0.7  # mix attention sim with value norm when available
    importance_normalize: bool = True  # normalize scores to [0,1]
    adaptive_budget_cap: float = 2.5
    adaptive_budget_floor: float = 0.5
    adaptive_variance_smoothing: float = 0.1
    coverage_segments: int = 4  # split middle into segments for coverage-aware allocation
    coverage_priority: float = 0.3  # fraction of budget reserved for broad coverage


class LowRankCompressor(nn.Module):
    """Low-rank approximation compressor for KV tensors"""
    
    def __init__(self, rank: int = 16, max_tokens: Optional[int] = None):
        super().__init__()
        self.rank = rank
        self.max_tokens = max_tokens
        
    def compress(self, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress K, V using low-rank SVD approximation
        
        Args:
            K: Key tensor [d_k, seq_len]
            V: Value tensor [d_v, seq_len] 
            
        Returns:
            Compressed K, V tensors
        """
        if K.shape[1] == 0:
            return K, V
            
        device = K.device
        dtype = K.dtype
        
        # Fast path: if already small enough, just subsample
        if K.shape[1] <= self.rank * 2:
            step = max(1, K.shape[1] // self.rank)
            return K[:, ::step], V[:, ::step]
        
        try:
            # Pre-subsample to bound SVD cost
            subsample_size = self.max_tokens if self.max_tokens is not None else self.rank * 8
            if K.shape[1] > subsample_size:
                idx = torch.linspace(0, K.shape[1] - 1, subsample_size, dtype=torch.long, device=K.device)
                K = K[:, idx]
                V = V[:, idx]
            
            # SVD compression for K - keep top singular components
            U_k, S_k, Vh_k = torch.linalg.svd(K.float(), full_matrices=False)
            r_k = min(self.rank, min(S_k.shape[0], K.shape[1]))
            if r_k > 0:
                # Reconstruct with top-r components, keep all reconstructed columns
                K_compressed = (U_k[:, :r_k] @ torch.diag(S_k[:r_k]) @ Vh_k[:r_k, :]).to(dtype)
            else:
                K_compressed = K
            
            # SVD compression for V - keep top singular components
            U_v, S_v, Vh_v = torch.linalg.svd(V.float(), full_matrices=False)
            r_v = min(self.rank, min(S_v.shape[0], V.shape[1]))
            if r_v > 0:
                # Reconstruct with top-r components, keep all reconstructed columns
                V_compressed = (U_v[:, :r_v] @ torch.diag(S_v[:r_v]) @ Vh_v[:r_v, :]).to(dtype)
            else:
                V_compressed = V
                
        except Exception as e:
            # Fallback: simple subsampling if SVD fails
            step = max(1, K.shape[1] // self.rank)
            K_compressed = K[:, ::step]
            V_compressed = V[:, ::step]
        
        return K_compressed, V_compressed


class VQCompressor(nn.Module):
    """Vector Quantization compressor for KV tensors"""
    
    def __init__(self, num_clusters: int = 64):
        super().__init__()
        self.num_clusters = num_clusters
        
    def compress(self, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress using vector quantization (simplified k-means)
        
        Args:
            K: Key tensor [d_k, seq_len]
            V: Value tensor [d_v, seq_len]
            
        Returns:
            Compressed K, V tensors
        """
        if K.shape[1] == 0:
            return K, V
            
        seq_len = K.shape[1]
        n_clusters = min(self.num_clusters, seq_len)
        
        if n_clusters >= seq_len:
            return K, V
        
        # Simple uniform sampling as cluster representatives
        # In practice, would use proper k-means clustering
        indices = torch.linspace(0, seq_len - 1, n_clusters, dtype=torch.long, device=K.device)
        
        K_compressed = K[:, indices]
        V_compressed = V[:, indices]
        
        return K_compressed, V_compressed


class SimplePolicy(nn.Module):
    """Simple policy network for token importance scoring"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def score_candidates(self, candidates: torch.Tensor) -> torch.Tensor:
        """
        Score candidate tokens for compression priority
        
        Args:
            candidates: [seq_len, d_model] tensor of token representations
            
        Returns:
            scores: [seq_len] tensor of importance scores
        """
        if candidates.shape[0] == 0:
            return torch.tensor([], device=candidates.device)
            
        scores = self.scorer(candidates).squeeze(-1)  # [seq_len]
        return scores


class ADMSKVCache:
    """
    Adaptive Dual-Memory Streaming KV Cache
    
    Maintains three tiers of memory:
    1. Sink cache: First few tokens (attention sinks)
    2. Compressed cache: Compressed middle tokens  
    3. Recent cache: Recent tokens (full precision)
    """
    
    def __init__(self, config: ADMSConfig):
        self.config = config
        self.start_size = config.start_size
        self.recent_size = config.recent_size
        self.compressed_budget = config.compressed_budget
        self.k_seq_dim = config.k_seq_dim
        self.v_seq_dim = config.v_seq_dim
        
        # Dynamic sink sizing based on context length
        if config.enable_dynamic_sink:
            # Scale sink size: 1% of max context length (min: start_size)
            # For 2K: ~20 tokens, For 8K: ~80 tokens, For 32K: ~320 tokens
            self.dynamic_start_size = max(
                self.start_size,
                int(0.01 * config.max_seq_length)
            )
        else:
            self.dynamic_start_size = self.start_size
        
        # Initialize compressors
        self.compressors = {}
        self.policies = {}
        self._variance_ema: Dict[Tuple[int, int], float] = {}
        
        # Current sequence length
        self.current_length = 0
        self.last_compress_len = 0  # throttling checkpoint
        
        # Diagnostic stats
        self.stats = {
            "total_compressions": 0,
            "total_exact_kept": 0,
            "total_compressed_kept": 0,
            "total_middle_dropped": 0,
        }
        
        # Print config with flush to ensure it appears immediately
        print(f"\n{'='*60}")
        print(f"ADMS CACHE INITIALIZED")
        print(f"{'='*60}")
        print(f"  Static sink size:  {self.start_size}")
        if config.enable_dynamic_sink:
            print(f"  Dynamic sink size: {self.dynamic_start_size} (scaled for {config.max_seq_length} context)")
        print(f"  Recent window:     {self.recent_size}")
        print(f"  Compressed budget: {self.compressed_budget}")
        print(f"  Compressor type:   {config.compressor_type}")
        print(f"  Importance ratio:  {config.importance_ratio}")
        print(f"  Compression interval: {config.compression_interval}")
        effective_sink = self.dynamic_start_size if config.enable_dynamic_sink else self.start_size
        print(f"  Expected max cache size: {effective_sink + self.compressed_budget + self.recent_size}")
        print(f"{'='*60}\n", flush=True)
        
    def _get_compressor(self, layer: int, head: int):
        """Get compressor for specific layer/head"""
        key = (layer, head)
        if key not in self.compressors:
            if self.config.compressor_type == "low_rank":
                self.compressors[key] = LowRankCompressor(self.config.rank, max_tokens=self.config.svd_max_tokens)
            elif self.config.compressor_type == "vq":
                self.compressors[key] = VQCompressor(self.config.num_clusters)
            else:
                raise ValueError(f"Unknown compressor type: {self.config.compressor_type}")
        return self.compressors[key]
    
    def _get_policy(self, layer: int, head: int, d_k: int):
        """Get policy network for specific layer/head"""
        key = (layer, head)
        if key not in self.policies:
            self.policies[key] = SimplePolicy(d_k)
        return self.policies[key]

    def _normalize_scores(self, scores: torch.Tensor) -> torch.Tensor:
        if scores.numel() == 0 or not getattr(self.config, "importance_normalize", False):
            return scores
        if scores.isfinite().all():
            score_min = scores.min()
            score_range = scores.max() - score_min
            if score_range > 1e-6:
                scores = (scores - score_min) / score_range
            else:
                scores = scores - score_min
        return scores

    def _balanced_topk(
        self,
        scores: torch.Tensor,
        total_budget: int,
        segments_hint: Optional[int] = None,
    ) -> torch.Tensor:
        """Select indices while reserving part of the budget for timeline coverage."""

        if total_budget <= 0 or scores.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=scores.device)

        num_tokens = scores.shape[0]
        segments_cfg = max(1, int(getattr(self.config, "coverage_segments", 1)))
        segments = max(1, min(segments_cfg, num_tokens))
        if segments_hint is not None:
            segments = max(1, min(segments_hint, segments))

        coverage_priority = float(getattr(self.config, "coverage_priority", 0.0))
        coverage_priority = float(min(max(coverage_priority, 0.0), 1.0))
        coverage_budget = int(round(total_budget * coverage_priority))
        if coverage_priority > 0.0:
            coverage_budget = max(coverage_budget, min(total_budget, segments))
        coverage_budget = min(total_budget, coverage_budget)

        selected_indices: List[torch.Tensor] = []
        selected_mask = torch.zeros(num_tokens, dtype=torch.bool, device=scores.device)

        if coverage_budget > 0 and segments > 1:
            base_len = num_tokens // segments
            len_remainder = num_tokens % segments
            base_budget = coverage_budget // segments
            budget_remainder = coverage_budget % segments

            start = 0
            for seg in range(segments):
                seg_len = base_len + (1 if seg < len_remainder else 0)
                if seg_len <= 0:
                    continue
                end = min(num_tokens, start + seg_len)
                seg_scores = scores[start:end]
                seg_budget = base_budget + (1 if seg < budget_remainder else 0)
                seg_budget = min(seg_budget, seg_len, total_budget - sum(s.numel() for s in selected_indices))
                if seg_budget > 0:
                    topk = torch.topk(seg_scores, k=seg_budget, largest=True).indices + start
                    selected_indices.append(topk)
                    selected_mask[topk] = True
                start = end

        selected = torch.cat(selected_indices) if selected_indices else torch.empty(0, dtype=torch.long, device=scores.device)

        remaining_budget = total_budget - selected.numel()
        if remaining_budget > 0:
            available_indices = torch.nonzero(~selected_mask, as_tuple=False).squeeze(-1)
            if available_indices.numel() > 0:
                avail_scores = scores[available_indices]
                take = min(remaining_budget, available_indices.numel())
                topk = torch.topk(avail_scores, k=take, largest=True).indices
                selected = torch.cat([selected, available_indices[topk]]) if selected.numel() > 0 else available_indices[topk]

        if selected.numel() > total_budget:
            sel_scores = scores[selected]
            top = torch.topk(sel_scores, k=total_budget, largest=True).indices
            selected = selected[top]

        if selected.numel() == 0:
            return selected

        selected = torch.unique(selected, sorted=False)
        selected, _ = torch.sort(selected)
        return selected

    def _update_head_variance(self, key: Tuple[int, int], value: float) -> float:
        smoothing = float(getattr(self.config, "adaptive_variance_smoothing", 0.1))
        smoothing = min(max(smoothing, 0.0), 1.0)
        previous = self._variance_ema.get(key, value)
        updated = (1.0 - smoothing) * previous + smoothing * value
        self._variance_ema[key] = updated
        return updated

    def _scaled_budget(self, layer_idx: int, head_idx: int, middle_k: torch.Tensor, middle_v: torch.Tensor) -> int:
        base_budget = int(self.compressed_budget)
        if base_budget <= 0:
            return 0
        if not getattr(self.config, "use_adaptive_budget", False):
            return base_budget

        length_scale = middle_k.shape[0] / max(1, self.config.min_middle_size_for_compress)
        var_value = 0.0
        if middle_v.numel() > 0:
            token_norms = torch.norm(middle_v, dim=1)
            var_value = float(torch.var(token_norms, unbiased=False).item())
        ema = self._update_head_variance((layer_idx, head_idx), var_value)
        variance_scale = 1.0 + (ema / (ema + 1.0))  # in [1,2)
        scale = length_scale * variance_scale
        floor = float(getattr(self.config, "adaptive_budget_floor", 0.5))
        cap = float(getattr(self.config, "adaptive_budget_cap", 2.5))
        if cap < floor:
            cap = floor
        scale = max(floor, min(cap, scale))
        scaled_budget = int(round(base_budget * scale))
        return max(1, scaled_budget)

    def _importance_scores(self, keys: torch.Tensor, values: torch.Tensor, recent_keys: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute per-token importance scores based on configured metric."""
        if keys.shape[0] == 0:
            return torch.tensor([], device=keys.device)

        metric = getattr(self.config, "importance_metric", "value_norm")

        if metric == "attention" and recent_keys is not None:
            window_size = min(recent_keys.shape[0], getattr(self.config, "attention_window", 128))
            if window_size == 0:
                scores = torch.norm(values, dim=1)
            else:
                queries = recent_keys[-window_size:]
                sim = torch.matmul(keys.float(), queries.float().T)
                sim = sim / float(keys.shape[1] ** 0.5)
                att_scores = sim.max(dim=1).values
                val_scores = torch.norm(values, dim=1)
                weight = float(getattr(self.config, "attention_blend", 1.0))
                weight = max(0.0, min(1.0, weight))
                scores = weight * att_scores + (1.0 - weight) * val_scores
        elif metric == "key_norm":
            scores = torch.norm(keys, dim=1)
        elif metric == "mixed":
            key_scores = torch.norm(keys, dim=1)
            val_scores = torch.norm(values, dim=1)
            scores = 0.5 * val_scores + 0.5 * key_scores
        else:  # default value_norm
            scores = torch.norm(values, dim=1)

        return self._normalize_scores(scores)
        
    def _anchor_positions(self, original_positions: List[int], num_anchors: int) -> torch.Tensor:
        """
        Assign synthetic positions to compressed tokens
        
        Args:
            original_positions: List of original token positions
            num_anchors: Number of anchor positions needed
            
        Returns:
            anchor_positions: Tensor of anchor positions
        """
        if len(original_positions) == 0 or num_anchors == 0:
            return torch.tensor([], dtype=torch.long)
            
        start_pos = self.start_size
        end_pos = max(self.current_length - self.recent_size, start_pos + 1)
        
        if self.config.anchor_mode == "grid":
            if num_anchors <= 1:
                positions = torch.tensor([start_pos + (end_pos - start_pos) // 2], dtype=torch.long)
            else:
                positions = torch.linspace(start_pos, end_pos, num_anchors, dtype=torch.long)
        elif self.config.anchor_mode == "mean":
            # Group original positions and take means
            if num_anchors >= len(original_positions):
                positions = torch.tensor(original_positions, dtype=torch.long)
            else:
                group_size = max(1, len(original_positions) // num_anchors)
                positions = []
                for i in range(num_anchors):
                    start_idx = i * group_size
                    end_idx = min((i + 1) * group_size, len(original_positions))
                    if start_idx < len(original_positions):
                        group_mean = sum(original_positions[start_idx:end_idx]) // (end_idx - start_idx)
                        positions.append(group_mean)
                positions = torch.tensor(positions, dtype=torch.long)
        elif self.config.anchor_mode == "hybrid":
            positions = torch.linspace(start_pos, end_pos, num_anchors, dtype=torch.long)
            # Add small random jitter
            jitter = torch.randint(-2, 3, (num_anchors,))
            positions = positions + jitter
            positions = torch.clamp(positions, start_pos, end_pos)
        else:
            raise ValueError(f"Unknown anchor mode: {self.config.anchor_mode}")
            
        return positions
        
    def __call__(self, past_key_values):
        """
        Apply ADMS compression to past key-value states
        
        Args:
            past_key_values: List of (key, value) tuples for each layer
            
        Returns:
            Compressed past_key_values with ADMS applied
        """
        if past_key_values is None:
            return None

        # Fast path: if sequence length hasn't exceeded start+recent, skip any work
        try:
            first_k = past_key_values[0][0]
            # Expect [batch, heads, seq_len, dim]
            seq_len = first_k.shape[2]
            effective_sink = self.dynamic_start_size if self.config.enable_dynamic_sink else self.start_size
            if seq_len <= (effective_sink + self.recent_size):
                return past_key_values
            middle_len = seq_len - (effective_sink + self.recent_size)
            if middle_len <= 0:
                return past_key_values
            interval = max(1, self.config.compression_interval)
            threshold = max(0, getattr(self.config, "compression_middle_threshold", 0))
            debt_triggered = threshold > 0 and middle_len >= threshold
            # Throttle: only compress every N tokens unless middle debt is high
            if (seq_len - self.last_compress_len) < interval and not debt_triggered:
                # IMPORTANT: Still need to enforce budget even when skipping compression
                # Otherwise cache grows unbounded between compression steps
                if seq_len > (effective_sink + self.recent_size) and self.compressed_budget > 0:
                    # Quick eviction: keep start + budget + recent
                    max_cache_size = effective_sink + self.compressed_budget + self.recent_size
                    if first_k.shape[2] > max_cache_size:
                        # Simple truncation: keep first effective_sink, drop middle to budget, keep last recent_size
                        truncated_past = []
                        for k, v in past_key_values:
                            # k, v shape: [batch, heads, seq, dim]
                            keep_start_k = k[:, :, :effective_sink, :]
                            keep_start_v = v[:, :, :effective_sink, :]
                            keep_recent_k = k[:, :, -self.recent_size:, :]
                            keep_recent_v = v[:, :, -self.recent_size:, :]
                            # Sample middle evenly
                            middle_start = effective_sink
                            middle_end = k.shape[2] - self.recent_size
                            middle_indices = torch.linspace(
                                middle_start,
                                middle_end - 1,
                                min(self.compressed_budget, middle_end - middle_start),
                                dtype=torch.long,
                                device=k.device,
                            )
                            keep_middle_k = k[:, :, middle_indices, :]
                            keep_middle_v = v[:, :, middle_indices, :]

                            trunc_k = torch.cat([keep_start_k, keep_middle_k, keep_recent_k], dim=2)
                            trunc_v = torch.cat([keep_start_v, keep_middle_v, keep_recent_v], dim=2)
                            truncated_past.append((trunc_k, trunc_v))
                        return truncated_past
                return past_key_values
            # If we reach here, compression should run - fall through to main loop below
        except Exception:
            # If unexpected structure, fall back to standard path
            pass
            
        new_past_key_values = []
        
        for layer_idx, (key_states, value_states) in enumerate(past_key_values):
            batch_size, num_heads, seq_len, d_k = key_states.shape
            d_v = value_states.shape[-1]
            
            # Update current length
            self.current_length = seq_len
            
            # Use dynamic sink size for this layer
            effective_sink = self.dynamic_start_size if self.config.enable_dynamic_sink else self.start_size
            
            if seq_len <= effective_sink + self.recent_size:
                # No compression needed yet (should have been caught by fast path)
                new_past_key_values.append((key_states, value_states))
                continue
            
            # Apply ADMS compression
            new_key_states = []
            new_value_states = []
            
            for head_idx in range(num_heads):
                k = key_states[0, head_idx]  # [seq_len, d_k]
                v = value_states[0, head_idx]  # [seq_len, d_v]
                
                # Split into three tiers
                sink_k = k[:effective_sink]  # [effective_sink, d_k]
                sink_v = v[:effective_sink]  # [effective_sink, d_v]
                
                recent_start = seq_len - self.recent_size
                recent_k = k[recent_start:]  # [recent_size, d_k]
                recent_v = v[recent_start:]  # [recent_size, d_v]
                
                # Middle region for compression
                middle_k = k[effective_sink:recent_start]  # [middle_len, d_k]
                middle_v = v[effective_sink:recent_start]  # [middle_len, d_v]

                # Default: drop middle entirely (like StreamingLLM)
                middle_combined_k = middle_k[:0]  # Empty tensor
                middle_combined_v = middle_v[:0]  # Empty tensor

                if middle_k.shape[0] >= self.config.min_middle_size_for_compress and self.compressed_budget > 0:
                    middle_positions = torch.arange(
                        effective_sink,
                        recent_start,
                        device=middle_k.device,
                        dtype=torch.long,
                    )

                    # Reserve part of the budget for exact top tokens based on importance
                    topk_k = middle_k[:0]
                    topk_v = middle_v[:0]
                    topk_positions = middle_positions[:0]
                    remaining_positions = middle_positions
                    middle_k_for_compress = middle_k
                    middle_v_for_compress = middle_v

                    adapted_budget = self._scaled_budget(layer_idx, head_idx, middle_k, middle_v)
                    importance_ratio = max(0.0, getattr(self.config, "importance_ratio", 0.0))
                    importance_count = 0
                    if importance_ratio > 0.0 and middle_k.shape[0] > 0:
                        raw_count = int(adapted_budget * importance_ratio)
                        min_tokens = getattr(self.config, "min_importance_tokens", 0)
                        if min_tokens > 0:
                            raw_count = max(raw_count, min(adapted_budget, min_tokens))
                        importance_count = min(raw_count, adapted_budget, middle_k.shape[0])

                    if importance_count > 0:
                        scores = self._importance_scores(middle_k, middle_v, recent_k)
                        if scores.numel() > 0:
                            topk_indices = self._balanced_topk(scores, importance_count)
                            if topk_indices.numel() > 0:
                                importance_count = topk_indices.shape[0]
                                topk_k = middle_k[topk_indices]
                                topk_v = middle_v[topk_indices]
                                topk_positions = middle_positions[topk_indices]

                                mask = torch.ones(middle_k.shape[0], dtype=torch.bool, device=middle_k.device)
                                mask[topk_indices] = False
                                middle_k_for_compress = middle_k[mask]
                                middle_v_for_compress = middle_v[mask]
                                remaining_positions = middle_positions[mask]

                    importance_count = min(importance_count, adapted_budget)

                    remaining_budget = max(adapted_budget - importance_count, 0)

                    comp_k = middle_k[:0]
                    comp_v = middle_v[:0]
                    comp_positions = remaining_positions[:0]

                    if remaining_budget > 0 and middle_k_for_compress.shape[0] > 0:
                        # Transpose for compressor: [d_k, seq_len]
                        middle_k_t = middle_k_for_compress.transpose(0, 1)
                        middle_v_t = middle_v_for_compress.transpose(0, 1)

                        compressor = self._get_compressor(layer_idx, head_idx)
                        comp_k_t, comp_v_t = compressor.compress(middle_k_t, middle_v_t)

                        comp_k = comp_k_t.transpose(0, 1)
                        comp_v = comp_v_t.transpose(0, 1)

                        if comp_k.shape[0] > remaining_budget:
                            comp_scores = self._importance_scores(comp_k, comp_v, recent_k)
                            keep = self._balanced_topk(comp_scores, remaining_budget)
                            if keep.numel() > 0:
                                comp_k = comp_k[keep]
                                comp_v = comp_v[keep]
                                remaining_positions = remaining_positions[keep]
                        if comp_k.shape[0] > 0:
                            if comp_k.shape[0] == remaining_positions.shape[0]:
                                comp_positions = remaining_positions
                            else:
                                anchor_source = remaining_positions.detach().cpu().tolist()
                                comp_positions = self._anchor_positions(anchor_source, comp_k.shape[0]).to(comp_k.device)

                    # Merge exact and compressed selections and sort by synthetic/original positions
                    components_k = []
                    components_v = []
                    component_positions = []

                    if topk_k.shape[0] > 0:
                        components_k.append(topk_k)
                        components_v.append(topk_v)
                        component_positions.append(topk_positions)

                    if comp_k.shape[0] > 0:
                        components_k.append(comp_k)
                        components_v.append(comp_v)
                        component_positions.append(comp_positions.to(comp_k.device))

                    if component_positions:
                        stacked_positions = torch.cat(component_positions)
                        stacked_k = torch.cat(components_k, dim=0)
                        stacked_v = torch.cat(components_v, dim=0)
                        order = torch.argsort(stacked_positions)
                        middle_combined_k = stacked_k[order]
                        middle_combined_v = stacked_v[order]
                        
                        # Track stats
                        self.stats["total_compressions"] += 1
                        self.stats["total_exact_kept"] += topk_k.shape[0]
                        self.stats["total_compressed_kept"] += comp_k.shape[0]
                        self.stats["total_middle_dropped"] += middle_k.shape[0] - topk_k.shape[0] - comp_k.shape[0]
                    else:
                        # If no candidates survived selection, drop middle entirely (StreamingLLM behavior)
                        middle_combined_k = middle_k[:0]
                        middle_combined_v = middle_v[:0]
                        self.stats["total_middle_dropped"] += middle_k.shape[0]
                else:
                    # Middle too small or budget=0 -> drop middle entirely (acts like StreamingLLM)
                    pass  # Already initialized to empty above

                # Concatenate all tiers: sink + compressed + recent
                k_out = torch.cat([sink_k, middle_combined_k, recent_k], dim=0)
                v_out = torch.cat([sink_v, middle_combined_v, recent_v], dim=0)
                
                new_key_states.append(k_out)
                new_value_states.append(v_out)
            
            # Stack heads back together with correct dimension order:
            # new_key_states/new_value_states are lists of [new_seq, d_k] per head.
            # We need [batch=1, heads, seq, head_dim]. Stack along dim=0 to make [heads, seq, dim].
            layer_k = torch.stack(new_key_states, dim=0).unsqueeze(0)  # [1, num_heads, new_seq, d_k]
            layer_v = torch.stack(new_value_states, dim=0).unsqueeze(0)  # [1, num_heads, new_seq, d_v]
            
            new_past_key_values.append((layer_k, layer_v))
        # Update last compressed length after successful compression
        self.last_compress_len = seq_len
        
        # CRITICAL: Enforce max cache size even after compression
        # In case adaptive budgets or other logic caused cache to exceed limit
        effective_sink = self.dynamic_start_size if self.config.enable_dynamic_sink else self.start_size
        max_cache_size = effective_sink + self.compressed_budget + self.recent_size
        actual_cache_size = new_past_key_values[0][0].shape[2]
        
        if actual_cache_size > max_cache_size:
            print(f"[ADMS Post-compress @ {seq_len}] Cache {actual_cache_size} > {max_cache_size}, enforcing limit", flush=True)
            # Truncate: keep start, sample middle to budget, keep recent
            final_past = []
            for k, v in new_past_key_values:
                keep_start_k = k[:, :, :effective_sink, :]
                keep_start_v = v[:, :, :effective_sink, :]
                keep_recent_k = k[:, :, -self.recent_size:, :]
                keep_recent_v = v[:, :, -self.recent_size:, :]
                
                middle_start = effective_sink
                middle_end = k.shape[2] - self.recent_size
                middle_size = middle_end - middle_start
                
                if middle_size > self.compressed_budget:
                    # Sample middle evenly to fit budget
                    middle_indices = torch.linspace(
                        middle_start, middle_end - 1,
                        self.compressed_budget,
                        dtype=torch.long,
                        device=k.device
                    )
                    keep_middle_k = k[:, :, middle_indices, :]
                    keep_middle_v = v[:, :, middle_indices, :]
                else:
                    keep_middle_k = k[:, :, middle_start:middle_end, :]
                    keep_middle_v = v[:, :, middle_start:middle_end, :]
                
                final_k = torch.cat([keep_start_k, keep_middle_k, keep_recent_k], dim=2)
                final_v = torch.cat([keep_start_v, keep_middle_v, keep_recent_v], dim=2)
                final_past.append((final_k, final_v))
            
            new_past_key_values = final_past
            actual_cache_size = new_past_key_values[0][0].shape[2]
            print(f"[ADMS Post-compress @ {seq_len}] Enforced to {actual_cache_size}", flush=True)
        
        # Print stats more frequently for debugging (every 512 tokens or at key milestones)
        log_interval = 512
        if (seq_len % log_interval == 0 or seq_len in [256, 512, 1024, 2048, 4096, 8192, 15000]) and self.stats["total_compressions"] > 0:
            avg_exact = self.stats["total_exact_kept"] / self.stats["total_compressions"]
            avg_comp = self.stats["total_compressed_kept"] / self.stats["total_compressions"]
            avg_dropped = self.stats["total_middle_dropped"] / self.stats["total_compressions"]
            
            # Calculate actual cache size
            actual_cache_size = new_past_key_values[0][0].shape[2]  # [batch, heads, seq, dim]
            effective_sink = self.dynamic_start_size if self.config.enable_dynamic_sink else self.start_size
            expected_max = effective_sink + self.compressed_budget + self.recent_size
            
            print(f"[ADMS Stats @ {seq_len}] Avg per head: exact={avg_exact:.1f}, compressed={avg_comp:.1f}, dropped={avg_dropped:.1f}", flush=True)
            print(f"[ADMS Cache @ {seq_len}] Actual size: {actual_cache_size}, Expected max: {expected_max}, Ratio: {actual_cache_size/seq_len:.2%}", flush=True)
            
            # WARNING if cache is unexpectedly large
            if actual_cache_size > expected_max * 1.5:
                print(f"[ADMS WARNING] Cache size {actual_cache_size} exceeds expected {expected_max} by >50%!", flush=True)
        
        return new_past_key_values