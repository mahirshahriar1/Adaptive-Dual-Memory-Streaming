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
    enable_dual_fidelity: bool = False
    sketch_budget: int = 0
    sketch_reduction: str = "mean"
    enable_residual_replay: bool = False
    replay_budget: int = 0
    energy_replay_threshold: float = 0.9
    enable_position_calibration: bool = False
    calibration_window: int = 512
    calibration_regularization: float = 0.1
    enable_adaptive_controller: bool = False
    controller_gain: float = 0.3
    controller_energy_floor: float = 0.8
    controller_energy_ceiling: float = 0.97
    controller_group_size: int = 1


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
        
        # Dual-Fidelity: sketch bank storage per head
        self.sketch_bank: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        self.sketch_energies: Dict[Tuple[int, int], torch.Tensor] = {}  # residual energies per sketch token
        
        # Residual Replay: original token storage for replay
        self.replay_bank: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        self.replay_energies: Dict[Tuple[int, int], float] = {}  # reconstruction energy per head
        
        # Position Calibration: anchor state tracking
        self.calibration_anchors: Dict[Tuple[int, int], torch.Tensor] = {}  # recent anchor positions
        self.calibration_offsets: Dict[Tuple[int, int], torch.Tensor] = {}  # learned position offsets
        
        # Adaptive Controller: per-head budget state
        self.controller_budgets: Dict[Tuple[int, int], int] = {}  # current budget per head
        self.controller_energy_ema: Dict[Tuple[int, int], float] = {}  # reconstruction energy EMA
        self.controller_attention_mass: Dict[Tuple[int, int], float] = {}  # attention mass EMA
        
        # Current sequence length
        self.current_length = 0
        self.last_compress_len = 0  # throttling checkpoint
        
        # Diagnostic stats
        self.stats = {
            "total_compressions": 0,
            "total_exact_kept": 0,
            "total_compressed_kept": 0,
            "total_middle_dropped": 0,
            "total_sketch_tokens": 0,
            "total_replay_tokens": 0,
            "total_sketch_promotions": 0,
            "total_replay_triggers": 0,
            "total_calibrations": 0,
            "total_budget_adjustments": 0,
        }

        self.middle_allocation = self.compressed_budget
        if getattr(self.config, "enable_dual_fidelity", False):
            self.middle_allocation += int(getattr(self.config, "sketch_budget", 0))
        if getattr(self.config, "enable_residual_replay", False):
            self.middle_allocation += int(getattr(self.config, "replay_budget", 0))
        self.middle_allocation = max(0, int(self.middle_allocation))
        
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
        print(f"  Expected max cache size: {effective_sink + self.middle_allocation + self.recent_size}")
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

    def _update_controller_state(self, layer_idx: int, head_idx: int, 
                                   reconstruction_energy: float, attention_mass: float):
        """
        Update adaptive controller state with reconstruction energy and attention mass
        
        Args:
            layer_idx, head_idx: Head identifier
            reconstruction_energy: Energy ratio (lower = need more budget)
            attention_mass: Attention weight sum for this head
        """
        if not self.config.enable_adaptive_controller:
            return
        
        key = (layer_idx, head_idx)
        gain = self.config.controller_gain
        
        # Update energy EMA
        if key not in self.controller_energy_ema:
            self.controller_energy_ema[key] = reconstruction_energy
        else:
            self.controller_energy_ema[key] = (1 - gain) * self.controller_energy_ema[key] + gain * reconstruction_energy
        
        # Update attention mass EMA
        if key not in self.controller_attention_mass:
            self.controller_attention_mass[key] = attention_mass
        else:
            self.controller_attention_mass[key] = (1 - gain) * self.controller_attention_mass[key] + gain * attention_mass
    
    def _get_adaptive_budget(self, layer_idx: int, head_idx: int, base_budget: int) -> int:
        """
        Get adaptive budget based on reconstruction energy and attention mass
        
        Args:
            layer_idx, head_idx: Head identifier
            base_budget: Base budget allocation
            
        Returns:
            Adjusted budget
        """
        if not self.config.enable_adaptive_controller:
            return base_budget
        
        # Head grouping: share state across groups
        group_size = max(1, self.config.controller_group_size)
        group_id = (layer_idx, head_idx // group_size)
        
        # Collect energy and attention from heads in this group
        group_energies = []
        group_attention = []
        
        for h in range((head_idx // group_size) * group_size, 
                       (head_idx // group_size + 1) * group_size):
            head_key = (layer_idx, h)
            if head_key in self.controller_energy_ema:
                group_energies.append(self.controller_energy_ema[head_key])
            if head_key in self.controller_attention_mass:
                group_attention.append(self.controller_attention_mass[head_key])
        
        if not group_energies:
            # No history yet
            return base_budget
        
        # Use group average for shared state
        avg_energy = sum(group_energies) / len(group_energies)
        avg_attention = sum(group_attention) / len(group_attention) if group_attention else 1.0
        
        # Decision logic
        floor = self.config.controller_energy_floor
        ceiling = self.config.controller_energy_ceiling
        
        if avg_energy < floor:
            # Low energy = poor reconstruction, increase budget
            scale = 1.0 + (floor - avg_energy) * 2.0  # Up to 2x more
        elif avg_energy > ceiling:
            # High energy = good reconstruction, can reduce budget
            scale = 0.8  # Reduce by 20%
        else:
            # In acceptable range
            scale = 1.0
        
        # Adjust by attention mass (high attention = more important head)
        attention_scale = 0.8 + 0.4 * min(1.0, avg_attention)  # Range [0.8, 1.2]
        
        adjusted_budget = int(base_budget * scale * attention_scale)
        adjusted_budget = max(1, min(adjusted_budget, base_budget * 2))  # Clamp to [1, 2*base]
        
        # Cache the adjusted budget
        key = (layer_idx, head_idx)
        self.controller_budgets[key] = adjusted_budget
        self.stats["total_budget_adjustments"] += 1
        
        return adjusted_budget

    def _scaled_budget(self, layer_idx: int, head_idx: int, middle_k: torch.Tensor, middle_v: torch.Tensor) -> int:
        base_budget = int(self.compressed_budget)
        if base_budget <= 0:
            return 0
        
        # If adaptive controller is enabled, use it instead of variance-based scaling
        if self.config.enable_adaptive_controller:
            # Check if we have controller state
            key = (layer_idx, head_idx)
            if key in self.controller_budgets:
                return self.controller_budgets[key]
            # Otherwise fall through to base budget
            return base_budget
        
        # Legacy variance-based scaling (kept for backward compatibility)
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

    def _apply_sketch_reduction(self, tokens_k: torch.Tensor, tokens_v: torch.Tensor, reduction: str = "mean") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply sketch reduction to overflow tokens
        
        Args:
            tokens_k: Key tokens to sketch [seq_len, d_k]
            tokens_v: Value tokens to sketch [seq_len, d_v]
            reduction: 'mean', 'sum', or 'first'
            
        Returns:
            Sketched single token (k, v)
        """
        if tokens_k.shape[0] == 0:
            return tokens_k[:0], tokens_v[:0]
        
        if reduction == "mean":
            sketch_k = tokens_k.mean(dim=0, keepdim=True)  # [1, d_k]
            sketch_v = tokens_v.mean(dim=0, keepdim=True)  # [1, d_v]
        elif reduction == "sum":
            sketch_k = tokens_k.sum(dim=0, keepdim=True)
            sketch_v = tokens_v.sum(dim=0, keepdim=True)
        elif reduction == "first":
            sketch_k = tokens_k[:1]
            sketch_v = tokens_v[:1]
        else:
            # Fallback to mean
            sketch_k = tokens_k.mean(dim=0, keepdim=True)
            sketch_v = tokens_v.mean(dim=0, keepdim=True)
        
        return sketch_k, sketch_v
    
    def _compute_residual_energy(self, original_k: torch.Tensor, original_v: torch.Tensor, 
                                  approx_k: torch.Tensor, approx_v: torch.Tensor) -> float:
        """
        Compute residual energy between original and approximation
        
        Args:
            original_k, original_v: Original tensors
            approx_k, approx_v: Approximated tensors
            
        Returns:
            Energy ratio (0=perfect reconstruction, 1=complete loss)
        """
        if original_k.shape[0] == 0 or approx_k.shape[0] == 0:
            return 0.0
        
        # Compute reconstruction error
        if original_k.shape == approx_k.shape:
            k_error = torch.norm(original_k - approx_k).item()
            v_error = torch.norm(original_v - approx_v).item()
        else:
            # Different sizes - measure coverage loss
            k_error = torch.norm(original_k).item()
            v_error = torch.norm(original_v).item()
        
        k_total = torch.norm(original_k).item() + 1e-8
        v_total = torch.norm(original_v).item() + 1e-8
        
        # Energy ratio: higher = more loss
        energy = (k_error + v_error) / (k_total + v_total)
        return min(1.0, max(0.0, energy))
    
    def _update_sketch_bank(self, layer_idx: int, head_idx: int, overflow_k: torch.Tensor, 
                             overflow_v: torch.Tensor, overflow_positions: torch.Tensor):
        """
        Update sketch bank with overflow tokens
        
        Args:
            layer_idx, head_idx: Head identifier
            overflow_k, overflow_v: Overflow tokens
            overflow_positions: Original positions of overflow tokens
        """
        if not self.config.enable_dual_fidelity:
            return
        
        if overflow_k.shape[0] == 0:
            return
        
        key = (layer_idx, head_idx)
        reduction = getattr(self.config, "sketch_reduction", "mean")
        
        # Create sketch from overflow
        sketch_k, sketch_v = self._apply_sketch_reduction(overflow_k, overflow_v, reduction)
        
        # Compute residual energy for this sketch
        energy = self._compute_residual_energy(overflow_k, overflow_v, sketch_k, sketch_v)
        
        # Store in sketch bank
        if key not in self.sketch_bank:
            self.sketch_bank[key] = {"keys": [], "values": [], "positions": [], "original_k": [], "original_v": []}
            self.sketch_energies[key] = []
        
        self.sketch_bank[key]["keys"].append(sketch_k)
        self.sketch_bank[key]["values"].append(sketch_v)
        # Use mean position for sketch
        sketch_position = overflow_positions.float().mean().long()
        self.sketch_bank[key]["positions"].append(sketch_position)
        
        # Store original tokens for potential promotion
        self.sketch_bank[key]["original_k"].append(overflow_k)
        self.sketch_bank[key]["original_v"].append(overflow_v)
        self.sketch_energies[key].append(energy)
        
        self.stats["total_sketch_tokens"] += 1
    
    def _promote_sketches_to_precision(self, layer_idx: int, head_idx: int, promotion_budget: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Promote high-energy sketches back to precision bank
        
        Args:
            layer_idx, head_idx: Head identifier  
            promotion_budget: Number of sketches to promote
            
        Returns:
            Promoted (keys, values, positions) or empty tensors
        """
        if not self.config.enable_dual_fidelity or promotion_budget <= 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        key = (layer_idx, head_idx)
        if key not in self.sketch_bank or not self.sketch_energies[key]:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        # Sort sketches by energy (highest first)
        energies = torch.tensor(self.sketch_energies[key])
        if energies.numel() == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        # Select top-k high-energy sketches to promote
        num_promote = min(promotion_budget, len(self.sketch_energies[key]))
        if num_promote == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        top_indices = torch.topk(energies, k=num_promote, largest=True).indices
        
        promoted_k = []
        promoted_v = []
        promoted_pos = []
        
        # Promote selected sketches (use original tokens, not sketch)
        for idx in sorted(top_indices.tolist(), reverse=True):  # Remove from back to front
            original_k = self.sketch_bank[key]["original_k"].pop(idx)
            original_v = self.sketch_bank[key]["original_v"].pop(idx)
            position = self.sketch_bank[key]["positions"].pop(idx)
            
            # Remove from sketch storage
            self.sketch_bank[key]["keys"].pop(idx)
            self.sketch_bank[key]["values"].pop(idx)
            self.sketch_energies[key].pop(idx)
            
            # Subsample if original is too large
            max_tokens_per_sketch = 8  # Limit promoted tokens per sketch
            if original_k.shape[0] > max_tokens_per_sketch:
                indices = torch.linspace(0, original_k.shape[0] - 1, max_tokens_per_sketch, dtype=torch.long, device=original_k.device)
                original_k = original_k[indices]
                original_v = original_v[indices]
            
            promoted_k.append(original_k)
            promoted_v.append(original_v)
            promoted_pos.append(position.repeat(original_k.shape[0]))
            
            self.stats["total_sketch_promotions"] += 1
        
        if not promoted_k:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        # Concatenate all promoted tokens
        promoted_k = torch.cat(promoted_k, dim=0)
        promoted_v = torch.cat(promoted_v, dim=0)
        promoted_pos = torch.cat(promoted_pos, dim=0)
        
        return promoted_k, promoted_v, promoted_pos
    
    def _get_sketch_tokens(self, layer_idx: int, head_idx: int, budget: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve sketch tokens for attention, up to budget
        
        Args:
            layer_idx, head_idx: Head identifier
            budget: Maximum number of sketch tokens to return
            
        Returns:
            (keys, values, positions) tensors
        """
        if not self.config.enable_dual_fidelity or budget <= 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        key = (layer_idx, head_idx)
        if key not in self.sketch_bank:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        num_sketches = len(self.sketch_bank[key]["keys"])
        if num_sketches == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        # Return up to budget sketches
        num_return = min(budget, num_sketches)
        
        keys = torch.cat(self.sketch_bank[key]["keys"][:num_return], dim=0)
        values = torch.cat(self.sketch_bank[key]["values"][:num_return], dim=0)
        positions = torch.stack(self.sketch_bank[key]["positions"][:num_return])
        
        return keys, values, positions
    
    def _update_replay_bank(self, layer_idx: int, head_idx: int, middle_k: torch.Tensor, 
                             middle_v: torch.Tensor, middle_positions: torch.Tensor):
        """
        Store original tokens in replay bank for potential replay
        
        Args:
            layer_idx, head_idx: Head identifier
            middle_k, middle_v: Middle region tokens
            middle_positions: Original positions
        """
        if not self.config.enable_residual_replay:
            return
        
        key = (layer_idx, head_idx)
        
        # Store limited number of recent middle tokens for replay
        max_replay_storage = self.config.replay_budget * 4  # Store 4x budget for selection
        
        if key not in self.replay_bank:
            self.replay_bank[key] = {"keys": [], "values": [], "positions": []}
        
        # Add new tokens
        self.replay_bank[key]["keys"].append(middle_k)
        self.replay_bank[key]["values"].append(middle_v)
        self.replay_bank[key]["positions"].append(middle_positions)
        
        # Limit storage size (FIFO)
        while len(self.replay_bank[key]["keys"]) > max_replay_storage:
            self.replay_bank[key]["keys"].pop(0)
            self.replay_bank[key]["values"].pop(0)
            self.replay_bank[key]["positions"].pop(0)
    
    def _compute_spectral_energy(self, approx_k: torch.Tensor, approx_v: torch.Tensor,
                                  original_k: torch.Tensor, original_v: torch.Tensor) -> float:
        """
        Compute spectral energy ratio to detect reconstruction quality
        
        Args:
            approx_k, approx_v: Compressed approximation
            original_k, original_v: Original tokens
            
        Returns:
            Energy ratio (lower = more loss, triggers replay)
        """
        if approx_k.shape[0] == 0 or original_k.shape[0] == 0:
            return 1.0
        
        try:
            # Compute spectral energy (top singular values)
            _, S_approx, _ = torch.linalg.svd(approx_k.float().T, full_matrices=False)
            _, S_orig, _ = torch.linalg.svd(original_k.float().T, full_matrices=False)
            
            # Take top-k singular values
            k = min(8, min(S_approx.shape[0], S_orig.shape[0]))
            if k == 0:
                return 1.0
            
            energy_approx = torch.sum(S_approx[:k] ** 2).item()
            energy_orig = torch.sum(S_orig[:k] ** 2).item()
            
            # Ratio: high = good preservation, low = trigger replay
            ratio = energy_approx / (energy_orig + 1e-8)
            return min(1.0, max(0.0, ratio))
            
        except Exception:
            # Fallback: Frobenius norm ratio
            norm_approx = torch.norm(approx_k).item()
            norm_orig = torch.norm(original_k).item()
            return norm_approx / (norm_orig + 1e-8)
    
    def _trigger_replay(self, layer_idx: int, head_idx: int, energy_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Trigger replay of original tokens if energy drops below threshold
        
        Args:
            layer_idx, head_idx: Head identifier
            energy_ratio: Current spectral energy ratio
            
        Returns:
            Replayed (keys, values, positions) or empty tensors
        """
        if not self.config.enable_residual_replay:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        threshold = self.config.energy_replay_threshold
        if energy_ratio >= threshold:
            # Energy is good, no replay needed
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        key = (layer_idx, head_idx)
        if key not in self.replay_bank or not self.replay_bank[key]["keys"]:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        # Replay budget: more tokens when energy is worse
        urgency = (threshold - energy_ratio) / threshold  # 0 to 1
        replay_count = max(1, int(self.config.replay_budget * urgency))
        
        # Select most recent tokens from replay bank
        replay_k = []
        replay_v = []
        replay_pos = []
        
        tokens_to_replay = min(replay_count, len(self.replay_bank[key]["keys"]))
        
        for i in range(-tokens_to_replay, 0):  # Take from end (most recent)
            replay_k.append(self.replay_bank[key]["keys"][i])
            replay_v.append(self.replay_bank[key]["values"][i])
            replay_pos.append(self.replay_bank[key]["positions"][i])
        
        if not replay_k:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        # Concatenate and subsample if needed
        replay_k = torch.cat(replay_k, dim=0)
        replay_v = torch.cat(replay_v, dim=0)
        replay_pos = torch.cat(replay_pos, dim=0)
        
        if replay_k.shape[0] > self.config.replay_budget:
            # Subsample to budget
            indices = torch.linspace(0, replay_k.shape[0] - 1, self.config.replay_budget, dtype=torch.long, device=replay_k.device)
            replay_k = replay_k[indices]
            replay_v = replay_v[indices]
            replay_pos = replay_pos[indices]
        
        self.stats["total_replay_triggers"] += 1
        self.stats["total_replay_tokens"] += replay_k.shape[0]
        
        return replay_k, replay_v, replay_pos
    
    def _update_calibration_anchors(self, layer_idx: int, head_idx: int, recent_positions: torch.Tensor):
        """
        Update sliding anchor window for position calibration
        
        Args:
            layer_idx, head_idx: Head identifier
            recent_positions: Recent token positions
        """
        if not self.config.enable_position_calibration:
            return
        
        key = (layer_idx, head_idx)
        window_size = self.config.calibration_window
        
        # Store recent positions for calibration
        if key not in self.calibration_anchors:
            self.calibration_anchors[key] = recent_positions[-window_size:]
        else:
            # Concatenate and keep last window_size
            combined = torch.cat([self.calibration_anchors[key], recent_positions])
            self.calibration_anchors[key] = combined[-window_size:]
    
    def _calibrate_synthetic_positions(self, layer_idx: int, head_idx: int, 
                                        synthetic_positions: torch.Tensor,
                                        compressed_keys: torch.Tensor) -> torch.Tensor:
        """
        Calibrate synthetic positions using lightweight linear solve
        
        Args:
            layer_idx, head_idx: Head identifier
            synthetic_positions: Initial synthetic position assignments [num_tokens]
            compressed_keys: Compressed key vectors [num_tokens, d_k]
            
        Returns:
            Calibrated positions with drift correction
        """
        if not self.config.enable_position_calibration:
            return synthetic_positions
        
        if synthetic_positions.numel() == 0 or compressed_keys.shape[0] == 0:
            return synthetic_positions
        
        key = (layer_idx, head_idx)
        
        # Need anchor positions for calibration
        if key not in self.calibration_anchors or self.calibration_anchors[key].numel() < 10:
            # Not enough anchors yet, return uncalibrated
            return synthetic_positions
        
        try:
            # Compute expected vs actual position drift
            anchor_positions = self.calibration_anchors[key]
            
            # Simple linear model: offset = a * position + b
            # Solve least squares to find drift correction
            
            # Use recent anchors as ground truth
            X = anchor_positions.float().unsqueeze(1)  # [n, 1]
            X = torch.cat([X, torch.ones_like(X)], dim=1)  # [n, 2] with bias term
            
            # Target: positions should align with sequence order
            y = torch.arange(anchor_positions.shape[0], dtype=torch.float32, device=X.device).unsqueeze(1)
            
            # Solve: X @ theta = y  =>  theta = (X^T X)^{-1} X^T y
            XtX = X.T @ X
            Xty = X.T @ y
            
            # Add regularization for stability
            reg = self.config.calibration_regularization
            XtX = XtX + reg * torch.eye(2, device=XtX.device)
            
            theta = torch.linalg.solve(XtX, Xty)  # [2, 1]
            
            # Apply learned offset to synthetic positions
            synth_X = synthetic_positions.float().unsqueeze(1)
            synth_X = torch.cat([synth_X, torch.ones_like(synth_X)], dim=1)  # [m, 2]
            
            calibrated = (synth_X @ theta).squeeze(1)  # [m]
            
            # Store learned offset for tracking
            self.calibration_offsets[key] = theta
            
            # Blend calibrated with original (avoid over-correction)
            blend = 0.7  # Trust calibration but keep some original structure
            calibrated = blend * calibrated + (1 - blend) * synthetic_positions.float()
            
            # Ensure monotonic and within reasonable range
            calibrated = calibrated.long()
            calibrated = torch.clamp(calibrated, 
                                      min=self.start_size, 
                                      max=self.current_length - self.recent_size)
            
            self.stats["total_calibrations"] += 1
            
            return calibrated
            
        except Exception as e:
            # Calibration failed, return original
            return synthetic_positions
        
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

    def _coerce_head_sequence_length(
        self,
        key_seq: torch.Tensor,
        value_seq: torch.Tensor,
        effective_sink: int,
        target_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Trim or subsample middle tokens so a head sequence matches the target length."""

        seq_len = key_seq.shape[0]
        if seq_len <= target_length:
            return key_seq, value_seq, 0

        recent_len = min(self.recent_size, max(0, seq_len - effective_sink))
        target_middle = max(0, target_length - (effective_sink + recent_len))

        middle_start = effective_sink
        middle_end = seq_len - recent_len

        if target_middle <= 0 or middle_end <= middle_start:
            trimmed_keys = torch.cat([key_seq[:effective_sink], key_seq[-recent_len:]], dim=0)
            trimmed_values = torch.cat([value_seq[:effective_sink], value_seq[-recent_len:]], dim=0)
        else:
            middle_indices = torch.linspace(
                float(middle_start),
                float(max(middle_start, middle_end - 1)),
                steps=target_middle,
                device=key_seq.device,
                dtype=torch.float64,
            ).round().long()
            middle_indices = torch.clamp(middle_indices, middle_start, max(middle_start, middle_end - 1))
            if middle_indices.numel() < target_middle:
                pad_needed = target_middle - middle_indices.numel()
                pad_idx = middle_indices[-1:].repeat(pad_needed) if middle_indices.numel() > 0 else torch.full(
                    (pad_needed,), middle_start, dtype=torch.long, device=key_seq.device
                )
                middle_indices = torch.cat([middle_indices, pad_idx])
            elif middle_indices.numel() > target_middle:
                middle_indices = middle_indices[:target_middle]

            middle_keys = key_seq[middle_indices]
            middle_values = value_seq[middle_indices]

            trimmed_keys = torch.cat([key_seq[:effective_sink], middle_keys, key_seq[-recent_len:]], dim=0)
            trimmed_values = torch.cat([value_seq[:effective_sink], middle_values, value_seq[-recent_len:]], dim=0)

        if trimmed_keys.shape[0] > target_length:
            trimmed_keys = trimmed_keys[:target_length]
            trimmed_values = trimmed_values[:target_length]

        dropped = seq_len - trimmed_keys.shape[0]
        return trimmed_keys, trimmed_values, max(0, dropped)
        
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
                if seq_len > (effective_sink + self.recent_size) and self.middle_allocation > 0:
                    # Quick eviction: keep start + budget + recent
                    max_cache_size = effective_sink + self.middle_allocation + self.recent_size
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
                            mid_steps = min(self.middle_allocation, max(0, middle_end - middle_start))
                            if mid_steps > 0:
                                middle_indices = torch.linspace(
                                    middle_start,
                                    middle_end - 1,
                                    mid_steps,
                                    dtype=torch.long,
                                    device=k.device,
                                )
                                keep_middle_k = k[:, :, middle_indices, :]
                                keep_middle_v = v[:, :, middle_indices, :]
                            else:
                                keep_middle_k = k[:, :, middle_start:middle_start, :]
                                keep_middle_v = v[:, :, middle_start:middle_start, :]

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

                if middle_k.shape[0] >= self.config.min_middle_size_for_compress and self.middle_allocation > 0:
                    middle_positions = torch.arange(
                        effective_sink,
                        recent_start,
                        device=middle_k.device,
                        dtype=torch.long,
                    )

                    # Store original tokens in replay bank
                    self._update_replay_bank(layer_idx, head_idx, middle_k, middle_v, middle_positions)
                    
                    # Update calibration anchors with recent positions
                    recent_positions = torch.arange(recent_start, seq_len, device=recent_k.device, dtype=torch.long)
                    self._update_calibration_anchors(layer_idx, head_idx, recent_positions)

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
                    original_k_for_energy = middle_k_for_compress  # Save for energy calculation
                    
                    # Initialize replay variables (always, to avoid UnboundLocalError)
                    replay_k = torch.tensor([], dtype=middle_k.dtype, device=middle_k.device)
                    replay_v = torch.tensor([], dtype=middle_v.dtype, device=middle_v.device)
                    replay_pos = torch.tensor([], dtype=middle_positions.dtype, device=middle_positions.device)

                    if remaining_budget > 0 and middle_k_for_compress.shape[0] > 0:
                        # Transpose for compressor: [d_k, seq_len]
                        middle_k_t = middle_k_for_compress.transpose(0, 1)
                        middle_v_t = middle_v_for_compress.transpose(0, 1)

                        compressor = self._get_compressor(layer_idx, head_idx)
                        comp_k_t, comp_v_t = compressor.compress(middle_k_t, middle_v_t)

                        comp_k = comp_k_t.transpose(0, 1)
                        comp_v = comp_v_t.transpose(0, 1)
                        
                        # Compute spectral energy for replay triggering
                        if self.config.enable_residual_replay:
                            energy_ratio = self._compute_spectral_energy(comp_k, comp_v, middle_k_for_compress, middle_v_for_compress)
                            replay_k, replay_v, replay_pos = self._trigger_replay(layer_idx, head_idx, energy_ratio)
                        else:
                            energy_ratio = 1.0
                        
                        # Compute attention mass (approximation using value norms)
                        attention_mass = torch.norm(middle_v, dim=1).sum().item() / max(1.0, middle_v.numel())
                        
                        # Update adaptive controller state
                        self._update_controller_state(layer_idx, head_idx, energy_ratio, attention_mass)

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
                                
                            # Apply position calibration to compressed positions
                            if self.config.enable_position_calibration and comp_positions.numel() > 0:
                                comp_positions = self._calibrate_synthetic_positions(layer_idx, head_idx, comp_positions, comp_k)
                    
                    # Handle sketch bank and overflow for dual-fidelity
                    sketch_k = torch.tensor([])
                    sketch_v = torch.tensor([])
                    sketch_positions = torch.tensor([])
                    promoted_k = torch.tensor([])
                    promoted_v = torch.tensor([])
                    promoted_pos = torch.tensor([])
                    
                    if self.config.enable_dual_fidelity:
                        sketch_budget = getattr(self.config, "sketch_budget", 0)
                        
                        # Promote high-energy sketches to precision
                        promotion_count = max(1, sketch_budget // 4) if sketch_budget > 0 else 0
                        if promotion_count > 0:
                            promoted_k, promoted_v, promoted_pos = self._promote_sketches_to_precision(
                                layer_idx, head_idx, promotion_count
                            )
                        
                        # Compute overflow (tokens that didn't make it to precision bank)
                        total_precision = topk_k.shape[0] + comp_k.shape[0] + promoted_k.shape[0]
                        if middle_k.shape[0] > total_precision:
                            # Create overflow from remaining tokens
                            overflow_mask = torch.ones(middle_k.shape[0], dtype=torch.bool, device=middle_k.device)
                            if topk_k.shape[0] > 0:
                                # Mark topk as not overflow
                                topk_orig_indices = topk_indices if importance_count > 0 else torch.tensor([], dtype=torch.long, device=middle_k.device)
                                if topk_orig_indices.numel() > 0:
                                    overflow_mask[topk_orig_indices] = False
                            
                            # Collect overflow tokens
                            overflow_indices = torch.nonzero(overflow_mask, as_tuple=False).squeeze(-1)
                            if overflow_indices.numel() > sketch_budget * 2:  # Limit overflow storage
                                overflow_indices = overflow_indices[:sketch_budget * 2]
                            
                            if overflow_indices.numel() > 0:
                                overflow_k = middle_k[overflow_indices]
                                overflow_v = middle_v[overflow_indices]
                                overflow_pos = middle_positions[overflow_indices]
                                self._update_sketch_bank(layer_idx, head_idx, overflow_k, overflow_v, overflow_pos)
                        
                        # Get sketch tokens for attention
                        if sketch_budget > 0:
                            sketch_k, sketch_v, sketch_positions = self._get_sketch_tokens(layer_idx, head_idx, sketch_budget)

                    # Merge exact, compressed, promoted, replayed, and sketch tokens
                    components_k = []
                    components_v = []
                    component_positions = []

                    if topk_k.shape[0] > 0:
                        components_k.append(topk_k)
                        components_v.append(topk_v)
                        component_positions.append(topk_positions)
                    
                    if promoted_k.shape[0] > 0:
                        components_k.append(promoted_k)
                        components_v.append(promoted_v)
                        component_positions.append(promoted_pos)

                    if comp_k.shape[0] > 0:
                        components_k.append(comp_k)
                        components_v.append(comp_v)
                        component_positions.append(comp_positions.to(comp_k.device))
                    
                    if replay_k.shape[0] > 0:
                        components_k.append(replay_k)
                        components_v.append(replay_v)
                        component_positions.append(replay_pos.to(replay_k.device))
                    
                    if sketch_k.shape[0] > 0:
                        components_k.append(sketch_k)
                        components_v.append(sketch_v)
                        component_positions.append(sketch_positions.to(sketch_k.device))

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
                        self.stats["total_middle_dropped"] += middle_k.shape[0] - topk_k.shape[0] - comp_k.shape[0] - promoted_k.shape[0] - replay_k.shape[0] - sketch_k.shape[0]
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
            
            if new_key_states:
                seq_lengths = [key.shape[0] for key in new_key_states]
                target_len = min(seq_lengths)
                max_allowed = effective_sink + self.middle_allocation + self.recent_size
                target_len = min(target_len, max_allowed)

                min_required = effective_sink + min(
                    self.recent_size,
                    target_len - effective_sink if target_len > effective_sink else 0,
                )
                target_len = max(target_len, min_required)

                if any(length != target_len for length in seq_lengths):
                    for idx, (head_k, head_v) in enumerate(zip(new_key_states, new_value_states)):
                        if head_k.shape[0] != target_len:
                            trimmed_k, trimmed_v, dropped = self._coerce_head_sequence_length(
                                head_k,
                                head_v,
                                effective_sink,
                                target_len,
                            )
                            new_key_states[idx] = trimmed_k
                            new_value_states[idx] = trimmed_v
                            if dropped > 0:
                                self.stats["total_middle_dropped"] += dropped

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
        max_cache_size = effective_sink + self.middle_allocation + self.recent_size
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
                
                if middle_size > self.middle_allocation:
                    # Sample middle evenly to fit budget
                    steps = min(self.middle_allocation, middle_size)
                    if steps > 0:
                        middle_indices = torch.linspace(
                            middle_start, middle_end - 1,
                            steps,
                            dtype=torch.long,
                            device=k.device
                        )
                        keep_middle_k = k[:, :, middle_indices, :]
                        keep_middle_v = v[:, :, middle_indices, :]
                    else:
                        keep_middle_k = k[:, :, middle_start:middle_start, :]
                        keep_middle_v = v[:, :, middle_start:middle_start, :]
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
            expected_max = effective_sink + self.middle_allocation + self.recent_size
            
            print(f"[ADMS Stats @ {seq_len}] Avg per head: exact={avg_exact:.1f}, compressed={avg_comp:.1f}, dropped={avg_dropped:.1f}", flush=True)
            
            # ADM++ feature stats
            if self.config.enable_dual_fidelity:
                total_sketches = self.stats.get("total_sketch_tokens", 0)
                total_promotions = self.stats.get("total_sketch_promotions", 0)
                print(f"[ADMS Dual-Fidelity @ {seq_len}] Sketches: {total_sketches}, Promotions: {total_promotions}", flush=True)
            
            if self.config.enable_residual_replay:
                total_replays = self.stats.get("total_replay_triggers", 0)
                total_replay_tokens = self.stats.get("total_replay_tokens", 0)
                print(f"[ADMS Replay @ {seq_len}] Triggers: {total_replays}, Tokens: {total_replay_tokens}", flush=True)
            
            if self.config.enable_position_calibration:
                total_calibrations = self.stats.get("total_calibrations", 0)
                print(f"[ADMS Calibration @ {seq_len}] Total calibrations: {total_calibrations}", flush=True)
            
            if self.config.enable_adaptive_controller:
                total_adjustments = self.stats.get("total_budget_adjustments", 0)
                print(f"[ADMS Controller @ {seq_len}] Budget adjustments: {total_adjustments}", flush=True)
            
            print(f"[ADMS Cache @ {seq_len}] Actual size: {actual_cache_size}, Expected max: {expected_max}, Ratio: {actual_cache_size/seq_len:.2%}", flush=True)
            
            # WARNING if cache is unexpectedly large
            if actual_cache_size > expected_max * 1.5:
                print(f"[ADMS WARNING] Cache size {actual_cache_size} exceeds expected {expected_max} by >50%!", flush=True)
        
        return new_past_key_values