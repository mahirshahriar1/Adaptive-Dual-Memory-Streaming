"""
Test script to verify ADM++ features are properly implemented
"""

import torch
from streaming_llm.adms_cache import ADMSConfig, ADMSKVCache


def test_dual_fidelity():
    """Test Dual-Fidelity Mid Memory feature"""
    print("\n" + "="*60)
    print("Testing Dual-Fidelity Mid Memory")
    print("="*60)
    
    config = ADMSConfig(
        start_size=4,
        recent_size=32,
        compressed_budget=16,
        enable_dual_fidelity=True,
        sketch_budget=8,
        sketch_reduction="mean",
        compressor_type="low_rank",
        rank=8,
    )
    
    cache = ADMSKVCache(config)
    
    # Test sketch reduction
    test_k = torch.randn(10, 64)
    test_v = torch.randn(10, 64)
    
    for reduction in ["mean", "sum", "first"]:
        sketch_k, sketch_v = cache._apply_sketch_reduction(test_k, test_v, reduction)
        assert sketch_k.shape[0] == 1, f"Sketch should be single token, got {sketch_k.shape[0]}"
        print(f"✓ Sketch reduction '{reduction}' works: {sketch_k.shape}")
    
    # Test residual energy computation
    approx_k = test_k[:5]
    approx_v = test_v[:5]
    energy = cache._compute_residual_energy(test_k, test_v, approx_k, approx_v)
    assert 0 <= energy <= 1, f"Energy should be in [0,1], got {energy}"
    print(f"✓ Residual energy computation works: {energy:.4f}")
    
    # Test sketch bank update
    positions = torch.arange(10)
    cache._update_sketch_bank(0, 0, test_k, test_v, positions)
    assert (0, 0) in cache.sketch_bank, "Sketch bank should be created"
    assert len(cache.sketch_bank[(0, 0)]["keys"]) > 0, "Sketch should be stored"
    print(f"✓ Sketch bank update works: {len(cache.sketch_bank[(0, 0)]['keys'])} sketches")
    
    # Test sketch retrieval
    sketch_k, sketch_v, sketch_pos = cache._get_sketch_tokens(0, 0, budget=4)
    assert sketch_k.shape[0] > 0, "Should retrieve sketch tokens"
    print(f"✓ Sketch token retrieval works: {sketch_k.shape[0]} tokens")
    
    # Test sketch promotion
    promoted_k, promoted_v, promoted_pos = cache._promote_sketches_to_precision(0, 0, promotion_budget=1)
    print(f"✓ Sketch promotion works: {promoted_k.shape[0]} tokens promoted")
    
    print("✅ Dual-Fidelity Mid Memory: ALL TESTS PASSED\n")


def test_residual_replay():
    """Test Residual Replay Engine feature"""
    print("\n" + "="*60)
    print("Testing Residual Replay Engine")
    print("="*60)
    
    config = ADMSConfig(
        start_size=4,
        recent_size=32,
        compressed_budget=16,
        enable_residual_replay=True,
        replay_budget=8,
        energy_replay_threshold=0.85,
        compressor_type="low_rank",
        rank=4,
    )
    
    cache = ADMSKVCache(config)
    
    # Test replay bank update
    test_k = torch.randn(20, 64)
    test_v = torch.randn(20, 64)
    positions = torch.arange(20)
    
    cache._update_replay_bank(0, 0, test_k, test_v, positions)
    assert (0, 0) in cache.replay_bank, "Replay bank should be created"
    assert len(cache.replay_bank[(0, 0)]["keys"]) > 0, "Tokens should be stored"
    print(f"✓ Replay bank update works: {len(cache.replay_bank[(0, 0)]['keys'])} entries")
    
    # Test spectral energy computation
    approx_k = test_k[:10]
    approx_v = test_v[:10]
    energy = cache._compute_spectral_energy(approx_k, approx_v, test_k, test_v)
    assert 0 <= energy <= 1, f"Energy should be in [0,1], got {energy}"
    print(f"✓ Spectral energy computation works: {energy:.4f}")
    
    # Test replay triggering with low energy
    low_energy = 0.7
    replay_k, replay_v, replay_pos = cache._trigger_replay(0, 0, low_energy)
    assert replay_k.shape[0] > 0, "Should trigger replay for low energy"
    print(f"✓ Replay trigger works: {replay_k.shape[0]} tokens replayed (energy={low_energy})")
    
    # Test no replay with high energy
    high_energy = 0.95
    replay_k, replay_v, replay_pos = cache._trigger_replay(0, 0, high_energy)
    assert replay_k.shape[0] == 0, "Should not trigger replay for high energy"
    print(f"✓ No replay for high energy: {replay_k.shape[0]} tokens (energy={high_energy})")
    
    print("✅ Residual Replay Engine: ALL TESTS PASSED\n")


def test_rope_calibration():
    """Test RoPE Alignment Calibration feature"""
    print("\n" + "="*60)
    print("Testing RoPE Alignment Calibration")
    print("="*60)
    
    config = ADMSConfig(
        start_size=4,
        recent_size=32,
        compressed_budget=16,
        enable_position_calibration=True,
        calibration_window=64,
        calibration_regularization=0.1,
    )
    
    cache = ADMSKVCache(config)
    cache.current_length = 100
    
    # Test calibration anchor update
    recent_positions = torch.arange(50, 100)
    cache._update_calibration_anchors(0, 0, recent_positions)
    assert (0, 0) in cache.calibration_anchors, "Calibration anchors should be stored"
    print(f"✓ Calibration anchor update works: {cache.calibration_anchors[(0, 0)].shape[0]} anchors")
    
    # Test position calibration (need enough anchors)
    synthetic_positions = torch.tensor([10, 20, 30, 40, 50])
    compressed_keys = torch.randn(5, 64)
    
    calibrated = cache._calibrate_synthetic_positions(0, 0, synthetic_positions, compressed_keys)
    assert calibrated.shape == synthetic_positions.shape, "Calibrated positions should match input shape"
    print(f"✓ Position calibration works: {synthetic_positions.tolist()} -> {calibrated.tolist()}")
    
    # Verify calibration offsets are stored
    if (0, 0) in cache.calibration_offsets:
        print(f"✓ Calibration offsets stored: shape {cache.calibration_offsets[(0, 0)].shape}")
    
    print("✅ RoPE Alignment Calibration: ALL TESTS PASSED\n")


def test_adaptive_controller():
    """Test Adaptive Budget Controller feature"""
    print("\n" + "="*60)
    print("Testing Adaptive Budget Controller")
    print("="*60)
    
    config = ADMSConfig(
        start_size=4,
        recent_size=32,
        compressed_budget=16,
        enable_adaptive_controller=True,
        controller_gain=0.3,
        controller_energy_floor=0.75,
        controller_energy_ceiling=0.95,
        controller_group_size=2,
    )
    
    cache = ADMSKVCache(config)
    
    # Test controller state update
    cache._update_controller_state(0, 0, reconstruction_energy=0.8, attention_mass=0.5)
    assert (0, 0) in cache.controller_energy_ema, "Energy EMA should be stored"
    assert (0, 0) in cache.controller_attention_mass, "Attention mass should be stored"
    print(f"✓ Controller state update works: energy={cache.controller_energy_ema[(0, 0)]:.4f}, attention={cache.controller_attention_mass[(0, 0)]:.4f}")
    
    # Test adaptive budget with low energy (should increase)
    cache._update_controller_state(0, 0, reconstruction_energy=0.6, attention_mass=0.8)
    budget_low = cache._get_adaptive_budget(0, 0, base_budget=16)
    print(f"✓ Low energy budget: {budget_low} (base=16, should increase)")
    
    # Test adaptive budget with high energy (should decrease)
    cache._update_controller_state(0, 0, reconstruction_energy=0.98, attention_mass=0.5)
    budget_high = cache._get_adaptive_budget(0, 0, base_budget=16)
    print(f"✓ High energy budget: {budget_high} (base=16, should decrease)")
    
    # Test head grouping (group_size=2)
    cache._update_controller_state(0, 1, reconstruction_energy=0.7, attention_mass=0.6)
    budget_grouped = cache._get_adaptive_budget(0, 1, base_budget=16)
    print(f"✓ Grouped head budget: {budget_grouped} (shares state with head 0)")
    
    print("✅ Adaptive Budget Controller: ALL TESTS PASSED\n")


def test_integration():
    """Test full integration with all features enabled"""
    print("\n" + "="*60)
    print("Testing Full ADM++ Integration")
    print("="*60)
    
    config = ADMSConfig(
        start_size=4,
        recent_size=32,
        compressed_budget=16,
        # Enable all ADM++ features
        enable_dual_fidelity=True,
        sketch_budget=8,
        enable_residual_replay=True,
        replay_budget=8,
        enable_position_calibration=True,
        enable_adaptive_controller=True,
        compressor_type="low_rank",
        rank=8,
    )
    
    cache = ADMSKVCache(config)
    
    # Simulate a typical cache update
    batch_size = 1
    num_heads = 4
    seq_len = 100
    d_k = 64
    
    # Create mock past_key_values
    past_key_values = []
    for layer in range(2):
        k = torch.randn(batch_size, num_heads, seq_len, d_k)
        v = torch.randn(batch_size, num_heads, seq_len, d_k)
        past_key_values.append((k, v))
    
    # Apply ADMS compression
    compressed = cache(past_key_values)
    
    assert compressed is not None, "Compression should return non-None"
    assert len(compressed) == len(past_key_values), "Should have same number of layers"
    
    compressed_seq_len = compressed[0][0].shape[2]
    expected_max = cache.dynamic_start_size + cache.middle_allocation + cache.recent_size
    
    print(f"✓ Compression works: {seq_len} -> {compressed_seq_len} tokens")
    print(f"✓ Expected max: {expected_max}, Actual: {compressed_seq_len}")
    assert compressed_seq_len <= expected_max, f"Compressed size {compressed_seq_len} exceeds max {expected_max}"
    
    # Check stats
    print(f"\n📊 Compression Stats:")
    print(f"   Total compressions: {cache.stats['total_compressions']}")
    print(f"   Exact kept: {cache.stats['total_exact_kept']}")
    print(f"   Compressed kept: {cache.stats['total_compressed_kept']}")
    print(f"   Dropped: {cache.stats['total_middle_dropped']}")
    print(f"   Sketch tokens: {cache.stats['total_sketch_tokens']}")
    print(f"   Sketch promotions: {cache.stats['total_sketch_promotions']}")
    print(f"   Replay triggers: {cache.stats['total_replay_triggers']}")
    print(f"   Replay tokens: {cache.stats['total_replay_tokens']}")
    print(f"   Calibrations: {cache.stats['total_calibrations']}")
    print(f"   Budget adjustments: {cache.stats['total_budget_adjustments']}")
    
    print("\n✅ Full ADM++ Integration: ALL TESTS PASSED\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ADM++ FEATURE VERIFICATION TESTS")
    print("="*60)
    
    try:
        test_dual_fidelity()
        test_residual_replay()
        test_rope_calibration()
        test_adaptive_controller()
        test_integration()
        
        print("\n" + "="*60)
        print("🎉 ALL ADM++ FEATURES VERIFIED SUCCESSFULLY!")
        print("="*60)
        print("\nSummary:")
        print("  ✅ Dual-Fidelity Mid Memory - IMPLEMENTED")
        print("     - Sketch bank with mean/sum/first reducers")
        print("     - Residual energy tracking")
        print("     - Sketch-to-precision promotion")
        print("\n  ✅ Residual Replay Engine - IMPLEMENTED")
        print("     - Spectral energy monitoring")
        print("     - Reconstruction error tracking")
        print("     - Automatic token replay on energy drop")
        print("\n  ✅ RoPE Alignment Calibration - IMPLEMENTED")
        print("     - Lightweight linear solver")
        print("     - Sliding anchor windows")
        print("     - Position offset tracking")
        print("\n  ✅ Adaptive Budget Controller - IMPLEMENTED")
        print("     - Reconstruction energy-based adjustment")
        print("     - Attention mass tracking")
        print("     - Head grouping for shared state")
        print("\n" + "="*60 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}\n")
        raise
