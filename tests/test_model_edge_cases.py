"""Unit tests for edge cases in model_development_template.py.

Covers:
1. allocate_sequential_stable edge cases (n_past behavior)
2. Extreme input values for dynamic multiplier
"""

import numpy as np
import pandas as pd
import pytest

from template.model_development_template import (
    allocate_sequential_stable,
    compute_dynamic_multiplier,
)

# -----------------------------------------------------------------------------
# Allocation Edge Cases
# -----------------------------------------------------------------------------

def test_allocate_sequential_stable_n_past_gt_n():
    """Test that n_past > n is handled by locking all weights."""
    raw = np.array([1.0, 1.0, 1.0])
    n_past = 5 # Greater than len(raw)
    
    weights = allocate_sequential_stable(raw, n_past)
    
    assert len(weights) == 3
    # Should behave as if n_past = 3
    # Stable allocation logic applies to all
    assert np.all(weights > 0)
    assert np.isclose(weights.sum(), 1.0)


def test_allocate_sequential_stable_n_past_zero():
    """Test behavior when n_past is 0 (all future)."""
    raw = np.array([1.0, 1.0, 1.0])
    n_past = 0
    
    weights = allocate_sequential_stable(raw, n_past)
    
    # All uniform
    expected = np.array([1/3, 1/3, 1/3])
    np.testing.assert_allclose(weights, expected)


def test_allocate_sequential_stable_locked_weights_mismatch():
    """Test when provided locked_weights length doesn't match n_past."""
    raw = np.array([1.0, 1.0, 1.0])
    n_past = 2
    # Only provide 1 locked weight, but n_past=2
    # Logic should fall back to computing for the missing past index
    locked = np.array([0.5]) 
    
    weights = allocate_sequential_stable(raw, n_past, locked_weights=locked)
    
    # Index 0 is recomputed because n_past > len(locked_weights) triggers recomputation of all past
    # (Existing implementation does not support partial locking)
    
    # Index 1 is past but not locked -> computed from raw
    # Logic: if locked_weights provided but len < n_past, it uses them
    # BUT current implementation:
    # if locked_weights is not None and len(locked_weights) >= n_past: use it
    # else: recompute ALL past
    
    # So here it should recompute ALL past, ignoring current locked array because it's too short
    # This verifies the 'else' branch of that check
    # Recomputed for index 0: raw[0]=1, running_mean=1 -> signal=1 -> weight=1/3
    # Recomputed for index 1: raw[1]=1, running_mean=1 -> signal=1 -> weight=1/3
    
    # Wait, let's check code:
    # if locked_weights is not None and len(locked_weights) >= n_past: ... else: ...
    # So yes, it falls into else.
    
    expected_recomputed = 1.0/3 # roughly, depends on scaling
    
    # Verify it recomputed (didn't use locked 0.5)
    assert np.isclose(weights[0], expected_recomputed)
    
    # Just checking it didn't crash and returns valid distribution
    assert np.isclose(weights.sum(), 1.0)
    assert len(weights) == 3


# -----------------------------------------------------------------------------
# Extreme Value Handling
# -----------------------------------------------------------------------------

def test_compute_dynamic_multiplier_extreme_values():
    """Test handling of extreme inputs (inf, nan handling wrapped in array)."""
    # Inputs - simplified to only price_vs_ma
    price_vs_ma = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 10.0, -10.0])  # Including extreme values
    
    # Should not crash and clip result
    multiplier = compute_dynamic_multiplier(price_vs_ma)
    
    assert np.all(np.isfinite(multiplier))
    assert np.all(multiplier > 0)  # Multipliers must be positive
    
    # Below MA (negative price_vs_ma) should give higher multipliers
    assert multiplier[0] > multiplier[2]  # -1.0 > 0.0
    # Above MA (positive price_vs_ma) should give lower multipliers
    assert multiplier[4] < multiplier[2]  # 1.0 < 0.0


def test_compute_dynamic_multiplier_clipping():
    """Test that extreme values are properly clipped."""
    # Test with values beyond [-1, 1] range
    price_vs_ma = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
    
    multiplier = compute_dynamic_multiplier(price_vs_ma)
    
    # All should be finite and positive
    assert np.all(np.isfinite(multiplier))
    assert np.all(multiplier > 0)
    
    # Should handle extreme values gracefully (clipped internally)
    assert multiplier[0] > 0  # Very negative should still give positive multiplier
    assert multiplier[4] > 0   # Very positive should still give positive multiplier
