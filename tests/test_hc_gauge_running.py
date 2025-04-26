import numpy as np
import pytest
from code.hc_gauge_running import run_RGE, run_with_threshold

def test_run_RGE_identity():
    # If start and end scales are equal, RGE should return initial α unchanged
    α0 = np.array([0.03, 0.03, 0.03])
    μ0 = 1e3
    α = run_RGE(α0.copy(), μ0, μ0, b_vec=np.array([1,1,1]), steps=10)
    np.testing.assert_allclose(α, α0, atol=0, rtol=0)

def test_run_RGE_steps_effect():
    # More steps should yield closer result; test decreasing step count changes output
    α0 = np.array([0.03, 0.03, 0.03])
    μ0, μ1 = 1e3, 1e4
    α_few = run_RGE(α0.copy(), μ0, μ1, b_vec=np.array([1,1,1]), steps=10)
    α_more = run_RGE(α0.copy(), μ0, μ1, b_vec=np.array([1,1,1]), steps=100)
    # Results should differ under low vs high steps
    assert not np.allclose(α_few, α_more)

def test_run_with_threshold_no_triplet():
    # Without threshold, run_with_threshold should equal run_RGE
    from code.hc_gauge_running import beta
    α0 = np.array([0.03, 0.03, 0.03])
    α_thr = run_with_threshold(α0.copy(), add_triplet=False)
    α_direct = run_RGE(α0.copy(), 1.221e28 * np.exp(-10.967714943872613), 91.1876e9, b_vec=np.array([41/10,-19/6,-7]), steps=1000)
    np.testing.assert_allclose(α_thr, α_direct, rtol=1e-5)
