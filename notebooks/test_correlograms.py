import numpy as np
from task5 import fast_histogram
from task5 import brute_hist
from numpy.testing import assert_array_equal

def test_fast_histogram():
    # Test Case 1: Basic Autocorrelogram (No Zero-Lag)
    t = np.array([0.1, 0.2, 0.3])
    edges = np.linspace(-0.1, 0.1, 21)  # 10ms bins
    result = fast_histogram(t, t, edges)
    zero_bin = len(edges)//2  # Middle bin (0ms)
    assert result[zero_bin] == 0, "Autocorrelogram zero-lag bin not zero!"
    assert np.sum(result) == 6, "Should have 3*2 valid pairs (excluding self)"

    # Test Case 2: Perfectly Synchronized Spikes (Cross-Correlogram)
    tA = np.array([0.1, 0.2, 0.3])
    tB = np.array([0.1, 0.2, 0.3])
    edges = np.linspace(-0.05, 0.05, 11)  # 10ms bins
    result = fast_histogram(tA, tB, edges)
    assert result[5] == 3, "3 coincident spikes at 0ms"

    # Test Case 3: Refractory Period Simulation (Autocorrelogram)
    t = np.arange(0, 1.0, 0.0025)  # 400Hz firing (2.5ms interval)
    edges = np.linspace(-0.01, 0.01, 21)  # 1ms bins
    result = fast_histogram(t, t, edges)
    assert result[10] == 0, "Zero-lag should be excluded"
    assert np.all(result[9:12] == 0), "2ms refractory period"

    # Test Case 4: Edge Spikes (Cross-Correlogram)
    tA = np.array([0.0])
    tB = np.array([-0.03, 0.03])
    edges = np.linspace(-0.03, 0.03, 7)  # 10ms bins
    result = fast_histogram(tA, tB, edges)
    assert result[0] == 1, "-30ms bin"
    assert result[-1] == 1, "+30ms bin"

    # Test Case 5: Empty Inputs
    assert np.all(fast_histogram([], np.arange(10), edges) == 0)
    assert np.all(fast_histogram(np.arange(10), [], edges) == 0)

    # Test Case 6: Single Spike (Autocorrelogram)
    t = np.array([0.5])
    result = fast_histogram(t, t, edges)
    assert np.sum(result) == 0, "No pairs to count"

    # Test Case 7: Bin Alignment Check
    tA = np.array([0.0])
    tB = np.array([0.009, 0.010, 0.011])
    edges = np.linspace(-0.01, 0.01, 21)  # 1ms bins
    result = fast_histogram(tA, tB, edges)
    assert result[10+9] == 1, "9ms bin"
    assert result[10+10] == 1, "10ms bin (edge)"
    assert result[10+11] == 0, "11ms excluded"

    # Test Case 8: Non-symmetric Window Warning
    edges = np.linspace(-0.03, 0.031, 61)
    #with pytest.warns(UserWarning):
    fast_histogram(tA, tB, edges)

    # Test Case 9: Large Dataset Validation
    np.random.seed(42)
    tA = np.sort(np.random.uniform(0, 100, 10_000))
    tB = np.sort(np.random.uniform(0, 100, 10_000))
    edges = np.linspace(-5, 5, 11)  # 1s bins
    fast_counts = fast_histogram(tA, tB, edges)
    brute_counts = brute_hist(tA, tB, edges)
    assert np.array_equal(fast_counts, brute_counts), "Large dataset mismatch"

    print("All test cases passed!")

# Add to your test suite
if __name__ == "__main__":
    import pytest
    pytest.main([__file__])