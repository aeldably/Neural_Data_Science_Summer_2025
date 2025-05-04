#%%
from sklearn.mixture import GaussianMixture
import  numpy as np
import matplotlib.pyplot as plt
#%%
# replace by path to your solutions
b = np.load("../data/nds_cl_1_features.npy")
s = np.load("../data/nds_cl_1_spiketimes_s.npy")
t = np.load("../data/nds_cl_1_spiketimes_t.npy")
w = np.load("../data/nds_cl_1_waveforms.npy")

assert b.shape == (33983, 12), "b should be (33983, 12)"
assert s.shape == (33983,), "s should be (33983,)"
assert t.shape == (33983,), "t should be (33983,)"

print(f"w.shape = {w.shape}")
#%%
assert w.shape == (33983, 30, 4), "w should be (33983, 30, 4)"

#%%  
# Give intuitive names to the variables

print(f"b: {b.shape}, s: {s.shape}, t: {t.shape}, w: {w.shape}")
raw_wave_forms = (
    w  # (N, 30, 4) # - 33,983 spikes \times 30 time samples \times 4 channels
)
spike_features = b  # (N, 12) - 12 features per spikes  (3 principal components \times 4 channes = 12)
spike_indices = s  # 33,983 spikes, sample numbers in raw file – NOT 0…N-1 rows.
spike_seconds = t  # in seconds of spikes.
print(
    f"raw_wave_forms: {raw_wave_forms.shape}, spike_features: {spike_features.shape}, sample_indices: {spike_indices.shape}, spike_seconds: {spike_seconds.shape}"
)

#%%

# 1) Fit sklearn GMMs & compute BIC for K=1…Kmax
Kmax = 15
bics = []
models = []  # will store the fitted GMM for each K
ranges = np.arange(15, Kmax + 1)
for K in ranges:
    gmm = GaussianMixture(
        n_components=K,
        covariance_type="full",
        random_state=0,
        init_params="kmeans",  # more robust than random
        max_iter=200,
    ).fit(
        spike_features
    )  # b is your (33983×12) feature matrix

    bic = gmm.bic(spike_features)
    bics.append(bic)
    models.append(gmm)
    print(f"K={K:2d}   BIC={bic:10.1f}")

# 2) Pick best K and model
best_idx = int(np.argmin(bics))
best_K = ranges[best_idx] 
best_gmm = models[best_idx]
print(f"\n→ best K = {best_K}")

# 3) Extract labels and parameters
best_labels = best_gmm.predict(b)
best_means = best_gmm.means_
best_covs = best_gmm.covariances_
best_weights = best_gmm.weights_

#%% Assert the sizes and shapes of the variables
assert best_labels.shape == (33983,), "best_labels should be (33983,)"
assert best_means.shape == (best_K, 12), "best_means should be (K, 12)"
assert best_covs.shape == (best_K, 12, 12), "best_covs should be (K, 12, 12)"
assert best_weights.shape == (best_K,), "best_weights should be (K,)"

print(
    f"best_labels: {best_labels.shape}, best_means: {best_means.shape}, best_covs: {best_covs.shape}, best_weights: {best_weights.shape}"
)
#%%
assert best_K == 15, "Based on BIC, best_K should be 15"
#%%
def group_spike_indices_by_cluster(
    spike_indices: np.ndarray,
    labels: np.ndarray,
    n_clusters: int | None = None,
) -> np.ndarray:
    """
    Return an object-dtype array where element k is the 1-D int array of
    spike-row indices that belong to cluster k.

    All inputs **must already be NumPy arrays**.
    The output is a (n_clusters,) NumPy array with dtype=object.
    """
    # --- sanity & shape ---
    spike_indices = np.asarray(spike_indices, dtype=int).ravel()
    labels = np.asarray(labels, dtype=int).ravel()
    if spike_indices.shape != labels.shape:
        raise ValueError("spike_indices and labels need identical shape")

    if n_clusters is None:
        n_clusters = labels.max() + 1

    # --- allocate result container (object array, not Python list) ---
    grouped = np.empty(n_clusters, dtype=object)

    # --- fill per cluster ---
    for k in range(n_clusters):
        grouped[k] = spike_indices[labels == k]

    return grouped

#%%
idx_by_cluster = group_spike_indices_by_cluster(
    spike_indices, best_labels, n_clusters=best_K
)
assert idx_by_cluster.shape == (15,), "idx_by_cluster should be (15,)"

# Assert that clusters are of size from 500 to 5000 spikes per cluster.
for k in range(best_K):
    assert 100 <= idx_by_cluster[k].shape[0] <= 10000, f"Cluster {k} has {idx_by_cluster[k].shape[0]} spikes"

for k in range(best_K):
    print(f"Cluster {k} has {idx_by_cluster[k].shape[0]} in the range  {idx_by_cluster[k][0]} to {idx_by_cluster[k][-1]}")
    
#%%
# Hence we have 15 clusters of spikes, each containing between 
# 500 and 5000 spikes.
#
# idx_by_cluster[k] is a 1-D array of spike indices that 
# belong to the cluster.
#%%
## Task 5: Cluster separation and Correlograms
#
# As postprocessing, implement the calculation of auto- and 
# cross correlograms over the spike times.
#
# Plot the (auto-/cross-) correlograms, displaying a 
# time frame of -30ms to +30ms. 
#
# Choose a good bin size and interpret the resulting diagrams.
# 
# _Grading: 3 pts_
# 
#### Hints
#
# It is faster to calculate the histogram only over the 
# spike-times that are in the displayed range. 
# 
# Filter the spike times before calculating the histogram!_
#
# For the auto-correlogram, make sure not to include the 
# time difference between a spike and itself (which would be exactly 0)_
#
# For the correlogram an efficient 
# implementation is very important - looping over all 
# spike times is not feasible. 
# Instead, make use of numpy vectorization and broadcasting 
# - you can use functions such as tile or repeat.
#
# The cross-correlogram function
# R_{uv}(\tau) = \int u(t) v(t - \tau) dt
# Histogram of $\tau = t_i - t_j$ for all spike times $t_i$ and $t_j$.
# 
# Time series where events happen at discrete time points.  
# The difference in all pairwise differences in spike times.  
#
# 0. Start with a target and reference spike train. 
# 1. Start from the reference spike train, 
# 2. Then for a given reference_spike[n] we find all when relative to this spike all the 
#    other spikes happened, that is what are the differences in timing between this 
#    and the other spike.
# 3. Then we add to the bins where the spike was we do a histogram and add it up.
# 4. Then we move up ,  
# If we do that for a spike with itself we get the autocorrelogram.
# If we do that between two neurons, we call that teh cross-correlleograms.
# So we will look at the cross and auto-correllogram between the clusters. 
# 
# We can detect single unit clusters, by looking at auto-correlograms.
#%%
#%%
def brute_histogram(tA, tB, edges):
    dt = (tB[:,None] - tA[None,:]).ravel()
    dt = dt[(dt != 0) & (dt >= edges[0]) & (dt < edges[-1])]
    return np.histogram(dt, bins=edges)[0]

def simple_histogram(tA, tB, edges):
    """
    Return the histogram of (tB - tA) in the half-open bins given by *edges*.
    •  tA, tB must be 1-D and sorted (ascending seconds)
    •  dt = 0 is discarded ⇒ works for auto- and cross-correlograms
    """
    w    = edges[-1]            # half-width of the window  (edges are symmetric)
    bw   = edges[1] - edges[0]  # bin width
    hist = np.zeros(len(edges) - 1, dtype=int)

    for t0 in tA:
        # indices of spikes in tB that lie within [t0-w , t0+w]
        lo = np.searchsorted(tB, t0 - w, side='left')
        hi = np.searchsorted(tB, t0 + w, side='right')

        # vectorised differences, remove zeros (auto-corr)
        dt  = tB[lo:hi] - t0
        dt  = dt[dt != 0]

        # map each dt to a bin and accumulate
        bins = ((dt - edges[0]) // bw).astype(int)
        np.add.at(hist, bins, 1)

    return hist

# --- The function to test ---
def fast_histogram(tA, tB, edges):
    """
    Calculates histogram counts for time differences (tB - tA)
    within the range defined by edges. Assumes tA and tB are sorted.

    Efficiently avoids calculating differences outside the window.
    Excludes zero differences (for autocorrelograms).
    
    """
    # Assumption 1: Inputs are sorted
    assert np.all(np.diff(tA) >= 0), "tA must be sorted!"
    assert np.all(np.diff(tB) >= 0), "tB must be sorted!"
    
    # Assumption 2: Edges are uniformly spaced
    bin_width = edges[1] - edges[0]
    assert np.allclose(np.diff(edges), bin_width), "Edges must be uniformly spaced!"
    
    # Assumption 3: Window is symmetric (optional but recommended)
    assert np.isclose(edges[0], -edges[-1]), "Edges should be symmetric around 0!"
    
    # Assumption 4: Inputs are 1D arrays
    assert tA.ndim == 1 and tB.ndim == 1, "tA/tB must be 1D arrays!"
         
    counts = np.zeros(edges.size-1, int)
    j_left = 0
    # Determine window half-width 'w' based on edges assuming symmetric window
    # A more robust way might be needed if edges aren't symmetric around 0
    if not (np.isclose(edges[0], -edges[-1])):
         print("Warning: Window edges might not be symmetric. Assuming edges[-1] defines positive window.")
    w = edges[-1] # Use the positive edge limit as the window extent
    bin_width = edges[1] - edges[0] # Assumes uniform bins

    for t0 in tA:
        # Advance j_left past spikes in tB too early for the window
        while j_left < len(tB) and tB[j_left] < t0 - w:
            j_left += 1

        # Iterate through spikes in tB potentially within the window
        j = j_left
        while j < len(tB) and tB[j] <= t0 + w:
            dt = tB[j] - t0

            # Exclude self-comparison (dt=0) and check if within overall edge bounds
            if dt != 0 and dt >= edges[0] and dt < edges[-1]:
                 # Calculate bin index based on edges
                 # This assumes uniform bins starting at edges[0]
                 bin_idx = int(np.floor((dt - edges[0]) / bin_width))
                 # Ensure bin_idx is within the valid range (can happen due to float precision)
                 bin_idx = max(0, min(bin_idx, len(counts) - 1))
                 counts[bin_idx] += 1
            j += 1
    return counts

#%%
# --- Test Cases ---
# Test Case 1: Simple Autocorrelogram (tA == tB)
print("--- Test Case 1: Autocorrelogram ---")
tA_auto = np.array([0.1, 0.101, 0.105, 0.2, 0.202])
tB_auto = tA_auto
# Window: -0.005s to +0.005s (-5ms to +5ms), Bin size: 1ms
edges_auto = np.arange(-0.005, 0.005 + 0.001, 0.001)
# Edges: [-0.005, -0.004, -0.003, -0.002, -0.001,  0.   ,  0.001,  0.002,  0.003,  0.004,  0.005]
# Bins:  | -5:-4 | -4:-3 | -3:-2 | -2:-1 | -1:0  | 0:1   | 1:2   | 2:3   | 3:4   | 4:5   |

# Expected differences (tB - tA, excluding 0):
# 0.101 - 0.1   =  0.001 (bin 5: 0-1ms)
# 0.105 - 0.1   =  0.005 (outside < edge[-1])
# 0.1   - 0.101 = -0.001 (bin 4: -1-0ms)
# 0.105 - 0.101 =  0.004 (bin 8: 3-4ms)
# 0.1   - 0.105 = -0.005 (outside >= edge[0])
# 0.101 - 0.105 = -0.004 (bin 1: -4--3ms)
# 0.202 - 0.2   =  0.002 (bin 6: 1-2ms)
# 0.2   - 0.202 = -0.002 (bin 3: -2--1ms)
# Other differences are > 0.005 or < -0.005

# Expected counts:
# Bin Indices: 0      1      2      3      4      5      6      7      8      9
# Expected:   [1,     1,     1,     1,     0,     0,     1,     1,     1,     1]

print("fast :", fast_histogram(tA_auto, tB_auto, edges_auto))
print("brute:", brute_histogram      (tA_auto, tB_auto, edges_auto))
assert np.array_equal(fast_histogram(tA_auto, tB_auto, edges_auto), brute_histogram(tA_auto, tB_auto, edges_auto)), "Test Case -1: Brute and fast histogram do not match!"
#%%
def test_random_histograms():
    rng = np.random.default_rng(0)
    tA = np.sort(rng.uniform(0, 2, 6_000))   
    tB = np.sort(rng.uniform(0, 2, 6_000))   

    bin_size = 0.001
    window   = 0.050
    
    edges    = np.arange(-window, window + bin_size, bin_size)

    h_fast   = fast_histogram   (tA, tB, edges)
    h_brute  = brute_histogram (tA, tB, edges)
    h_simple = simple_histogram(tA, tB, edges)

    print(np.allclose(h_fast, h_simple))   
    assert np.allclose(h_fast, h_simple), "Random histograms do not match, h_fast, h_simple!"
    print(np.allclose(h_fast, h_brute))
    assert np.allclose(h_fast, h_brute), "Random histograms do not match!, h_fast, h_brute"

test_random_histograms()
#%%
expected_counts_auto = np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1])
result_auto = fast_histogram(tA_auto, tB_auto, edges_auto)

print(f"Edges: {edges_auto}")
print(f"Result:   {result_auto}")
print(f"Expected: {expected_counts_auto}")
assert np.array_equal(result_auto, expected_counts_auto), "Test Case 1 Failed!"
print("Test Case 1 Passed!\n")

#%% # 0. choose bin size and window once
bin_size = 0.0005       # .1 ms
window   = 0.030       # ±30 ms
# TODO: OFF by ONE ?
num_bins = int((2 * window) / bin_size)  # Ensures exact window coverage
edges    = np.arange(-window, window + bin_size, bin_size)
# Find the zero-lag bin (where dt=0 would fall, even though excluded)
# No idea why?
zero_lag_bin = np.searchsorted(edges, 0) - 1

# 1. convert sample indices → seconds, if you haven’t yet
fs = 30_000 # 30 kHz is the sampling rate
times_by_cluster = []
for idx in idx_by_cluster:
    # keep only one entry per sample index
    unique_idx = np.unique(idx)
    times_by_cluster.append(unique_idx / fs)      # seconds

# 2. build the K × K correlogram matrix
centres = edges[:-1] + bin_size/2

K      = best_K
n_bins = len(edges) - 1
corr   = np.empty((K, K, n_bins), int)

for i in range(K):
    for j in range(K):
        corr[i, j] = fast_histogram(times_by_cluster[i],
                                    times_by_cluster[j],
                                    edges)
 
np.save("corr_raw_counts.npy", corr)

#%%
fig, axes = plt.subplots(K, K, figsize=(3*K, 3*K))
for i in range(K):
    for j in range(K):
        ax = axes[i, j]
        ax.bar(centres*1e3, corr[i,j], width=bin_size*1e3, color='k')
        ax.set_xlim(-30, 30)
        ax.axvline(0, color='grey', lw=0.6)
        if i == K-1: ax.set_xlabel('lag (ms)')
        if j == 0:   ax.set_ylabel(f'cl {i}')
        ax.set_xticks([-30, 0, 30]); ax.set_yticks([])
plt.tight_layout(); 
plt.suptitle(f'Cross-correlograms: bin-size:{bin_size}, window:{window}, K:{K} ', fontsize=16, y=1.02)
plt.savefig("correlograms.png", dpi=300)
plt.show()
#%% 
# Plot Auto-Correlograms Separately
def plot_autocorrelograms(corr      : np.ndarray,
                          centres   : np.ndarray,
                          *,
                          max_lag_ms : float = 30,
                          bin_size_s : float = 1e-3,
                          cols       : int   = 4,
                          figsize    : tuple = (12, 8),
                          y_label    : str   = "count",
                          x_label    : str   = "lag (ms)"):
    """
    Draw the K auto-correlograms sitting on the diagonal of *corr* and
    attach nice axis labels.

    Parameters
    ----------
    corr : (K, K, n_bins) int array
        Correlogram cube (full cross-matrix).  We plot corr[k, k].
    centres : 1-D float array
        Bin-centre positions **in seconds** (same length as the last dim).
    max_lag_ms, bin_size_s, cols, figsize … see earlier version
    """
    K        = corr.shape[0]
    rows     = int(np.ceil(K / cols))
    width_ms = bin_size_s * 1e3

    fig, axes = plt.subplots(rows, cols,
                             figsize=figsize,
                             sharex=True, sharey=True,
                             squeeze=False)

    zero_bin  = np.argmin(np.abs(centres))        # locate the 0-lag bin

    for k in range(K):
        r, c  = divmod(k, cols)
        ax    = axes[r, c]

        # make a local copy so we don’t overwrite the cube
        counts           = corr[k, k].copy()
        counts[zero_bin] = 0                     # hide 0-lag spike

        ax.bar(centres * 1e3, counts,            # convert x to ms
               width=width_ms, color='k')

        ax.set_xlim(-max_lag_ms, max_lag_ms)
        ax.axvline(0, color='grey', lw=.6)
        ax.set_title(f"Cluster {k}", fontsize=9)

        # Put tick labels only on the outer panels to avoid clutter
        if r == rows - 1:         # bottom row – show x-tick labels
            ax.set_xlabel(x_label)
            ax.set_xticks([-max_lag_ms, 0, max_lag_ms])
        else:
            ax.set_xticklabels([])

        if c == 0:                # left column – show y-tick labels
            ax.set_ylabel(y_label)
            ax.set_yticks(ax.get_yticks())       # keep ticks
        else:
            ax.set_yticklabels([])

    # delete unused axes if K is not a multiple of *cols*
    for k in range(K, rows * cols):
        fig.delaxes(axes.flat[k])
        
    # fig.supxlabel(x_label, fontsize=12)              
    # fig.supylabel(y_label, fontsize=12, x=0.04)     
    fig.suptitle(f"Auto-correlograms: Clusters:{K} with a bin size of {bin_size_s*1e3:.0f} ms, over a window:$\pm${max_lag_ms:.0f} ms", 
                 fontsize=16, y=1.02)


    fig.tight_layout()
    return fig


def run_plot_autocorrelograms(corr, bin_size: float = 0.001, window: float = 0.030):
    """
    Run the autocorrelogram plotting function with a given bin size and window.
    """
    edges      = np.arange(-window, window + bin_size, bin_size)
    centres    = edges[:-1] + bin_size/2        # seconds
    max_lag_ms = window*1e3  # ms
    bin_size_s = bin_size
    cols       = 5     # 4 plots per row
    figsize    = (18, 9)
    fig = plot_autocorrelograms(corr, centres, max_lag_ms=max_lag_ms,
                                bin_size_s=bin_size_s, cols=cols,
                                figsize=figsize)
    fig.savefig("autocorrelograms.png", dpi=300)
    plt.show()
    return fig

run_plot_autocorrelograms(corr, bin_size=bin_size, window=window)
# %%
