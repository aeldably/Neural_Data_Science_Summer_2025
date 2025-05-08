# %% [markdown]
# _Neural Data Science_
# 
# Lecturer: Dr. Jan Lause, Prof. Dr. Philipp Berens
# 
# Tutors: Jonas Beck, Fabio Seel, Julius Würzler
# 
# Summer term 2025
# 
# Student names: <span style='background: yellow'>*FILL IN YOUR NAMES HERE* </span>
# 
# LLM Disclaimer: <span style='background: yellow'>*Did you use an LLM to solve this exercise? If yes, which one and where did you use it? [Copilot, Claude, ChatGPT, etc.]* </span>

# %% [markdown]
# # Coding Lab 3
# 
# ![image-3.png](attachment:image-3.png)

# %% [markdown]
# In this notebook you will work with 2 photon calcium recordings from mouse V1 
# and retina. 
#
# For details see [Chen et al. 2013](https://www.nature.com/articles/nature12354) 
# and [Theis et al. 2016](https://www.cell.com/neuron/pdf/S0896-6273(16)30073-3.pdf). 
# Two-photon imaging is widely used to study computations in populations of 
# neurons. 
# 
# In this exercise sheet we will study properties of different indicators and 
# work on methods to infer spikes from calcium traces. 
#
# All data is provided at a sampling rate of 100 Hz. 
# For easier analysis, please resample it to 25 Hz. 
#
# `scipy.signal.decimate` can help here, but note that it is only meant for continous signals. 
# 
# __Data__: Download the data file ```nds_cl_3_*.csv``` from 
# ILIAS and save it in a subfolder ```../data/```. 
#
# Note, some recordings were of shorter duration, hence their 
# columns are padded. 
#
# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from __future__ import annotations

%matplotlib inline

%load_ext jupyter_black

%load_ext watermark
%watermark --time --date --timezone --updated --python --iversions --watermark -p sklearn

# %%
plt.style.use("../matplotlib_style.txt")

# %% [markdown]
# ## Load data

# %%
# ogb dataset from Theis et al. 2016 Neuron
ogb_calcium = pd.read_csv("../data/nds_cl_3_ogb_calcium.csv", header=0)
ogb_spikes = pd.read_csv("../data/nds_cl_3_ogb_spikes.csv", header=0)
print(f"[OGB] calcium: {ogb_calcium.shape}, spikes: {ogb_spikes.shape}")

# gcamp dataset from Chen et al. 2013 Nature
gcamp_calcium = pd.read_csv("../data/nds_cl_3_gcamp2_calcium.csv", header=0)
gcamp_spikes = pd.read_csv("../data/nds_cl_3_gcamp2_spikes.csv", header=0)
print(f"[GCaMP] calcium: {gcamp_calcium.shape}, spikes: {gcamp_spikes.shape}")

# spike dataframe
ogb_spikes.head()

# %% [markdown]
# ## Task 1: Visualization of calcium and spike recordings
# 
# We start again by plotting the raw data - calcium and spike traces in 
# this case. One dataset has been recorded using the synthetic calcium 
# indicator OGB-1 at population imaging zoom 
# (~100 cells in a field of view) and the other one using the 
# genetically encoded indicator GCamp6f zooming in on individual cells. 
#
# Plot the traces of an example cell from each dataset to show how spikes 
# and calcium signals are related. A good example cell for the OGB-dataset 
# is cell 5. For the CGamp-dataset a good example is cell 6. Align the traces 
# by eye (add a small offset to the plot) such that a valid comparison is 
# possible and zoom in on a small segment of tens of seconds.
# 
# *Grading: 3 pts*
# %%
# --------------------------------
# Resample and prepare data (1 pt)
# --------------------------------
import scipy.signal

sampling_rate = 100  # Hz
# Resample to 25 Hz
new_sampling_rate = 25  # Hz

# Resample the data
resampled_ogb_calcium = scipy.signal.decimate(
    ogb_calcium.iloc[:, 1:].values, sampling_rate // new_sampling_rate, axis=0
)

resampled_gcamp_calcium = scipy.signal.decimate(
    gcamp_calcium.iloc[:, 1:].values, sampling_rate // new_sampling_rate, axis=0
)

# Resample the spikes
fig, axs = plt.subplots(
    2, 2, figsize=(9, 5), height_ratios=[3, 1], layout="constrained"
)
#%% Resample the spikes
def bin_spikes(spike_array, factor: int) -> np.ndarray:
    """Downsample spikes by summing over non-overlapping bins.

    Parameters
    ----------
    spike_df : pd.DataFrame
        Original spike data at high resolution (100 Hz).
    factor : int
        Downsampling factor (e.g. 4 for 100 Hz -> 25 Hz).

    Returns
    -------
    np.ndarray
        Binned spike matrix (n_bins, n_cells).
    """
    print(f"Original shape: {spike_array.shape}")
    # spike_array = spike_df.iloc[:, 1:].values  # Drop time/index column if present
    n_bins = spike_array.shape[0] // factor
    n_cells = spike_array.shape[1]
    spike_array = spike_array[:n_bins * factor]  # Trim to full bins
    binned = spike_array.reshape(n_bins, factor, n_cells).sum(axis=1)
    return binned

#%%
# Downsample factor: 100 Hz → 25 Hz → factor = 4
downsample_factor = sampling_rate // new_sampling_rate

# Binned spike arrays
resampled_ogb_spikes = bin_spikes(ogb_spikes.iloc[:,1:].values, downsample_factor)
resampled_gcamp_spikes = bin_spikes(gcamp_spikes.iloc[:, 1:].values, downsample_factor)


# Check shape
print(f"Binned OGB spikes shape: {resampled_ogb_spikes.shape}")
print(f"Binned GCaMP spikes shape: {resampled_gcamp_spikes.shape}")

#%%
# --------------------
# Plot OGB data (1 pt)
# --------------------
# zoom 10 seconds of ogb resampled data
ogb_cell = 5
ogb_start = 1000
ogb_end = ogb_start + 10 * new_sampling_rate
time = np.arange(ogb_start, ogb_end) / new_sampling_rate

assert len(time) == 10 * new_sampling_rate # 10 seconds

print(f"Duration of resampled OGB calcium: {len(time) / new_sampling_rate} seconds")
print(f"Number of samples in OGB calcium: {len(resampled_ogb_calcium[ogb_cell, ogb_start:ogb_end])}")

assert len(resampled_ogb_calcium[ogb_start:ogb_end, ogb_cell]) == 10 * new_sampling_rate # 10 seconds

axs[0, 0].plot(
    np.arange(ogb_start, ogb_end) / new_sampling_rate,
    resampled_ogb_calcium[ogb_start:ogb_end, ogb_cell],
    label=f"OGB-1, Cell {ogb_cell}",
)
# Plot spikes
# TODO: Need to make sure the spikes are aligned with the calcium signal.
axs[1, 0].plot(
    np.arange(ogb_start, ogb_end) / new_sampling_rate,
    ogb_spikes.iloc[ogb_start:ogb_end, ogb_cell],
    label=f"Spikes, Cell {ogb_cell}",
    color="orange",
    alpha=0.5,
)

# ----------------------
# Plot GCamp data (1 pt)
# ----------------------
gcamp_cell = 6
gcamp_start = 1000
gcamp_end = gcamp_start + 10 * new_sampling_rate
axs[0, 1].plot(
    np.arange(gcamp_start, gcamp_end) / new_sampling_rate,
    resampled_gcamp_calcium[gcamp_start:gcamp_end, gcamp_cell],
    label=f"GCamp6f, Cell {gcamp_cell}",
)

# Plot spikes
axs[1, 1].plot(
    np.arange(gcamp_start, gcamp_end) / new_sampling_rate,
    gcamp_spikes.iloc[gcamp_start:gcamp_end, gcamp_cell],
    label=f"Spikes, Cell {gcamp_cell}",
    color="orange",
    alpha=0.5,
)
# --------------------------------
plt.show()

# %% [markdown]
# ## Bonus Task (Optional): Calcium preprocessing
# 
# To improve the quality of the inferred spike trains, further preprocessing steps can undertaken. This includes filtering and smoothing of the calcium trace.
# 
# Implement a suitable filter and local averaging procedure as discussed in the lecture. Explain your choices and discuss how it helps!
# 
# _Grading: 1 BONUS point_
# 
# _BONUS Points do not count for this individual coding lab, but sum up to 5% of your **overall coding lab grade**. There are 4 BONUS points across all coding labs._

# %%

# %% [markdown]
# ## Task 2: Simple deconvolution
# 
# It is clear from the above plots that the calcium events 
# happen in relationship to the spikes. As a first simple 
# algorithm implement a deconvolution approach like presented in the 
# lecture in the function `deconv_ca`. 
#
# Assume an exponential kernel where the decay constant depends on the 
# indicator ($\tau_{OGB}= 0.5 s$, $\tau_{GCaMP}= 0.1 s$). 
#
# Note there can be no negative rates! Plot the kernel as 
# well as an example cell with true and deconvolved spike rates. 
# 
# Scale the signals such as to facilitate comparisons. You can use functions from `scipy` for this. Explain your results and your choice of kernel.
# 
# *Grading: 6 pts*
# 
#%%
def deconv_ca(ca: np.ndarray, tau: float, dt: float) -> np.ndarray:
    """Compute the deconvolution of the calcium signal.

    Parameters
    ----------

    ca: np.array, (n_points,)
        Calcium trace

    tau: float
        decay constant of conv kernel

    dt: float
        sampling interval.

    Return
    ------

    sp_hat: np.array
    """
    # --------------------------------------------
    # Apply de-convolution to calcium signal (1 pt)
    # --------------------------------------------
    # Define kernel duration to cover 5*tau
    kernel_len = int(np.ceil(5 * tau / dt))
    t = np.arange(kernel_len) * dt
    
    # Create exponential decay kernel
    kernel = np.exp(-t / tau)
    
    # Normalize kernel to have unit area
    kernel /= kernel.sum()
    
    # Use scipy to deconvolve
    sp_hat, _ = signal.deconvolve(ca, kernel)

    # Pad the output to match original size (deconvolve returns shorter output)
    sp_hat = np.pad(sp_hat, (0, ca.shape[0] - sp_hat.shape[0]), mode="constant")

    # Clip negative values
    sp_hat = np.clip(sp_hat, 0, None)
    
    return sp_hat

#%%
def run_deconvolution(calcium: np.ndarray, 
                      tau: float, dt: float) -> np.ndarray:
    """
    Run deconvolution on calcium data for all cells.

    Parameters
    ----------
    calcium : np.ndarray
        Calcium data of shape (time, n_cells).
    tau : float
        Decay constant for the algorithm.
    dt : float
        Sampling interval.

    Returns
    -------
    np.ndarray
        Inferred spike data of shape (time, n_cells).
    """
    inferred_spikes = np.zeros_like(calcium)
    for cell in range(calcium.shape[1]):  # Iterate over each cell
        inferred_spikes[:, cell] = deconv_ca(calcium[:, cell], tau=tau, dt=dt)
    return inferred_spikes

# %%
# -------------------------
# Plot the 2 kernels (1 pt)
# -------------------------
fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
# Plotting exponential kernels for OGB and GCaMP
dt = 1 / new_sampling_rate  # 0.04 s
t_max = 2.0  # seconds
t = np.arange(0, t_max, dt)

tau_ogb = 0.5
tau_gcamp = 0.1

kernel_ogb = np.exp(-t / tau_ogb)
kernel_ogb /= kernel_ogb.sum()

kernel_gcamp = np.exp(-t / tau_gcamp)
kernel_gcamp /= kernel_gcamp.sum()

plt.figure(figsize=(6, 4))
plt.plot(t, kernel_ogb, label="OGB-1 Kernel (τ = 0.5s)")
plt.plot(t, kernel_gcamp, label="GCaMP6f Kernel (τ = 0.1s)")
plt.xlabel("Time (s)")
plt.ylabel("Normalized kernel")
plt.title("Exponential Decay Kernels")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ### Questions (1 pt)
# 1) Explain how you constructed the kernels
# 
# YOUR ANSWER HERE
# 
# 2) How do the indicators / kernels compare?
# 
# YOUR ANSWER HERE
# 
# 3) What are pros and cons of each indicator?
# 
# YOUR ANSWER HERE

# %%
# ----------------------------------------------------------------------
# Compare true and deconvolved spikes rates for the OGB and GCamP cells.
# What do you notice? Why is that? (3 pts)
# ----------------------------------------------------------------------
# Sampling parameters
dt = 1 / new_sampling_rate  # 0.04 s

# OGB example: cell 5
cell_ogb = 5
tau_ogb = 0.5  # seconds
start = 1000
end = start + 100 * new_sampling_rate

# Calcium and ground truth
ca_ogb = resampled_ogb_calcium[start:end, cell_ogb]
true_spikes_ogb = resampled_ogb_spikes[start:end, cell_ogb] 

# Run deconvolution
sp_hat_ogb = deconv_ca(ca_ogb, tau=tau_ogb, dt=dt)

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(8, 5), sharex=True)
time = np.arange(start, end) / new_sampling_rate

axs[0].plot(time, ca_ogb, label="Calcium")
axs[0].set_ylabel("Ca")
axs[0].legend()

axs[1].plot(time, sp_hat_ogb, label="Deconv. spikes", color="green")
axs[1].set_ylabel("Deconv")
axs[1].legend()

axs[2].plot(time, true_spikes_ogb, label="True spikes", color="orange", alpha=0.6)
axs[2].set_ylabel("True")
axs[2].set_xlabel("Time (s)")
axs[2].legend()

plt.suptitle("OGB - Cell 5: Calcium, Deconv. Spikes, Ground Truth")
plt.tight_layout()
plt.savefig("../plots/ogb_deconv.png", dpi=300)
plt.show()


#%%
# GCaMP example: cell 6
cell_gcamp = 6
tau_gcamp = 0.1  # seconds

ca_gcamp = resampled_gcamp_calcium[start:end, cell_gcamp]
true_spikes_gcamp = resampled_gcamp_spikes[start:end, cell_gcamp] 

# Run deconvolution
sp_hat_gcamp = deconv_ca(ca_gcamp, tau=tau_gcamp, dt=dt)

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(8, 5), sharex=True)
time = np.arange(start, end) / new_sampling_rate

axs[0].plot(time, ca_gcamp, label="Calcium")
axs[0].set_ylabel("Ca")
axs[0].legend()

axs[1].plot(time, sp_hat_gcamp, label="Deconv. spikes", color="green")
axs[1].set_ylabel("Deconv")
axs[1].legend()

axs[2].plot(time, true_spikes_gcamp, label="True spikes", color="orange", alpha=0.6)
axs[2].set_ylabel("True")
axs[2].set_xlabel("Time (s)")
axs[2].legend()

plt.suptitle("GCaMP6f - Cell 6: Calcium, Deconv. Spikes, Ground Truth")
plt.tight_layout()
plt.savefig("../plots/gcamp_deconv.png", dpi=300)
plt.show()

#%% [markdown]
# ## Task 3: Run more complex algorithm
# 
# As reviewed in the lecture, a number of more complex algorithms for inferring spikes from calcium traces have been developed. Run an implemented algorithm on the data and plot the result. There is a choice of algorithms available, for example:
# 
# * Vogelstein: [oopsi](https://github.com/liubenyuan/py-oopsi)
# * Theis: [c2s](https://github.com/lucastheis/c2s)
# * Friedrich: [OASIS](https://github.com/j-friedrich/OASIS)
# 
# *Grading: 3 pts*
# 
# 

# %%
# run this cell to download the oopsi.py file and put it in the 
# same folder as this notebook
!wget https://raw.githubusercontent.com/liubenyuan/py-oopsi/master/oopsi.py
import oopsi

# %%
# ----------------------------------------------------------------------
# Apply one of the advanced algorithms to the OGB and GCamp Cells (1 pt)
# ----------------------------------------------------------------------

# %%
# -------------------------------------------------------------------------------
# Plot the results for the OGB and GCamp Cells and describe the results (1+1 pts)
# -------------------------------------------------------------------------------

fig, axs = plt.subplots(
    3, 1, figsize=(6, 4), height_ratios=[1, 1, 1], gridspec_kw=dict(hspace=0)
)

# OGB Cell

fig, axs = plt.subplots(
    3, 1, figsize=(6, 4), height_ratios=[1, 1, 1], gridspec_kw=dict(hspace=0)
)

# GCamP Cell

# %% [markdown]
# ## Task 4: Evaluation of algorithms
# 
# To formally evaluate the algorithms on the two datasets run the 
# deconvolution algorithm and the more complex one on all cells and 
# compute the correlation between true and inferred spike trains. 
# `DataFrames` from the `pandas` package are a useful tool for 
# aggregating data and later plotting it. Create a dataframe 
# with columns.
# 
# * algorithm
# * correlation
# * indicator
# 
# and enter each cell. Plot the results using `stripplot` and/or `boxplot` 
# in the `seaborn` package. Note these functions provide useful options for 
# formatting the plots. See their documentation, i.e. `sns.boxplot?`.
# 
# *Grading: 5 pts*
pd.set_option("display.max_columns", None)
eval_results = pd.DataFrame(
    {
        "algorithm": ["deconv", "oopsi"],
        "correlation": [None, None],
        "indicator": ["OGB", "GCaMP"],
    }
)


# %% [markdown]
# First, evaluate on OGB data and create OGB dataframe. 
# Then repeat for GCamp and combine the two dataframes.
def evaluate_algorithm(
    calcium: np.ndarray,
    spikes: np.ndarray,
    algorithm: str,
    tau: float,
    dt: float,
    indicator: str  # Add this to label which dataset (OGB or GCaMP)
) -> pd.DataFrame:
    """
    Evaluate the algorithm on calcium and spike data for all cells.

    Parameters
    ----------
    calcium : np.ndarray
        Calcium data of shape (time, n_cells).
    spikes : np.ndarray
        Binned spike data of shape (time, n_cells).
    algorithm : str
        Algorithm to use ("deconv" or "oopsi").
    tau : float
        Decay constant for the algorithm.
    dt : float
        Sampling interval.
    indicator : str
        Indicator label (e.g. "OGB" or "GCaMP").

    Returns
    -------
    pd.DataFrame
        DataFrame with correlation results for each cell.
    """
    # Run the algorithm
    if algorithm == "deconv":
        inferred_spikes = run_deconvolution(calcium, tau=tau, dt=dt)
    elif algorithm == "oopsi":
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Ensure same shape
    assert inferred_spikes.shape == spikes.shape, \
        f"Shape mismatch: inferred {inferred_spikes.shape}, true {spikes.shape}"
    
    # Compute correlation for each cell
    results = []
    for cell in range(spikes.shape[1]):
        true_cell = spikes[:, cell]
        inferred_cell = inferred_spikes[:, cell]

        # Optional: handle NaNs if any
        if np.isnan(true_cell).any() or np.isnan(inferred_cell).any():
            corr = np.nan
        else:
            corr = np.corrcoef(true_cell, inferred_cell)[0, 1]

        results.append({
            "algorithm": algorithm,
            "correlation": corr,
            "indicator": indicator,
            "cell": cell
        })

    return pd.DataFrame(results)
# %%
# ----------------------------------------------------------
# Evaluate the algorithms on the OGB and GCamp cells (2 pts)
# ----------------------------------------------------------
# Ensure dt, tau_ogb, and tau_gcamp are defined (likely from Task 2)
# List to store individual DataFrame results
all_results_list = []

algorithms_to_evaluate = ["deconv"]
# If you implement oopsi and want to evaluate it, add "oopsi" to the list:
# algorithms_to_evaluate = ["deconv", "oopsi"]

for alg in algorithms_to_evaluate:
    for indicator, calcium_data, spike_data in zip(
        ["OGB", "GCaMP"],
        [resampled_ogb_calcium, resampled_gcamp_calcium],
        [resampled_ogb_spikes, resampled_gcamp_spikes]
    ):
        print(f"\nEvaluating Algorithm: '{alg}' for Indicator: '{indicator}'")
        print(f"Initial calcium data shape: {calcium_data.shape}")
        print(f"Initial spike data shape: {spike_data.shape}")

        # Determine the correct tau for the current indicator
        current_tau = tau_ogb if indicator == "OGB" else tau_gcamp

        # Ensure calcium and spike data have the same number of time points (rows)
        # This can be important if decimation and binning led to slight length differences
        min_rows = min(calcium_data.shape[0], spike_data.shape[0])
        
        aligned_calcium = calcium_data[:min_rows, :]
        aligned_spikes = spike_data[:min_rows, :]
        
        print(f"Aligned calcium shape for evaluation: {aligned_calcium.shape}")
        print(f"Aligned spikes shape for evaluation: {aligned_spikes.shape}")

        if alg == "oopsi":
            # Special handling for oopsi if its input requirements differ
            # or if it processes one cell at a time differently.
            # The provided oopsi.py might take (Time,) for calcium.
            # For now, this example assumes evaluate_algorithm handles it.
            # If oopsi.oopsi itself handles multiple cells, great.
            # If not, you might need a loop inside evaluate_algorithm for oopsi too,
            # or call it cell by cell here.
            
            # Example: if oopsi.oopsi needs 1D calcium array per cell
            inferred_spikes_algo = np.zeros_like(aligned_calcium)
            if aligned_calcium.shape[1] > 0 : # Check if there are cells
                 for cell_idx in range(aligned_calcium.shape[1]):
                    # Note: The oopsi function from the master branch of py-oopsi
                    # seems to take F (calcium trace, 1d), and dt. Tau is estimated.
                    # You might need to adjust how tau is passed or used if using that version.
                    # For this exercise, we are following the structure of deconv_ca.
                    # If your oopsi function in evaluate_algorithm is set up like deconv_ca, this is fine.
                    if 'oopsi' in globals() and hasattr(oopsi, 'oopsi'): # Check if oopsi is imported and has the function
                        # This is a placeholder, adapt if your oopsi function works differently
                        # The evaluate_algorithm function already loops through cells for "oopsi"
                        pass # evaluate_algorithm will handle it based on its internal logic
                    else:
                        print(f"OOPSI algorithm selected, but 'oopsi.oopsi' function not available. Skipping OOPSI for {indicator}.")
                        continue 
            else:
                print(f"No cells found for {indicator}. Skipping OOPSI.")
                continue
        # Call the evaluation function
        df_result = evaluate_algorithm(
            calcium=aligned_calcium,
            spikes=aligned_spikes,
            algorithm=alg,
            tau=current_tau,
            dt=dt,
            indicator=indicator
        )

        all_results_list.append(df_result)

# Concatenate all results into the final DataFrame
if all_results_list:
    eval_results_df = pd.concat(all_results_list, ignore_index=True)
else:
    eval_results_df = pd.DataFrame() # Create an empty DataFrame if no results

print("\n--- Final Evaluation DataFrame ---")
print(eval_results_df)

# Now eval_results_df is ready for Task 4 plotting
# %%
# ----------------------------------------------------------
# Evaluate the algorithms on the OGB and GCamp cells (2 pts)
# ----------------------------------------------------------
# Run the evaluation. 

# %%
# -------------------------------
# Construct the dataframe (1 pts)
# -------------------------------

# %% [markdown]
# Combine both dataframes. Plot the performance of each indicator and algorithm. 
# You should only need a single plot for this.
# %%
# ----------------------------------------------------------------------------
# Create Strip/Boxplot for both cells and algorithms Cell as described. (1 pt)
# Describe and explain the results briefly. (1 pt)
# ----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
# %%
# ----------------------------------------------------------------------------
# Create Strip/Boxplot for both cells and algorithms Cell as described. (1 pt)
# Describe and explain the results briefly. (1 pt)
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 7), layout="constrained") # Increased figure size for better readability
# Check if the DataFrame is empty or if the 'correlation' column is all NaN
if eval_results_df.empty or eval_results_df['correlation'].isnull().all():
    ax.text(0.5, 0.5, "No valid correlation data to plot.",
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, fontsize=12, color='red')
    ax.set_title("Evaluation of Spike Inference Algorithms")
    ax.set_xlabel("Indicator")
    ax.set_ylabel("Correlation")
else:
    # Create a combined boxplot and stripplot
    sns.boxplot(
        x="indicator",
        y="correlation",
        hue="algorithm",  # Even if only one algorithm, hue makes it expandable
        data=eval_results_df,
        ax=ax,
        palette="Set2",
        fliersize=0 # Hide outlier markers from boxplot as stripplot will show all points
    )
    sns.stripplot(
        x="indicator",
        y="correlation",
        hue="algorithm", # Even if only one algorithm, hue makes it expandable
        data=eval_results_df,
        ax=ax,
        dodge=True, # Dodge points along the categorical axis for hue
        jitter=True, # Add some jitter to prevent points from overlapping
        alpha=0.7,
        palette="Set2",
        linewidth=0.5,
        edgecolor='gray'
    )

    ax.set_title("Performance of Spike Inference Algorithms")
    ax.set_xlabel("Calcium Indicator")
    ax.set_ylabel("Correlation (Inferred vs. True Spikes)")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Improve legend if there are multiple algorithms, or simplify if only one
    handles, labels = ax.get_legend_handles_labels()
    # Depending on how many algorithms, handles/labels might be duplicated.
    # Take unique labels. For stripplot and boxplot, hue creates two sets of legend items.
    # We only need one set for the legend.
    if handles: # Check if there are any legend handles
        # A common way to get unique legend items when hue is used for both box and strip plots
        # is to take the first half (or second half) if they are duplicated.
        # For a single algorithm, this also works.
        unique_handles = handles[:len(handles)//2]
        unique_labels = labels[:len(labels)//2]
        if not unique_handles: # If the above logic fails (e.g. only one type of plot had hue)
            unique_handles = handles
            unique_labels = labels
        ax.legend(unique_handles, unique_labels, title="Algorithm", loc="best")
    else:
        # If no legend handles (e.g., no 'hue' used or no data), remove any potential empty legend
        if ax.get_legend() is not None:
            ax.get_legend().remove()

plt.savefig("../plots/evaluation_results.png", dpi=300)
plt.show()

print("\n--- Final Evaluation DataFrame Used for Plot ---")
print(eval_results_df)
# %%
