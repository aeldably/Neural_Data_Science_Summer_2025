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
# It is clear from the above plots that the calcium events happen in relationship to the spikes. As a first simple algorithm implement a deconvolution approach like presented in the lecture in the function `deconv_ca`. Assume an exponential kernel where the decay constant depends on the indicator ($\tau_{OGB}= 0.5 s$, $\tau_{GCaMP}= 0.1 s$). Note there can be no negative rates! Plot the kernel as well as an example cell with true and deconvolved spike rates. Scale the signals such as to facilitate comparisons. You can use functions from `scipy` for this. Explain your results and your choice of kernel.
# 
# *Grading: 6 pts*
# 

# %%
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
    # apply devonvolution to calcium signal (1 pt)
    # --------------------------------------------

    return sp_hat

# %%
# -------------------------
# Plot the 2 kernels (1 pt)
# -------------------------
fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")


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

fig, axs = plt.subplots(
    3, 1, figsize=(6, 4), height_ratios=[1, 1, 1], gridspec_kw=dict(hspace=0)
)

# OGB Cell

fig, axs = plt.subplots(
    3, 1, figsize=(6, 4), height_ratios=[1, 1, 1], gridspec_kw=dict(hspace=0)
)

# GCamp Cell


# %% [markdown]
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
# run this cell to download the oopsi.py file and put it in the same folder as this notebook
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
# To formally evaluate the algorithms on the two datasets run the deconvolution algorithm and the more complex one on all cells and compute the correlation between true and inferred spike trains. `DataFrames` from the `pandas` package are a useful tool for aggregating data and later plotting it. Create a dataframe with columns
# 
# * algorithm
# * correlation
# * indicator
# 
# and enter each cell. Plot the results using `stripplot` and/or `boxplot` in the `seaborn` package. Note these functions provide useful options for formatting the
# plots. See their documentation, i.e. `sns.boxplot?`.
# 
# *Grading: 5 pts*
# 

# %% [markdown]
# First, evaluate on OGB data and create OGB dataframe. Then repeat for GCamp and combine the two dataframes.

# %%
# ----------------------------------------------------------
# Evaluate the algorithms on the OGB and GCamp cells (2 pts)
# ----------------------------------------------------------

# %%
# -------------------------------
# Construct the dataframe (1 pts)
# -------------------------------

# %% [markdown]
# Combine both dataframes. Plot the performance of each indicator and algorithm. You should only need a single plot for this.

# %%
# ----------------------------------------------------------------------------
# Create Strip/Boxplot for both cells and algorithms Cell as described. (1 pt)
# Describe and explain the results briefly. (1 pt)
# ----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")



