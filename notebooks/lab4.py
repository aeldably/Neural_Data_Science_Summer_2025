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
# # Coding Lab 4
# 
# In this notebook you will work with preprocessed 2 photon calcium recordings, that have already been converted into spike counts for a population of cells from the Macaque V1. During the experiment the animal has been presented with several drifting grating stimuli, in response to which the neural activity was recorded. In this exercise sheet we will study how you can visualize the activity of multiple neural spike trains and assess whether a neuron is selective to a specific stimulus type.
# 
# Download the data files ```nds_cl_4_*.csv``` from ILIAS and save it in the subfolder ```../data/```. We recommend you to use a subset of the data for testing and debugging, ideally focus on a single cell (e.g. cell number x). The spike times and stimulus conditions are read in as pandas data frames. You can solve the exercise by making heavy use of that, allowing for many quite compact computations. See [documentation](http://pandas.pydata.org/pandas-docs/stable/index.html) and several good [tutorials](https://www.datacamp.com/community/tutorials/pandas-tutorial-dataframe-python#gs.L37i87A) on how to do this. Of course, converting the data into classical numpy arrays is also valid.

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.optimize as opt

from scipy import signal as signal
from typing import Tuple

import itertools

#%matplotlib inline

#%load_ext jupyter_black

#%load_ext watermark
#%watermark --time --date --timezone --updated --python --iversions --watermark -p sklearn

# %%
#plt.style.use("../../matplotlib_style.txt")

# %% [markdown]
# ## Load data

# %%
import os
os.chdir("./notebooks")
#%%
spikes = pd.read_csv("../data/nds_cl_4_spiketimes.csv")  # neuron id, spike time
stims = pd.read_csv("../data/nds_cl_4_stimulus.csv")  # stimulus onset in ms, direction

stimDur = 2000.0  # in ms
nTrials = 11  # number of trials per condition
nDirs = 16  # number of conditions
deltaDir = 22.5  # difference between conditions

stims["StimOffset"] = stims["StimOnset"] + stimDur

# %% [markdown]
# We require some more information about the spikes for 
# the plots and analyses we intend to make later. With a 
# solution based on dataframes, it is natural to compute 
# this information here and add it as additional columns 
# to the `spikes` dataframe by combining it with the 
# `stims` dataframe. 
#
# We later need to know which condition (`Dir`) and trial (`Trial`) a spike 
# was recorded in, the relative spike times compared 
# to stimulus onset of the stimulus it was recorded in 
# (`relTime`) and whether a spike was during the stimulation period 
# (`stimPeriod`). 
#
# But there are many options how to solve this exercise 
# and you are free to choose any of them.

# %%
# you may add computations as specified above
spikes["Dir"] = np.nan
spikes["relTime"] = np.nan
spikes["Trial"] = np.nan
spikes["stimPeriod"] = np.nan

#%%
dirs = np.unique(stims["Dir"])
trialcounter = np.zeros_like(dirs)

for i, row in stims.iterrows():
    trialcounter[dirs == row["Dir"]] += 1

    i0 = spikes["SpikeTimes"] > row["StimOnset"]
    i1 = spikes["SpikeTimes"] < row["StimOffset"]

    select = i0.values & i1.values

    spikes.loc[select, "Dir"] = row["Dir"]
    spikes.loc[select, "Trial"] = trialcounter[dirs == row["Dir"]][0]
    spikes.loc[select, "relTime"] = spikes.loc[select, "SpikeTimes"] - row["StimOnset"]
    spikes.loc[select, "stimPeriod"] = True

spikes = spikes.dropna()

# %%
spikes.head()
# %%
# I want to group the spikes by trial and direction
# and count the number of spikes in each trial
# I will use the groupby function to do this
spk_by_dir = (
    spikes.groupby(["Dir", "Trial"])["stimPeriod"]
    .sum()
    .astype(int)
    .reset_index()
)

#%%
# %% [markdown]
# ## Task 1: Plot spike rasters
# 
# In a raster plot, each spike is shown by a small tick at 
# the time it occurs relative to stimulus onset. 
# Implement a function `plotRaster()` that plots the spikes 
# of one cell as one trial per row, sorted by conditions 
# (similar to what you saw in the lecture). 
# Why are there no spikes in some conditions and 
# many in others?
# 
# If you opt for a solution without a dataframe, 
# you need to change the interface of the function.
# 
# *Grading: 3 pts*
# 
# %%
def plotRaster(spikes: pd.DataFrame, neuron: int):
    """plot spike rasters for a single neuron sorted by condition

    Parameters
    ----------

    spikes: pd.DataFrame
        Pandas DataFrame with columns
            Neuron | SpikeTimes | Dir | relTime | Trial | stimPeriod

    neuron: int
        Neuron ID


    Note
    ----

    this function does not return anything, it just creates a plot!
    """
    # -------------------------------------------------
    # Write a raster plot function for the data (2 pts)
    # -------------------------------------------------
    df = spikes[(spikes["Neuron"] == neuron) & (spikes["stimPeriod"] == True)]
    dir_order = np.sort(df["Dir"].unique())

    plt.figure(figsize=(10, 6))
    trial_offset = 0
    yticks = []
    yticklabels = []

    for dir_val in dir_order:
        dir_df = df[df["Dir"] == dir_val]
        trials = np.sort(dir_df["Trial"].unique())

        for trial in trials:
            trial_df = dir_df[dir_df["Trial"] == trial]
            plt.vlines(trial_df["relTime"], trial_offset, trial_offset + 1, color="black", linewidth=0.5)
            trial_offset += 1

        # Add a horizontal line to separate conditions
        plt.axhline(y=trial_offset, color="gray", linestyle="--", linewidth=0.5)

        # Record middle of block for yticks
        yticks.append(trial_offset - len(trials) / 2)
        yticklabels.append(f"{int(dir_val)}°")

    plt.xlabel("Time relative to stimulus onset (ms)")
    plt.ylabel("Trials grouped by direction")
    plt.yticks(yticks, yticklabels)
    plt.title(f"Spike Raster Plot — Neuron {neuron}")
    plt.tight_layout()
    plt.show()

#%%
for i in range(40):
    plotRaster(spikes, i)

# %% [markdown]
# Find examples of 
# 1. A direction selective neuron
# 2. An orientation selective neuron 
# 3. Neither
# 
# And explain your choices.



# %%
# ---------------------------------
# Find and explain examples? (1 pt)
# ---------------------------------

# %% [markdown]
# ## Task 2: Plot spike density functions
# 
# Compute an estimate of the spike rate against time 
# relative to stimulus onset. 
#
# There are two ways:
#
# * Discretize time: Decide on a bin size, 
# count the spikes in each bin and average 
# across trials. 
#
# * Directly estimate the probability of spiking 
# using a density estimator with specified 
# kernel width. 
# 
# For full points, the *optimal* kernel- or bin-width 
# needs to be computed.
# 
# Implement one of them in the function `plotPSTH()`. 
#
# If you dont use a dataframe you may need to 
# change the interface of the function.
# 
# *Grading: 4 pts*
# %%
def plotPSTH(spikes: pd.DataFrame, neuron: int):
    """Plot PSTH for a single neuron sorted by 
    condition

    Parameters
    ----------

    spikes: pd.DataFrame
        Pandas DataFrame with columns
            Neuron | SpikeTimes | Dir | relTime | Trial | stimPeriod

    neuron: int
        Neuron ID


    Note
    ----

    This function does not return anything, it just 
    creates a plot!
    """
    # -------------------------------------------------
    # Implement one of the spike rate estimates (3 pts)
    # -------------------------------------------------
    for row, dir in enumerate(dirs):
        print("direction", dir)
        # ---------------------------------------------
        # Plot the obtained spike rate estimates (1 pt)
        # ---------------------------------------------
        continue
    neuron_spikes = spikes[
        (spikes["Neuron"] == neuron) & (spikes["stimPeriod"] == True)
    ]
    if neuron_spikes.empty:
        print(f"No spikes found for neuron {neuron} during stimulation periods.")
        return

    unique_dirs = np.sort(neuron_spikes["Dir"].unique())
    n_dirs = len(unique_dirs)
    n_trials_per_dir = spikes.groupby(["Neuron", "Dir"])["Trial"].nunique().loc[neuron].max()

    stim_dur_ms = stimDur
    bin_width_ms = 50  # Set bin width in milliseconds
    # Determine the number of bins
    bins = np.arange(0, stim_dur_ms + bin_width_ms, bin_width_ms)
    bin_centers = bins[:-1] + bin_width_ms / 2

    # Create subplots: one row per direction, or a grid
    # For many directions, a grid might be better, or overlaying plots
    # Let's try a grid layout
    n_cols = 4
    n_rows = int(np.ceil(n_dirs / n_cols))
    
    if n_dirs == 0:
        print(f"No stimulus directions found for neuron {neuron}.")
        return

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True, sharey=True
    )
    axes = axes.flatten() # Flatten in case of single row/col

    max_rate = 0 # To keep track of max rate for y-axis scaling

    for i, direction in enumerate(unique_dirs):
        ax = axes[i]
        dir_spikes = neuron_spikes[neuron_spikes["Dir"] == direction]

        # Aggregate all relative spike times for this direction
        all_rel_times_for_dir = dir_spikes["relTime"].values

        if len(all_rel_times_for_dir) > 0:
            # Create histogram of spike counts
            counts, _ = np.histogram(all_rel_times_for_dir, bins=bins)

            # Number of trials for this specific direction
            # This is important if some directions had fewer trials recorded or if trials are missing spikes
            current_n_trials = dir_spikes["Trial"].nunique()
            if current_n_trials == 0 and len(all_rel_times_for_dir) > 0:
                 # This case should ideally not happen if data is consistent
                 # but as a fallback if somehow trials are not correctly numbered for all spikes.
                 # A better way is to use the global nTrials if it's guaranteed to be constant.
                 # For now, let's use the known nTrials constant
                 current_n_trials = nTrials


            # Convert to rate: counts / (num_trials * bin_width_seconds)
            rate = counts / (current_n_trials * (bin_width_ms / 1000.0))
            max_rate = max(max_rate, np.max(rate))

            ax.plot(bin_centers, rate, linestyle="-", marker="")
            # ax.bar(bin_centers, rate, width=bin_width_ms, align='center') # Alternative: bar plot

        ax.set_title(f"Direction {int(direction)}°")
        ax.axvspan(0, stim_dur_ms, color="gray", alpha=0.2, label="Stimulus ON") # Mark stimulus period
        if i >= (n_rows -1) * n_cols : # Only x-label for bottom plots
             ax.set_xlabel("Time relative to stimulus onset (ms)")
        if i % n_cols == 0: # Only y-label for left-most plots
            ax.set_ylabel("Firing Rate (spikes/s)")


    # Post-loop adjustments
    for i in range(n_dirs): # Set ylim for all relevant axes
        axes[i].set_ylim(0, max_rate * 1.1 if max_rate > 0 else 1) # Add 10% margin
        axes[i].grid(True, linestyle=":", alpha=0.7)


    # Hide unused subplots
    for i in range(n_dirs, len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle(f"PSTH for Neuron {neuron} (Bin Width: {bin_width_ms} ms)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show()
    
plotPSTH(spikes, 1)
# %% [markdown]
# Plot the same 3 examples you selected in Task 1

# %%


# %% [markdown]
# ## Task 3: Fit and plot tuning functions
# 
# The goal is to visualize the activity of each neuron as a function of stimulus direction. 
# First, compute the spike counts of each neuron for each direction of motion and trial.  
# The result should be a matrix `x`, where $x_{jk}$ represents the spike count of the $j$-th response to the $k$-th direction of motion (i.e. each column contains the spike counts for all trials with one direction of motion).	If you used dataframes above, the `groupby()` function allows to implement this very compactly. Make sure you don't loose trials with zero spikes though. Again, other implementations are completely fine.
# 
# Fit the tuning curve, i.e. the average spike count per direction, using a von Mises model. To capture the non-linearity and direction selectivity of the neurons, we will fit a modified von Mises function:
# 
# $$ f(\theta) = \exp(\alpha + \kappa (\cos (2*(\theta-\phi))-1) + \nu (\cos (\theta-\phi)-1))$$
# 
# Here, $\theta$ is the stimulus direction. Implement the von Mises function in `vonMises()` and plot it to understand how to interpret its parameters $\phi$, $\kappa$, $\nu$, $\alpha$. Perform a non-linear least squares fit using a package/function of your choice. Implement the fitting in `tuningCurve()`. 
# 
# Plot the average number of spikes per direction, the spike counts from individual trials as well as your optimal fit.
# 
# Select two cells that show nice tuning to test your code.
# 
# *Grading: 5 pts*

# %%
def vonMises(θ: np.ndarray, α: float, κ: float, ν: float, ϕ: float) -> np.ndarray:
    """Evaluate the parametric von Mises tuning curve with parameters p at locations theta.

    Parameters
    ----------

    θ: np.array, shape=(N, )
        Locations. The input unit is degree.

    α, κ, ν, ϕ : float
        Function parameters

    Return
    ------
    f: np.array, shape=(N, )
        Tuning curve.
    """

    # -----------------------------------
    # Implement the Mises model (0.5 pts)
    # -----------------------------------

    pass

# %% [markdown]
# Plot the von Mises function while varying the parameters systematically.

# %%
# ------------------------------------------------------------------------------
# plot von Mises curves with varying parameters and explain what they do (2 pts)
# ------------------------------------------------------------------------------

# %%
def tuningCurve(counts: np.ndarray, dirs: np.ndarray, show: bool = True) -> np.ndarray:
    """Fit a von Mises tuning curve to the spike counts in count with direction dir using a least-squares fit.

    Parameters
    ----------

    counts: np.array, shape=(total_n_trials, )
        the spike count during the stimulation period

    dirs: np.array, shape=(total_n_trials, )
        the stimulus direction in degrees

    show: bool, default=True
        Plot or not.


    Return
    ------
    p: np.array or list, (4,)
        parameter vector of tuning curve function
    """

    # ----------------------------------------
    # Compute the spike count matrix (0.5 pts)
    # ----------------------------------------

    # ------------------------------------------------------------
    # fit the von Mises tuning curve to the spike counts (0.5 pts)
    # ------------------------------------------------------------


    if show:
        # --------------------------------------------
        # plot the data and fitted tuning curve (1 pt)
        # --------------------------------------------
        pass

# %% [markdown]
# Plot tuning curve and fit for different neurons. Good candidates to check are 28, 29 or 37. 
# %%
def get_data(spikes, neuron):
    spk_by_dir = (
        spikes[spikes["Neuron"] == neuron]
        .groupby(["Dir", "Trial"])["stimPeriod"]
        .sum()
        .astype(int)
        .reset_index()
    )

    dirs = spk_by_dir["Dir"].values
    counts = spk_by_dir["stimPeriod"].values

    # because we count spikes only when they are present, some zero entries in the count vector are missing
    for i, Dir in enumerate(np.unique(spikes["Dir"])):
        m = nTrials - np.sum(dirs == Dir)
        if m > 0:
            dirs = np.concatenate((dirs, np.ones(m) * Dir))
            counts = np.concatenate((counts, np.zeros(m)))

    idx = np.argsort(dirs)
    dirs_sorted = dirs[idx]  # sorted dirs
    counts_sorted = counts[idx]

    return dirs_sorted, counts_sorted

# %%
# ----------------------------------------------------------
# plot the average number of spikes per direction, the spike 
# counts from individual trials as well as your optimal fit 
# for different neurons (0.5 pts)
# ----------------------------------------------------------

# %% [markdown]
# ## Task 4: Permutation test for direction tuning
# 
# Implement a permutation test to quantitatively assess whether a 
# neuron is direction/orientation selective. 
# 
# To do so, project the vector of average spike counts, 
#       $m_k=\frac{1}{N}\sum_j x_{jk}$ 
# on a complex exponential with two cycles, 
#       $v_k = \exp(\psi i \theta_k)$, 
# where $\theta_k$ is the $k$-th direction of motion in radians and 
# $\psi \in 1,2$ is the fourier component to test 
# (1: direction, 2: orientation). 
#
# Denote the projection by $q=m^Tv$. The magnitude $|q|$ tells you how 
# much power there is in the $\psi$-th fourier component. 
# 
# Estimate the distribution of |q| under the null hypothesis that the neuron fires randomly across directions by running 1000 iterations where you repeat the same calculation as above but on a random permutation of the trials (that is, randomly shuffle the entries in the spike count matrix x). The fraction of iterations for which you obtain a value more extreme than what you observed in the data is your p-value. Implement this procedure in the function ```testTuning()```. 
# 
# Illustrate the test procedure for one of the cells from above. Plot the sampling distribution of |q| and indicate the value observed in the real data in your plot. 
# 
# How many cells are tuned at p < 0.01?
# 
# *Grading: 3 pts*
# 

# %%
def testTuning(
    counts: np.ndarray,
    dirs: np.ndarray,
    psi: int = 1,
    niters: int = 1000,
    show: bool = False,
    random_seed: int = 2046,
) -> Tuple[float, float, np.ndarray]:
    """Plot the data if show is True, otherwise just return the fit.

    Parameters
    ----------

    counts: np.array, shape=(total_n_trials, )
        the spike count during the stimulation period

    dirs: np.array, shape=(total_n_trials, )
        the stimulus direction in degrees

    psi: int
        fourier component to test (1 = direction, 2 = orientation)

    niters: int
        Number of iterations / permutation

    show: bool
        Plot or not.

    random_seed: int
        Random seed for reproducibility.

    Returns
    -------
    p: float
        p-value
    q: float
        magnitude of second Fourier component

    qdistr: np.array
        sampling distribution of |q| under the null hypothesis

    """
    
    # -------------------------------
    # calculate m, nu and q (0.5 pts)
    # -------------------------------

    # -------------------------------------------------------------------------
    # Estimate the distribution of q under the H0 and obtain the p value (1 pt)
    # -------------------------------------------------------------------------
    # ensure reproducibility using a random number generator
    # hint: access random functions of this generator
    rng = np.random.default_rng(random_seed)


    if show:
        # add plotting code here
        pass

# %% [markdown]
# Show null distribution for the example cell:

# %%
# ---------------------------------------------------------
# Plot null distributions for example cells 28 & 29. (1 pt)
# ---------------------------------------------------------

# %% [markdown]
# Test all cells for orientation and direction tuning

# %%
# --------------------------------------------------
# Test all cells for orientation / direction tuning. 
# Which ones are selective? (0.5 pts)
# --------------------------------------------------

# %% [markdown]
# Number of direction tuned neurons:

# %%


# %% [markdown]
# Number of orientation tuned neurons:

# %%



