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
# In this notebook you will work with preprocessed 2 
# photon calcium recordings, that have already been converted 
# into spike counts for a population of cells from the 
# Macaque V1. 
#
# During the experiment the animal has been presented with 
# several drifting grating stimuli, in response to which the 
# neural activity was recorded. 
#
# In this exercise sheet we will study how you can visualize the 
# activity of multiple neural spike trains and assess whether a 
# neuron is selective to a specific stimulus type.
# 
# Download the data files ```nds_cl_4_*.csv``` from ILIAS and save it in the subfolder ```../data/```. We recommend you to use a subset of the data for testing and debugging, ideally focus on a single cell (e.g. cell number x). The spike times and stimulus conditions are read in as pandas data frames. You can solve the exercise by making heavy use of that, allowing for many quite compact computations. See [documentation](http://pandas.pydata.org/pandas-docs/stable/index.html) and several good [tutorials](https://www.datacamp.com/community/tutorials/pandas-tutorial-dataframe-python#gs.L37i87A) on how to do this. Of course, converting the data into classical numpy arrays is also valid.

#%%
from scipy import signal as signal
from scipy.stats import gaussian_kde # For SDF example

from typing import Tuple

import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
import seaborn as sns
import os
import logging

#%%
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Starting the script...")
#%matplotlib inline

#%load_ext jupyter_black

#%load_ext watermark
#%watermark --time --date --timezone --updated --python --iversions --watermark -p sklearn

# %%
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# %%
#plt.style.use("../../matplotlib_style.txt")

# %% [markdown]
# ## Load data

#%%
spikes = pd.read_csv("../data/nds_cl_4_spiketimes.csv")  # neuron id, spike time
stims = pd.read_csv("../data/nds_cl_4_stimulus.csv")  # stimulus onset in ms, direction

stimDur = 2000.0  # in ms
nTrials = 11  # number of trials per condition
nDirs = 16  # number of conditions
deltaDir = 22.5  # difference between conditions

stims["StimOffset"] = stims["StimOnset"] + stimDur

# %% [markdown]
#
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
# (`relTime`) and whether a spike was during the 
# stimulation period (`stimPeriod`). 
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
for i in range(1,10):
    plotRaster(spikes, i)

# %% [markdown]
# Find examples of:
#
# 1. A direction selective neuron
# 2. An orientation selective neuron 
# 3. Neither
# And explain your choices.



# %%
# ---------------------------------
# Find and explain examples? (1 pt)
# ---------------------------------
#%% My library code: 

# --- Helper function for optimal bin width (example: Freedman-Diaconis) ---
# TODO: Consider other binning methods.
def calculate_fd_bin_width(data: np.ndarray) -> float:
    if len(data) < 4:
        return 50.0
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        data_range = np.max(data) - np.min(data)
        return data_range / 10 if data_range > 0 else 20.0
    bin_width = 2 * iqr * (len(data) ** (-1/3))
    return bin_width if bin_width > 0 else 20.0

# --- Computation for Binned PSTH ---
def compute_binned_psth(
    spike_times: np.ndarray,
    stim_dur_ms: float,
    num_trials: int,
    bin_width_ms: float = None,
    min_bin_width_ms: float = 10.0,
    max_bin_ratio: float = 0.1, # Max bin width as ratio of stim_dur_ms
    calculate_bin_width=calculate_fd_bin_width # Optional: pass a custom bin width function
):
    """
    Computes a binned Peristimulus Time Histogram (PSTH).

    Parameters:
    -----------
    spike_times : np.ndarray
        Array of spike times relative to stimulus onset (in ms) for a specific condition.
    stim_dur_ms : float
        Total duration of the stimulus period to analyze (in ms).
    num_trials : int
        Number of trials the spike_times are aggregated over.
    bin_width_ms : float, optional
        Desired bin width in ms. If None, Freedman-Diaconis rule is used.
    min_bin_width_ms : float
        Minimum allowed bin width.
    max_bin_ratio : float
        Maximum allowed bin width as a fraction of stim_dur_ms.
        
    calculate_bin_width : function
        Function to calculate bin width. Default is Freedman-Diaconis rule.

    Returns:
    --------
    tuple: (bin_centers, rates, actual_bin_width_ms)
        - bin_centers (np.ndarray): Time points for the center of each bin.
        - rates (np.ndarray): Firing rate in spikes/second for each bin.
        - actual_bin_width_ms (float): The bin width used for calculation.
    """
    if num_trials == 0: # Should not happen if spike_times is not empty, but good check
        return np.array([]), np.array([]), bin_width_ms or 50.0

    if bin_width_ms is None:
        if len(spike_times) > 1:
            calculated_width = calculate_bin_width(spike_times)
            actual_bin_width_ms = np.clip(calculated_width, min_bin_width_ms, stim_dur_ms * max_bin_ratio)
        else:
            actual_bin_width_ms = 50.0 # Default if not enough data
    else:
        actual_bin_width_ms = bin_width_ms

    if actual_bin_width_ms <= 0:
        actual_bin_width_ms = 50.0 # Fallback

    bins = np.arange(0, stim_dur_ms + actual_bin_width_ms / 2.0, actual_bin_width_ms)
    if len(bins) < 2: # Not enough bins, possibly due to very small stim_dur_ms or large bin_width_ms
        return np.array([]), np.array([]), actual_bin_width_ms

    counts, _ = np.histogram(spike_times, bins=bins)
    
    rates = counts / (num_trials * (actual_bin_width_ms / 1000.0)) # Convert to spikes/sec
    bin_centers = bins[:-1] + actual_bin_width_ms / 2.0
    
    return bin_centers, rates, actual_bin_width_ms

# --- Computation for Spike Density Function (SDF) ---
def compute_sdf(
    spike_times: np.ndarray,
    stim_dur_ms: float,
    num_trials: int,
    kernel_bandwidth_ms: float = None, # Bandwidth for Gaussian KDE
    num_points: int = 200 # Number of points to evaluate the SDF
):
    """
    Computes a Spike Density Function (SDF) using 
    Gaussian Kernel Density Estimation.

    Parameters:
    -----------
    spike_times : np.ndarray
        Array of spike times relative to stimulus onset (in ms) for a specific condition.
    stim_dur_ms : float
        Total duration of the stimulus period (used for defining evaluation range).
    num_trials : int
        Number of trials the spike_times are aggregated over.
    kernel_bandwidth_ms : float, optional
        Bandwidth for the Gaussian kernel in ms. If None, Scott's rule or Silverman's
        heuristic is used by gaussian_kde, adjusted for scale.
    num_points : int
        Number of points at which to evaluate the SDF.

    Returns:
    --------
    tuple: (time_points, density_rate)
        - time_points (np.ndarray): Time points at which the SDF is evaluated.
        - density_rate (np.ndarray): Estimated firing rate in spikes/second.
    """
    if len(spike_times) == 0 or num_trials == 0:
        time_points_eval = np.linspace(0, stim_dur_ms, num_points)
        return time_points_eval, np.zeros_like(time_points_eval)

    # gaussian_kde expects data in its original scale for bandwidth estimation.
    # The output of kde.evaluate is a probability density.
    # To get spikes/sec: density * (total number of spikes / num_trials) / (bandwidth_in_seconds_if_kernel_was_normalized_to_integrate_to_1_over_bandwidth)
    # More simply: density * (num_spikes_per_trial) -> gives probability of spike per ms if kernel is for ms
    # Then multiply by 1000 for spikes/sec.
    # Or: total density * N_spikes / (N_trials * kernel_width_s) - this gets tricky.

    # Alternative: Scale by total number of spikes, divide by trials, divide by evaluation window length if density sums to 1
    # The output of gaussian_kde is a PDF. To convert to firing rate:
    # Rate(t) = sum_over_spikes K(t - t_spike) / N_trials
    # If K is a Gaussian kernel, gaussian_kde estimates sum_over_spikes K(t - t_spike) / N_spikes_total
    # So, Rate(t) = kde_estimate(t) * N_spikes_total / N_trials
    
    kde = gaussian_kde(spike_times, bw_method=kernel_bandwidth_ms / np.std(spike_times) if kernel_bandwidth_ms and np.std(spike_times) > 0 else 'scott')
    # bw_method in gaussian_kde is a factor multiplied by std of data.
    # If you provide kernel_bandwidth_ms, you want it to be the actual sigma of the Gaussian.
    # So, bw_factor = kernel_bandwidth_ms / np.std(spike_times)

    time_points_eval = np.linspace(0, stim_dur_ms, num_points)
    density_estimate = kde.evaluate(time_points_eval) # This is a probability density

    # Scale to firing rate: (density values * total number of spikes) / number of trials
    # This gives an estimate of the instantaneous rate.
    # The density integrates to 1. We want the rate such that integrating it over time gives total spikes / N_trials
    sdf_rate = density_estimate * (len(spike_times) / num_trials) * 1000 # Convert from spikes/ms to spikes/s

    return time_points_eval, sdf_rate

#%%
def plot_single_psth_sdf_on_axes(
    ax: plt.Axes,
    psth_bin_centers: np.ndarray,
    psth_rates: np.ndarray,
    sdf_time_points: np.ndarray = None,
    sdf_rates: np.ndarray = None,
    stim_dur_ms: float = None,
    title: str = "",
    psth_color: str = 'gray',
    sdf_color: str = 'blue',
    show_legend: bool = True
):
    """
    Plots a single binned PSTH and optionally an SDF on a given Matplotlib Axes.
    """
    if len(psth_bin_centers) > 0 and len(psth_rates) > 0:
        # Calculate bin width from bin_centers for bar plot width
        if len(psth_bin_centers) > 1:
            bar_width = psth_bin_centers[1] - psth_bin_centers[0]
        elif len(psth_bin_centers) == 1: # Only one bin
            bar_width = stim_dur_ms or psth_bin_centers[0]*2 # Guess
        else: # no bins
            bar_width = 0

        ax.bar(psth_bin_centers, psth_rates, width=bar_width*0.9,
               color=psth_color, alpha=0.6, label='Binned PSTH')

    if sdf_time_points is not None and sdf_rates is not None and len(sdf_time_points) > 0:
        ax.plot(sdf_time_points, sdf_rates, color=sdf_color, linewidth=1.5, label='SDF (KDE)')

    if stim_dur_ms:
        ax.axvspan(0, stim_dur_ms, color='lightgray', alpha=0.3, zorder=-1, label='Stimulus ON')

    ax.set_title(title)
    ax.set_xlabel("Time relative to stimulus onset (ms)")
    ax.set_ylabel("Firing Rate (spikes/s)")
    ax.grid(True, linestyle=':', alpha=0.7)
    if show_legend:
        ax.legend(fontsize='small')
    ax.set_xlim(left=-100 if sdf_time_points is not None and sdf_time_points.min()<0 else 0, # Adjust if showing pre-stim
                right = stim_dur_ms + 100 if stim_dur_ms else None)


def plot_all_conditions_psth(
    spikes_df: pd.DataFrame,
    neuron_id: int,
    stim_dur_ms: float, # e.g., global stimDur
    n_trials_per_cond: int, # e.g., global nTrials
    compute_sdf_flag: bool = True
):
    """
    Computes and plots PSTHs (and optionally SDFs) for all stimulus conditions for a neuron.
    """
    neuron_data = spikes_df[(spikes_df["Neuron"] == neuron_id) & (spikes_df["stimPeriod"] == True)]
    if neuron_data.empty:
        print(f"No spikes for neuron {neuron_id} during stimulus period.")
        return

    unique_dirs = np.sort(neuron_data["Dir"].unique())
    if len(unique_dirs) == 0:
        print(f"No directions found for neuron {neuron_id}.")
        return

    n_cols = 5
    n_rows = int(np.ceil(len(unique_dirs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows),
                             sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.flatten()
    max_overall_rate = 0.0

    # --- First pass to compute all data and find max rate for y-scaling ---
    all_plot_data = []
    psth_bin_width_used = None # To store the calculated bin width if FD rule is used once

    for i, direction in enumerate(unique_dirs):
        dir_spikes_rel_times = neuron_data[neuron_data["Dir"] == direction]["relTime"].dropna().values

        # Compute Binned PSTH
        # For consistent bin width across plots for the same neuron, calculate once if not specified
        if i == 0 and psth_bin_width_used is None: # Calculate optimal bin width once for the neuron
             all_neuron_rel_times = neuron_data["relTime"].dropna().values
             if len(all_neuron_rel_times) > 1:
                 psth_bin_width_used = np.clip(calculate_fd_bin_width(all_neuron_rel_times), 10.0, stim_dur_ms * 0.1)
             else:
                 psth_bin_width_used = 50.0
             print(f"Neuron {neuron_id}: Using PSTH bin width: {psth_bin_width_used:.2f} ms")


        psth_bc, psth_r, _ = compute_binned_psth(
            dir_spikes_rel_times, stim_dur_ms, n_trials_per_cond, bin_width_ms=psth_bin_width_used
        )
        current_max = np.max(psth_r) if len(psth_r) > 0 else 0

        # Compute SDF
        sdf_tp, sdf_r = None, None
        if compute_sdf_flag:
            # For SDF bandwidth, could also calculate globally or use a fixed sensible value
            # For simplicity here, let's use a fixed fraction of the PSTH bin width or None for KDE's default
            sdf_bw = psth_bin_width_used / 2.0 if psth_bin_width_used else 25.0
            sdf_tp, sdf_r = compute_sdf(
                dir_spikes_rel_times, stim_dur_ms, n_trials_per_cond, kernel_bandwidth_ms=sdf_bw
            )
            if len(sdf_r) > 0:
                current_max = max(current_max, np.max(sdf_r))

        max_overall_rate = max(max_overall_rate, current_max)
        all_plot_data.append({
            "psth_bc": psth_bc, "psth_r": psth_r,
            "sdf_tp": sdf_tp, "sdf_r": sdf_r,
            "direction": direction
        })

    # --- Second pass to plot with consistent y-scaling ---
    for i, data_dict in enumerate(all_plot_data):
        ax = axes_flat[i]
        is_left_col = (i % n_cols == 0)
        is_last_active_row_plot = (i // n_cols == (len(unique_dirs) -1) // n_cols)

        plot_single_psth_sdf_on_axes(
            ax,
            data_dict["psth_bc"], data_dict["psth_r"],
            data_dict["sdf_tp"], data_dict["sdf_r"],
            stim_dur_ms,
            title=f"Direction {int(data_dict['direction'])}°",
            show_legend= (i==0) # Show legend only on first plot
        )
        ax.set_ylim(0, max_overall_rate * 1.1 if max_overall_rate > 0 else 1.0)
        if not is_left_col:
            ax.set_ylabel("") # Remove y-label for non-left plots
        if not is_last_active_row_plot:
            ax.set_xlabel("") # Remove x-label for non-bottom plots


    # Hide unused subplots
    for i in range(len(all_plot_data), len(axes_flat)):
        fig.delaxes(axes_flat[i])

    fig.suptitle(f"PSTH & SDF for Neuron {neuron_id}", fontsize=16)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.show()
#%%
# --- Example Usage ---
# Select 5 neurons at random and plot them. 
for i in range(5):
    target_neuron = np.random.choice(np.unique(spikes['Neuron']))
    # Call the main plotting function
    # This function internally calls the computation functions
    plot_all_conditions_psth(
        spikes_df=spikes, # Your main spikes DataFrame
        neuron_id=target_neuron,
        stim_dur_ms=stimDur,
        n_trials_per_cond=nTrials,
        compute_sdf_flag=True # Set to False if you only want binned PSTH
    )

# TODO: bin width 
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
    """Evaluate the parametric von Mises tuning curve 
        with parameters p at locations theta.

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
    return np.exp(α + κ * (np.cos(2 * (θ - ϕ)) - 1) \
        + ν * (np.cos(θ - ϕ) - 1))
    

# %% [markdown]
# Plot the von Mises function while 
# varying the parameters systematically.
sns.set_theme(style="whitegrid")
for alpha_range in np.linspace(-10, 10, 5):
    for kappa_range in np.linspace(0, 10, 5): 
        plt.figure(figsize=(20, 15))
        for nu_range in np.linspace(0, 10, 5): 
            for phi_range in [0, 360]:
                theta = np.linspace(0, 360, 100)
                theta_rad = np.deg2rad(theta)
                ϕ_rad = np.deg2rad(phi_range)
                
                y = vonMises(theta_rad, alpha_range, kappa_range, nu_range, ϕ_rad)
                plt.plot(theta, y, label=f"α={alpha_range}, κ={kappa_range}, ν={nu_range}, ϕ={phi_range}")
        plt.title("Von Mises Tuning Curve")
        plt.xlabel("Direction (degrees)")
        plt.ylabel("Firing Rate")
        plt.xlim(0, 360)
        plt.legend()
        plt.grid()
        plt.show()

# %%
# ------------------------------------------------------------------------------
# Plot von Mises curves with varying parameters 
# and explain what they do (2 pts).
# ------------------------------------------------------------------------------


#%% Provided Template Code
def tuningCurve(counts: np.ndarray, dirs: np.ndarray, show: bool = True) -> np.ndarray:
    """Fit a von Mises tuning curve to the spike counts in count with 
    direction dir using a **least-squares fit**.

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
#%%
def compute_spike_count_matrix(counts: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Compute the spike count matrix from the counts and dirs.

    Parameters
    ----------
    counts: np.ndarray
        The spike counts for each trial.

    dirs: np.ndarray
        The stimulus directions for each trial.

    Returns
    -------
    spike_count_matrix: np.ndarray
        The computed spike count matrix.
    """
    unique_stim_directions_deg = np.unique(dirs)  # Shape: (nDirs,)
    num_unique_directions = len(unique_stim_directions_deg)
    logger.debug(f"Unique stimulus directions: {unique_stim_directions_deg}")
    logger.debug(f"Number of unique stimulus directions: {num_unique_directions}")
 
    # Get the unique stimulus directions, sorted. These will be the columns of our matrix.
    unique_stim_directions_deg = np.unique(dirs)  # Shape: (nDirs,)
    num_unique_directions = len(unique_stim_directions_deg)
    # Initialize the spike count matrix `x` with shape (nTrials, nDirs)
    # x_jk: j-th trial (row), k-th direction (column)
    spike_count_matrix_x = np.zeros((nTrials, num_unique_directions))    
    # Populate the matrix
    for k_idx, direction_value in enumerate(unique_stim_directions_deg):
        # Extract all spike counts from the 1D 'counts' array that correspond to the current 'direction_value'
        counts_for_this_direction = counts[dirs == direction_value]
        # The get_data function should ensure that 'counts_for_this_direction'
        # has exactly 'nTrials' elements.
        if len(counts_for_this_direction) == nTrials:
            spike_count_matrix_x[:, k_idx] = counts_for_this_direction
        else:
            # This part handles unexpected lengths, though get_data should prevent this.
            # If fewer than nTrials, pad with zeros (already done by get_data, but good for robustness).
            # If more than nTrials (unlikely), take the first nTrials.
            actual_trials_found = len(counts_for_this_direction)
            if actual_trials_found >= nTrials:
                spike_count_matrix_x[:, k_idx] = counts_for_this_direction[:nTrials]
            else: # actual_trials_found < nTrials
                spike_count_matrix_x[:actual_trials_found, k_idx] = counts_for_this_direction
                # The remaining (nTrials - actual_trials_found) elements will stay zero 
                # due to initialization with np.zeros.
    return spike_count_matrix_x
#%%
def inital_von_mises_params(mean_counts_to_fit: np.ndarray, unique_dirs_rad: np.ndarray) -> tuple:
    """Initial guess for the von Mises parameters based on mean counts.
    Parameters
    ----------
    mean_counts_to_fit: np.ndarray
        The mean counts for each direction.
    unique_dirs_rad: np.ndarray
        The unique directions in radians.
    Returns
    -------
    tuple: (alpha_guess, kappa_guess, nu_guess, phi_guess_rad)
        Initial guesses for the parameters of the von Mises function.
    """
    # Robust initial guesses:
    if not np.any(mean_counts_to_fit > 1e-9): # Handle cases where all mean counts are zero or tiny
        alpha_guess = np.log(1e-6) # A small baseline
        phi_guess_rad = 0.0        # Default preferred direction
    else:
        # For alpha, use log of mean of positive counts, or log of max if all else fails
        positive_mean_counts = mean_counts_to_fit[mean_counts_to_fit > 1e-9]
        if len(positive_mean_counts) > 0:
            alpha_guess = np.log(np.maximum(1e-6, np.mean(positive_mean_counts)))
        else: # Should be caught by the outer if, but as a fallback
            alpha_guess = np.log(np.maximum(1e-6, np.max(mean_counts_to_fit)))
        
        phi_guess_rad = unique_dirs_rad[np.argmax(mean_counts_to_fit)]

    kappa_guess = 1.0  # Initial guess for bimodal strength
    nu_guess = 1.0     # Initial guess for unimodal strength 
    return alpha_guess, kappa_guess, nu_guess, phi_guess_rad

#%%
def tuningCurve(counts: np.ndarray, dirs: np.ndarray, show: bool = True) -> np.ndarray:
    """Fit a von Mises tuning curve to the spike counts in count with 
    direction dir using a **least-squares fit**.

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
    logger.info("Fitting tuning curve...")
    logger.info(f"Counts: {counts.shape}")
    logger.info(f"Dirs: {dirs.shape}")

    spike_count_matrix_x = compute_spike_count_matrix(counts, dirs)
    
    logger.info(f"Spike count matrix shape: {spike_count_matrix_x.shape}") 
    logger.info(f"Spike count matrix: {spike_count_matrix_x}")

    # ------------------------------------------------------------
    # fit the von Mises tuning curve to the spike counts (0.5 pts)
    # ------------------------------------------------------------


    # 1. Calculate mean spike counts per direction
    mean_counts_to_fit = np.mean(spike_count_matrix_x, axis=0)
    logger.info(f"Mean counts to fit: {mean_counts_to_fit}")
    
    # 2. Get unique directions (degrees) and convert to radians
    # 'dirs' is the original 1D array of directions for all trials passed to tuningCurve
    unique_stim_directions_deg = np.unique(dirs) 
    unique_dirs_rad = np.deg2rad(unique_stim_directions_deg)
    logger.info(f"Unique directions (radians) for fitting: {unique_dirs_rad}")

    # Check if there's enough data to fit (at least as many points as parameters)
    if len(unique_dirs_rad) < 4: # vonMises has 4 parameters
        logger.warning(f"Not enough unique directions ({len(unique_dirs_rad)}) to fit the von Mises model. Need at least 4. Skipping fit.")
        p_opt = None # Indicate fit failed
    else:
        alpha_guess, kappa_guess, nu_guess, phi_guess_rad = inital_von_mises_params(mean_counts_to_fit, unique_dirs_rad)
        p0 = [alpha_guess, kappa_guess, nu_guess, phi_guess_rad]
        logger.info(f"Initial parameter guesses (p0): {p0}")
        # Bounds: alpha, kappa>=0, nu>=0, phi in [0, 2*pi]
        bounds = ([-np.inf, 0, 0, 0], [np.inf, np.inf, np.inf, 2 * np.pi])
        # Fit the von Mises function to the mean counts
        # 4. Perform the non-linear least squares fit
        try:
            p_opt, p_cov = opt.curve_fit(
                f=vonMises,          # Your vonMises function (make sure it's defined and accessible)
                xdata=unique_dirs_rad,
                ydata=mean_counts_to_fit,
                p0=p0,
                bounds=bounds,
                maxfev=5000          # Maximum number of function evaluations
            )
            logger.info(f"Optimized parameters (p_opt): {p_opt}")
        except RuntimeError:
            logger.warning("RuntimeError: Optimal parameters not found during curve_fit. Fit failed.")
            p_opt = None # Or assign np.full(4, np.nan) if you prefer NaNs for failed fits
        except ValueError as e:
            logger.warning(f"ValueError during curve_fit: {e}. Fit failed.")
            p_opt = None

 
        if show:
            # --------------------------------------------
            # plot the data and fitted tuning curve (1 pt)
            # --------------------------------------------
            pass
        
        return p_opt

# %% [markdown]
# Plot tuning curve and fit for different neurons. Good candidates to 
# check are 28, 29 or 37. 
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

#%%
neurons_to_plot = [28, 29, 37]
for neuron in neurons_to_plot:
    dirs_sorted, counts_sorted = get_data(spikes, neuron)
    result = tuningCurve(counts_sorted, dirs_sorted, show=True)
    if len(result) == 4:
        print(f"Neuron {neuron}: dirs_sorted.shape = {dirs_sorted.shape}, counts_sorted.shape = {counts_sorted.shape}")
    else:
        print(f"Neuron {neuron}: No result from tuningCurve()")
    """
    if result is not None:
        print(f"Neuron {neuron}: dirs_sorted.shape = {dirs_sorted.shape}, counts_sorted.shape = {counts_sorted.shape}")
    else:
        print(f"Neuron {neuron}: No result from tuningCurve()")
    """
    
# %%
# ----------------------------------------------------------
# Plot the average number of spikes per direction, the spike 
# counts from individual trials as well as your optimal fit 
# for different neurons (0.5 pts)
# ----------------------------------------------------------

# %% [markdown]
# ## Task 4: Permutation test for direction tuning
# 
# Implement a permutation test to quantitatively assess 
# whether a neuron is direction/orientation selective. 
# 
# To do so, project the vector of average spike counts, 
#       $m_k=\frac{1}{N}\sum_j x_{jk}$ 
# on a complex exponential with two cycles, 
#       $v_k = \exp(\psi i \theta_k)$, 
# where $\theta_k$ is the $k$-th direction of motion in radians and 
# $\psi \in 1,2$ is the fourier component to test 
# (1: direction, 2: orientation). 
#
# Denote the projection by $q=m^Tv$. 
# The magnitude $|q|$ tells you how much power there is in the 
# $\psi$-th fourier component. 
#
# Estimate the distribution of |q| under the 
# null hypothesis that the neuron fires randomly across 
# directions by running 1000 iterations where you repeat the 
# same calculation as above but on a random permutation of the 
# trials (that is, randomly shuffle the entries in the 
# spike count matrix x). 
#
# The fraction of iterations for which you obtain a 
# value more extreme than what you observed in the 
# data is your p-value. 
#
# Implement this procedure in the function ```testTuning()```. 
# 
# Illustrate the test procedure for one of the cells from above. 
# Plot the sampling distribution of |q| and indicate the 
# value observed in the real data in your plot. 
# 
# How many cells are tuned at p < 0.01?
# 
# *Grading: 3 pts*
# 
# %% Provided Template Code
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

# %% AN - Code 
def compute_null_distribution(
    counts: np.ndarray,
    dirs: np.ndarray,
    v_k: np.ndarray,
    niters: int,
   rng: np.random.Generator) -> np.ndarray:
    """Compute the null distribution of |q| under the null hypothesis.

    Parameters
    ----------
    counts: np.ndarray
        The spike counts for each trial.

    dirs: np.ndarray
        The stimulus directions for each trial.

    v_k: np.ndarray
        The complex exponential vector for the specified psi.

    niters: int
        Number of iterations for the permutation test.

    rng: np.random.Generator
        Random number generator for reproducibility. 
    Returns
    -------
    q_distribution_null: np.ndarray
        The computed null distribution of |q|.

    """
    # Initialize an array to store the |q| values from each permutation
    q_distribution_null = np.zeros(niters)
    logger.info(f"Starting permutation test with {niters} iterations...")
    
    # Loop over the number of iterations
    for i in range(niters):
        # 1. Permute the data: Shuffle the spike counts randomly across all trials.
        shuffled_trial_counts = rng.permutation(counts)
        
        # 2. Recalculate m_k (average spike count per direction) for this permuted dataset.
        permuted_spike_matrix = compute_spike_count_matrix(shuffled_trial_counts, dirs)
        m_k_permuted = np.mean(permuted_spike_matrix, axis=0)
        
        # 3. Recalculate |q| using the permuted m_k_permuted and the *original* v_k.
        # v_k does not change because it depends on the stimulus directions and psi,
        # which are fixed.
        q_complex_permuted = np.dot(m_k_permuted, v_k)
        q_magnitude_permuted = np.abs(q_complex_permuted)
        
        # 4. Store the magnitude from this permutation.
        q_distribution_null[i] = q_magnitude_permuted
        
        if (i + 1) % (niters // 10) == 0: # Log progress every 10%
            logger.debug(f"Permutation iteration {i+1}/{niters} completed.")

    logger.info("Permutation test finished.")
    
    return q_distribution_null

# %% AN - Code 
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
    # Calculate m, nu and q (0.5 pts)
    # -------------------------------
    # m - This is the vector of average spike counts for each unique stimulus direction.
    spike_count_matrix = compute_spike_count_matrix(counts, dirs) # 'counts' and 'dirs' are 1D arrays of all trials
    m_k = np.mean(spike_count_matrix, axis=0)  # Shape: (nUniqueDirs,)

    # Get unique directions and convert to radians for v_k
    unique_stim_directions_deg = np.unique(dirs) # These are the directions corresponding to columns of spike_count_matrix
    theta_k_rad = np.deg2rad(unique_stim_directions_deg) # Shape: (nUniqueDirs,)
        
    # v_k - This is the complex exponential vector for the specified psi.
    # It should be based on the unique radian directions theta_k_rad.
    v_k = np.exp(1j * psi * theta_k_rad)  # Corrected: Shape: (nUniqueDirs,)
        
    # q - This is the projection of m_k onto v_k.
    q_complex_observed = np.dot(m_k, v_k)

    # Magnitude of the projection for the observed data.
    # This is the 'q' to be returned and tested against null distribution.
    q_observed_magnitude = np.abs(q_complex_observed) 

    logger.debug(f"Observed q magnitude: {q_observed_magnitude}")
    logger.debug(f"Observed q complex: {q_complex_observed}")
    logger.debug(f"Observed m_k: {m_k}")
    logger.debug(f"Observed v_k: {v_k}")
    
    # -------------------------------------------------------------------------
    # Estimate the distribution of q under the H0 and obtain the p value (1 pt)
    # -------------------------------------------------------------------------
    # Ensure reproducibility using a random number generator
    # Hint: Access random functions of this generator
    rng = np.random.default_rng(random_seed)

    # 1-4. Compute the null distribution of |q| under the null hypothesis
    #    by running niters iterations where you repeat the same calculation as above
    #    but on a random permutation of the trials (that is, randomly shuffle the entries in the spike count matrix x).
    #    This is done in the compute_null_distribution function.
    #    The q_distribution_null is the array of |q| values from the null distribution.
    #    The function compute_null_distribution is defined above.
    #    It takes the counts, dirs, v_k, niters, and rng as inputs.
    q_distribution_null = compute_null_distribution(counts=counts, dirs=dirs,
                                                    v_k=v_k, niters=niters, rng=rng)
    
    # 5. Calculate the p-value.
    #    This is the proportion of permuted |q| values that are as extreme as,
    #    or more extreme than, the |q| observed from the original data.
    # We use smoothing to avoid p-values of 0 or 1.
    p_value = (np.sum(q_distribution_null >= q_observed_magnitude) + 1) / (niters + 1)
    #p_value = np.sum(q_distribution_null >= q_observed_magnitude) / niters

    # For a slightly more robust p-value, especially if niters is not huge or q_observed_magnitude is very extreme:
    # p_value_corrected = (np.sum(q_distribution_null >= q_observed_magnitude) + 1) / (niters + 1)
    # You can choose which one to use; the simpler one is fine for this lab typically.
    logger.info(f"Observed |q|: {q_observed_magnitude:.4f}, Calculated p-value: {p_value:.35f}")

    if show:
        # Add plotting code here
        pass
    
    # The array q_distribution_null is the 'qdistr' to be returned.
    qdistr = q_distribution_null
    return p_value, q_observed_magnitude, qdistr


#%%
neurons_to_plot = [28, 29, 37]
for neuron in neurons_to_plot:
    dirs_sorted, counts_sorted = get_data(spikes, neuron)
    result = tuningCurve(counts_sorted, dirs_sorted, show=True)
    for psi in [0, 1, 2]:
        p_value, q_observed_magnitude, qdistr = testTuning(counts_sorted, dirs_sorted, 
                                                           psi=psi, show=True, niters=10000)
        logger.info(f"Neuron:{neuron} psi:{psi} ->  p-value: {p_value}, q_observed_magnitude: {q_observed_magnitude}")

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


