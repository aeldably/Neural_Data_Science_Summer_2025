# %% [markdown]
# _Neural Data Science_
# 
# Lecturer: Dr. Jan Lause, Prof. Dr. Philipp Berens
# 
# Tutors: Jonas Beck, Fabio Seel, Julius Würzler
# 
# Summer term 2025
# 
# Student names: <span style='background: yellow'>Aakarsh Nair, Andreas Kotzur, Ahmed Eldably</span>
# 
# LLM Disclaimer: <span style='background: yellow'>
# Google Gemini , Google Gemini Diffiusion - Planning, Coding, and Verification.</span>

# %% [markdown]
# # Coding Lab 8: Neural Morphologies

# %% [markdown]
# ## Introduction
# 
# The anatomical shape of a neuron — its morphology — has fascinated scientists ever since the pioneering work of Cajal (Ramon y Cajal, 1911). A neuron's dendritic and axonal processes naturally decide what other neurons it can connect to, hence, its shape plays an important role for its function in the circuit. In particular, different functional types of neurons have fundamentally different morphologies.
# 
# This notebook will introduce you to the analysis of neural morphologies using the dendrites of over $500$ retinal ganglion cells. The aim is to teach you two different ways of representing morphologies and give you an impression of their repsective strengths and weaknesses.
# 
# ![image.png](attachment:image.png)
# 
# ### 1. Data
# 
# The data set contains morphological reconstructions of $599$ retinal ganglion cell dendrites with cell type label and projection target to either the parabigeminal (Pbg) or the pulvinar nucleus (LP)([Reinhard et al. (2019)](https://elifesciences.org/articles/50697)). 
# Here we only keep cells that map to clusters with more than six cells per cluster which leads to $550$ remaining reconstructions. 
# 
# Download the data file `nds_cl_8.zip` from ILIAS and unzip it in a subfolder `../data/`
# 
# 
# ### 2. Toolbox
# 
# We will use MorphoPy (Laturnus, et al., 2020; https://github.com/berenslab/MorphoPy) for this exercise. We recommend to use the Github version, as it is more up-to-date:
# 
# ```
# git clone https://github.com/berenslab/MorphoPy
# pip install -e MorphoPy
# ```
# 
# Most of the computations and even some plottings will be handled by MorphoPy. You can learn more about MorphoPy's APIs in this [tutorial](https://nbviewer.jupyter.org/github/berenslab/MorphoPy/blob/master/notebooks/MORPHOPY%20Tutorial.ipynb). 

# %%
import pandas as pd
import numpy as np
import os

from morphopy.neurontree import NeuronTree as nt

from morphopy.computation import file_manager
from morphopy.computation import file_manager as fm

from morphopy.neurontree.plotting import show_threeview
from morphopy.neurontree import NeuronTree as nt

import warnings

warnings.filterwarnings("ignore")

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

%load_ext jupyter_black

%load_ext watermark
%watermark --time --date --timezone --updated --python --iversions --watermark -p sklearnv

# %%
plt.style.use("../matplotlib_style.txt")

# %% [markdown]
# # Inspect the raw data

# %% [markdown]
# #### File format
# 
# Morphological reconstructions are typically stored in the SWC file format, a simple text file that holds node information in each row and connects nodes through the `parent` node id. A parent id of -1 indicates no parent, so the starting point of the tree graph, also called the root. 
# The `type` label indicates the node type (1: somatic , 2: axonal, 3: dendritic (basal), 4: dendritic (apical), 5+: custom).
# The code snippet below loads in one swc file and prints its head. 
# 
# You can find a more detailed specification of SWC and SWC+ [here](http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html) and [here](https://neuroinformatics.nl/swcPlus/).
# 
# 

# %%
def load_swc(filepath: str) -> pd.DataFrame:
    """Loads in the swc located at filepath as a pandas dataframe.

    Args:
        filepath (str): The path to the swc file.

    Returns:
        pd.DataFrame: A pandas dataframe containing the swc file.
    """
    swc = pd.read_csv(
        filepath,
        delim_whitespace=True,
        comment="#",
        names=["n", "type", "x", "y", "z", "radius", "parent"],
        index_col=False,
    )
    return swc


# define color for each cluster
colors = sns.color_palette("rainbow_r", n_colors=14)

# %%
# import swc file
PATH = "../data/nds_cl_8/"
data_path = PATH + "reconstructions/soma-centered/"
filename = "0006_00535_4L_C02_01.swc"
filepath = data_path + filename

swc = load_swc(filepath)
swc.head()

# %% [markdown]
# The labels `x`, `y`, and `z` hold a node's 3D coordinate in tracing space (here in microns). For reasons of simplicity we will work with reconstructions that are soma centered in XY.
# 
# The assigned cell type labels are stored in the file `rgc_labels.csv` and indexed by their `Cell_nr`. In this file you find three different cluster assignments: `clusterA` is the assignment of the authors (clus1 -- clus14), `clusterB` is the respective cluster identifier of the [Eyewire museum](http://museum.eyewire.org) (also see [Bae et al. 2018](https://www.sciencedirect.com/science/article/pii/S0092867418305725)), and `clusterC` are molecular or functional label names when available. 
# We have formatted the cluster assignments of the authors (`clusterA`) into integer values and stored them in the column `cluster`, which we will use in the following.

# %%
labels = pd.read_csv(PATH + "rgc_labels.csv", index_col=0)

cluster_label, cluster_counts = np.unique(labels["cluster"], return_counts=True)
labels.head()

# %% [markdown]
# ## Task 1: Plotting individual morphologies
# 
# Load data using `file_manager` and plot individual morphologie 
# using `show_threeview` of from `MorphoPy`. 
#
# It plots all three planar views on the reconstruction. 
# 
# Here, XY shows the planar view on top of the retina, and 
# Z denotes the location within the inner plexiform layer (IPL).
# 
# Noted, by default, the `file_manager` loads data with 
# `pca_rot=True` and `soma_center=True`. 
# For the all the exercise in this Coding Lab, it's better to set 
# both of them as `False`. 
# 
# *Grading: 2pts*
data_path = "../data/nds_cl_8/reconstructions/soma-centered/"
neuron_path = data_path + filename 
neuron = fm.load_swc_file(
    neuron_path,
    soma_center=False,
    pca_rot=False
)

fig = plt.figure(figsize=(10, 10))
show_threeview(neuron, fig)
plt.suptitle(f"Three-view plot of {filename}") # Add a title
plt.show()

# %%
# ----------------------------------------------------------------
# load the example cell "0060_00556_3R_C02_01" with `file_manager`
# from morphology (0.5 pts)
# ----------------------------------------------------------------

# -------------------------------------
# plot all three planar views (0.5 pts)
# -------------------------------------

# %% [markdown]
# ### Questions (0.5 pts)
# 
# 1) Describe the dendritic structure of this neuron. How is it special? Can you even give a technical term for its appearance?
# 
# **Answer:**

# %% [markdown]
# SWC files are a compact way for storing neural morphologies but their graph structure makes them difficult to handle for current machine learning methods. We, therefore, need to convert our reconstructions into a reasonable vector-like representations. 
# 
# Here we will present two commonly chosen representations: Morphometric statistics and density maps
# 

# %%
# load all reconstructions. Note: files are sorted by cell number
def load_files(path: str) -> list[nt]:
    """Returns list of NeuronTrees for all .swc files in `path`.
    The reconstructions should be sorted ascendingly by their filename.

    Args:
        path (str): The path to the folder containing the reconstructions.

    Returns:
        list[nt]: An object array of NeuronTrees containing all reconstructions at `path`.
    """
    neurons = []
    # ----------------------------------------------------------
    # use `file_manager` to import all reconstructions (0.5 pts)
    # Note the list should be sorted by filename.
    # ----------------------------------------------------------
    # 1. Find all swc files in the directory
    files = os.listdir(path)
    print(f"Found {len(files)} files in {path}")
    # 2. Filter for swc files
    files = [f for f in files if f.endswith(".swc")]
    files.sort()  # Sort files by filename
    for file in files:
        neuron = fm.load_swc_file(f"{path}/{file}", soma_center=False, pca_rot=False)
        neurons.append(neuron)
    return neurons


neurons = load_files(data_path)
print("Number of reconstructions: ", len(neurons))


# %% [markdown]
# ## Task 2: Morphometric statistics
# 
# Morphometric statistics denote a set of hand-crafted single valued features such as `soma radius`, `number of tips` or `average branch angle`. For a more detailed explanation of morphometrics please refer to the [MorphoPy documentation](https://github.com/berenslab/MorphoPy#morphometric-statistics).
# 
# *Grading: 4pts*

# %% [markdown]
# First, let's compute the feature-based representation for each cell using the function `compute_morphometric_statistics` of the MorphoPy package which computes a predefined set of $28$ statistics.
# 

# %%
from morphopy.computation.feature_presentation import compute_morphometric_statistics


# --------------------------------------------------------------------------
# 1. extraction the morphometric statistics for the entire data set (0.5 pts)
# --------------------------------------------------------------------------
ms_list = [compute_morphometric_statistics(neuron) for neuron in neurons]

# -----------------------------------------------------------------------------------
# 2. concatenate data into one pd.DataFrame and set the `Cell_nr`` as index (0.5 pts)
# -----------------------------------------------------------------------------------
morphometric_statistics = pd.concat(ms_list, axis=0)
morphometric_statistics.reset_index(inplace=True)
morphometric_statistics.index.name = "Cell_nr"

# %% [markdown]
# Now let's visualize the data.

# %%
features = morphometric_statistics.columns.values

fig, axes = plt.subplots(4, 7, figsize=(30, 10))
axes = axes.flatten()

# -----------------------------------------------------------
# Create a scatter/strip plot for each morphometric statistic
# showing how it varies across clusters. (2 pts)
# -----------------------------------------------------------
for feature, ax in zip(features, axes):
    # Use seaborn's stripplot to show the distribution for each cluster
    sns.stripplot(
        x=labels["cluster"],  # Categorical data for the x-axis
        y=morphometric_statistics[feature],  # Numerical data for the y-axis
        ax=ax,  # Tell seaborn which subplot to draw on
        palette=colors,  # Use the predefined color palette
        s=3, # Make the points a bit smaller to avoid overplotting
        alpha=0.7 # Add some transparency
    )
    # Set the title of the subplot to the name of the feature
    ax.set_title(feature, fontsize=8)
    ax.set_xlabel("") # Hide x-axis label for cleanliness
    ax.set_ylabel("") # Hide y-axis label for cleanliness
    ax.tick_params(axis='x', rotation=45) # Rotate cluster labels slightly

# Improve the layout to prevent titles from overlapping
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Questions (1 pt)
# 
# 1) Which statistics separate clusters well? Which can be removed? (tips: there are 5 uninformative features)
# 
# **Answer:**
# 
# 2) More generally, what do morphometric statistics capture well? What are their advantages, what might be their downsides? Briefly explain.
# 
# **Answer:**

# %% [markdown]
# ## Task 3: Density maps
# 
# Density maps project a neuron's 3D point cloud ($x$, $y$, $z$) onto a plane or an axis, and bin the projected point cloud into a fixed number of bins. Hereby, the binning controls how much global or local information is kept, which majorly affects the results.
# 
# **Exercise:** Compute the density maps of all neurons onto all cardinal planes and axes using the method `compute_density_maps`. You can manipulate the parameters for the density maps via the dictonary `config`. 
# Make sure that you normalize the density maps globally and bin each direction into $20$ bins.
# You are welcome to explore, how the different projections look like but we will only use the z-projection for further analysis.
# 
# 
# Possible parameters to pass are:
# 
# - `distance`: (default=1, in microns) determines the resampling distance.
# - `bin_size`: (default=20, in microns). If set the number of bins will be computed such that one bin 
# spans `bin_size` microns. This is overwritten when `n_bins_x/y/z` is set!
# - `n_bins_x/y/z`: (default=None) specifies the number of bins for each dimension. If set it will overwrite the 
# `bin_size` flag.
# - `density`: (default=True) bool to specify if a density or counts are returned.
# - `smooth`: (default=True) bool to trigger Gaussian smoothing.
# - `sigma`: (default=1) determines std of the Gaussian used for smoothing. The bigger the sigma the more smoothing occurs. If smooth is set to False this parameter is ignored. 
# - `r_min_x/y/z`: (in microns) minimum range for binning of x, y, and z. This value will correspond to the 
# minimal histogram edge. 
# - `r_max_x/y/z`: (in microns) maximum range for binning for x, y, and z. This value will correspond to the 
# maximal histogram edge. 
# 
# *Grading: 4pts*

# %%
# For further analysis we will remove uninformative features and z-score along each statistic
features_to_drop = [
    "avg_thickness",
    "max_thickness",
    "total_surface",
    "total_volume",
    "log_min_tortuosity",
]
morphometric_data = morphometric_statistics.drop(features_to_drop, axis=1)

# z-score morphometrics and remove nans and uninformative features
morphometric_data = (
    morphometric_data - morphometric_data.mean()
) / morphometric_data.std()
morphometric_data[morphometric_data.isna()] = 0
morphometric_data = morphometric_data.values

# %%
# ------------------------------------------------------------------------------------
# Find the minimal and maximal x,y,z - coordinates of the reconstructions to normalize
# the density maps globally using r_min_x/y/z and r_max_x/y/z and print them  for 
# each direction. (1 pt)
# ------------------------------------------------------------------------------------

# %%
from morphopy.computation.feature_presentation import compute_density_maps

config_global = dict(
# ---------------------------------------------------------------------------------
# complete the config dict and compute the z-density maps for each neuron (1 pts)
# ---------------------------------------------------------------------------------
    distance=,
    n_bins_x=,
    n_bins_y=,
    n_bins_z=,
    density=,
    smooth=,
    sigma=,
    r_max_x=,
    r_max_y=,
    r_max_z=,
    r_min_x=,
    r_min_y=,
    r_min_z=,
)

density_maps = 

# extract the z density map
dm_z = 

# %%
# --------------------------------------------------------------------
# plot the Z-density maps and their means sorted by class label (1 pt)
# Note: make sure the clusters are comparable.
# --------------------------------------------------------------------

# %% [markdown]
# ### Questions (1 pt)
# 
# 1) What does the Z-density map tell you about the cell types? Can you identify a trend in the density maps?
# 
# **Answer:**
# 
# 2) Which cluster(s) would you expect the cell from Task 1 to come from and why?
# 
# **Answer:**
# 

# %% [markdown]
# ## Task 4: 2D embedding using t-SNE
# 
# 
# Embed both data, the morphometric statistics and the density maps, in 2D using t-SNE and color each embedded point by its cluster assignment.
# 
# *Grading: 3 pts*

# %%
from openTSNE import TSNE

# ----------------------------------------------------------------------
# Fit t-SNE with morphometric statistics and density maps (0.5 + 0.5 pt)
# Note that this can take a bit to run. (use perplexity=100 
# and a random state of 17)
# ----------------------------------------------------------------------

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")

# ----------------------------------------------------------------------------
# plot tsne fits for both morpometric statistics and z-projected density maps.
# Color the points appropriately and answer the questions below. (2 pt)
# ----------------------------------------------------------------------------



# %% [markdown]
# ### Questions:
# 1) Which representation produces the better clustering? Why could this be the case?
# 
# **Answer:**
# 
# 2) What are the advantages of morphometric statistics over density maps 
# 
# **Answer:**
# 
# 3) What are the advantages of density maps over morphometric statistics 
# 
# **Answer:**

# %% [markdown]
# ## Task 5: Predicting the projection site

# %% [markdown]
# The relationship between neuronal morphology and functional specialization is well-established in neurobiology. Hence, we expect distinct functional domains within the thalamus to exhibit corresponding morphological signatures. In this analysis, we aim to predict the thalamic projection site (`labels['projection_site']`) of individual neurons based on their morphological characteristics. Fit a logistic regression on both morphological representations and report its average cross validated (cv=5) prediction accuracy for each. Which representation works better to recover the prediction target? Which features are most relevant for that prediction?
# 
# You can use `LogisticRegressionCV` of the scikit-learn library directly. To understand the relevance of individual features plot the fitted linear coefficients. Note, since the classes are imbalanced make sure to report the balanced prediction accuracy.
# 
# *Grading: 2 pts*

# %%
from sklearn.linear_model import LogisticRegressionCV


# -----------------------------------------------------------------------------
# Fit a logistic regressor to predict the projection site based on both feature
# representations and print the avg. prediction accuracy (1 pt)
# -----------------------------------------------------------------------------

# %% [markdown]
# While Z density maps allow for better recovery of cell type labels, they are worse than morphometric statistics on predicting the projection target. 

# %%
# ---------------------------------------------------------------------------
# Plot the fitted linear coefficients for both of the feature representations
# and answer the question below. (1 pt)
# ---------------------------------------------------------------------------

# %% [markdown]
# ### Question:
# 
# 1) Which morphometrics are informative on the projection site?
# 
# **Answer:**

# %% [markdown]
# ## Further references
# 
# Other ways to represent and compare morphologies are
# * Persistence: [Description](https://link.springer.com/article/10.1007/s12021-017-9341-1) and [application on somatosensory pyramidal cell dendrites](https://academic.oup.com/cercor/article/29/4/1719/5304727) by Kanari et al. 2018
# 
# * Tree edit distance: [Heumann et al. 2009](https://link.springer.com/article/10.1007/s12021-009-9051-4)
# 
# * Sequential encoding inspired by BLAST: [Encoding](https://link.springer.com/article/10.1186/s12859-015-0604-2) and [similarity analysis on cortical dendrites](https://link.springer.com/article/10.1186/s12859-015-0605-1) by Gilette et al. 2015
# 
# * Vector point clouds: [BlastNeuron: Wan et al. 2015](https://link.springer.com/article/10.1007/s12021-015-9272-7)


