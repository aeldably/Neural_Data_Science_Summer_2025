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
# # Coding Lab 7 : Transcriptomics

# %%
import numpy as np
import pylab as plt
import pandas as pd
import matplotlib.pyplot as plt

# We recommend using openTSNE for experiments with t-SNE
# https://github.com/pavlin-policar/openTSNE
from openTSNE import TSNE

%matplotlib inline

%load_ext jupyter_black

%load_ext watermark
%watermark --time --date --timezone --updated --python --iversions --watermark -p sklearn

# %%
plt.style.use("../matplotlib_style.txt")

# %% [markdown]
# # Introduction

# %% [markdown]
# In this notebook you are going to work with 
# transcriptomics data, in particular single-cell RNA 
# sequencing (scRNA-seq) data from the paper by 
# [Harris et al. (2018)](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.2006387). 
# They recorded the transcriptomes of 3,663 inhibitory cells 
# in the hippocampal area CA1. Their analysis divided these 
# cells into 49 fine-scale clusters coresponding to 
# different cell subtypes. They asigned names to these 
# cluster in a hierarchical fashion according to strongly 
# expressed gene in each clusters. The figure below shows 
# the details of their classification. 
# 
# You will first analyze some of the most 
# relevant statistics of UMI gene counts 
# distributions, and afterwards follow the 
# standard pipeline in the field to produce a 
# visualization of the data.

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# ## Load data

# %% [markdown]
# Download the data from ILIAS, move it to the `data/` directory and unzip it there.
# The read counts can be found in `counts`, with rows corresponding to cells and columns to genes.
# The cluster assignments for every individual cell can be found in  `clusters`, along with the colors used in the publication in  `clusterColors`.

# %%
# LOAD HARRIS ET AL DATA

# Load gene counts
data = pd.read_csv("../data/nds_cl_7/harris-data/expression.tsv", sep="\t")
genes = data.values[:, 0]
cells = data.columns[1:-1]
counts = data.values[:, 1:-1].transpose().astype("int")
data = []

# Kick out all genes with all counts = 0
genes = genes[counts.sum(axis=0) > 0]
counts = counts[:, counts.sum(axis=0) > 0]
print(counts.shape)

# Load clustering results
data = pd.read_csv("../data/nds_cl_7/harris-data/analysis_results.tsv", sep="\t")
clusterNames, clusters = np.unique(data.values[0, 1:-1], return_inverse=True)

# Load cluster colors
data = pd.read_csv("../data/nds_cl_7/harris-data/colormap.txt", sep="\s+", header=None)
clusterColors = data.values

# Note: the color order needs to be 
# reversed to match the publication
clusterColors = clusterColors[::-1]

# Taken from Figure 1 - we need cluster order to get 
# correct color order.
clusterOrder = [
    "Sst.No",
    "Sst.Npy.C",
    "Sst.Npy.Z",
    "Sst.Npy.S",
    "Sst.Npy.M",
    "Sst.Pnoc.Calb1.I",
    "Sst.Pnoc.Calb1.P",
    "Sst.Pnoc.P",
    "Sst.Erbb4.R",
    "Sst.Erbb4.C",
    "Sst.Erbb4.T",
    "Pvalb.Tac1.N",
    "Pvalb.Tac1.Ss",
    "Pvalb.Tac1.Sy",
    "Pvalb.Tac1.A",
    "Pvalb.C1ql1.P",
    "Pvalb.C1ql1.C",
    "Pvalb.C1ql1.N",
    "Cacna2d1.Lhx6.R",
    "Cacna2d1.Lhx6.V",
    "Cacna2d1.Ndnf.N",
    "Cacna2d1.Ndnf.R",
    "Cacna2d1.Ndnf.C",
    "Calb2.Cry",
    "Sst.Cry",
    "Ntng1.S",
    "Ntng1.R",
    "Ntng1.C",
    "Cck.Sema",
    "Cck.Lmo1.N",
    "Cck.Calca",
    "Cck.Lmo1.Vip.F",
    "Cck.Lmo1.Vip.C",
    "Cck.Lmo1.Vip.T",
    "Cck.Ly",
    "Cck.Cxcl14.Calb1.Tn",
    "Cck.Cxcl14.Calb1.I",
    "Cck.Cxcl14.S",
    "Cck.Cxcl14.Calb1.K",
    "Cck.Cxcl14.Calb1.Ta",
    "Cck.Cxcl14.V",
    "Vip.Crh.P",
    "Vip.Crh.C1",
    "Calb2.Vip.G",
    "Calb2.Vip.I",
    "Calb2.Vip.Nos1",
    "Calb2.Cntnap5a.R",
    "Calb2.Cntnap5a.V",
    "Calb2.Cntnap5a.I",
]

reorder = np.zeros(clusterNames.size) * np.nan
for i, c in enumerate(clusterNames):
    for j, k in enumerate(clusterOrder):
        if c[: len(k)] == k:
            reorder[i] = j
            break
clusterColors = clusterColors[reorder.astype(int)]

# %% [markdown]
# # Task 1: Data inspection
# Before we use t-SNE or any other advanced visualization 
# methods on the data, we first want to have a closer look on the 
# data and plot some statistics. For most of the analysis 
# we will compare the data to a Poisson distribution.

# %% [markdown]
# ###  1.1. Relationship between expression mean and fraction 
# of zeros. Compute actual and predicted gene expression. 
# The higher the average expression of a gene, 
# the smaller fraction of cells will show a 0 count. 
# 
# Plot the data and explain what you see in the plot.
# 
# 
# _(3 pts)_



# %%
# ------------------------------------------------------
# Compute actual and predicted gene expression (1 pt)
# ------------------------------------------------------


# Compute the average expression for each gene
# Compute the fraction of zeros for each gene
mean_expression = np.mean(counts, axis=0)
fraction_zeros = np.mean(counts == 0, axis=0)


# %%
# Compute the Poisson prediction
# (what is the expected fraction of zeros in a 
# Poisson distribution with a given mean?)

poisson_prediction = np.exp(-mean_expression)

# %%
# --------------------------------------------------
# plot the data and the Poisson prediction (1 pt)
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 4))
# add plot

# 1. Plot the actual data as a scatter plot
ax.scatter(mean_expression, fraction_zeros, alpha=0.5, label='Actual Data')

# 2. Plot the Poisson prediction as a line plot
#    (Sorting the values makes the line clean)
sorted_indices = np.argsort(mean_expression)
ax.plot(mean_expression[sorted_indices], poisson_prediction[sorted_indices], color='red', linestyle='--', label='Poisson Prediction')

# 3. Set scales and labels
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Mean expression')
ax.set_ylabel('Fraction of Zeros')
ax.set_title('Mean expression vs. Fraction of Zeros')
ax.grid(True, which="both", linestyle='--', linewidth=0.5)
ax.legend()

plt.savefig("../images/lab7-mean_vs_fraction_zeros.png", dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# _Explanation (1 pt)_
# ...

# %% [markdown]
# ### 1.2. Mean-variance relationship
# 
# If the expression follows Poisson distribution, then the mean should 
# be equal to the variance. Plot the mean-variance relationship and 
# interpret the plot.
# 
# _(2.5 pts)_


# %%
# -------------------------------------------------------------------
# Compute the variance of the expression counts of each gene (0.5 pt)
# -------------------------------------------------------------------

# %%
# -------------------------------------------------------------
# Plot the mean-variance relationship on a log-log plot (1 pt)
# Plot the Poisson prediction as a line
# -------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 4))

# %% [markdown]
# _Explanation (1 pt)_
# ...

# %% [markdown]
# ### 1.3. Relationship between the mean and the Fano factor
# 
# Compute the Fano factor for each gene and make a scatter plot of expression mean vs. Fano factor in log-log coordinates, and interpret what you see in the plot. If the expression follows the Poisson distribution, then the Fano factor (variance/mean) should be equal to 1 for all genes.
# 
# _(2.5 pts)_

# %%
# --------------------------------------------
# Compute the Fano factor for each gene (0.5 pt)
# --------------------------------------------

fano =

# %%
# -------------------------------
# plot fano-factor vs mean (1 pt)
# incl. fano factor
# -------------------------------
# Plot a Poisson prediction as line
# Use the same style of plot as above.

fig, ax = plt.subplots(figsize=(6, 4))

# %% [markdown]
# _Explanation (1 pt)_
# ...

# %% [markdown]
# ### 1.4. Histogram of sequencing depths
# 
# Different cells have different sequencing depths (sum of counts across all genes) because the efficiency can change from droplet to droplet due to some random expreimental factors. Make a histogram of sequencing depths.
# 
# _(1.5 pts)_

# %%
# -------------------------------
# Compute sequencing depth (0.5 pt)
# -------------------------------


# %%
# ------------------------------------------
# Plot histogram of sequencing depths (1 pt)
# ------------------------------------------

fig, ax = plt.subplots(figsize=(6, 4))

# %% [markdown]
# ### 1.5. Fano factors after normalization
# 
# Normalize counts by the sequencing depth of each cell and multiply by the median sequencing depth. Then make the same expression vs Fano factor plot as above. After normalization by sequencing depth, Fano factor should be closer to 1 (i.e. variance even more closely following the mean). This can be used for feature selection.
# 
# _(2.5 pts)_

# %%
# -------------------------------------------------
# compute normalized counts and fano factor (1 pt)
# -------------------------------------------------

# %%
# ----------------------------------------------------------
# plot normalized counts and find the top 10 genes (1 pt)
# hint: keep appropriate axis scaling in mind
# ----------------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 4))
# add plot

# %%
#--------------------------------------------------------------------
# Find top-10 genes with the highest normalized Fano factor (0.5 pts)
# Print them sorted by the Fano factor starting from the highest
# Gene names are stored in the `genes` array
#--------------------------------------------------------------------

# %% [markdown]
# # Task 2: Low dimensional visualization
# 
# In this task we will construct a two dimensional visualization of the data. 
# First we will normalize the data with some variance stabilizing 
# transformation and study the effect that different approaches have on the 
# data. 
# Second, we will reduce the dimensionality of the data to a more 
# feasible number of dimensions (e.g. $d=50$) using PCA. And last, 
# we will project the PCA-reduced data to two dimensions using t-SNE.



# %% [markdown]
# ### 2.1. PCA with and without transformations
# 
# Here we look at the influence of variance-stabilizing transformations on PCA. We will focus on the following transformations: 
# - Square root (`sqrt(X)`): it is a variance-stabilizing transformation for the Poisson data. 
# - Log-transform (`log2(X+1)`): it is also often used in the transcriptomic community. 
# 
# We will only work with the most important genes. For that, transform the counts into normalized counts (as above) and select all genes with normalized Fano factor above 3 and remove the rest. We will look at the effect that both transformations have in the PCA-projected data by visualizing the first two components. Interpret qualitatively what you see in the plot and compare the different embeddings making use of the ground truth clusters.
# 
# _(3.5 pts)_

# %%
# --------------------------------
# Select important genes (0.5 pts)
# --------------------------------


# %%
# --------------------------------------
# transform data and apply PCA (1 pt)
# --------------------------------------

from sklearn.decomposition import PCA

# perform PCA

# %%
# ------------------------------------------------
# plot first 2 PCs for each transformation (1 pt)
# ------------------------------------------------

fig, axs = plt.subplots(1, 3, figsize=(9, 3))
# add plot

# %% [markdown]
# _Explanation (1 pt)_
# ...

# %% [markdown]
# ### 2.2. tSNE with and without transformations
# 
# Now, we will reduce the dimensionality of the PCA-reduced data further to two dimensions using t-SNE. We will use only $n=50$ components of the PCA-projected data. Plot the t-SNE embedding for the three versions of the data and interpret the plots. Do the different transformations have any effect on t-SNE?
# 
# _(1.5 pts)_

# %%
# -----------------------
# Perform tSNE (0.5 pts)
# -----------------------

# %%
# -----------------------------------------------
# plot t-SNE embedding for each dataset (1 pt)
# -----------------------------------------------

fig, axs = plt.subplots(1, 3, figsize=(9, 3))
# add plot

# %% [markdown]
# ### 2.3. Leiden clustering
# 
# Now we will play around with some clustering and see whether the clustering methods can produce similar results to the original clusters from the publication. We will apply Leiden clustering (closely related to the Louvain clustering), which is standard in the field and works well even for very large datasets. 
# 
# Choose one representation of the data (best transformation based in your results from the previous task) to use further in this task and justify your choice. Think about which level of dimensionality would be sensible to use to perform clustering. Visualize in the two-dimensional embedding the resulting clusters and compare to the original clusters. 
# 
# _(1.5 pts)_

# %%
# To run this code you need to install leidenalg and igraph
# conda install -c conda-forge python-igraph leidenalg

import igraph as ig
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
import leidenalg as la

# %%
# Define some contrast colors

clusterCols = [
    "#FFFF00",
    "#1CE6FF",
    "#FF34FF",
    "#FF4A46",
    "#008941",
    "#006FA6",
    "#A30059",
    "#FFDBE5",
    "#7A4900",
    "#0000A6",
    "#63FFAC",
    "#B79762",
    "#004D43",
    "#8FB0FF",
    "#997D87",
    "#5A0007",
    "#809693",
    "#FEFFE6",
    "#1B4400",
    "#4FC601",
    "#3B5DFF",
    "#4A3B53",
    "#FF2F80",
    "#61615A",
    "#BA0900",
    "#6B7900",
    "#00C2A0",
    "#FFAA92",
    "#FF90C9",
    "#B903AA",
    "#D16100",
    "#DDEFFF",
    "#000035",
    "#7B4F4B",
    "#A1C299",
    "#300018",
    "#0AA6D8",
    "#013349",
    "#00846F",
    "#372101",
    "#FFB500",
    "#C2FFED",
    "#A079BF",
    "#CC0744",
    "#C0B9B2",
    "#C2FF99",
    "#001E09",
    "#00489C",
    "#6F0062",
    "#0CBD66",
    "#EEC3FF",
    "#456D75",
    "#B77B68",
    "#7A87A1",
    "#788D66",
    "#885578",
    "#FAD09F",
    "#FF8A9A",
    "#D157A0",
    "#BEC459",
    "#456648",
    "#0086ED",
    "#886F4C",
    "#34362D",
    "#B4A8BD",
    "#00A6AA",
    "#452C2C",
    "#636375",
    "#A3C8C9",
    "#FF913F",
    "#938A81",
    "#575329",
    "#00FECF",
    "#B05B6F",
    "#8CD0FF",
    "#3B9700",
    "#04F757",
    "#C8A1A1",
    "#1E6E00",
    "#7900D7",
    "#A77500",
    "#6367A9",
    "#A05837",
    "#6B002C",
    "#772600",
    "#D790FF",
    "#9B9700",
    "#549E79",
    "#FFF69F",
    "#201625",
    "#72418F",
    "#BC23FF",
    "#99ADC0",
    "#3A2465",
    "#922329",
    "#5B4534",
    "#FDE8DC",
    "#404E55",
    "#0089A3",
    "#CB7E98",
    "#A4E804",
    "#324E72",
    "#6A3A4C",
    "#83AB58",
    "#001C1E",
    "#D1F7CE",
    "#004B28",
    "#C8D0F6",
    "#A3A489",
    "#806C66",
    "#222800",
    "#BF5650",
    "#E83000",
    "#66796D",
    "#DA007C",
    "#FF1A59",
    "#8ADBB4",
    "#1E0200",
    "#5B4E51",
    "#C895C5",
    "#320033",
    "#FF6832",
    "#66E1D3",
    "#CFCDAC",
    "#D0AC94",
    "#7ED379",
    "#012C58",
]

clusterCols = np.array(clusterCols)

# %%
# ------------------------------------------------------
# create graph and run leiden clustering on it (0.5 pts)
# hint: use `la?`, `la.find_partition?` and `ig.Graph?`
# to find out more about the provided packages.
# ------------------------------------------------------

# Construct kNN graph with k=15

A =

# Transform it into an igraph object

sources, targets = A.nonzero()

# Run Leiden clustering
# you can use `la.RBConfigurationVertexPartition` as the partition type

# %%
# --------------------------
# Plot the results (1 pt)
# --------------------------

fig, ax = plt.subplots(figsize=(4, 4))

# %% [markdown]
# ### 2.4. Change the clustering resolution
# 
# The number of clusters can be changed by modifying the resolution parameter. How many clusters did we get with the default value? Change the resolution parameter to yield 2x more and 2x fewer clusters
# Plot all three results as t-SNE overlays (same as above).
# 
# _(1.5 pts)_

# %%
# ------------------------------------------------------------------
# run the clustering for 3 different resolution parameters (0.5 pts)
# ------------------------------------------------------------------

# %%
# --------------------------
# Plot the results (1 pt)
# --------------------------

fig, axs = plt.subplots(1, 3, figsize=(9, 3))
# add plot


