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
# # Coding Lab 5

# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import scipy.io as io

#%load_ext jupyter_black

#%load_ext watermark
#%watermark --time --date --timezone --updated --python --iversions --watermark -p sklearn

#%matplotlib inline
#plt.style.use("../matplotlib_style.txt")

# %% [markdown]
# # Task 1: Fit RF on simulated data
# 
# We will start  with toy data generated from an LNP model neuron to make sure everything works right. The model LNP neuron consists of one Gaussian linear filter, an exponential nonlinearity and a Poisson spike count generator. We look at it in discrete time with time bins of width $\delta t$. The model is:
# 
# $$
# c_t \sim Poisson(r_t)\\
# r_t = \exp(w^T s_t) \cdot \Delta t \cdot R
# $$
# 
# Here, $c_t$ is the spike count in time window $t$ of length $\Delta t$, $s_t$ is the stimulus and $w$ is the receptive field of the neuron. The receptive field variable `w` is 15 × 15 pixels and normalized to $||w||=1$. A stimulus frame is a 15 × 15 pixel image, for which we use uncorrelated checkerboard noise (binary) with a stimulus intesity of 5 (peak to peak). R can be used to bring the firing rate into the right regime (e.g. by setting $R=50$).      
# 
# For computational ease, we reformat the stimulus and the receptive field in a 225 by 1 array. The function ```sample_lnp``` can be used to generate data from this model. It returns a spike count vector `c` with samples from the model (dimensions: 1 by nT = $T/\Delta t$), a stimulus matrix `s` (dimensions: 225 × nT) and the mean firing rate `r` (dimensions: nT × 1). 
# 
# Here we assume that the receptive field influences the spike count instantaneously just as in the above equations. Implement a Maximum Likelihood approach to fit the receptive field. 
# 
# To this end derive mathematically and implement the log-likelihood function $L(w)$ and its gradient $\frac{L(w)}{dw}$ with respect to $w$ (`negloglike_lnp`). The log-likelihood of the model is
# $$L(w) = \log \prod_t \frac{r_t^{c_t}}{c_t!}\exp(-r_t).$$
# 
# Make sure you include intermediate steps of the mathematical derivation in your answer, and you give as final form the maximally simplified expression, substituting the corresponding variables.
# 
# Plot the stimulus for one frame, the cell's response over time and the spike count vs firing rate. Plot the true and the estimated receptive field. 
# 
# *Grading: 2 pts (calculations) + 4 pts (generation) + 4 pts (implementation)*
# 

# %% [markdown]
# ### Calculations (2 pts)
# _You can add your calculations in_ $\LaTeX$ _here_. 
# 
# $$
# \begin{align}
# L(\omega) &= \sum_t \log \left[ \frac{r_t^{c_t}}{c_t!} \exp(-r_t) \right] \\
#           &= \sum_t \log \left[ (r_t)^{c_t} \exp(-r_t)   -\log(c_t!) \right]\\
#           &= \sum_t c_t \log(r_t) + \log(\exp(-r_t))  -\log(c_t!)\\
#           &= \sum_t c_t \log(\exp(w^Ts_t)) -r_t - \log(c_t!) + c_t \log(\Delta t \cdot R) \\
#           &= \sum_t c_t (w^T s_t) - r_t - \log(c_t!) + c_t \log(\Delta t \cdot R) \\
#           &= \sum_t c_t w^Ts_t - \exp(w^T s_t) \Delta \cdot R - \log(c_t!) + c_t \log (\Delta t \cdot R)
# \end{align}
# $$
# 
# $$
# \begin{align}
# \frac{d L(\omega)}{d\omega} &= D_w \left[  \sum_t c_t w^T s_t  - \exp(w^T, s_t) \Delta t \cdot R  - \log(c_t!) + c_t \log(\Delta t \cdot R) \right] \\
# &=  \left[  \sum_t D_w \left(c_t w^T s_t  - \exp(w^T s_t) \Delta t \cdot R  - \log(c_t!) + c_t \log(\Delta t \cdot R)  \right) \right] \\ 
# &= \left[ \sum_t c_t s_t  - \exp(w^T s_t) s_t \Delta t \cdot R + 0 + 0 + 0 \right] \\ 
# &= \left[ \sum_t c_t s_t -\exp(w^T s_t) s_t \Delta t \cdot R \right] \\ 
# &= \sum_t (s_t (c_t - r_t)) 
# \end{align}
# $$

# %% [markdown]
# ### Generate data (2 pts)

# %%
def gen_gauss_rf(D: int, width: float, center: tuple = (0, 0)) -> np.ndarray:
    """
    Generate a Gaussian receptive field.

    Args:
        D (int): Size of the receptive field (DxD).
        width (float): Width parameter of the Gaussian.
        center (tuple, optional): Center coordinates of the receptive field. Defaults to (0, 0).

    Returns:
        np.ndarray: Gaussian receptive field.
    """

    sz = (D - 1) / 2
    x, y = np.meshgrid(np.arange(-sz, sz + 1), np.arange(-sz, sz + 1))
    x = x + center[0]
    y = y + center[1]
    w = np.exp(-(x**2 / width + y**2 / width))
    w = w / np.sum(w.flatten())

    return w

#%% Generate a Gaussian RF
w = gen_gauss_rf(15, 7, (1, 1))

vlim = np.max(np.abs(w))
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(w, cmap="bwr", vmin=-vlim, vmax=vlim)
ax.set_title("Gaussian RF")

# %%
def sample_lnp(
    w: np.array, nT: int, dt: float, R: float, s_i: float, random_seed: int = 10
):
    """Generate samples from an instantaneous LNP model neuron with
    receptive field kernel w.

    Parameters
    ----------

    w: np.array, (Dx * Dy, )
        (flattened) receptive field kernel.

    nT: int
        number of time steps

    dt: float
        duration of a frame in s

    R: float
        rate parameter

    s_i: float
        stimulus intensity peak to peak

    random_seed: int
        seed for random number generator

    Returns
    -------

    c: np.array, (nT, )
        sampled spike counts in time bins

    r: np.array, (nT, )
        mean rate in time bins

    s: np.array, (Dx * Dy, nT)
        stimulus frames used

    Note
    ----

    See equations in task description above for a precise definition
    of the individual parameters.

    """

    rng = np.random.default_rng(random_seed)

    # ------------------------------------------------
    # Generate samples from an instantaneous LNP model
    # neuron with receptive field kernel w. (1 pt)
    # ------------------------------------------------

    # Store all the stimulus frames in a 2D array
    s_all = np.zeros((w.shape[0], nT))

    # Store the mean rate in a 1D array
    r_all = np.zeros(nT)

    # Store the sampled spike counts in a 1D array
    # (spike counts in time bins)
    c_all = np.zeros(nT)
    

    pass
#%% Create the stimulus matrix
def generate_all_stimulus_frames(num_pixels: int, nT: int, s_i: float, rng: np.random.Generator) -> np.ndarray:
    """
    Generates all stimulus frames in one shot.

    Args:
        num_pixels (int): The total number of pixels in the flattened frame.
        nT (int): The number of time steps (frames).
        s_i (float): The peak-to-peak intensity of the stimulus.
        rng (np.random.Generator): A NumPy random number generator instance.

    Returns:
        np.ndarray: A 2D array of shape (num_pixels, nT) representing all stimulus frames.
    """
    # Generate random binary values (-1 or 1) for all pixels and all time steps
    random_binary_patterns = rng.choice([-1, 1], size=(num_pixels, nT))
    
    # Scale to achieve the desired peak-to-peak intensity
    s_all = random_binary_patterns * (s_i / 2.0)
    
    return s_all

#%% Calculate the firing rate for each frame 
def calculate_firing_rate(stim_frame: np.ndarray, w: np.ndarray, dt: float, R: float) -> float:
    """
    Calculates the instantaneous firing rate r_t for an LNP neuron.

    Args:
        stim_frame (np.ndarray): The current flattened stimulus frame.
        w (np.ndarray): The flattened receptive field kernel.
        dt (float): Duration of a frame in s.
        R (float): Rate parameter.

    Returns:
        float: The calculated mean firing rate r_t.
    """
    # Linear Filtering (w^T s_t)
    linear_response = np.dot(w, stim_frame)
    
    # Non-linearity and Rate Calculation
    # r_t = exp(w^T s_t) * dt * R
    rate_t = np.exp(linear_response) * dt * R
    
    return rate_t

#%%
# %%
def sample_lnp(
    w: np.array, nT: int, dt: float, R: float, s_i: float, random_seed: int = 10
):
    """Generate samples from an instantaneous LNP model neuron with
    receptive field kernel w.

    Parameters
    ----------

    w: np.array, (Dx * Dy, )
        (flattened) receptive field kernel.

    nT: int
        number of time steps

    dt: float
        duration of a frame in s

    R: float
        rate parameter

    s_i: float
        stimulus intensity peak to peak

    random_seed: int
        seed for random number generator

    Returns
    -------

    c: np.array, (nT, )
        sampled spike counts in time bins

    r: np.array, (nT, )
        mean rate in time bins

    s: np.array, (Dx * Dy, nT)
        stimulus frames used

    Note
    ----

    See equations in task description above for a precise definition
    of the individual parameters.

    """
    
    rng = np.random.default_rng(random_seed)

    # ------------------------------------------------
    # Generate samples from an instantaneous LNP model
    # neuron with receptive field kernel w. (1 pt)
    # ------------------------------------------------

    # Store all the stimulus frames in a 2D array
    s_all = generate_all_stimulus_frames(w.shape[0], nT, s_i, rng)

    # Store the mean rate in a 1D array
    r_all = np.zeros(nT)

    # Store the sampled spike counts in a 1D array
    # (spike counts in time bins)
    c_all = np.zeros(nT)

    for t in range(nT):
        r_all[t] = calculate_firing_rate(s_all[:, t], w, dt, R)
        c_all[t] = rng.poisson(r_all[t])

    return c_all, r_all, s_all

# %%
D = 15  # number of pixels
nT = 1000  # number of time bins
dt = 0.1  # bins of 100 ms
R = 50  # firing rate in Hz
s_i = 5 # stimulus intensity

w = gen_gauss_rf(D, 7, (1, 1))
w = w.flatten()

c, r, s = sample_lnp(w, nT, dt, R, s_i)

# %% [markdown]
# Plot the stimulus for one frame, the cell's response 
# over time and the spike count vs firing rate.

# %%
mosaic = [["stim", "responses", "count/rate"]]

fig, ax = plt.subplot_mosaic(mosaic=mosaic, figsize=(15, 4))

# -----------------------------------------------------------------------------------------------------------
# Plot the stimulus for one frame, the cell's responses 
# over time and spike count vs firing rate (1 pt)
# -----------------------------------------------------------------------------------------------------------

# %% [markdown]
# ### Implementation (3 pts)
# 
# Implement the negative log-likelihood of the LNP and its gradient with respect to the receptive field using the simplified equations you calculated earlier _(1 pt)_

# %%
def negloglike_lnp(
    w: np.array, c: np.array, s: np.array, dt: float = 0.1, R: float = 50
) -> float:
    """Implements the negative (!) log-likelihood of the LNP model

    Parameters
    ----------

    w: np.array, (Dx * Dy, )
      current receptive field

    c: np.array, (nT, )
      spike counts

    s: np.array, (Dx * Dy, nT)
      stimulus matrix


    Returns
    -------

    f: float
      function value of the negative log likelihood at w

    """

    # ------------------------------------------------
    # Implement the negative log-likelihood of the LNP
    # ------------------------------------------------

    pass


def deriv_negloglike_lnp(
    w: np.array, c: np.array, s: np.array, dt: float = 0.1, R: float = 50
) -> np.array:
    """Implements the gradient of the negative log-likelihood of the LNP model

    Parameters
    ----------

    see negloglike_lnp

    Returns
    -------

    df: np.array, (Dx * Dy, )
      gradient of the negative log likelihood with respect to w

    """

    # --------------------------------------------------------------
    # Implement the gradient with respect to the receptive field `w`
    # --------------------------------------------------------------

    pass

# %% [markdown]
# The helper function `check_grad` in `scipy.optimize` can help you to make sure your equations and implementations are correct. It might be helpful to validate the gradient before you run your optimizer.

# %%
# Check gradient

# %% [markdown]
# Fit receptive field maximizing the log likelihood.
# 
# The scipy.optimize package also has suitable functions for optimization. If you generate a large number of samples, the fitted receptive field will look more similar to the true receptive field. With more samples, the optimization takes longer, however.

# %%
# ------------------------------------------
# Estimate the receptive field by maximizing
# the log-likelihood (or more commonly,
# minimizing the negative log-likelihood).
#
# Tips: use scipy.optimize.minimize(). (1 pt)
# ------------------------------------------

# %% [markdown]
# Plot the true and the estimated receptive field.

# %%
# ------------------------------------
# Plot the ground truth and estimated
# `w` side by side. (1 pt)
# ------------------------------------

mosaic = [["True", "Estimated"]]
fig, ax = plt.subplot_mosaic(mosaic=mosaic, figsize=(12, 5))

# make sure to add a colorbar. 'bwr' is a reasonable choice for the cmap.

# %% [markdown]
# # Task 2: Apply to real neuron
# 
# Download the dataset for this task from Ilias (`nds_cl_5_data.mat`). It contains a stimulus matrix (`s`) in the same format you used before and the spike times. In addition, there is an array called `trigger` which contains the times at which the stimulus frames were swapped.
# 
# * Generate an array of spike counts at the same temporal resolution as the stimulus frames
# * Fit the receptive field with time lags of 0 to 4 frames. Fit them one lag at a time (the ML fit is very sensitive to the number of parameters estimated and will not produce good results if you fit the full space-time receptive field for more than two time lags at once). 
# * Plot the resulting filters
# 
# *Grading: 3.5 pts*
# 

# %%
var = io.loadmat("../data/nds_cl_5_data.mat")

# t contains the spike times of the neuron
t = var["DN_spiketimes"].flatten()

# trigger contains the times at which the stimulus flipped
trigger = var["DN_triggertimes"].flatten()

# contains the stimulus movie with black and white pixels
s = var["DN_stim"]
s = s.reshape((300, 1500))  # the shape of each frame is (20, 15)
s = s[:, 1 : len(trigger)]

# %% [markdown]
# Create vector of spike counts

# %%
# ------------------------------------------
# Bin the spike counts at the same temporal
# resolution as the stimulus (0.5 pts)
# ------------------------------------------

# %% [markdown]
# Fit receptive field for each frame separately

# %%
# ------------------------------------------
# Fit the receptive field with time lags of
# 0 to 4 frames separately (1 pt)
#
# The final receptive field (`w_hat`) should
# be in the shape of (Dx * Dy, 5)
# ------------------------------------------

# specify the time lags
delta = [0, 1, 2, 3, 4]

# fit for each delay

# %% [markdown]
# Plot the frames one by one and explain what you see.

# %%
# --------------------------------------------
# Plot all 5 frames of the fitted RFs (1 pt)
# --------------------------------------------

fig, ax = plt.subplot_mosaic(mosaic=[delta], figsize=(10, 4), constrained_layout=True)

# %% [markdown]
# _Explanation (1 pt)_
# ...

# %% [markdown]
# # Task 3: Separate space/time components
# 
# The receptive field of the neuron can be decomposed into a spatial and a temporal component. Because of the way we computed them, both are independent and the resulting spatio-temporal component is thus called separable. As discussed in the lecture, you can use singular-value decomposition to separate these two: 
# 
# $$
# W = u_1 s_1 v_1^T
# $$
# 
# Here $u_1$ and $v_1$ are the singular vectors belonging to the 1st singular value $s_1$ and provide a long rank approximation of W, the array with all receptive fields. It is important that the mean is subtracted before computing the SVD.  
# 
# Plot the first temporal component and the first spatial component. You can use a Python implementation of SVD. The results can look a bit puzzling, because the sign of the components is arbitrary.
# 
# *Grading: 1.5 pts*

# %%
# --------------------------------------------
# Apply SVD to the fitted receptive field,
# you can use either numpy or sklearn (0.5 pt)
# --------------------------------------------



# %%
# -------------------------------------------------
# Plot the spatial and temporal components (1 pt)
# -------------------------------------------------

fig, ax = plt.subplot_mosaic(
    mosaic=[["Spatial", "Temporal"]], figsize=(10, 4), constrained_layout=True
)
# add plot

# %% [markdown]
# # Task 4: Regularized receptive field
# 
# As you can see, maximum likelihood estimation of linear receptive fields can be quite noisy, if little data is available. 
# 
# To improve on this, one can regularize the receptive field vector and a term to the cost function
# 
# 
# $$
# C(w) = L(w) + \alpha ||w||_p^2
# $$
# 
# Here, the $p$ indicates which norm of $w$ is used: for $p=2$, this is shrinks all coefficient equally to zero; for $p=1$, it favors sparse solutions, a penality also known as lasso. Because the 1-norm is not smooth at zero, it is not as straightforward to implement "by hand". 
# 
# Use a toolbox with an implementation of the lasso-penalization and fit the receptive field. Possibly, you will have to try different values of the regularization parameter $\alpha$. Plot your estimates from above and the lasso-estimates. How do they differ? What happens when you increase or decrease $alpha$?
# 
# If you want to keep the Poisson noise model, you can use the implementation in [`pyglmnet`](https://pypi.python.org/pypi/pyglmnet). Otherwise, you can also resort to the linear model from `sklearn` which assumes Gaussian noise (which in my hands was much faster).
# 
# *Grading: 3 pts*
# 

# %%
from sklearn import linear_model

# ------------------------------------------
# Fit the receptive field with time lags of
# 0 to 4 frames separately (the same as before)
# with sklearn or pyglmnet for different values
# of alpha (1 pt)
# ------------------------------------------

delta = [0, 1, 2, 3, 4]
alphas= []

# %%
# ------------------------------------------
# plot the estimated receptive fields (1 pt)
# ------------------------------------------


fig, ax = plt.subplots(
    len(alphas), len(delta), figsize=(10, 4), constrained_layout=True
)# add plot

# %% [markdown]
# _Explanation (1 pt)_
# ...

# %% [markdown]
# ## Bonus Task (Optional): Spike Triggered Average
# 
# Instead of the Maximum Likelihood implementation above, estimate the receptive field using the spike triggered average.
# Use it to increase the temporal resolution of your receptive field estimate.
# Perform the SVD analysis for your STA-based receptive field and plot the spatial and temporal kernel as in Task 3.
# 
# **Questions:**
# 1. Explain how / why you chose a specific time delta.
# 2. Reconsider what you know about STA. Is it suitable to use STA for this data? Why/why not? What are the (dis-)advantages of using the MLE based method from above?
# 
# _Grading: 1 BONUS Point._
# 
# 
# _BONUS Points do not count for this individual coding lab, but sum up to 5% of your **overall coding lab grade**. There are 4 BONUS points across all coding labs._


