########################################################################################################################
# Packages
########################################################################################################################

# General
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# Special
from category_encoders import target_encoder
from sklearn.base import BaseEstimator, TransformerMixin

# Custom functions and classes
import utils_plots as up



########################################################################################################################
# Parameter
########################################################################################################################

# Locations
DATALOC = "../data/"
PLOTLOC = "../output/"

# Number of cpus
N_JOBS = 4

# Util
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 20)

# Plot
sns.set(style="whitegrid")
plt.rcParams["axes.edgecolor"] = "black"

# Colors
COLORTWO = ["green", "red"]
COLORTHREE = ["green", "yellow", "red"]
COLORMANY = np.delete(np.array(list(mcolors.BASE_COLORS.values()) + list(mcolors.CSS4_COLORS.values()), dtype=object),
                      np.array([4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 26]))
# sel = np.arange(50); fig, ax = plt.subplots(figsize=(5,15)); ax.barh(sel.astype("str"), 1, color=COLORMANY[sel])
COLORBLIND = list(sns.color_palette("colorblind").as_hex())
COLORDEFAULT = list(sns.color_palette("tab10").as_hex())

