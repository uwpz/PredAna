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


########################################################################################################################
# Custom functions and classes
########################################################################################################################

def identity(x):
    return x


def nume_log(x):  
    return np.log(x + 1)


class Etl(BaseEstimator, TransformerMixin):
    def __init__(self, derive_day_of_month=True, miss=None,
                 cate_fill_na=None, toomany=None, df_util=None, target_name="target"):
        self.derive_day_of_month = derive_day_of_month
        self.miss = miss
        self.cate_fill_na = cate_fill_na
        self.toomany = toomany
        self.df_util = df_util
        self.target_name = target_name

    def fit(self, df, *_):
        if self.toomany is not None:
            self._toomany_encoder = (target_encoder.TargetEncoder(cols=self.toomany)
                                     .fit(self.df_util[self.toomany] if self.df_util is not None 
                                          else df[self.toomany],
                                          self.df_util[self.target_name] if self.df_util is not None 
                                          else df[self.target_name]))
        return self

    def transform(self, df, *_):
        if self.derive_day_of_month:
            df["day_of_month"] = df["dteday"].dt.day.astype("str").str.zfill(2)
        if self.miss is not None:
            df[up.add("MISS_", self.miss)] = pd.DataFrame(np.where(df[self.miss].isnull(), "No", "Yes"))
        if self.cate_fill_na is not None:
            df[self.cate_fill_na] = df[self.cate_fill_na].fillna("(Missing)").replace("nan", "(Missing)")
        if self.toomany is not None:
            df[up.add(self.toomany, "_ENCODED")] = self._toomany_encoder.transform(df[self.toomany])
        return df
