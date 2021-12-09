########################################################################################################################
# Initialize: Packages, functions, parameters
########################################################################################################################

# --- Packages ------------------------------------------------------------------------------------

# General
from scipy.interpolate import splev, splrep, make_interp_spline
import seaborn as sns
import matplotlib.colors as mcolors
import os  # sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # ,matplotlib
import pickle
import importlib  # importlib.reload(my)
import time

# Special
import hmsPM.plotting as hms_plot

# Custom functions and classes
import my_utils as my

a = pd.Series(np.array(['a', 'b']))
b = pd.Series(np.array(['a', 1]))
pd.api.types.is_object_dtype(a)
pd.api.types.is_string_dtype(a)
pd.api.types.is_object_dtype(b)
pd.api.types.is_string_dtype(b)




# --- Parameter --------------------------------------------------------------------------

# Main parameter
TARGET_TYPE = "CLASS"

# Specific parameters
n_jobs = 4

# Locations
DATALOC = "../data/"
PLOTLOC = "../output/"

# Load results from exploration
df = nume_standard = cate_standard = cate_binned = nume_encoded = None
with open(DATALOC + "1_explore.pkl", "rb") as file:
    d_pick = pickle.load(file)
for key, val in d_pick.items():
    exec(key + "= val")

# Adapt targets
df["cnt_CLASS"] = df["cnt_CLASS"].str.slice(0, 1).astype("int")
df["cnt_MULTICLASS"] = df["cnt_MULTICLASS"].str.slice(0, 1).astype("int")


########################################################################################################################
# Scratch
########################################################################################################################

# blub
a = df[nume].apply(lambda x:plt.plot(x)).iloc[0,:].values
%matplotlib inline
a.iloc[0,1]
plt.plot(1)

# fill missing
df = pd.DataFrame({"a": range(1,5), "b": [1, 2, 3, np.nan]})
df.dtypes
df = df.astype("object")
df = df.astype("str").replace("nan", None)
df = df.fillna("missing")
df
df.replace("nan", "Missing")

# Exchange axes
%matplotlib inline
TARGET_TYPE = "CLASS"  
distr_nume_plots = (hms_plot.MultiFeatureDistributionPlotter(n_rows=2, n_cols=3, w=18, h=12,
                                                             show_regplot=True)
                    .plot(features=df[nume],
                          target=df["cnt_" + TARGET_TYPE],
                          file_path=PLOTLOC + "distr_nume__" + TARGET_TYPE + ".pdf"))

page = 0
old_page = distr_nume_plots[page]
old_fig = old_page[0]
old_axes = old_page[1]
old_ax = old_axes[0,2]
old_ax.set_title("My New Title")
old_fig
old_fig.set_size_inches(6,6)
old_fig.tight_layout()
old_fig

old_fig.savefig("blub.pdf")

distr_cate_plots = (hms_plot.MultiFeatureDistributionPlotter(n_rows=2, n_cols=3, w=18, h=12)
                            .plot(features=df[np.append(cate, ["MISS_" + miss])],
                                  target=df["cnt_" + TARGET_TYPE],
                                  varimps=varperf_cate,
                                  file_path=PLOTLOC + "distr_cate__" + TARGET_TYPE + ".pdf"))

for page in distr_cate_plots:
    fig = page[0]
    axes = page[1]
    for i, ax in enumerate(axes.flatten()):
        ax.set_title(ax.get_title().replace("VI", "AUC"))
        if i >= 0:
            if ax.get_legend() is not None:           
                ax.get_legend().remove()
    #fig.tight_layout()
hms_plot.save_plot_grids_to_pdf(distr_cate_plots, "blub1.pdf")
for page in distr_cate_plots:
    display(page[0])

# DO NOT CHANGE GEOMETRY
new_fig, new_axes = plt.subplots(3, 2)
new_ax = new_axes[2,1]
new_fig.set_size_inches(6, 6)
new_fig.tight_layout()
old_ax.change_geometry(*(new_ax.get_geometry()))
#old_ax._position = new_ax._position
#old_ax.pchanged()
new_ax.remove()
old_ax.figure = new_fig
new_fig.add_axes(old_ax)
new_fig


'''
# Fancy!

fig, ax = plt.subplots(1,2)
ax[1].plot(1,1)
ax[1].patch.set_facecolor('xkcd:light yellow')


from hmsPM.plotting.output import save_plot_grids_to_pdf
from hmsPM.plotting.distribution import FeatureDistributionPlotter
from hmsPM.plotting.grid import PlotGridBuilder
from hmsPM.datatypes import PlotFunctionCall
plot_calls = []
features = np.concatenate([nume, cate[[0, 4, 5, 6, 7]]])
for row in features:
    for col in features:
        if row == col:
            plot_calls.append(PlotFunctionCall(FeatureDistributionPlotter(show_regplot=True).plot,
                                               kwargs=dict(feature=df[row], target=df["cnt_CLASS"])))
        else:
            plot_calls.append(PlotFunctionCall(FeatureDistributionPlotter(show_regplot=True).plot,
                                               kwargs=dict(feature=df[row], target=df[col])))
plot_grids = PlotGridBuilder(n_rows=len(features), n_cols=len(features), h=60, w=60).build(plot_calls=plot_calls)
for i in range(len(features)):
    plot_grids[0][1][i,i].set_facecolor('xkcd:light yellow')

save_plot_grids_to_pdf(plot_grids, PLOTLOC + "fancy.pdf")
'''


fig, ax = plt.subplots(1, 1)
ax_act = ax
tmp_scale = 1
tmp_cmap = mcolors.LinearSegmentedColormap.from_list("wh_bl_yl_rd",
                                                     [(1, 1, 1, 0), "blue", "yellow", "red"])
p = ax_act.hexbin(df["temp"], df["cnt_REGR"],
                  gridsize=(int(50 * tmp_scale), 50),
                  cmap=tmp_cmap)
plt.colorbar(p, ax=ax_act)
sns.regplot(x, y, lowess=True, scatter=False, color="black", ax=ax_act)

df_tmp = df[["cnt_REGR", "temp"]].groupby("temp")[["cnt_REGR"]].mean().reset_index().sort_values("temp")

x = df_tmp["temp"].values
y = df_tmp["cnt_REGR"].values
spl = splrep(x, y)
inter = make_interp_spline(x, y)
x2 = np.quantile(x, np.arange(0.01, 1, 0.01))
y2 = splev(x2, spl)
y2
y2 = inter(x2)
y2
ax_act.plot(x2, y2, color="r")


df["cnt_CLASS"][4] = np.nan
df["weathersit"][4:10] = np.nan

up.colorblind[1]

f_nume = "temp"
f_cate = "weathersit"

#%%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
up.plot_feature_target(ax, df[f_cate].values, df["cnt_REGR"].values,
                       feature_name="feature", target_name="target",
                       target_category="0_low",
                       #target_lim=(2,6),
                       feature_lim=(10, 30), add_boxplot=True,
                       add_feature_distribution=True,
                       add_target_distribution=True,
                       min_width=0.5, inset_size=0.2, n_bins=50,
                       title="title",
                       add_miss_info=True)
