########################################################################################################################
# Packages
########################################################################################################################

# General
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from joblib import Parallel, delayed
import copy
import warnings
import time

# Scikit
from sklearn.metrics import (make_scorer, roc_auc_score, accuracy_score, roc_curve, confusion_matrix,
                             precision_recall_curve, average_precision_score)
from sklearn.model_selection import cross_val_score, GridSearchCV, check_cv, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, MinMaxScaler
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils import _safe_indexing
from sklearn.base import BaseEstimator, TransformerMixin, clone  # ClassifierMixin
from sklearn.calibration import calibration_curve

# ML
import xgboost as xgb
import lightgbm as lgbm
from itertools import product  # for GridSearchCV_xlgb

# Other
from scipy.interpolate import splev, splrep


########################################################################################################################
# Parameter
########################################################################################################################

# Colors
twocol = ["red", "green"]
threecol = ["green", "yellow", "red"]
manycol = np.delete(np.array(list(mcolors.BASE_COLORS.values()) + list(mcolors.CSS4_COLORS.values()), dtype=object),
                    np.array([4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 26]))
colorblind = sns.color_palette("colorblind", as_cmap=True)
# sel = np.arange(50); fig, ax = plt.subplots(figsize=(5,15)); ax.barh(sel.astype("str"), 1, color=manycol[sel])


########################################################################################################################
# General Functions
########################################################################################################################

# --- General ----------------------------------------------------------------------------------------

def debugtest(a=1, b=2):
    print(a)
    print(b)
    print("blub")
    #print("blub2")
    #print("blub3")
    print(1)
    print(2)
    return "done"


def diff(a, b):
    return np.setdiff1d(a, b, True)


def logit(p):
    return(np.log(p / (1 - p)))


def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))


# Show closed figure again
def show_figure(fig):
    # create a dummy figure and use its manager to display "fig"
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    

# Plot list of tuples (plot_call, kwargs)
def plot_function_calls(l_calls, n_rows=2, n_cols=3, figsize=(18, 12), pdf_path=None):
    
    # TODO: return fig-list
    # TODO: l_calls -> calls can be list or dict

    # Open pdf
    if pdf_path is not None:
        pdf_pages = PdfPages(pdf_path)
    else:
        pdf_pages = None

    for i, (plot_func, kwargs) in enumerate(l_calls):
        # Init new page
        if i % (n_rows * n_cols) == 0:
            fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
            i_ax = 0

        # Plot call
        plot_func(ax=ax.flat[i_ax] if (n_rows * n_cols > 1) else ax, **kwargs)
        fig.tight_layout()
        i_ax += 1

        # "Close" page
        if (i_ax == n_rows * n_cols) or (i == len(l_calls) - 1):
            # Remove unused axes
            if (i == len(l_calls) - 1):
                for k in range(i_ax, n_rows * n_cols):
                    ax.flat[k].axis("off")

            # Write pdf
            if pdf_path is not None:
                pdf_pages.savefig(fig)

    # Close pdf
    if pdf_path is not None:
        pdf_pages.close()


# --- Metrics ----------------------------------------------------------------------------------------

# Regr

def spear(y_true, y_pred):
    # Also for classification
    if y_pred.ndim == 2:
        if y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
    return pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).corr(method="spearman").values[0, 1]


def pear(y_true, y_pred):
    # Also for classification
    if y_pred.ndim == 2:
        if y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
    return pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).corr(method="pearson").values[0, 1]


def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))


def ame(y_true, y_pred):
    return np.abs(np.mean(y_true - y_pred))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

#def myrmse(y_true, y_pred):
#    return np.sqrt(np.mean(np.power(y_true + 0.03 - y_pred, 2)))


# Class + Multiclass

def auc(y_true, y_pred):
    if y_pred.ndim == 2:
        if y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
            
    # Also for regression
    if y_pred.ndim == 1:
        if (np.min(y_pred) < 0) | (np.max(y_pred) > 1):
            y_pred = MinMaxScaler().fit_transform(y_pred.reshape(-1, 1))[:, 0]
    if y_true.ndim == 1:
        if np.max(y_true) > 1:
            y_true = np.where(y_true>1, 1, np.where(y_true<1, 0, y_true))
 
    return roc_auc_score(y_true, y_pred, multi_class="ovr")


def acc(y_true, y_pred):
    if y_pred.ndim > 1:
        y_pred = y_pred.argmax(axis=1)
    if y_true.ndim > 1:
        y_true = y_true.values.argmax(axis=1)
    return accuracy_score(y_true, y_pred)


# Scoring metrics
d_scoring = {"REGR": {"spear": make_scorer(spear, greater_is_better=True),
                      "rmse": make_scorer(rmse, greater_is_better=False),
                      "ame": make_scorer(ame, greater_is_better=False),
                      "mae": make_scorer(mae, greater_is_better=False)},
             "CLASS": {"auc": make_scorer(auc, greater_is_better=True, needs_proba=True),
                       "acc": make_scorer(acc, greater_is_better=True)},
             "MULTICLASS": {"auc": make_scorer(auc, greater_is_better=True, needs_proba=True),
                            "acc": make_scorer(acc, greater_is_better=True)}}


########################################################################################################################
# Explore
#######################################################################################################################

# --- Non-plots --------------------------------------------------------------------------

# Overview of values
def value_counts(df, topn=5, dtypes=["object"]):
    df_tmp = df.select_dtypes(dtypes)
    return pd.concat([(df_tmp[catname].value_counts().iloc[: topn].reset_index()
                       .rename(columns={"index": catname, catname: "#"}))
                      for catname in df_tmp.columns.values],
                     axis=1).fillna("")


# Univariate model performance
def variable_performance(feature, target, scorer, target_type=None, splitter=KFold(5), groups=None):

    # Drop all missings
    df_hlp = pd.DataFrame().assign(feature=feature, target=target)
    if groups is not None:
        df_hlp["groups_for_split"] = groups
    df_hlp = df_hlp.dropna().reset_index(drop=True)
    
    # Detect types
    if target_type is None:
        target_type = dict(continuous="REGR", binary="CLASS", 
                           multiclass="MULTICLASS")[type_of_target(df_hlp["target"])]
    numeric_feature = pd.api.types.is_numeric_dtype(df_hlp["feature"])
    
    # Calc performance
    perf = np.mean(cross_val_score(
        estimator=(LinearRegression() if target_type == "REGR" else LogisticRegression()),
        X=(KBinsDiscretizer().fit_transform(df_hlp[["feature"]]) if numeric_feature else 
           OneHotEncoder().fit_transform(df_hlp[["feature"]])),
        y=df_hlp["target"],
        cv=splitter.split(df_hlp, groups=df_hlp["groups_for_split"] if groups is not None else None),
        scoring=scorer))
    
    return perf
    

# Winsorize
class Winsorize(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=None, upper_quantile=None):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        #self._private_attr = whatever

    def fit(self, X, *_):
        X = pd.DataFrame(X)
        if self.lower_quantile is not None:
            self.a_lower_ = np.nanquantile(X, q=self.lower_quantile, axis=0)
        else:
            self.a_lower_ = None
        if self.upper_quantile is not None:
            self.a_upper_ = np.nanquantile(X, q=self.upper_quantile, axis=0)
        else:
            self.a_uppe_ = None
        return self

    def transform(self, X, *_):
        if (self.lower_quantile is not None) or (self.upper_quantile is not None):
            X = np.clip(X, a_min=self.a_lower_, a_max=self.a_upper_)
        return X
    

# Map Non-topn frequent members of a string column to "other" label
class Collapse(BaseEstimator, TransformerMixin):
    def __init__(self, n_top=10, other_label="_OTHER_"):
        self.n_top = n_top
        self.other_label = other_label      

    def fit(self, X, *_):
        self.d_top_ = pd.DataFrame(X).apply(lambda x: x.value_counts().index.values[:self.n_top])
        return self

    def transform(self, X):
        X = pd.DataFrame(X).apply(lambda x: x.where(np.in1d(x, self.d_top_[x.name]),
                                                    other=self.other_label)).values
        return X 
    

# Impute Mode (simpleimputer is too slow)
class ImputeMode(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X):
        self.impute_values_ = pd.DataFrame(X).mode().iloc[0]
        return self

    def transform(self, X):
        X = pd.DataFrame(X).fillna(self.impute_values_).values
        return X
  
  
# --- Plots --------------------------------------------------------------------------

def helper_calc_barboxwidth(feature, target, min_width=0.2):
    df_hlp = pd.crosstab(feature, target)
    df_barboxwidth = (df_hlp.div(df_hlp.sum(axis=1), axis=0)
                      .assign(w=df_hlp.sum(axis=1))
                      .reset_index()
                      .assign(pct=lambda z: 100 * z["w"] / df_hlp.values.sum())
                      .assign(w=lambda z: 0.9 * z["w"] / z["w"].max())
                      .assign(**{feature.name + "_fmt":
                                 lambda z: z[feature.name] + z["pct"].map(" ({:,.1f} %)".format)})
                      .assign(w=lambda z: np.where(z["w"] < min_width, min_width, z["w"])))
    return df_barboxwidth


def helper_adapt_feature_target(feature, target, feature_label, target_label):
    # Convert to Series and adapt labels
    if not isinstance(feature, pd.Series):
        feature = pd.Series(feature)
        feature.name = feature_label if feature_label is not None else "x"
    if not isinstance(target, pd.Series):
        target = pd.Series(target)
        target.name = target_label if target_label is not None else "y"
    return (feature, target)


def helper_inner_barplot(ax, x, y, inset_size=0.2): 
    xticks = ax.get_xticks()
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 1.5 * inset_size * (xlim[1] - xlim[0]))
    inset_ax = ax.inset_axes([0, 0, inset_size, 1], zorder=10)
    inset_ax.set_axis_off()
    ax.axvline(xlim[0], color="black")
    ax.get_shared_y_axes().join(ax, inset_ax)
    inset_ax.barh(x, y,
                  color="lightgrey", edgecolor="black", linewidth=1)
    _ = ax.set_xticks(xticks[xticks > xlim[0]])


def plot_cate_CLASS(ax,
                    feature, target,
                    feature_label=None, target_label=None, target_category=None,
                    target_lim=None,
                    min_width=0.2, inset_size=0.2, refline=True,
                    title=None, varimp=None, varimp_fmt="0.2f",
                    color=colorblind[1]):

    # Adapt feature and target
    feature, target = helper_adapt_feature_target(feature, target, feature_label, target_label)

    # Add title
    if title is None:
        title = feature.name
    #if varimp is not None:
    #    title = title + " (VI: " + format(varimp, varimp_fmt) + ")"

    # Get "1" class
    if target_category is None:
        target_category = target.value_counts().sort_values().index.values[0]

    # Prepare data
    df_plot = helper_calc_barboxwidth(feature, target, min_width=min_width)

    # Barplot
    ax.barh(df_plot[feature.name + "_fmt"], df_plot[target_category], height=df_plot["w"],
            color=color, edgecolor="black", alpha=0.5, linewidth=1)
    ax.set_xlabel("avg(" + target.name + ")")
    ax.set_title(title)
    if target_lim is not None:
        ax.set_xlim(target_lim)

    # Refline
    if refline:
        ax.axvline((target == target_category).sum() / len(target), ls="dotted", color="black")

    # Inner barplot
    helper_inner_barplot(ax, x=df_plot[feature.name + "_fmt"], y=df_plot["pct"], inset_size=inset_size)


def plot_cate_REGR(ax, feature, target, feature_label=None, target_label=None,
                   target_lim=None,
                   min_width=0.2, inset_size=0.2, refline=True,
                   title=None, varimp=None, varimp_fmt="0.2f",
                   color=colorblind[1]):

    # Adapt feature and target
    feature, target = helper_adapt_feature_target(feature, target, feature_label, target_label)

    # Add title
    if title is None:
        title = feature.name
    #if varimp is not None:
    #    title = title + " (VI: " + format(varimp, varimp_fmt) + ")"

    # Prepare data
    df_plot = helper_calc_barboxwidth(feature, np.tile("dummy", len(feature)),
                                      min_width=min_width)

    # Barplot
    _ = ax.boxplot([target[feature == value] for value in df_plot[feature.name].values],
                   labels=df_plot[feature.name + "_fmt"].values,
                   widths=df_plot["w"].values,
                   vert=False,
                   patch_artist=True,
                   showmeans=True,
                   boxprops=dict(facecolor=color, alpha=0.5),
                   #capprops=dict(color=color),
                   #whiskerprops=dict(color=color),
                   medianprops=dict(color="black"),
                   meanprops=dict(marker="x",
                                  markeredgecolor="black"),
                   flierprops=dict(marker="."))
    ax.set_xlabel(target.name)
    ax.set_title(title)
    if target_lim is not None:
        ax.set_xlim(target_lim)

    # Refline
    if refline:
        ax.axvline(target.mean(), ls="dotted", color="black")
 
    # Inner barplot
    helper_inner_barplot(ax, x=np.arange(len(df_plot)) + 1, y=df_plot["pct"], 
                         inset_size=inset_size)    
    
def plot_nume_CLASS(ax, 
                    feature, target, 
                    feature_label=None, target_label=None, target_category=None,
                    feature_lim=None, 
                    n_bins=20, 
                    title=None,
                    color=colorblind,
                    inset_size=0.2):
    
    # Adapt feature and target
    feature, target = helper_adapt_feature_target(feature, target, feature_label, target_label)

    # Add title
    if title is None:
        title = feature.name
    #if varimp is not None:
    #    title = title + " (VI: " + format(varimp, varimp_fmt) + ")"
    
    # Adapt color
    color = color[:target.nunique()]

    # Distribution plot
    sns.histplot(ax=ax, x=feature, hue=target, stat="density", common_norm=False, kde=True, bins=n_bins, 
                 palette=color)
    ax.set_ylabel("Density")
    ax.set_title(title)
    if feature_lim is not None:
        ax.set_xlim(feature_lim)

    # Inner Boxplot
    yticks = ax.get_yticks()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0] - 1.5 * inset_size * (ylim[1] - ylim[0]))
    inset_ax = ax.inset_axes([0, 0, 1, inset_size])
    inset_ax.set_axis_off()
    ax.axhline(ylim[0], color="black")
    ax.get_shared_x_axes().join(ax, inset_ax)
    sns.boxplot(ax=inset_ax, x=feature, y=target, orient="h", palette=color,
                showmeans=True, meanprops={"marker": "x", "markerfacecolor": "black", "markeredgecolor": "black"})
    _ = ax.set_yticks(yticks[yticks > ylim[0]])


# Scatterplot as heatmap
def plot_nume_REGR(ax,  
                   feature, target,
                   feature_label=None, target_label=None,
                   feature_lim=None, target_lim=None,
                   regplot=False, smooth=0.5,    
                   refline=True,               
                   title=None,
                   add_colorbar=True,
                   inset_size=0.2,
                   add_y_density=True, add_x_density=True, n_bins=20, add_boxplot=True,                  
                   color=LinearSegmentedColormap.from_list("bl_yl_rd", ["blue", "yellow", "red"])):

    # Adapt feature and target
    feature, target = helper_adapt_feature_target(feature, target, feature_label, target_label)
        
    # Add title
    if title is None:
        title = feature.name

    '''
    # Helper for scaling of heat-points
    heat_scale = 1
    if ylim is not None:
        ax.set_ylim(ylim)
        heat_scale = heat_scale * (ylim[1] - ylim[0]) / (np.max(y) - np.min(y))
    if xlim is not None:
        ax.set_xlim(xlim)
        heat_scale = heat_scale * (xlim[1] - xlim[0]) / (np.max(feature) - np.min(feature))
    '''

    # Heatmap
    #p = ax.hexbin(feature, target, gridsize=(int(50 * heat_scale), 50), mincnt=1, cmap=color)
    p = ax.hexbin(feature, target, mincnt=1, cmap=color)
    if add_colorbar:
        plt.colorbar(p, ax=ax)

    # Spline
    if regplot:
        if len(feature) < 1000:
            sns.regplot(x=feature, y=target, lowess=True, scatter=False, color="black", ax=ax)
        else:
            df_spline = (pd.DataFrame({"x": feature, "y": target})
                         .groupby("x")[["y"]].agg(["mean", "count"])
                         .pipe(lambda x: x.set_axis([a + "_" + b for a, b in x.columns],
                                                    axis=1, inplace=False))
                         .assign(w=lambda x: np.sqrt(x["y_count"]))
                         .sort_values("x")
                         .reset_index())
            spl = splrep(x=df_spline["x"].values, y=df_spline["y_mean"].values, w=df_spline["w"].values,
                         s=len(df_spline) * smooth)
            x2 = np.quantile(df_spline["x"].values, np.arange(0.01, 1, 0.01))
            y2 = splev(x2, spl)
            ax.plot(x2, y2, color="black")

    # Set labels
    #ax.set_ylabel(target.name)
    #ax.set_xlabel(feature.name)
    if title is not None:
        ax.set_title(title)
        
        
    # Get limits before any insetting
    if feature_lim is None:
        feature_lim = ax.get_xlim()
    if target_lim is None:
        target_lim = ax.get_ylim()
        
    # Add y density
    if add_y_density:
        # Inner Histogram on y
        inset_ax_y = ax.inset_axes([0, 0, inset_size, 1], zorder=10)
        inset_ax_y.get_xaxis().set_visible(False)
        ax.get_shared_y_axes().join(ax, inset_ax_y)
        sns.histplot(y=target, color="grey", stat="density", kde=True, bins=n_bins, ax=inset_ax_y)

        if add_boxplot:
            # Inner-inner Boxplot on y
            xlim_inner = inset_ax_y.get_xlim()
            inset_ax_y.set_xlim(xlim_inner[0] - 1.5 * inset_size * (xlim_inner[1] - xlim_inner[0]))
            inset_inset_ax_y = inset_ax_y.inset_axes([0, 0, inset_size, 1])
            inset_inset_ax_y.set_axis_off()
            inset_ax_y.get_shared_y_axes().join(inset_ax_y, inset_inset_ax_y)
            sns.boxplot(y=target, color="lightgrey", orient="v",
                        showmeans=True, meanprops={"marker": "x",
                                                   "markerfacecolor": "white", "markeredgecolor": "white"},
                        ax=inset_inset_ax_y)

    # Add x density
    if add_x_density:
        # Inner Histogram on x
        inset_ax_x = ax.inset_axes([0, 0, 1, inset_size], zorder=10)
        inset_ax_x.get_yaxis().set_visible(False)
        ax.get_shared_x_axes().join(ax, inset_ax_x)
        sns.histplot(x=feature, color="grey", stat="density", kde=True, bins=n_bins, ax=inset_ax_x)

        if add_boxplot:
            # Inner-inner Boxplot on x
            ylim_inner = inset_ax_x.get_ylim()
            inset_ax_x.set_ylim(ylim_inner[0] - 1.5 * inset_size * (ylim_inner[1] - ylim_inner[0]))
            inset_inset_ax_x = inset_ax_x.inset_axes([0, 0, 1, inset_size])
            inset_inset_ax_x.set_axis_off()
            inset_ax_x.get_shared_x_axes().join(inset_ax_x, inset_inset_ax_x)
            sns.boxplot(x=feature, color="lightgrey",
                        showmeans=True, meanprops={"marker": "x", "markerfacecolor": "white",
                                                   "markeredgecolor": "white"},
                        ax=inset_inset_ax_x)

    '''
    if ylim is not None:
        inset_inset_ax_y.set_ylim(ylim)
        inset_ax_y.set_ylim(ylim)
        
    if xlim is not None:
        inset_inset_ax_x.set_xlim(xlim)
        inset_ax_x.set_xlim(xlim)
    '''

    # Set limits
    #xticks = ax.get_xticks()
    #yticks = ax.get_yticks()
    ax.set_xlim(feature_lim[0] - 1.5 * inset_size * (feature_lim[1] - feature_lim[0]), feature_lim[1])
    ax.set_ylim(target_lim[0] - 1.5 * inset_size * (target_lim[1] - target_lim[0]), target_lim[1])
    #_ = ax.set_xticks(xticks[xticks > feature_lim[0]])
    #_ = ax.set_yticks(yticks[yticks > target_lim[0]])
    
    # Hide intersection
    if add_y_density and add_x_density:
        inset_ax_over = ax.inset_axes([0, 0, inset_size, inset_size], zorder=20)
        inset_ax_over.set_facecolor("white")
        inset_ax_over.get_xaxis().set_visible(False)
        inset_ax_over.get_yaxis().set_visible(False)


# Scatterplot as heatmap
def plot_biscatter(ax, x, y, xlabel=None, ylabel=None,
                   title=None, xlim=None, ylim=None,
                   regplot=False, smooth=0.5,
                   add_y_density=True, add_x_density=True, n_bins=20,
                   add_boxplot=True,
                   inset_size=0.2,
                   add_colorbar=True):

    ax = ax

    # Remove names
    if (xlabel is not None) and isinstance(x, pd.Series):
        x.name = xlabel
    if (ylabel is not None) and isinstance(y, pd.Series):
        y.name = ylabel

    '''
    # Helper for scaling of heat-points
    heat_scale = 1
    if ylim is not None:
        ax.set_ylim(ylim)
        heat_scale = heat_scale * (ylim[1] - ylim[0]) / (np.max(y) - np.min(y))
    if xlim is not None:
        ax.set_xlim(xlim)
        heat_scale = heat_scale * (xlim[1] - xlim[0]) / (np.max(x) - np.min(x))
    '''

    # Heatmap
    heat_cmap = LinearSegmentedColormap.from_list("bl_yl_rd", ["blue", "yellow", "red"])
    #p = ax.hexbin(x, y, gridsize=(int(50 * heat_scale), 50), mincnt=1, cmap=heat_cmap)
    p = ax.hexbin(x, y, mincnt=1, cmap=heat_cmap)
    if add_colorbar:
        plt.colorbar(p, ax=ax)

    # Spline
    if regplot:
        if len(x) < 1000:
            sns.regplot(x=x, y=y, lowess=True, scatter=False, color="black", ax=ax)
        else:
            df_spline = (pd.DataFrame({"x": x, "y": y})
                         .groupby("x")[["y"]].agg(["mean", "count"])
                         .pipe(lambda x: x.set_axis([a + "_" + b for a, b in x.columns],
                                                    axis=1, inplace=False))
                         .assign(w=lambda x: np.sqrt(x["y_count"]))
                         .sort_values("x")
                         .reset_index())
            spl = splrep(x=df_spline["x"].values, y=df_spline["y_mean"].values, w=df_spline["w"].values,
                         s=len(x) * smooth)
            x2 = np.quantile(df_spline["x"].values, np.arange(0.01, 1, 0.01))
            y2 = splev(x2, spl)
            ax.plot(x2, y2, color="black")

    # Set labels
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if title is not None:
        ax.set_title(title)

    # Get limits before any insetting
    if ylim is None:
        ylim = ax.get_ylim()
    if xlim is None:
        xlim = ax.get_xlim()

    # Add y density
    if add_y_density:
        # Inner Histogram on y
        inset_ax_y = ax.inset_axes([0, 0, inset_size, 1], zorder=10)
        inset_ax_y.get_xaxis().set_visible(False)
        ax.get_shared_y_axes().join(ax, inset_ax_y)
        sns.histplot(y=y, color="grey", stat="density", kde=True, bins=n_bins, ax=inset_ax_y)

        if add_boxplot:
            # Inner-inner Boxplot on y
            xlim_inner = inset_ax_y.get_xlim()
            inset_ax_y.set_xlim(xlim_inner[0] - 1.5 * inset_size * (xlim_inner[1] - xlim_inner[0]))
            inset_inset_ax_y = inset_ax_y.inset_axes([0, 0, inset_size, 1])
            inset_inset_ax_y.set_axis_off()
            inset_ax_y.get_shared_y_axes().join(inset_ax_y, inset_inset_ax_y)
            sns.boxplot(y=y, color="lightgrey", orient="v",
                        showmeans=True, meanprops={"marker": "x",
                                                   "markerfacecolor": "white", "markeredgecolor": "white"},
                        ax=inset_inset_ax_y)

    # Add x density
    if add_x_density:
        # Inner Histogram on x
        inset_ax_x = ax.inset_axes([0, 0, 1, inset_size], zorder=10)
        inset_ax_x.get_yaxis().set_visible(False)
        ax.get_shared_x_axes().join(ax, inset_ax_x)
        sns.histplot(x=x, color="grey", stat="density", kde=True, bins=n_bins, ax=inset_ax_x)

        if add_boxplot:
            # Inner-inner Boxplot on x
            ylim_inner = inset_ax_x.get_ylim()
            inset_ax_x.set_ylim(ylim_inner[0] - 1.5 * inset_size * (ylim_inner[1] - ylim_inner[0]))
            inset_inset_ax_x = inset_ax_x.inset_axes([0, 0, 1, inset_size])
            inset_inset_ax_x.set_axis_off()
            inset_ax_x.get_shared_x_axes().join(inset_ax_x, inset_inset_ax_x)
            sns.boxplot(x=x, color="lightgrey",
                        showmeans=True, meanprops={"marker": "x", "markerfacecolor": "white",
                                                   "markeredgecolor": "white"},
                        ax=inset_inset_ax_x)

    '''
    if ylim is not None:
        inset_inset_ax_y.set_ylim(ylim)
        inset_ax_y.set_ylim(ylim)
        
    if xlim is not None:
        inset_inset_ax_x.set_xlim(xlim)
        inset_ax_x.set_xlim(xlim)
    '''

    # Set limits
    ax.set_ylim(ylim[0] - 1.5 * inset_size * (ylim[1] - ylim[0]), ylim[1])
    ax.set_xlim(xlim[0] - 1.5 * inset_size * (xlim[1] - xlim[0]), xlim[1])

    # Hide intersection
    if add_y_density and add_x_density:
        inset_ax_over = ax.inset_axes([0, 0, inset_size, inset_size], zorder=20)
        inset_ax_over.set_facecolor("white")
        inset_ax_over.get_xaxis().set_visible(False)
        inset_ax_over.get_yaxis().set_visible(False)





# TODO HERE
def plot_feature_target(ax,
                        feature, target, feature_type=None, target_type=None,
                        feature_label=None, target_label=None,
                        target_lim=None,
                        target_category=None,
                        min_width=0.2, inset_size=0.2, refline=True,
                        title=None, varimp=None, varimp_fmt="0.2f",
                        color=colorblind[1]):

    # Determine feature and target type
    if feature_type is not None:
        feature_type = "nume" if pd.api.types.is_numeric_dtype(feature) else "cate"
    if target_type is not None:
        target_type = dict(binary="CLASS", continuous="REGR", multiclass="MULTICLASS")[type_of_target(target)]

    # Call plot functions
    params = dict(ax=ax, feature=feature, target=target, feature_label=feature_label, target_label=target_label)
    if feature_type == "nume":
        if target_type in ["CLASS", "MULTICLASS"]:
            plot_nume_CLASS(**params)
        elif target_type == "REGR":
            plot_nume_REGR(**params)
    else:
        if target_type == "CLASS":
            plot_cate_CLASS(**params)
        elif target_type == "REGR":
            plot_cate_REGR(**params)
        else:
            warnings.warn("MULTICLASS not implemented")


########################################################################################################################
# Model Comparison
#######################################################################################################################

# --- Non-plots ---------------------------------------------------------------------------------------

# Undersample
def undersample(df, target, n_max_per_level, random_state=42):
    b_all = df[target].value_counts().values / len(df)
    df_under = (df.groupby(target).apply(lambda x: x.sample(min(n_max_per_level, x.shape[0]),
                                                            random_state=random_state))
                .sample(frac=1)  # shuffle
                .reset_index(drop=True))  
    b_sample = df_under[target].value_counts().values / len(df_under)
    return df_under, b_sample, b_all


# Kfold cross validation with strict separation between (prespecified) train and test-data
class KFoldSep(KFold):
    def __init__(self, shuffle, *args, **kwargs):
        super().__init__(shuffle=True, *args, **kwargs)

    def split(self, X, y=None, groups=None, test_fold=None):
        i_test_fold = np.arange(len(X))[test_fold]
        for i_train, i_test in super().split(X, y, groups):
            yield i_train[~np.isin(i_train, i_test_fold)], i_test[np.isin(i_test, i_test_fold)]
  
  
# Splitter: test==train fold, i.e. in-sample selection, needed for quick change of cross-validation code to non-cv
class InSampleSplit:
    def __init__(self, shuffle=True, random_state=42):
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, *args, **kwargs):
        i_df = np.arange(X.shape[0])
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(i_df)
        yield i_df, i_df  # train equals test

    def get_n_splits(self, *args):
        return 1
    

# Column selector: Workaround as scikit's ColumnTransformer currently needs same columns for fit and transform (bug!)
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, *args):
        return self

    def transform(self, df, *args):
        return df[self.columns]


# Incremental n_estimators (warm start) GridSearch for XGBoost and Lightgbm
# TODO adapt to single scorer shape of scikit
class GridSearchCV_xlgb(GridSearchCV):

    def fit(self, X, y=None, **fit_params):
        # pdb.set_trace()

        # Adapt grid: remove n_estimators
        n_estimators = self.param_grid["n_estimators"]
        param_grid = self.param_grid.copy()
        del param_grid["n_estimators"]
        df_param_grid = pd.DataFrame(product(*param_grid.values()), columns=param_grid.keys())

        # Materialize generator as this cannot be pickled for parallel
        self.cv = list(check_cv(self.cv, y).split(X))

        # TODO: Iterate also over split (see original fit method)
        def run_in_parallel(i):
            # for i in range(len(df_param_grid)):

            # Intialize
            df_results = pd.DataFrame()

            # Get actual parameter set
            d_param = df_param_grid.iloc[[i], :].to_dict(orient="records")[0]

            for fold, (i_train, i_test) in enumerate(self.cv):

                # pdb.set_trace()
                # Fit only once par parameter set with maximum number of n_estimators
                start = time.time()
                fit = (clone(self.estimator).set_params(**d_param,
                                                        n_estimators=int(max(n_estimators)))
                       .fit(_safe_indexing(X, i_train), _safe_indexing(y, i_train), **fit_params))
                fit_time = time.time() - start
                # Score with all n_estimators
                for ntree_limit in n_estimators:
                    start = time.time()
                    if isinstance(self.estimator, lgbm.sklearn.LGBMClassifier):
                        yhat_test = fit.predict_proba(_safe_indexing(X, i_test), num_iteration=ntree_limit)
                    elif isinstance(self.estimator, lgbm.sklearn.LGBMRegressor):
                        yhat_test = fit.predict(_safe_indexing(X, i_test), num_iteration=ntree_limit)
                    elif isinstance(self.estimator, xgb.sklearn.XGBClassifier):
                        yhat_test = fit.predict_proba(_safe_indexing(X, i_test), ntree_limit=ntree_limit)
                    else:
                        yhat_test = fit.predict(_safe_indexing(X, i_test), ntree_limit=ntree_limit)
                    score_time = time.time() - start

                    # Do it for training as well
                    if self.return_train_score:
                        if isinstance(self.estimator, lgbm.sklearn.LGBMClassifier):
                            yhat_train = fit.predict_proba(_safe_indexing(X, i_train), num_iteration=ntree_limit)
                        elif isinstance(self.estimator, lgbm.sklearn.LGBMRegressor):
                            yhat_train = fit.predict(_safe_indexing(X, i_train), num_iteration=ntree_limit)
                        elif isinstance(self.estimator, xgb.sklearn.XGBClassifier):
                            yhat_train = fit.predict_proba(_safe_indexing(X, i_train), ntree_limit=ntree_limit)
                        else:
                            yhat_train = fit.predict(_safe_indexing(X, i_train), ntree_limit=ntree_limit)
                            
                    # Get time metrics
                    df_results = df_results.append(pd.DataFrame(dict(fold_type="train", fold=fold,
                                                                     scorer="time", scorer_value=fit_time,
                                                                     n_estimators=ntree_limit, **d_param),
                                                                index=[0]))
                    df_results = df_results.append(pd.DataFrame(dict(fold_type="test", fold=fold,
                                                                     scorer="time", scorer_value=score_time,
                                                                     n_estimators=ntree_limit, **d_param),
                                                                index=[0]))
                    # Get performance metrics
                    for scorer in self.scoring:
                        scorer_value = self.scoring[scorer]._score_func(_safe_indexing(y, i_test), yhat_test)
                        df_results = df_results.append(pd.DataFrame(dict(fold_type="test", fold=fold,
                                                                         scorer=scorer, scorer_value=scorer_value,
                                                                         n_estimators=ntree_limit, **d_param),
                                                                    index=[0]))
                        if self.return_train_score:
                            scorer_value = self.scoring[scorer]._score_func(_safe_indexing(y, i_train), yhat_train)
                            df_results = df_results.append(pd.DataFrame(dict(fold_type="train", fold=fold,
                                                                             scorer=scorer,
                                                                             scorer_value=scorer_value,
                                                                             n_estimators=ntree_limit, **d_param),
                                                                        index=[0]))
            return df_results

        df_results = pd.concat(Parallel(n_jobs=self.n_jobs,
                                        max_nbytes='100M')(delayed(run_in_parallel)(row)
                                                           for row in range(len(df_param_grid))))

        # Transform results
        param_names = list(np.append(df_param_grid.columns.values, "n_estimators"))
        df_cv_results = pd.pivot_table(df_results,
                                       values="scorer_value",
                                       index=param_names,
                                       columns=["fold_type", "scorer"],
                                       aggfunc=["mean", "std"],
                                       dropna=False)
        df_cv_results.columns = ['_'.join(x) for x in df_cv_results.columns.values]
        scorer_names = np.array(list(self.scoring.keys()), dtype=object)
        df_cv_results["rank_test_" + scorer_names] = df_cv_results["mean_test_" + scorer_names].rank()
        df_cv_results = df_cv_results.rename(columns={"mean_train_time": "mean_fit_time",
                                                      "mean_test_time": "mean_score_time",
                                                      "std_train_time": "std_fit_time",
                                                      "std_test_time": "std_score_time"})
        df_cv_results = df_cv_results.reset_index()
        df_cv_results["params"] = df_cv_results[param_names].apply(lambda x: x.to_dict(), axis=1)
        df_cv_results = df_cv_results.rename(columns={name: "param_" + name for name in param_names})
        self.cv_results_ = df_cv_results.to_dict(orient="list")

        # Refit
        if self.refit:
            self.scorer_ = self.scoring
            self.multimetric_ = True
            self.best_index_ = df_cv_results["mean_test_" + self.refit].idxmax()
            self.best_score_ = df_cv_results["mean_test_" + self.refit].loc[self.best_index_]
            tmp = (df_cv_results[["param_" + name for name in param_names]].loc[[self.best_index_]]
                   .to_dict(orient="records")[0])
            self.best_params_ = {key.replace("param_", ""): value for key, value in tmp.items()}
            self.best_estimator_ = (clone(self.estimator).set_params(**self.best_params_).fit(X, y, **fit_params))

        return self


# --- Plots ---------------------------------------------------------------------------------------

# Plot model comparison
def plot_modelcomp(df_modelcomp_result, modelvar="model", runvar="run", scorevar="test_score", pdf=None):
    fig, ax = plt.subplots(1, 1)
    sns.boxplot(data=df_modelcomp_result, x=modelvar, y=scorevar, showmeans=True,
                meanprops={"markerfacecolor": "black", "markeredgecolor": "black"},
                ax=ax)
    sns.lineplot(data=df_modelcomp_result, x=modelvar, y=scorevar,
                 hue="#" + df_modelcomp_result[runvar].astype("str"), linewidth=0.5, linestyle=":",
                 legend=None, ax=ax)
    if pdf is not None:
        fig.savefig(pdf)



########################################################################################################################
# Interpret
#######################################################################################################################

# --- Non-Plots --------------------------------------------------------------------------

# Rescale predictions (e.g. to rewind undersampling)
def scale_predictions(yhat, b_sample=None, b_all=None):
    flag_1dim = False
    if b_sample is None:
        yhat_rescaled = yhat
    else:
        if yhat.ndim == 1:
            flag_1dim = True
            yhat = np.column_stack((1 - yhat, yhat))
        tmp = (yhat * b_all) / b_sample
        yhat_rescaled = tmp / tmp.sum(axis=1, keepdims=True)  
    if flag_1dim:
        yhat_rescaled = yhat_rescaled[:, 1]
    return yhat_rescaled


# Metaestimator which rescales the predictions
class ScalingEstimator(BaseEstimator):
    def __init__(self, estimator=None, b_sample=None, b_all=None):
        self.estimator = estimator
        self.b_sample = b_sample
        self.b_all = b_all
        self._estimator_type = estimator._estimator_type

    def fit(self, X, y, *args, **kwargs):
        self.classes_ = unique_labels(y)
        self.estimator.fit(X, y, *args, **kwargs)
        return self

    def predict(self, X, *args, **kwargs):
        return self.estimator.predict(X, *args, **kwargs)

    def predict_proba(self, X, *args, **kwargs):
        yhat = scale_predictions(self.estimator.predict_proba(X, *args, **kwargs),
                                 self.b_sample, self.b_all)
        return yhat


# Metaestimator for log-transformed target
class LogtrafoEstimator(BaseEstimator):
    def __init__(self, estimator=None):
        self.estimator = estimator
        self._varest = None
        self._estimator_type = estimator._estimator_type

    def fit(self, X, y, *args, **kwargs):
        self.estimator.fit(X, np.log(1 + y), *args, **kwargs)
        self._varest = np.var(self.estimator.predict(X) - np.log(1 + y))
        return self

    def predict(self, X, *args, **kwargs):
        return (np.exp(self.estimator.predict(X, *args, **kwargs) + self._varest / 2) - 1)


# Alternative to above with explicit classifier
class XGBClassifier_rescale(xgb.XGBClassifier):
    def __init__(self, b_sample=None, b_all=None, **kwargs):
        super().__init__(**kwargs)
        self.b_sample = b_sample
        self.b_all = b_all

    def predict_proba(self, X, *args, **kwargs):
        yhat = scale_predictions(super().predict_proba(X, *args, **kwargs),
                                 b_sample=self.b_sample, b_all=self.b_all)
        return yhat


# Convert result of scikit's variable importance to a dataframe
def varimp2df(varimp, features):
    df_varimp = (pd.DataFrame(dict(score_diff=varimp["importances_mean"], feature=features))
                 .sort_values(["score_diff"], ascending=False).reset_index(drop=True)
                 .assign(importance=lambda x: 100 * np.where(x["score_diff"] > 0,
                                                             x["score_diff"] / max(x["score_diff"]), 0),
                         importance_cum=lambda x: 100 * x["importance"].cumsum() / sum(x["importance"])))
    return df_varimp


# Dataframe based permutation importance which can select a subset of features for which to calculate VI
def variable_importance(estimator, df, y, features, target_type=None, scoring=None, 
                        n_jobs=None, random_state=None, **_):

    # Original performance
    if target_type is None:
        target_type = dict(continuous="REGR", binary="CLASS", multiclass="MULTICLASS")[type_of_target(y)]
    yhat = estimator.predict(df) if target_type == "REGR" else estimator.predict_proba(df)
    score_orig = scoring._score_func(y, yhat)

    # Performance per variable after permutation
    np.random.seed(random_state)
    i_perm = np.random.permutation(np.arange(len(df)))  # permutation vector
    
    def run_in_parallel(df, feature):
        #feature = features[0]
        df_copy = df.copy()
        df_copy[feature] = df_copy[feature].values[i_perm]
        yhat_perm = estimator.predict(df_copy) if target_type == "REGR" else estimator.predict_proba(df_copy)
        score = scoring._score_func(y, yhat_perm)      
        return score    
    scores = Parallel(n_jobs=n_jobs, max_nbytes='100M')(delayed(run_in_parallel)(df, feature)
                                                        for feature in features)    
    return varimp2df({"importances_mean": score_orig - scores}, features)




# Dataframe based patial dependence which can use a reference dataset for value-grid defintion
def partial_dependence(estimator, df, features,
                       df_ref=None, quantiles=np.arange(0.05, 1, 0.1),
                       n_jobs=4):
    #estimator=model; df=df_test[features]; features=features_top_test; df_ref=None; quantiles=np.arange(0.05, 1, 0.1)

    if df_ref is None:
        df_ref = df

    def run_in_parallel(feature):
        if pd.api.types.is_numeric_dtype(df_ref[feature]):
            values = np.unique(df_ref[feature].quantile(quantiles).values)
        else:
            values = df_ref[feature].unique()

        df_copy = df.copy()  # save original data

        #yhat_pd = np.array([]).reshape(0, 1 if estimator._estimator_type == "regressor" else len(estimator.classes_))
        df_return = pd.DataFrame()
        for value in values:
            df_copy[feature] = value
            df_return = df_return.append(
                pd.DataFrame(np.mean(estimator.predict_proba(df_copy) if hasattr(estimator, "predict_proba") else
                                     estimator.predict(df_copy), axis=0).reshape(1, -1)))
        # yhat_pd = np.append(yhat_pd,
        #                     np.mean(estimator.predict_proba(df_pd) if hasattr(estimator, "predict_proba") else
        #                             estimator.predict(df_pd), axis=0).reshape(1, -1), axis=0)
        df_return.columns = ["yhat"] if estimator._estimator_type == "regressor" else estimator.classes_
        df_return["value"] = values

        return df_return
    
    # Run in parallel and append
    l_pd = (Parallel(n_jobs=n_jobs, max_nbytes='100M')(delayed(run_in_parallel)(feature)
                                                       for feature in features))
    d_pd = dict(zip(features, l_pd))
    return d_pd


# Aggregate shapley to partial dependence
def shap2pd(shap_values, features,
            df_ref=None, n_bins=10, format_string=".2f"):

    if df_ref is None:
        df_ref = pd.DataFrame(shap_values.data, columns=shap_values.feature_names[0])

    d_pd = dict()
    for feature in features:
        # Location of feature in shap_values
        i_feature = np.argwhere(shap_values.feature_names[0] == feature)[0][0]
        intercept = shap_values.base_values[0]

        # Numeric features: Create bins
        if pd.api.types.is_numeric_dtype(df_ref[feature]):
            kbinsdiscretizer_fit = KBinsDiscretizer(n_bins=n_bins, encode="ordinal").fit(df_ref[[feature]])
            bin_edges = kbinsdiscretizer_fit.bin_edges_
            bin_labels = np.array([format(bin_edges[0][i], format_string) + " - " + 
                                   format(bin_edges[0][i + 1], format_string)
                                   for i in range(len(bin_edges[0]) - 1)])
            df_shap = pd.DataFrame({"value": bin_labels[(kbinsdiscretizer_fit
                                                         .transform(shap_values.data[:, [i_feature]])[:, 0])
                                                        .astype(int)],
                                    "yhat": shap_values.values[:, i_feature]})  # TODO: MULTICLASS
            
        # Categorical feature
        else:
            df_shap = pd.DataFrame({"value": shap_values.data[:, i_feature],
                                    "yhat": shap_values.values[:, i_feature]})

        # Aggregate and add intercept
        df_shap_agg = (df_shap.groupby("value").mean().reset_index()
                       .assign(yhat=lambda x: x["yhat"] + intercept))
        d_pd[feature] = df_shap_agg
    return d_pd


# Aggregate onehot encoded shapely values
def agg_shap_values(shap_values, df_explain, len_nume, l_map_onehot, round=2):
    '''
    df_explain: data frame used to create matrix which is send to shap explainer
    len_nume: number of numerical features building first columns of df_explain
    l_map_onehot:  like categories_ of onehot-encoder 
    '''
    
    # Copy
    shap_values_agg = copy.copy(shap_values)

    # Adapt feature_names
    shap_values_agg.feature_names = np.tile(df_explain.columns.values, (len(shap_values_agg), 1))

    # Adapt display data
    shap_values_agg.data = df_explain.round(round).values

    # Care for multiclass
    values_3d = np.atleast_3d(shap_values_agg.values)
    a_shap = np.empty((values_3d.shape[0], df_explain.shape[1], values_3d.shape[2]))
    #for k in range(a_shap.shape[2]):
        
    # Initilaize with nume shap valus (MUST BE AT BEGINNING OF df_explain)
    start_cate = len_nume
    a_shap[:, 0:start_cate, :] = values_3d[:, 0:start_cate, :].copy()

    # Aggregate cate shap values
    for i in range(len(l_map_onehot)):
        step = len(l_map_onehot[i])
        a_shap[:, len_nume + i, :] = values_3d[:, start_cate:(start_cate + step), :].sum(axis=1)
        start_cate = start_cate + step
        
    # Adapt non-multiclass    
    if a_shap.shape[2] == 1:
        a_shap = a_shap[:, :, 0]
    shap_values_agg.values = a_shap

    # Return
    return shap_values_agg
   

# --- Plots --------------------------------------------------------------------------

# Plot ROC curve
def plot_roc(ax, y, yhat):
    #y = df_test[target_name]
    #yhat = yhat_test

    ax = ax

    # also for regression
    if y.ndim == 1:
        if (np.min(y) < 0) | (np.max(y) > 1):
            y = MinMaxScaler().fit_transform(y.reshape(-1, 1))[:, 0]
    if yhat.ndim == 1:
        if (np.min(yhat) < 0) | (np.max(yhat) > 1):
            yhat = MinMaxScaler().fit_transform(yhat.reshape(-1, 1))[:, 0]

    # Roc curve
    fpr, tpr, cutoff = roc_curve(y, yhat)
    roc_auc = roc_auc_score(y, yhat)
    # sns.lineplot(fpr, tpr, ax=ax, palette=sns.xkcd_palette(["red"]))
    ax.plot(fpr, tpr)
    props = {'xlabel': r"fpr: P($\^y$=1|$y$=0)",
             'ylabel': r"tpr: P($\^y$=1|$y$=1)",
             'title': "ROC (AUC = " + format(roc_auc, "0.2f") + ")"}
    _ = ax.set(**props)


# Plot calibration
def plot_calibration(ax, y, yhat, n_bins=5):
    #y = df_test[target_name]
    #yhat = yhat_test

    ax = ax

    # Calibration curve
    #true, predicted = calibration_curve(y, yhat, n_bins=n_bins)
    # sns.lineplot(predicted, true, ax=ax, marker="o")

    df_plot = (pd.DataFrame({"y": y, "yhat": yhat})
               .assign(bin=lambda x: pd.qcut(x["yhat"], n_bins, duplicates="drop").astype("str"))
               .groupby(["bin"], as_index=False).agg("mean")
               .sort_values("yhat"))
    #sns.lineplot("yhat", "y", data = df_calib, ax = ax, marker = "o")
    ax.plot(df_plot["yhat"], df_plot["y"], "o-")
    props = {'xlabel': r"$\bar{\^y}$ in $\^y$-bin",
             'ylabel': r"$\bar{y}$ in $\^y$-bin",
             'title': "Calibration"}
    _ = ax.set(**props)

    # Add diagonal line
    minmin = min(df_plot["y"].min(), df_plot["yhat"].min())
    maxmax = max(df_plot["y"].max(), df_plot["yhat"].max())
    ax.plot([minmin, maxmax], [minmin, maxmax], linestyle="--", color="grey")

    # Focus
    #yhat_max = df_plot["yhat"].max()
    #ax.set_xlim(None, yhat_max)
    #ax.set_ylim(None, yhat_max)


# PLot confusion matrix
def plot_confusion(ax, y, yhat, threshold=0.5, cmap="Blues"):

    ax = ax

    # binary label
    yhat_bin = np.where(yhat > threshold, 1, 0)

    # accuracy and confusion calculation
    acc = accuracy_score(y, yhat_bin)
    df_conf = pd.DataFrame(confusion_matrix(y, yhat_bin))

    # plot
    sns.heatmap(pd.DataFrame(confusion_matrix(y, yhat_bin)),
                annot=True, fmt=".5g", cmap=cmap, ax=ax)
    props = {'xlabel': "Predicted label",
             'ylabel': "True label",
             'title': "Confusion Matrix ($Acc_{" + format(threshold, "0.2f") + "}$ = " + format(acc, "0.2f") + ")"}
    ax.set(**props)


'''
def plot_pred_distribution(ax, y, yhat, xlim=None):

    ax = ax
        
    # plot distribution
    sns.histplot(x=yhat, hue=y, stat="density", common_norm=False, kde=True, bins=20, ax=ax)
    props = {'xlabel': r"Predictions ($\^y$)",
             'ylabel': "Density",
             'title': "Distribution of Predictions",
             'xlim': xlim}
    ax.set(**props)
    #ax.legend(title="Target", loc="best")
'''

# Plot precision-recall curve


def plot_precision_recall(ax, y, yhat, annotate=True, fontsize=10):

    ax = ax

    # precision recall calculation
    prec, rec, cutoff = precision_recall_curve(y, yhat)
    cutoff = np.append(cutoff, 1)
    prec_rec_auc = average_precision_score(y, yhat)

    # plot
    ax.plot(rec, prec)
    props = {'xlabel': r"recall=tpr: P($\^y$=1|$y$=1)",
             'ylabel': r"precision: P($y$=1|$\^y$=1)",
             'title': "Precision Recall Curve (AUC = " + format(prec_rec_auc, "0.2f") + ")"}
    ax.set(**props)

    # annotate text
    if annotate:
        for thres in np.arange(0.1, 1, 0.1):
            i_thres = np.argmax(cutoff > thres)
            ax.annotate(format(thres, "0.1f"), (rec[i_thres], prec[i_thres]), fontsize=fontsize)


# Plot precision curve
def plot_precision(ax, y, yhat, annotate=True, fontsize=10):

    ax = ax

    # precision calculation
    pct_tested = np.array([])
    prec, _, cutoff = precision_recall_curve(y, yhat)
    cutoff = np.append(cutoff, 1)
    for thres in cutoff:
        pct_tested = np.append(pct_tested, [np.sum(yhat >= thres) / len(yhat)])

    # plot
    #sns.lineplot(pct_tested, prec[:-1], ax=ax, palette=sns.xkcd_palette(["red"]))
    ax.plot(pct_tested, prec)
    props = {'xlabel': "% Samples Tested",
             'ylabel': r"precision: P($y$=1|$\^y$=1)",
             'title': "Precision Curve"}
    ax.set(**props)

    # annotate text
    if annotate:
        for thres in np.arange(0.1, 1, 0.1):
            i_thres = np.argmax(cutoff > thres)
            if i_thres:
                ax.annotate(format(thres, "0.1f"), (pct_tested[i_thres], prec[i_thres]),
                                fontsize=fontsize)
                

# Plot model performance for CLASS target
def get_plotcalls_model_performance_CLASS(y, yhat,
                                          n_bins=5, threshold=0.5, cmap="Blues", annotate=True, fontsize=10):

    # yhat to 1-dim
    if ((yhat.ndim == 2) and (yhat.shape[1] == 2)):
        yhat = yhat[:, 1]

    # Define plot dict
    d_calls = dict()
    d_calls["roc"] = (plot_roc, dict(y=y, yhat=yhat))
    d_calls["confusion"] = (plot_confusion, dict(y=y, yhat=yhat, threshold=threshold, cmap=cmap))
    d_calls["distribution"] = (plot_bidistribution, dict(x=yhat, group=y, xlim=(0, 1)))
    d_calls["calibration"] = (plot_calibration, dict(y=y, yhat=yhat, n_bins=n_bins))
    d_calls["precision_recall"] = (plot_precision_recall, dict(y=y, yhat=yhat, annotate=annotate, fontsize=fontsize))
    d_calls["precision"] = (plot_precision, dict(y=y, yhat=yhat, annotate=annotate, fontsize=fontsize))

    return d_calls


# Plot model performance for CLASS target
def get_plotcalls_model_performance_REGR(y, yhat,
                                         ylim, regplot, n_bins):

    # yhat to 1-dim
    if ((yhat.ndim == 2) and (yhat.shape[1] == 2)):
        yhat = yhat[:, 1]

    # Define plot dict
    d_calls = dict()
    title = r"Observed vs. Fitted ($\rho_{Spearman}$ = " + format(spear(y, yhat), "0.2f") + ")"
    d_calls["observed_vs_fitted"] = (plot_biscatter, dict(x=yhat, y=y, xlabel=r"$\^y$", ylabel="y",
                                                          title=title,
                                                          xlim=ylim,
                                                          regplot=regplot))
    d_calls["calibration"] = (plot_calibration, dict(y=y, yhat=yhat, n_bins=n_bins))
    d_calls["distribution"] = (plot_bidistribution, dict(x=np.append(y, yhat),
                                                         group=np.append(np.tile("y", len(y)),
                                                                         np.tile(r"$\^y$", len(yhat))),
                                                         title="Distribution"))
    d_calls["residuals_vs_fitted"] = (plot_biscatter, dict(x=yhat, y=y - yhat,
                                                           xlabel=r"$\^y$",
                                                           ylabel=r"y-$\^y$",
                                                           title="Residuals vs. Fitted",
                                                           xlim=ylim,
                                                           regplot=regplot))

    d_calls["absolute_residuals_vs_fitted"] = (plot_biscatter, dict(x=yhat, y=abs(y - yhat),
                                                                    xlabel=r"$\^y$",
                                                                    ylabel=r"|y-$\^y$|",
                                                                    title="Absolute Residuals vs. Fitted",
                                                                    xlim=ylim,
                                                                    regplot=regplot))

    d_calls["relative_residuals_vs_fitted"] = (plot_biscatter, dict(x=yhat,
                                                                    y=np.where(y == 0, np.nan,
                                                                               abs(y - yhat) / abs(y)),
                                                                    xlabel=r"$\^y$",
                                                                    ylabel=r"|y-$\^y$|/|y|",
                                                                    title="Relative Residuals vs. Fitted",
                                                                    xlim=ylim,
                                                                    regplot=regplot))

    return d_calls


# Wrapper for plot_model_performance_<target_type>
def get_plotcalls_model_performance(y, yhat, target_type=None,
                                    n_bins=5, threshold=0.5, cmap="Blues", annotate=True, fontsize=10,
                                    ylim=None, regplot=True,
                                    l_plots=None,
                                    n_rows=2, n_cols=3, w=18, h=12, pdf_path=None):
    # Derive target type
    if target_type is None:
        target_type = dict(continuous="REGR", binary="CLASS", multiclass="MULTICLASS")[type_of_target(y)]

    # Plot
    if target_type == "CLASS":
        d_calls = get_plotcalls_model_performance_CLASS(
            y=y, yhat=yhat, n_bins=n_bins, threshold=threshold, cmap=cmap, annotate=annotate, fontsize=fontsize)
    elif target_type == "REGR":
        d_calls = get_plotcalls_model_performance_REGR(y=y, yhat=yhat,
                                                       ylim=ylim, regplot=regplot, n_bins=n_bins)
    elif target_type == "MULTICLASS":
        pass
        #plot_model_performance_MULTICLASS(y, yhat, n_bins, n_rows, n_cols, pdf_path)
    else:
        warnings.warn("Target type cannot be determined")

    # Filter plot dict
    if l_plots is not None:
        d_calls = {x: d_calls[x] for x in l_plots}

    return d_calls


# Plot permutation base variable importance
def plot_variable_importance(ax,
                             features, importance,
                             importance_cum=None, importance_se=None, max_score_diff=None,
                             category=None,
                             category_label="Importance",
                             category_color_palette=sns.xkcd_palette(["blue", "orange", "red"])):

    ax = ax
    sns.barplot(importance, features, hue=category,
                palette=category_color_palette, dodge=False, ax=ax)
    ax.set_title("Top{0: .0f} Feature Importances".format(len(features)))
    ax.set_xlabel(r"permutation importance")
    if max_score_diff is not None:
        ax.set_xlabel(ax.get_xlabel() + "(100 = " + str(max_score_diff) + r" score-$\Delta$)")
    if importance_cum is not None:
        ax.plot(importance_cum, features, color="black", marker="o")
        ax.set_xlabel(ax.get_xlabel() + " /\n" + r"cumulative in % (-$\bullet$-)")
    if importance_se is not None:
        ax.errorbar(x=importance, y=features, xerr=importance_se,
                        fmt=".", marker="s", fillstyle="none", color="grey")
        ax.set_title(ax.get_title() + r" (incl. SE (-$\boxminus$-))")
    '''
    if column_score_diff is not None:
        ax2 = ax.twiny()
        ax2.errorbar(x=df_varimp[column_score_diff], y=df_varimp[column_feature],
                     xerr=df_varimp[column_score_diff_se]*5,
                    fmt=".", marker="s", fillstyle="none", color="grey")
        ax2.grid(False)
    '''

# Plot partial dependence
def plot_pd(ax, feature_name, feature, yhat, feature_ref=None, yhat_err=None, refline=None, ylim=None,
            legend_labels=None, color="red", min_width=0.2):

    ax = ax
    numeric_feature = pd.api.types.is_numeric_dtype(feature)
    #if yhat.ndim == 1: 
    #    yhat = yhat.reshape(-1, 1)

    if numeric_feature:
        
        # Lineplot
        ax.plot(feature, yhat, marker=".", color=color)

        # Background density plot
        if feature_ref is not None:
            ax2 = ax.twinx()
            ax2.axis("off")
            sns.kdeplot(feature_ref, color="grey", 
                        shade=True, linewidth=0,  # hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 0},
                        ax=ax2)
        # Rugs
        sns.rugplot(feature, color="grey", ax=ax)

        # Refline
        if refline is not None:
            ax.axhline(refline, ls="dotted", color="black")  # priori line

        # Axis style
        ax.set_title(feature_name)
        ax.set_xlabel("")
        ax.set_ylabel(r"$\^y$")
        if ylim is not None:
            ax.set_ylim(ylim)

        # Crossvalidation
        if yhat_err is not None:
            ax.fill_between(feature, yhat - yhat_err, yhat + yhat_err, color=color, alpha=0.2)

    else:
        # Use DataFrame for calculation
        df_plot = pd.DataFrame({feature_name: feature, "yhat": yhat}).sort_values(feature_name).reset_index(drop=True)
        if yhat_err is not None:
            df_plot["yhat_err"] = yhat_err

        # Distribution
        if feature_ref is not None:
            df_plot = df_plot.merge(pd.DataFrame({feature_name: feature_ref}).assign(count=1)
                                    .groupby(feature_name, as_index=False)[["count"]].sum()
                                    .assign(pct=lambda x: x["count"] / x["count"].sum())
                                    .assign(width=lambda x: 0.9 * x["pct"] / x["pct"].max()), how="left")
            df_plot[feature_name] = df_plot[feature_name] + " (" + (df_plot["pct"] * 100).round(1).astype(str) + "%)"
            if min_width is not None:
                df_plot["width"] = np.where(df_plot["width"] < min_width, min_width, df_plot["width"])
            #ax2 = ax.twiny()
            #ax2.barh(df_plot[feature_name], df_plot["pct"], color="grey", edgecolor="grey", alpha=0.5, linewidth=0)

        # Bar plot
        ax.barh(df_plot[feature_name], df_plot["yhat"],
                    height=df_plot["width"] if feature_ref is not None else 0.8,
                    color=color, edgecolor="black", alpha=0.5, linewidth=1)
                
        # Refline
        if refline is not None:
            ax.axvline(refline, ls="dotted", color="black")  # priori line

        # Axis style
        ax.set_title(feature_name)
        ax.set_xlabel(r"$\^y$")
        if ylim is not None:
            ax.set_xlim(ylim)

        # Crossvalidation
        if yhat_err is not None:
            ax.errorbar(df_plot["yhat"], df_plot[feature_name], xerr=df_plot["yhat_err"],
                            fmt=".", marker="s", capsize=5, fillstyle="none", color="grey")


# Plot shap
def plot_shap(ax, shap_values, index, id, 
              y_str=None, yhat_str=None, 
              show_intercept=True, show_prediction=True, 
              color=["blue", "red"], n_top=10, multiclass_index=None):

    ax = ax

    # Subset in case of multiclass
    if multiclass_index is not None:
        base_values = shap_values.base_values[:, multiclass_index]
        values = shap_values.values[:, :, multiclass_index]
    else:
        base_values = shap_values.base_values
        values = shap_values.values

    # Shap values to dataframe
    df_shap = (pd.concat([pd.DataFrame({"variable": "intercept",
                                        "variable_value": np.nan,
                                        "shap": base_values[index]}, index=[0]),
                          pd.DataFrame({"variable": shap_values.feature_names[index],
                                        "variable_value": shap_values.data[index],
                                        "shap": values[index]})
                          .assign(tmp=lambda x: x["shap"].abs())
                          .sort_values("tmp", ascending=False)
                          .drop(columns="tmp")])
               .assign(yhat=lambda x: x["shap"].cumsum()))  # here a my.inv_logit might be added

    # Prepare for waterfall plot
    df_plot = (df_shap.assign(offset=lambda x: x["yhat"].shift(1).fillna(0),
                              bar=lambda x: x["yhat"] - x["offset"],
                              color=lambda x: np.where(x["variable"] == "intercept", "grey",
                                                       np.where(x["bar"] > 0, color[1], color[0])),
                              bar_label=lambda x: np.where(x["variable"] == "intercept",
                                                           x["variable"],
                                                           x["variable"] + " = " + x["variable_value"].astype("str")))
               .loc[:, ["bar_label", "bar", "offset", "color"]])

    # Aggreagte non-n_top shap values
    if n_top is not None:
        df_plot = pd.concat([df_plot.iloc[:(n_top + 1)],
                            pd.DataFrame(dict(bar_label="... the rest",
                                              bar=df_plot.iloc[(n_top + 1):]["bar"].sum(),
                                              offset=df_plot.iloc[(n_top + 1)]["offset"]),
                                         index=[0])
                            .assign(color=lambda x: np.where(x["bar"] > 0, color[1], color[0]))])

    # Add final prediction
    df_plot = (pd.concat([df_plot, pd.DataFrame(dict(bar_label="prediction", bar=df_plot["bar"].sum(),
                                                     offset=0, color="black"), index=[0])])
               .reset_index(drop=True))

    # Remove intercept and final prediction
    if not show_intercept:
        df_plot = df_plot.query("bar_label != 'intercept'")
    if not show_prediction:
        df_plot = df_plot.query("bar_label != 'prediction'")
    #df_plot = df_plot.query('bar_label not in ["intercept", "Prediction"]')
    #x_min = (df_plot["offset"]).min()
    #x_max = (df_plot["offset"]).max()
    x_min = min(0, (df_plot["offset"]).min())
    x_max = max(0, (df_plot["offset"]).max())

    # Plot bars
    ax.barh(df_plot["bar_label"], df_plot["bar"], left=df_plot["offset"], color=df_plot["color"],
                alpha=0.5,
                edgecolor="black")

    # Set axis limits
    ax.set_xlim(x_min - 0.1 * (x_max - x_min),
                    x_max + 0.1 * (x_max - x_min))

    # Annotate
    for i in range(len(df_plot)):
        # Text
        ax.annotate(df_plot.iloc[i]["bar"].round(3),
                        (df_plot.iloc[i]["offset"] + max(0, df_plot.iloc[i]["bar"]) + np.ptp(ax.get_xlim()) * 0.02,
                        df_plot.iloc[i]["bar_label"]),
                        # if ~df_plot.iloc[i][["bar_label"]].isin(["intercept", "Prediction"])[0] else "right",
                        ha="left",
                        va="center", size=10,
                        color="black")  # "white" if i == (len(df_plot) - 1) else "black")

        # Lines
        if i < (len(df_plot) - 1):
            df_line = pd.concat([pd.DataFrame(dict(x=df_plot.iloc[i]["offset"] + df_plot.iloc[i]["bar"],
                                                   y=df_plot.iloc[i]["bar_label"]), index=[0]),
                                pd.DataFrame(dict(x=df_plot.iloc[i]["offset"] + df_plot.iloc[i]["bar"],
                                                  y=df_plot.iloc[i + 1]["bar_label"]), index=[0])])
            ax.plot(df_line["x"], df_line["y"], color="black", linestyle=":")

    # Title and labels
    title = "id = " + str(id)
    if y_str is not None:
        title = title + " (y = " + y_str + ")"
    if yhat_str is not None:
        title = title + r" ($\^ y$ = " + yhat_str + ")"
    ax.set_title(title)  
    ax.set_xlabel("shap")


