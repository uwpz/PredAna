########################################################################################################################
# Packages
########################################################################################################################

# General
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
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
from sklearn.linear_model import LinearRegression, LogisticRegression, LinearRegression
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
from scipy.cluster.hierarchy import linkage
from pycorrcat.pycorrcat import corr as corrcat
from statsmodels.nonparametric.smoothers_lowess import lowess



########################################################################################################################
# General Functions
########################################################################################################################

# --- General ----------------------------------------------------------------------------------------

def debugtest(a=1, b=2):
    print(a)
    print(b)
    print("blub")
    # print("blub2")
    # print("blub3")
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
def plot_l_calls(l_calls, n_cols=2, n_rows=2, figsize=(16, 10), pdf_path=None, constrained_layout=False):

    # Open pdf
    if pdf_path is not None:
        pdf_pages = PdfPages(pdf_path)
    else:
        pdf_pages = None

    l_fig = list()
    for i, (plot_function, kwargs) in enumerate(l_calls):
        # Init new page
        if i % (n_rows * n_cols) == 0:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=constrained_layout)
            l_fig.append([(fig, axes)])
            i_ax = 0

        # Plot call
        plot_function(ax=axes.flat[i_ax] if (n_rows * n_cols > 1) else axes, **kwargs)
        i_ax += 1

        # "Close" page
        if (i_ax == n_rows * n_cols) or (i == len(l_calls) - 1):
            # Remove unused axes
            if (i == len(l_calls) - 1):
                for k in range(i_ax, n_rows * n_cols):
                    axes.flat[k].axis("off")

            # Write pdf
            if constrained_layout:
                fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=4 / 72, hspace=0.1, wspace=0.1)
            else:
                fig.tight_layout()
            if pdf_path is not None:
                pdf_pages.savefig(fig, bbox_inches="tight", pad_inches=0.2)

    # Close pdf
    if pdf_path is not None:
        pdf_pages.close()

    return l_fig


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


# Mean absolute error
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# def myrmse(y_true, y_pred):
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
        if type_of_target(y_true) == "continuous":
            if np.max(y_true) > 1:
                y_true = np.where(y_true > 1, 1, np.where(y_true < 1, 0, y_true))

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
    """
    Print summary of (categorical) varibles (similar to R's summary function)
    
    Parameters
    ----------
    df: Dataframe 
        Dataframe comprising columns to be summarized
    topn: integer, default=5
        Restrict number of member listings
    dtype: list, default=["object"]
        Determines Which dtypes should be summarized.  
        
    Returns
    ------- 
    dataframe which comprises summary of variables
    """
    
    df_tmp = df.select_dtypes(dtypes)
    return pd.concat([(df_tmp[catname].value_counts().iloc[: topn].reset_index()
                       .rename(columns={"index": catname, catname: "#"}))
                      for catname in df_tmp.columns.values],
                     axis=1).fillna("")


# Binning with correct label
def bin(feature, n_bins=5, precision=3):
    feature_binned = pd.qcut(feature, n_bins, duplicates="drop", precision=precision)
    feature_binned = ("q" + feature_binned.cat.codes.astype("str") + " " +
                      feature_binned.astype("str").str.replace("\(" + str(feature_binned.cat.categories[0].left),
                                                                "[" + str(pd.Series(feature).min()
                                                                          .round(precision))))
    return feature_binned


# Univariate model performance
def variable_performance(feature, target, scorer, target_type=None, splitter=KFold(5), groups=None):
    """
    Calculates univariate variable performance by applying LinearRegression or LogisticRegression (depending on target)
    on single feature model and calcuating scoring metric.\n 
    "Categorical" features are 1-hot encoded, numeric features are binned into 5 bins (in order to approximate 
    also nonlinear effect)

    Parameters
    ----------
    feature: Numpy array or Pandas series
        Feature for which to calculate importance
    target: Numpy array or Pandas series
        Target variable
    scorer: skleran.metrics scorer        
    target_type: "CLASS", "REGR", "MULTICLASS", None, default=None
        Overwrites function's determination of target type
    splitter: sklearn.model_selection splitter, default=KFold(5)
    groups: Numpy array or Pandas series
        Grouping variable in case of using Grouped splitter
    
    Returns
    -------
    Numeric value representing scoring value
    """

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
    print("Calculate univariate performance for", 
          "numeric" if numeric_feature else "categorical",
          "feature '" + feature.name + "' for " + target_type + " target '" + target.name + "'")

    # Calc performance
    perf = np.mean(cross_val_score(
        estimator=(LinearRegression() if target_type == "REGR" else LogisticRegression()),
        X=(KBinsDiscretizer().fit_transform(df_hlp[["feature"]]) if numeric_feature else
           OneHotEncoder().fit_transform(df_hlp[["feature"]])),
        y=df_hlp["target"],
        cv=splitter.split(df_hlp, groups=df_hlp["groups_for_split"] if groups is not None else None),
        scoring=scorer))

    return perf

"""
Description

Parameters
----------
var1: Dataframe
    Explanation
var2: integer
    Explanation

Returns
-------
Description
"""


# Winsorize
class Winsorize(BaseEstimator, TransformerMixin):
    """
    Winsorizing transformer for clipping outlier

    Parameters
    ----------
    lower_quantile: float, default=None
        Lower quantile (which must be between 0 and 1) at which to clip all columns
    upper_quantile: float, default=None
        Upper quantile (which must be between 0 and 1) at which to clip all columns

    Attributes
    ----------
    a_lower_: array of lower quantile values   
    a_upper_: array of upper quantile values  
    """
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
            self.a_upper_ = None
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


def helper_adapt_feature_target(feature, target, feature_name, target_name):
    # Convert to Series and adapt labels
    if not isinstance(feature, pd.Series):
        feature = pd.Series(feature)
        feature.name = feature_name if feature_name is not None else "x"
    if feature_name is not None:
        feature = feature.copy()
        feature.name = feature_name
    if not isinstance(target, pd.Series):
        target = pd.Series(target)
        target.name = target_name if target_name is not None else "y"
    if target_name is not None:
        target = target.copy()
        target.name = target_name

    print("Plotting " + feature.name + " vs. " + target.name)

    # Remove missings
    mask_target_miss = target.isna()
    n_target_miss = mask_target_miss.sum()
    if n_target_miss:
        target = target[~mask_target_miss]
        feature = feature[~mask_target_miss]
        print("ATTENTION: " + str(n_target_miss) + " records removed due to missing target!")
        mask_target_miss = target.notna()
    mask_feature_miss = feature.isna()
    n_feature_miss = mask_feature_miss.sum()
    pct_miss_feature = 100 * n_feature_miss / len(feature)
    if n_feature_miss:
        target = target[~mask_feature_miss]
        feature = feature[~mask_feature_miss]
        #warnings.warn(str(n_feature_miss) + " records removed due to missing feature!")

    return (feature, target, pct_miss_feature)


def helper_inner_barplot(ax, x, y, inset_size=0.2):
    # Memorize ticks and limits
    xticks = ax.get_xticks()
    xlim = ax.get_xlim()

    # Create space
    ax.set_xlim(xlim[0] - 1.2 * inset_size * (xlim[1] - xlim[0]), xlim[1])

    # Create shared inset axis
    inset_ax = ax.inset_axes([0, 0, inset_size, 1])
    # inset_ax.get_xaxis().set_visible(False)
    # inset_ax.get_yaxis().set_visible(False)
    inset_ax.set_xticklabels([])
    inset_ax.set_yticklabels([])
    ax.get_shared_y_axes().join(ax, inset_ax)

    # Plot
    inset_ax.barh(x, y,
                  color="lightgrey", edgecolor="black", linewidth=1)

    # More space for plot
    left, right = inset_ax.get_xlim()
    inset_ax.set_xlim(left, right * 1.2)

    # Border
    inset_ax.axvline(inset_ax.get_xlim()[1], color="black")

    # Remove senseless ticks
    yticks = inset_ax.get_yticks()
    if len(yticks) > len(x):
        _ = inset_ax.set_yticks(yticks[1::2])
    _ = ax.set_xticks(xticks[(xticks >= xlim[0]) & (xticks <= xlim[1])])
    
    
def helper_inner_barplot_rotated(ax, x, y, inset_size=0.2):
    # Memorize ticks and limits
    yticks = ax.get_yticks()
    ylim = ax.get_ylim()

    # Create space
    ax.set_ylim(ylim[0] - 1.2 * inset_size * (ylim[1] - ylim[0]), ylim[1])

    # Create shared inset axis
    inset_ax = ax.inset_axes([0, 0, 1, inset_size])
    inset_ax.set_yticklabels([])
    inset_ax.set_xticklabels([])
    ax.get_shared_x_axes().join(ax, inset_ax)

    # Plot
    inset_ax.bar(x, y,
                 color="lightgrey", edgecolor="black", linewidth=1)

    # More space for plot
    left, right = inset_ax.get_ylim()
    inset_ax.set_ylim(left, right * 1.2)

    # Border
    inset_ax.axhline(inset_ax.get_ylim()[1], color="black")

    # Remove senseless ticks
    xticks = inset_ax.get_xticks()
    if len(xticks) > len(y):
        _ = inset_ax.set_xticks(xticks[1::2])
    _ = ax.set_yticks(yticks[(yticks >= ylim[0]) & (yticks <= ylim[1])])



def plot_cate_CLASS(ax,
                    feature, target,
                    feature_name=None, target_name=None,
                    target_category=None,
                    target_lim=None,
                    min_width=0.2, inset_size=0.2, refline=True,
                    title=None,
                    add_miss_info=True,
                    color=list(sns.color_palette("colorblind").as_hex()),
                    **_):

    # Adapt feature and target
    feature, target, pct_miss_feature = helper_adapt_feature_target(feature, target, feature_name, target_name)

    # Adapt color
    if isinstance(color, list):
        color = color[1]

    # Add title
    if title is None:
        title = feature.name

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
        ax.axvline((target == target_category).sum() / len(target), linestyle="dotted", color="black")

    # Inner barplot
    helper_inner_barplot(ax, x=df_plot[feature.name + "_fmt"], y=df_plot["pct"], inset_size=inset_size)

    # Missing information
    if add_miss_info:
        ax.set_ylabel(feature.name)
        ax.set_ylabel(ax.get_ylabel() + " (" + format(pct_miss_feature, "0.1f") + "% NA)")


def plot_cate_MULTICLASS(ax,
                         feature, target,
                         feature_name=None, target_name=None,
                         target_category=None,
                         target_lim=None,
                         min_width=0.2, inset_size=0.2, refline=True,
                         title=None,
                         add_miss_info=True,
                         color=list(sns.color_palette("colorblind").as_hex()),
                         reverse=False,
                         exchange_x_y_axis=False,
                         **_):

    # Adapt feature and target
    feature, target, pct_miss_feature = helper_adapt_feature_target(feature, target, feature_name, target_name)

    # Add title
    if title is None:
        title = feature.name

    # Prepare data
    df_plot = helper_calc_barboxwidth(feature, target, min_width=min_width)
    
    # Reverse
    if reverse:
        df_plot = df_plot.iloc[::-1]

    # Segmented barplot
    offset = np.zeros(len(df_plot))
    for m, member in enumerate(np.sort(target.unique())):
        if not exchange_x_y_axis:
            ax.barh(y=df_plot[feature.name + "_fmt"], width=df_plot[member], height=df_plot["w"],
                    left=offset,
                    color=color[m], label=member, edgecolor="black", alpha=0.5, linewidth=1)
        else:
            ax.bar(x=df_plot[feature.name + "_fmt"], height=df_plot[member], width=df_plot["w"],
                   bottom=offset,
                   color=color[m], label=member, edgecolor="black", alpha=0.5, linewidth=1)
        offset = offset + df_plot[member].values
        ax.legend(title=target.name, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(title)

    # Inner barplot
    if not exchange_x_y_axis:
        helper_inner_barplot(ax, x=df_plot[feature.name + "_fmt"], y=df_plot["pct"], inset_size=inset_size)
    else:
        helper_inner_barplot_rotated(ax, x=df_plot[feature.name + "_fmt"], y=df_plot["pct"], inset_size=inset_size)
        
    # Missing information
    if add_miss_info:
        ax.set_ylabel(feature.name)
        ax.set_ylabel(ax.get_ylabel() + " (" + format(pct_miss_feature, "0.1f") + "% NA)")


def plot_cate_REGR(ax,
                   feature, target,
                   feature_name=None, target_name=None,
                   target_lim=None,
                   min_width=0.2, inset_size=0.2, refline=True,
                   title=None,
                   add_miss_info=True,
                   color=list(sns.color_palette("colorblind").as_hex()),
                   **_):

    # Adapt feature and target
    feature, target, pct_miss_feature = helper_adapt_feature_target(feature, target, feature_name, target_name)

    # Adapt color
    if isinstance(color, list):
        color = color[0]

    # Add title
    if title is None:
        title = feature.name

    # Prepare data
    df_plot = helper_calc_barboxwidth(feature, np.tile("dummy", len(feature)),
                                      min_width=min_width)

    # Boxplot
    _ = ax.boxplot([target[feature == value] for value in df_plot[feature.name].values],
                   labels=df_plot[feature.name + "_fmt"].values,
                   widths=df_plot["w"].values,
                   vert=False,
                   patch_artist=True,
                   showmeans=True,
                   boxprops=dict(facecolor=color, alpha=0.5, color="black"),
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
        ax.axvline(target.mean(), linestyle="dotted", color="black")

    # Inner barplot
    helper_inner_barplot(ax, x=np.arange(len(df_plot)) + 1, y=df_plot["pct"],
                         inset_size=inset_size)

    # Missing information
    if add_miss_info:
        ax.set_ylabel(feature.name)
        ax.set_ylabel(ax.get_ylabel() + " (" + format(pct_miss_feature, "0.1f") + "% NA)")


def plot_nume_CLASS(ax,
                    feature, target,
                    feature_name=None, target_name=None,
                    feature_lim=None,
                    inset_size=0.2, n_bins=20,
                    title=None,
                    add_miss_info=True,
                    rasterized=True,
                    color=list(sns.color_palette("colorblind").as_hex()),
                    **_):

    # Adapt feature and target
    feature, target, pct_miss_feature = helper_adapt_feature_target(feature, target, feature_name, target_name)

    # Add title
    if title is None:
        title = feature.name

    # Adapt color
    color = color[:target.nunique()]

    # Distribution plot
    sns.histplot(ax=ax, x=feature, hue=target, hue_order=np.sort(target.unique()),
                 stat="density", common_norm=False, kde=True, bins=n_bins,
                 palette=color)
    ax.set_ylabel("Density")
    ax.set_title(title)

    # Inner Boxplot
    yticks = ax.get_yticks()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0] - 1.5 * inset_size * (ylim[1] - ylim[0]))
    inset_ax = ax.inset_axes([0, 0, 1, inset_size])
    inset_ax.set_axis_off()
    ax.axhline(ylim[0], color="black")
    ax.get_shared_x_axes().join(ax, inset_ax)
    sns.boxplot(ax=inset_ax, x=feature, y=target, order=np.sort(target.unique()), orient="h", palette=color,
                showmeans=True, meanprops={"marker": "x", "markerfacecolor": "black", "markeredgecolor": "black"})
    _ = ax.set_yticks(yticks[(yticks >= ylim[0]) & (yticks <= ylim[1])])
    ax.set_rasterized(rasterized)

    # Add missing information
    if add_miss_info:
        ax.set_xlabel(ax.get_xlabel() + " (" + format(pct_miss_feature, "0.1f") + "% NA)")

    # Set feature_lim (must be after inner plot)
    if feature_lim is not None:
        ax.set_xlim(feature_lim)


def plot_nume_MULTICLASS(ax,
                         feature, target,
                         feature_name=None, target_name=None,
                         feature_lim=None,
                         inset_size=0.2, n_bins=20,
                         title=None,
                         add_miss_info=True,
                         rasterized=True,
                         color=list(sns.color_palette("colorblind").as_hex()),
                         **_):

    plot_nume_CLASS(ax=ax, feature=feature, target=target,
                    feature_name=feature_name, target_name=target_name,
                    feature_lim=feature_lim,
                    inset_size=inset_size, n_bins=n_bins,
                    title=title, add_miss_info=add_miss_info, rasterized=rasterized,
                    color=color, **_)


# Scatterplot as heatmap
def plot_nume_REGR(ax,
                   feature, target,
                   feature_name=None, target_name=None,
                   feature_lim=None, target_lim=None,
                   regplot=True, regplot_type="lowess", lowess_n_sample=1000, lowess_frac=2 / 3, spline_smooth=1,
                   refline=True,
                   title=None,
                   add_miss_info=True,
                   add_colorbar=True,
                   inset_size=0.2,
                   add_feature_distribution=True, add_target_distribution=True, n_bins=20, 
                   add_boxplot=True, rasterized=True,
                   colormap=LinearSegmentedColormap.from_list("bl_yl_rd", ["blue", "yellow", "red"]),
                   **_):

    # Adapt feature and target
    feature, target, pct_miss_feature = helper_adapt_feature_target(feature, target, feature_name, target_name)

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
    p = ax.hexbin(feature, target, mincnt=1, cmap=colormap)
    ax.set_xlabel(feature.name)
    ax.set_ylabel(target.name)
    if add_colorbar:
        plt.colorbar(p, ax=ax)

    # Spline
    if regplot:
        if regplot_type == "linear":
            sns.regplot(x=feature, y=target, lowess=False, scatter=False, color="black", ax=ax)
        elif regplot_type == "spline":
            df_spline = (pd.DataFrame({"x": feature, "y": target})
                         .groupby("x")[["y"]].agg(["mean", "count"])
                         .pipe(lambda x: x.set_axis([a + "_" + b for a, b in x.columns],
                                                    axis=1, inplace=False))
                         .assign(w=lambda x: np.sqrt(x["y_count"]))
                         .sort_values("x")
                         .reset_index())
            spl = splrep(x=df_spline["x"].values, y=df_spline["y_mean"].values, w=df_spline["w"].values,
                         s=len(df_spline) * spline_smooth)
            x2 = np.quantile(df_spline["x"].values, np.arange(0.01, 1, 0.01))
            y2 = splev(x2, spl)
            ax.plot(x2, y2, color="black")
            
            '''
            from patsy import cr, bs
            df_spline = pd.DataFrame({"x": feature, "y": target}).sort_values("x")
            spline_basis = cr(df_spline["x"], df=7, constraints='center')
            spline_basis = bs(df_spline["x"], df=4, include_intercept=True)
            df_spline["y_spline"] = LinearRegression().fit(spline_basis, target).predict(spline_basis)
            ax.plot(df_spline["x"], df_spline["y_spline"], color="red") 
            
            sns.regplot(x=feature, y=target,
                        lowess=True,
                        scatter=False, color="green", ax=ax)
            '''
        else:
            if regplot_type != "lowess":
                warnings.warn("Wrong regplot_type, used 'lowess'")                
            df_lowess = (pd.DataFrame({"x": feature, "y": target})
                         .pipe(lambda x: x.sample(min(lowess_n_sample, x.shape[0]), random_state=42))
                         .reset_index(drop=True)
                         .sort_values("x")
                         .assign(yhat=lambda x: lowess(x["y"], x["x"], frac=lowess_frac,
                                                       is_sorted=True, return_sorted=False)))
            ax.plot(df_lowess["x"], df_lowess["yhat"], color="black")
                        
    if add_miss_info:
        ax.set_xlabel(ax.get_xlabel() + " (" + format(pct_miss_feature, "0.1f") + "% NA)")
    if title is not None:
        ax.set_title(title)
    if target_lim is not None:
        ax.set_ylim(target_lim)
    if feature_lim is not None:
        ax.set_xlim(feature_lim)

    # Refline
    if refline:
        ax.axhline(target.mean(), linestyle="dashed", color="black")

    # Add y density
    if add_target_distribution:

        # Memorize ticks and limits
        xticks = ax.get_xticks()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Create space
        ax.set_xlim(xlim[0] - 1.2 * inset_size * (xlim[1] - xlim[0]), xlim[1])

        # Create shared inset axis
        inset_ax_y = ax.inset_axes([0, 0, inset_size, 1])  # , zorder=10)
        ax.get_shared_y_axes().join(ax, inset_ax_y)

        # Plot histogram
        sns.histplot(y=target, color="grey", stat="density", kde=True, bins=n_bins, ax=inset_ax_y)
        inset_ax_y.set_ylim(ylim)

        # Remove overlayed elements
        inset_ax_y.set_xticklabels([])
        inset_ax_y.set_yticklabels([])
        inset_ax_y.set_xlabel("")
        inset_ax_y.set_ylabel("")

        # More space for plot
        left, right = inset_ax_y.get_xlim()
        inset_ax_y.set_xlim(left, right * 1.2)

        # Border
        #inset_ax_y.axvline(inset_ax_y.get_xlim()[1], color="black")

        # Remove senseless ticks
        _ = ax.set_xticks(xticks[(xticks >= xlim[0]) & (xticks <= xlim[1])])

        # Add Boxplot
        if add_boxplot:

            # Create space
            xlim_inner = inset_ax_y.get_xlim()
            inset_ax_y.set_xlim(xlim_inner[0] - 2 * inset_size * (xlim_inner[1] - xlim_inner[0]))

            # Create shared inset axis without any elements
            inset_inset_ax_y = inset_ax_y.inset_axes([0, 0, inset_size, 1])
            inset_inset_ax_y.set_axis_off()
            inset_ax_y.get_shared_y_axes().join(inset_ax_y, inset_inset_ax_y)

            # Plot boxplot
            sns.boxplot(y=target, color="lightgrey", orient="v",
                        showmeans=True, meanprops={"marker": "x",
                                                   "markerfacecolor": "white", "markeredgecolor": "white"},
                        ax=inset_inset_ax_y)
            inset_inset_ax_y.set_ylim(ylim)
            inset_inset_ax_y.set_rasterized(rasterized)

            # More space for plot
            left, right = inset_inset_ax_y.get_xlim()
            range = right - left
            inset_inset_ax_y.set_xlim(left - range * 0.5, right)

    # Add x density
    if add_feature_distribution:

        # Memorize ticks and limits
        yticks = ax.get_yticks()
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()

        # Create space
        ax.set_ylim(ylim[0] - 1.2 * inset_size * (ylim[1] - ylim[0]), ylim[1])

        # Create shared inset axis
        inset_ax_x = ax.inset_axes([0, 0, 1, inset_size])  # , zorder=10)
        ax.get_shared_x_axes().join(ax, inset_ax_x)

        # Plot histogram
        sns.histplot(x=feature, color="grey", stat="density", kde=True, bins=n_bins, ax=inset_ax_x)
        inset_ax_x.set_xlim(xlim)

        # Remove overlayed elements
        inset_ax_x.set_xticklabels([])
        inset_ax_x.set_yticklabels([])
        inset_ax_x.set_xlabel("")
        inset_ax_x.set_ylabel("")

        # More space for plot
        left, right = inset_ax_x.get_ylim()
        inset_ax_x.set_ylim(left, right * 1.2)

        # Border
        #inset_ax_x.axhline(inset_ax_x.get_ylim()[1], color="black")

        # Remove senseless ticks
        _ = ax.set_yticks(yticks[(yticks >= ylim[0]) & (yticks <= ylim[1])])

        # Add Boxplot
        if add_boxplot:

            # Create space
            ylim_inner = inset_ax_x.get_ylim()
            inset_ax_x.set_ylim(ylim_inner[0] - 2 * inset_size * (ylim_inner[1] - ylim_inner[0]))

            # Create shared inset axis without any elements
            inset_inset_ax_x = inset_ax_x.inset_axes([0, 0, 1, inset_size])
            inset_inset_ax_x.set_axis_off()
            inset_ax_x.get_shared_x_axes().join(inset_ax_x, inset_inset_ax_x)

            # PLot boxplot
            sns.boxplot(x=feature, color="lightgrey",
                        showmeans=True, meanprops={"marker": "x", "markerfacecolor": "white",
                                                   "markeredgecolor": "white"},
                        ax=inset_inset_ax_x)
            inset_inset_ax_x.set_xlim(xlim)
            inset_inset_ax_y.set_rasterized(rasterized)

            # More space for plot
            left, right = inset_inset_ax_x.get_ylim()
            range = right - left
            inset_inset_ax_x.set_ylim(left - range * 0.5, right)

    # Hide intersection
    if add_feature_distribution and add_target_distribution:
        inset_ax_over = ax.inset_axes([0, 0, inset_size, inset_size])  # , zorder=20)
        inset_ax_over.set_facecolor("white")
        inset_ax_over.get_xaxis().set_visible(False)
        inset_ax_over.get_yaxis().set_visible(False)
        for pos in ["bottom", "left"]:
            inset_ax_over.spines[pos].set_edgecolor(None)


def plot_feature_target(ax,
                        feature, target, feature_type=None, target_type=None,
                        feature_name=None, target_name=None,
                        target_category=None,
                        feature_lim=None, target_lim=None,
                        min_width=0.2, inset_size=0.2, refline=True, n_bins=20,
                        regplot=True, regplot_type="lowess", lowess_n_sample=1000, lowess_frac=2 / 3, spline_smooth=1, 
                        add_colorbar=True,
                        add_feature_distribution=True, add_target_distribution=True, add_boxplot=True, rasterized=True,
                        title=None,
                        add_miss_info=True,
                        color=list(sns.color_palette("colorblind").as_hex()),
                        colormap=LinearSegmentedColormap.from_list("bl_yl_rd", ["blue", "yellow", "red"])):

    # Determine feature and target type
    if feature_type is None:
        feature_type = "nume" if pd.api.types.is_numeric_dtype(feature) else "cate"
    if target_type is None:
        target_type = (dict(binary="CLASS", continuous="REGR", multiclass="MULTICLASS")
                       [type_of_target(target[~pd.Series(target).isna()])])

    # Call plot functions
    params = dict(ax=ax, feature=feature, target=target,
                  feature_name=feature_name, target_name=target_name,
                  target_category=target_category,
                  feature_lim=feature_lim, target_lim=target_lim,
                  min_width=min_width, inset_size=inset_size, refline=refline, n_bins=n_bins,
                  regplot=regplot, regplot_type=regplot_type,
                  lowess_n_sample=lowess_n_sample, lowess_frac=lowess_frac, spline_smooth=spline_smooth, 
                  add_colorbar=add_colorbar,
                  add_feature_distribution=add_feature_distribution, add_target_distribution=add_target_distribution,
                  add_boxplot=add_boxplot, rasterized=rasterized,
                  title=title,
                  add_miss_info=add_miss_info,
                  color=color,
                  colormap=colormap)
    if feature_type == "nume":
        if target_type == "CLASS":
            plot_nume_CLASS(**params)
        elif target_type == "MULTICLASS":
            plot_nume_MULTICLASS(**params)
        elif target_type == "REGR":
            plot_nume_REGR(**params)
        else:
            raise Exception('Wrong TARGET_TYPE')

    else:
        if target_type == "CLASS":
            plot_cate_CLASS(**params)
        elif target_type == "MULTICLASS":
            plot_cate_MULTICLASS(**params)
        elif target_type == "REGR":
            plot_cate_REGR(**params)
        else:
            raise Exception('Wrong TARGET_TYPE')

    # Create Frame
    # for spine in ax.spines.values():
    #    spine.set_edgecolor('black')


# Plot correlation
def plot_corr(ax, df, method, absolute=True, cutoff=None, n_jobs=1):

    # Check for mixed types
    count_numeric_dtypes = df.apply(lambda x: pd.api.types.is_numeric_dtype(x)).sum()
    if count_numeric_dtypes not in [0, df.shape[1]]:
        raise Exception('Mixed dtypes.')

    # Metr
    if count_numeric_dtypes != 0:
        if method not in ["pearson", "spearman"]:
            raise Exception('False method for numeric values: Choose "pearson" or "spearman"')
        df_corr = df.corr(method=method)
        suffix = " (" + round(df.isnull().mean() * 100, 1).astype("str") + "% NA)"

    # Cate
    else:
        if method not in ["cramersv"]:
            raise Exception('False method for numeric values: Choose "cramersv"')
        n = df.shape[1]
        df_corr = pd.DataFrame(np.zeros([n, n]), index=df.columns.values, columns=df.columns.values)
        l_tup = [(i, j) for i in range(n) for j in range(i + 1, n)]
        result = Parallel(n_jobs=n_jobs, max_nbytes='100M')(delayed(corrcat)(df.iloc[:, i], df.iloc[:, j])
                                                            for i, j in l_tup)
        for k, (i, j) in enumerate(l_tup):
            df_corr.iloc[i, j] = result[k]
            #df_corr.iloc[i, j] = corrcat(df.iloc[:, i], df.iloc[:, j])
            df_corr.iloc[j, i] = df_corr.iloc[i, j]

        '''
        for i in range(n):
            for j in range(i+1, n):
                df_corr.iloc[i, j] = corrcat(df.iloc[:, i], df.iloc[:, j])
                df_corr.iloc[j, i] = df_corr.iloc[i, j]
        '''
        suffix = " (" + df.nunique().astype("str").values + ")"

    # Add info to names
    d_new_names = dict(zip(df_corr.columns.values, df_corr.columns.values + suffix))
    df_corr.rename(columns=d_new_names, index=d_new_names, inplace=True)

    # Absolute trafo
    if absolute:
        df_corr = df_corr.abs()

    # Filter out rows or cols below cutoff and then fill diagonal
    np.fill_diagonal(df_corr.values, 0)
    if cutoff is not None:
        i_cutoff = (df_corr.abs().max(axis=1) > cutoff).values
        df_corr = df_corr.loc[i_cutoff, i_cutoff]
    np.fill_diagonal(df_corr.values, 1)

    # Cluster df_corr
    tmp_order = linkage(1 - np.triu(df_corr),
                        method="average", optimal_ordering=False)[:, :2].flatten().astype(int)
    new_order = df_corr.columns.values[tmp_order[tmp_order < len(df_corr)]]
    df_corr = df_corr.loc[new_order, new_order]

    # Plot
    sns.heatmap(df_corr, annot=True, fmt=".2f", cmap="Reds" if absolute else "BLues",
                xticklabels=True, yticklabels=True, ax=ax)
    ax.set_yticklabels(labels=ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=90)
    ax.set_title(("Absolute " if absolute else "") + method.upper() + " Correlation" +
                 (" (cutoff: " + str(cutoff) + ")" if cutoff is not None else ""))

    return df_corr


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

        # TODO: Iterate in parallel also over splits (see original fit method)
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
                if hasattr(self.estimator, "subestimator"):
                    estimator = self.estimator.subestimator
                else:
                    estimator = self.estimator                
                for ntree_limit in n_estimators:
                    start = time.time()
                    if isinstance(estimator, lgbm.sklearn.LGBMClassifier):
                        yhat_test = fit.predict_proba(_safe_indexing(X, i_test), num_iteration=ntree_limit)
                    elif isinstance(estimator, lgbm.sklearn.LGBMRegressor):
                        yhat_test = fit.predict(_safe_indexing(X, i_test), num_iteration=ntree_limit)
                    elif isinstance(estimator, xgb.sklearn.XGBClassifier):
                        yhat_test = fit.predict_proba(_safe_indexing(X, i_test), ntree_limit=ntree_limit)
                    else:
                        yhat_test = fit.predict(_safe_indexing(X, i_test), ntree_limit=ntree_limit)
                    score_time = time.time() - start

                    # Do it for training as well
                    if self.return_train_score:
                        if isinstance(estimator, lgbm.sklearn.LGBMClassifier):
                            yhat_train = fit.predict_proba(_safe_indexing(X, i_train), num_iteration=ntree_limit)
                        elif isinstance(estimator, lgbm.sklearn.LGBMRegressor):
                            yhat_train = fit.predict(_safe_indexing(X, i_train), num_iteration=ntree_limit)
                        elif isinstance(estimator, xgb.sklearn.XGBClassifier):
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

# Plot cv results
def plot_cvresults(cv_results, metric, x_var, color_var=None, style_var=None, column_var=None, row_var=None,
                   show_gap=True, show_std=False, color=list(sns.color_palette("tab10").as_hex()),
                   height=6):

    # Transform results
    df_cvres = pd.DataFrame.from_dict(cv_results)
    df_cvres.columns = df_cvres.columns.str.replace("param_", "")
    #df_cvres[x_var] = df_cvres[x_var].astype("float")
    if show_gap:
        df_cvres["gap"] = df_cvres["mean_train_" + metric] - df_cvres["mean_test_" + metric]
        gap_range = (df_cvres["gap"].min(), df_cvres["gap"].max())

    # Define plot function to use in FacetGrid
    def tmp(x, y, y2=None, std=None, std2=None, data=None,
            hue=None, style=None, color=None,
            show_gap=False, gap_range=None):

        # Test results
        sns.lineplot(x=x, y=y, data=data, hue=hue,
                     style=style, linestyle="-", markers=True if style is not None else None, dashes=False,
                     marker=True if style is not None else "o",
                     palette=color)

        # Train results
        sns.lineplot(x=x, y=y2, data=data, hue=hue,
                     style=style, linestyle="--", markers=True if style is not None else None, dashes=False,
                     marker=True if style is not None else "o",
                     palette=color)

        # Std bands
        if std is not None or std2 is not None:
            if hue is None:
                data[hue] = "dummy"
            if style is None:
                data[style] = "dummy"
            data = data.reset_index(drop=True)
            for key, val in data.groupby([hue, style]).groups.items():
                data_group = data.iloc[val, :]
                color_group = list(np.array(color)[data[hue].unique() == key[0]])[0]
                if std is not None:
                    plt.gca().fill_between(data_group[x],
                                           data_group[y] - data_group[std], data_group[y] + data_group[std],
                                           color=color_group, alpha=0.1)
                if std2 is not None:
                    plt.gca().fill_between(data_group[x],
                                           data_group[y2] - data_group[std2], data_group[y2] + data_group[std2],
                                           color=color_group, alpha=0.1)

        # Generalization gap
        if show_gap:
            ax2 = plt.gca().twinx()
            sns.lineplot(x=x, y="gap", data=data, hue=hue,
                         style=style, linestyle=":", markers=True if style is not None else None, dashes=False,
                         marker=True if style is not None else "o",
                         palette=color,
                         ax=ax2)
            ax2.set_ylabel("")
            if ax2.get_legend() is not None:
                ax2.get_legend().remove()
            ax2.set_ylim(gap_range)
            # ax2.axis("off")
        return plt.gca()

    # Plot FacetGrid
    g = (sns.FacetGrid(df_cvres, col=column_var, row=row_var, margin_titles=False if show_gap else True, 
                       height=height, aspect=1)
         .map_dataframe(tmp, x=x_var, y="mean_test_" + metric, y2="mean_train_" + metric,
                        std="std_test_" + metric if show_std else None,
                        std2="std_train_" + metric if show_std else None,
                        hue=color_var, style=style_var, 
                        color=color[:df_cvres[color_var].nunique()] if color_var is not None else color[0],
                        show_gap=show_gap, gap_range=gap_range if show_gap else None)
         .set_xlabels(x_var)
         .add_legend(title=None if style_var is not None else color_var))

    # Beautify and title
    g.legend._legend_box.align = "left"
    g.fig.subplots_adjust(wspace=0.2, hspace=0.1)
    g.fig.subplots_adjust(top=0.9)
    _ = g.fig.suptitle(metric + ": test (-) vs train (--)" + (" and gap (:)" if show_gap else ""), fontsize=16)
    g.tight_layout()
    return g


# Plot model comparison
def plot_modelcomp(ax, df_modelcomp_result, modelvar="model", runvar="run", scorevar="test_score"):
    sns.boxplot(data=df_modelcomp_result, x=modelvar, y=scorevar, showmeans=True,
                meanprops={"markerfacecolor": "black", "markeredgecolor": "black"},
                ax=ax)
    sns.lineplot(data=df_modelcomp_result, x=modelvar, y=scorevar,
                 hue=df_modelcomp_result[runvar], linewidth=0.5, linestyle=":",
                 legend=None, ax=ax)


# Plot learning curve
def plot_learningcurve(ax, n_train, score_train, score_test, time_train,
                       add_time=True,
                       color=list(sns.color_palette("tab10").as_hex())):

    score_train_mean = np.mean(score_train, axis=1)
    score_train_std = np.std(score_train, axis=1)
    score_test_mean = np.mean(score_test, axis=1)
    score_test_std = np.std(score_test, axis=1)
    time_train_mean = np.mean(time_train, axis=1)
    time_train_std = np.std(time_train, axis=1)

    # Plot learning curve
    ax.plot(n_train, score_train_mean, label="Train", marker='o', color=color[0])
    ax.plot(n_train, score_test_mean, label="Test", marker='o', color=color[1])
    ax.fill_between(n_train, score_train_mean - score_train_std, score_train_mean + score_train_std,
                    alpha=0.1, color=color[0])
    ax.fill_between(n_train, score_test_mean - score_test_std, score_test_mean + score_test_std,
                    alpha=0.1, color=color[1])

    # Plot fitting time
    if add_time:
        ax2 = ax.twinx()
        ax2.plot(n_train, time_train_mean, label="Time", marker="x", linestyle=":", color="grey")
        # ax2.fill_between(n_train, time_train_mean - time_train_std, time_train_mean + time_train_std,
        #                 alpha=0.1, color="grey")
        ax2.set_ylabel("Fitting time [s]")
    ax.legend(loc="best")
    ax.set_ylabel("Score")
    ax.set_title("Learning curve")


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
    def __init__(self, subestimator=None, b_sample=None, b_all=None, **kwargs):
        self.subestimator = subestimator
        self.b_sample = b_sample
        self.b_all = b_all
        self._estimator_type = subestimator._estimator_type
        if kwargs:
            self.subestimator.set_params(**kwargs)
        
    def get_params(self, deep=True):
        return dict(subestimator=self.subestimator,
                    b_sample=self.b_sample,
                    b_all=self.b_all,
                    **self.subestimator.get_params())

    def set_params(self, **params):
        if "subestimator" in params:
            self.subestimator = params["subestimator"]
            del params["subestimator"]
        if "b_sample" in params:
            self.b_sample = params["b_sample"]
            del params["b_sample"]
        if "b_all" in params:
            self.b_all = params["b_all"]
            del params["b_all"]
        self.subestimator = self.subestimator.set_params(**params)
        return self

    def fit(self, X, y, *args, **kwargs):
        self.classes_ = unique_labels(y)
        self.subestimator.fit(X, y, *args, **kwargs)
        return self

    def predict(self, X, *args, **kwargs):
        return self.subestimator.predict(X, *args, **kwargs)

    def predict_proba(self, X, *args, **kwargs):
        yhat = scale_predictions(self.subestimator.predict_proba(X, *args, **kwargs),
                                 self.b_sample, self.b_all)
        return yhat


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


# Metaestimator which undersamples before training and resclaes the predictions accordingly
class UndersampleEstimator(BaseEstimator):
    def __init__(self, subestimator=None, n_max_per_level=np.inf, seed=42, **kwargs):
        self.subestimator = subestimator
        self.n_max_per_level = n_max_per_level
        self.seed = seed
        self._estimator_type = subestimator._estimator_type
        if kwargs:
            self.subestimator.set_params(**kwargs)
        
    def get_params(self, deep=True):
        return dict(subestimator=self.subestimator,
                    n_max_per_level=self.n_max_per_level,
                    seed=self.seed,
                    **self.subestimator.get_params())

    def set_params(self, **params):
        if "subestimator" in params:
            self.subestimator = params["subestimator"]
            del params["subestimator"]
        if "n_max_per_level" in params:
            self.b_sample = params["n_max_per_level"]
            del params["n_max_per_level"]
        if "seed" in params:
            self.b_all = params["seed"]
            del params["seed"]
        self.subestimator = self.subestimator.set_params(**params)
        return self

    def fit(self, X, y, *args, **kwargs):
        
        # Sample and set b_sample_, b_all_
        if type_of_target(y) == "continuous":
            df_tmp = (pd.DataFrame(dict(y=y)).reset_index(drop=True).reset_index()
                      .pipe(lambda x: x.sample(min(self.n_max_per_level, x.shape[0]))))
        else:
            self.classes_ = unique_labels(y)       
            df_tmp = pd.DataFrame(dict(y=y)).reset_index(drop=True).reset_index()
            self.b_all_ = df_tmp["y"].value_counts().values / len(df_tmp)
            df_tmp = (df_tmp.groupby("y")
                      .apply(lambda x: x.sample(min(self.n_max_per_level, x.shape[0]), random_state=self.seed))
                      .reset_index(drop=True))
            self.b_sample_ = df_tmp["y"].value_counts().values / len(df_tmp)
        y_under = df_tmp["y"].values
        X_under = X[df_tmp["index"].values, :]

        # Fit
        self.subestimator.fit(X_under, y_under, *args, **kwargs)
        return self

    def predict(self, X, *args, **kwargs):
        print("")
        return self.subestimator.predict(X, *args, **kwargs)

    def predict_proba(self, X, *args, **kwargs):
        yhat = scale_predictions(self.subestimator.predict_proba(X, *args, **kwargs),
                                 self.b_sample_, self.b_all_)
        return yhat


# Metaestimator for log-transformed target
class LogtrafoEstimator(BaseEstimator):
    def __init__(self, subestimator=None, variance_scaling_factor=1, **kwargs):
        self.subestimator = subestimator
        self.subestimator = self.subestimator.set_params(**kwargs)
        self.variance_scaling_factor = variance_scaling_factor
        self._estimator_type = subestimator._estimator_type
        if kwargs:
            self.subestimator.set_params(**kwargs)

    def get_params(self, deep=True):
        return dict(subestimator=self.subestimator,
                    variance_scaling_factor=self.variance_scaling_factor,
                    **self.subestimator.get_params())

    def set_params(self, **params):
        if "subestimator" in params:
            self.subestimator = params["subestimator"]
            del params["subestimator"]
        if "variance_scaling_factor" in params:
            self.b_sample = params["variance_scaling_factor"]
            del params["variance_scaling_factor"]
        self.subestimator = self.subestimator.set_params(**params)
        return self

    def fit(self, X, y, *args, **kwargs):
        self.subestimator.fit(X, np.log(1 + y), *args, **kwargs)
        res = self.subestimator.predict(X) - np.log(1 + y)
        print(np.std(res)**2)
        self.varest_ = np.var(res)
        return self

    def predict(self, X, *args, **kwargs):
        return (np.exp(self.subestimator.predict(X, *args, **kwargs) +
                       0.5 * self.variance_scaling_factor * self.varest_) - 1)


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
                pd.DataFrame(np.mean(estimator.predict_proba(df_copy) if estimator._estimator_type == "classifier" else
                                     estimator.predict(df_copy), axis=0).reshape(1, -1)))
        # yhat_pd = np.append(yhat_pd,
        #                     np.mean(estimator.predict_proba(df_pd) if estimator._estimator_type == "classifier" else
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
    # for k in range(a_shap.shape[2]):

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
def plot_roc(ax, y, yhat, 
             color=list(sns.color_palette("colorblind").as_hex()), target_labels=None):

    # also for regression
    if (y.ndim == 1) & (yhat.ndim == 1):
        if (np.min(y) < 0) | (np.max(y) > 1):
            y = MinMaxScaler().fit_transform(y.reshape(-1, 1))[:, 0]
        if (np.min(yhat) < 0) | (np.max(yhat) > 1):
            yhat = MinMaxScaler().fit_transform(yhat.reshape(-1, 1))[:, 0]

    # CLASS (and regression)
    if yhat.ndim == 1:
        # Roc curve
        fpr, tpr, _ = roc_curve(y, yhat)
        roc_auc = roc_auc_score(y, yhat)
        # sns.lineplot(fpr, tpr, ax=ax, palette=sns.xkcd_palette(["red"]))
        ax.plot(fpr, tpr)
        ax.set_title("ROC (AUC = " + format(roc_auc, "0.2f") + ")")
        
    # MULTICLASS
    else:
        n_classes = yhat.shape[1]
        aucs = np.array([round(roc_auc_score(np.where(y == i, 1, 0), yhat[:, i]), 2) for i in np.arange(n_classes)])
        for i in np.arange(n_classes):
            y_bin = np.where(y == i, 1, 0)
            fpr, tpr, _ = roc_curve(y_bin, yhat[:, i])
            if target_labels is not None:
                new_label = str(target_labels[i]) + " (" + str(aucs[i]) + ")"
            else:
                new_label = str(i) + " (" + str(aucs[i]) + ")"
            ax.plot(fpr, tpr, color=color[i], label=new_label)
        mean_auc = np.average(aucs).round(3)
        weighted_auc = np.average(aucs, weights=np.array(np.unique(y, return_counts=True))[1, :]).round(3)
        ax.set_title("ROC\n" + r"($AUC_{mean}$ = " + str(mean_auc) + r", $AUC_{weighted}$ = " +
                     str(weighted_auc) + ")")
        ax.legend(title=r"Target ($AUC_{OvR}$)", loc='best')

    ax.set_xlabel(r"fpr: P($\^y$=1|$y$=0)")
    ax.set_ylabel(r"tpr: P($\^y$=1|$y$=1)")


# Plot calibration
def plot_calibration(ax, y, yhat, n_bins=5, 
                     color=list(sns.color_palette("colorblind").as_hex()), target_labels=None):

    minmin = np.inf
    maxmax = -np.inf
    max_yhat = -np.inf
    
    if yhat.ndim > 1:
        n_classes = yhat.shape[1]
    else:
        n_classes = 1
    
    for i in np.arange(n_classes):
        # Plot
        df_plot = (pd.DataFrame({"y": np.where(y == i, 1, 0) if yhat.ndim > 1 else y,
                                 "yhat": yhat[:, i] if yhat.ndim > 1 else yhat})
                   .assign(bin=lambda x: pd.qcut(x["yhat"], n_bins, duplicates="drop").astype("str"))
                   .groupby(["bin"], as_index=False).agg("mean")
                   .sort_values("yhat"))
        ax.plot(df_plot["yhat"], df_plot["y"], "o-", color=color[i],
                label=target_labels[i] if target_labels is not None else str(i))
        
        # Get limits
        minmin = min(minmin, min(df_plot["y"].min(), df_plot["yhat"].min()))
        maxmax = max(maxmax, max(df_plot["y"].max(), df_plot["yhat"].max()))
        max_yhat = max(max_yhat, df_plot["yhat"].max())
        
    # Diagonal line   
    ax.plot([minmin, maxmax], [minmin, maxmax], linestyle="--", color="grey")
    
    # Focus       
    ax.set_xlim(None, maxmax + 0.05 * (maxmax - minmin))
    ax.set_ylim(None, maxmax + 0.05 * (maxmax - minmin))
        
    # Labels
    props = {'xlabel': r"$\bar{\^y}$ in $\^y$-bin",
             'ylabel': r"$\bar{y}$ in $\^y$-bin",
             'title': "Calibration"}
    _ = ax.set(**props)
    
    if yhat.ndim > 1:
        ax.legend(title="Target", loc='best')


# PLot confusion matrix
def plot_confusion(ax, y, yhat, threshold=0.5, cmap="Blues", target_labels=None):

    # binary label
    if yhat.ndim == 1:
        yhat_bin = np.where(yhat > threshold, 1, 0)
    else:
        yhat_bin = yhat.argmax(axis=1)
        
    # Confusion dataframe
    unique_y = np.unique(y)
    freq_y = np.unique(y, return_counts=True)[1]
    freqpct_y = np.round(np.divide(freq_y, len(y)) * 100, 1)
    freq_yhat = np.unique(np.concatenate((yhat_bin, unique_y)), return_counts=True)[1] - 1
    freqpct_yhat = np.round(np.divide(freq_yhat, len(y)) * 100, 1)
    m_conf = confusion_matrix(y, yhat_bin)
    if target_labels is None:
        target_labels = unique_y
    ylabels = [str(target_labels[i]) + " (" + str(freq_y[i]) + ": " + str(freqpct_y[i]) + "%)" for i in
               np.arange(len(target_labels))]
    xlabels = [str(target_labels[i]) + " (" + str(freq_yhat[i]) + ": " + str(freqpct_yhat[i]) + "%)" for i in
               np.arange(len(target_labels))]
    df_conf = (pd.DataFrame(m_conf, columns=target_labels, index=target_labels)
               .rename_axis(index="True label",
                            columns="Predicted label"))

    # accuracy and confusion calculation
    acc = accuracy_score(y, yhat_bin)

    # plot
    sns.heatmap(df_conf, annot=True, fmt=".5g", cmap=cmap, ax=ax,
                xticklabels=True, yticklabels=True, cbar=False)
    ax.set_yticklabels(labels=ylabels, rotation=0)
    ax.set_xticklabels(labels=xlabels, rotation=90 if yhat.ndim > 1 else 0, ha="center")
    ax.set_xlabel("Predicted label (#: %)")
    ax.set_ylabel("True label (#: %)")
    ax.set_title("Confusion Matrix ($Acc_{" +
                 (format(threshold, "0.2f") if yhat.ndim == 1 else "") +
                 "}$ = " + format(acc, "0.2f") + ")")
    for text in ax.texts[::len(target_labels) + 1]:
        text.set_weight('bold')


def plot_confusionbars(ax, y, yhat, type, target_labels=None):
    
    n_classes = yhat.shape[1]
    
    # Make series
    y = pd.Series(y, name="y").astype(str)
    yhat = pd.Series(yhat.argmax(axis=1), name="yhat").astype(str)
    
    # Map labels
    if target_labels is not None:
        d_map = {str(i): str(target_labels[i]) for i in np.arange(n_classes)}
        y = y.map(d_map)
        yhat = yhat.map(d_map)
        
    # Plot and adapt
    if type == "true":
        plot_cate_MULTICLASS(ax, feature=y, target=yhat, reverse=True)
        ax.set_xlabel("% Predicted label")
        ax.set_ylabel("True label")
    else:
        plot_cate_MULTICLASS(ax, feature=yhat, target=y, exchange_x_y_axis=True)
        ax.set_xlabel("True label")
        ax.set_ylabel("% Predicted label")
    ax.set_title("")
    ax.get_legend().remove()


def plot_multiclass_metrics(ax, y, yhat, target_labels=None):

    m_conf = confusion_matrix(y, yhat.argmax(axis=1))
    aucs = np.array([round(roc_auc_score(np.where(y == i, 1, 0), yhat[:, i]), 2) for i in np.arange(yhat.shape[1])])
    prec = np.round(np.diag(m_conf) / m_conf.sum(axis=0) * 100, 1)
    rec = np.round(np.diag(m_conf) / m_conf.sum(axis=1) * 100, 1)
    f1 = np.round(2 * prec * rec / (prec + rec), 1)
    
    # Top3 metrics
    if target_labels is None:
        target_labels = np.unique(y).tolist()
    df_metrics = (pd.DataFrame(np.column_stack((y, np.flip(np.argsort(yhat, axis=1), axis=1)[:, :3])),
                               columns=["y", "yhat1", "yhat2", "yhat3"])
                  .assign(acc_top1=lambda x: (x["y"] == x["yhat1"]).astype("int"),
                          acc_top2=lambda x: ((x["y"] == x["yhat1"]) | (x["y"] == x["yhat2"])).astype("int"),
                          acc_top3=lambda x: ((x["y"] == x["yhat1"]) | (x["y"] == x["yhat2"]) |
                                              (x["y"] == x["yhat3"])).astype("int"))
                  .assign(label=lambda x: np.array(target_labels, dtype="object")[x["y"].values])
                  .groupby(["label"])["acc_top1", "acc_top2", "acc_top3"].agg("mean").round(2)
                  .join(pd.DataFrame(np.stack((aucs, rec, prec, f1), axis=1),
                                     index=target_labels, columns=["auc", "recall", "precision", "f1"])))
    sns.heatmap(df_metrics.T, annot=True, fmt=".5g",
                cmap=ListedColormap(['white']), linewidths=2, linecolor="black", cbar=False,
                ax=ax, xticklabels=True, yticklabels=True)
    ax.set_yticklabels(labels=['Accuracy\n Top1', 'Accuracy\n Top2', 'Accuracy\n Top3', "AUC\n 1-vs-all",
                               'Recall\n' r"P($\^y$=k|$y$=k))", 'Precision\n' r"P($y$=k|$\^y$=k))", 'F1'])
    ax.xaxis.tick_top()  # x axis on top
    ax.xaxis.set_label_position('top')
    ax.tick_params(left=False, top=False)
    ax.set_xlabel("True label")


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
    d_calls["distribution"] = (plot_nume_CLASS, dict(feature=yhat, target=y, feature_lim=(0, 1), 
                                                     feature_name=r"Predictions ($\^y$)",
                                                     add_miss_info=False))
    d_calls["calibration"] = (plot_calibration, dict(y=y, yhat=yhat, n_bins=n_bins))
    d_calls["precision_recall"] = (plot_precision_recall, dict(y=y, yhat=yhat, annotate=annotate, fontsize=fontsize))
    d_calls["precision"] = (plot_precision, dict(y=y, yhat=yhat, annotate=annotate, fontsize=fontsize))

    return d_calls


# Plot model performance for CLASS target
def get_plotcalls_model_performance_MULTICLASS(y, yhat,
                                               n_bins=5, cmap="Blues", annotate=True, fontsize=10,
                                               target_labels=None):

    # Define plot dict
    d_calls = dict()
    d_calls["roc"] = (plot_roc, dict(y=y, yhat=yhat, target_labels=target_labels))
    d_calls["confusion"] = (plot_confusion, dict(y=y, yhat=yhat, threshold=None, cmap=cmap, 
                                                 target_labels=target_labels))
    d_calls["true_bars"] = (plot_confusionbars, dict(y=y, yhat=yhat, type="true", target_labels=target_labels))
    d_calls["calibration"] = (plot_calibration, dict(y=y, yhat=yhat, n_bins=n_bins, target_labels=target_labels))
    d_calls["pred_bars"] = (plot_confusionbars, dict(y=y, yhat=yhat, type="pred", target_labels=target_labels))
    d_calls["multiclass_metrics"] = (plot_multiclass_metrics, dict(y=y, yhat=yhat, target_labels=target_labels))

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
    d_calls["observed_vs_fitted"] = (plot_nume_REGR, dict(feature=yhat, target=y, 
                                                          feature_name=r"$\^y$", target_name="y",
                                                          title=title,
                                                          feature_lim=ylim,
                                                          regplot=regplot,
                                                          add_miss_info=False))
    d_calls["calibration"] = (plot_calibration, dict(y=y, yhat=yhat, n_bins=n_bins))
    d_calls["distribution"] = (plot_nume_CLASS, dict(feature=np.append(y, yhat),
                                                     target=np.append(np.tile("y", len(y)),
                                                                      np.tile(r"$\^y$", len(yhat))),
                                                     feature_name="",
                                                     target_name="",
                                                     title="Distribution",
                                                     add_miss_info=False,))
    d_calls["residuals_vs_fitted"] = (plot_nume_REGR, dict(feature=yhat, target=y - yhat,
                                                           feature_name=r"$\^y$",
                                                           target_name=r"y-$\^y$",
                                                           title="Residuals vs. Fitted",
                                                           feature_lim=ylim,
                                                           regplot=regplot,
                                                           add_miss_info=False))

    d_calls["absolute_residuals_vs_fitted"] = (plot_nume_REGR, dict(feature=yhat, target=abs(y - yhat),
                                                                    feature_name=r"$\^y$",
                                                                    target_name=r"|y-$\^y$|",
                                                                    title="Absolute Residuals vs. Fitted",
                                                                    feature_lim=ylim,
                                                                    regplot=regplot,
                                                                    add_miss_info=False))

    d_calls["relative_residuals_vs_fitted"] = (plot_nume_REGR, dict(feature=yhat,
                                                                    target=np.where(y == 0, np.nan,
                                                                                    abs(y - yhat) / abs(y)),
                                                                    feature_name=r"$\^y$",
                                                                    target_name=r"|y-$\^y$|/|y|",
                                                                    title="Relative Residuals vs. Fitted",
                                                                    feature_lim=ylim,
                                                                    regplot=regplot,
                                                                    add_miss_info=False))

    return d_calls


# Wrapper for plot_model_performance_<target_type>
def get_plotcalls_model_performance(y, yhat, target_type=None,
                                    n_bins=5, threshold=0.5, target_labels=None, 
                                    cmap="Blues", annotate=True, fontsize=10,                                    
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
        d_calls = get_plotcalls_model_performance_REGR(
            y=y, yhat=yhat, ylim=ylim, regplot=regplot, n_bins=n_bins)
    elif target_type == "MULTICLASS":
        d_calls = get_plotcalls_model_performance_MULTICLASS(
            y=y, yhat=yhat, n_bins=n_bins, target_labels=target_labels, cmap=cmap, annotate=annotate, fontsize=fontsize)
    else:
        warnings.warn("Target type cannot be determined")

    # Filter plot dict
    if l_plots is not None:
        d_calls = {x: d_calls[x] for x in l_plots}

    return d_calls


# Plot permutation base variable importance
def plot_variable_importance(ax,
                             features, importance,
                             importance_cum=None, importance_mean=None, importance_se=None, max_score_diff=None,
                             category=None,
                             category_label="Importance",
                             category_color_palette=sns.xkcd_palette(["blue", "orange", "red"]),
                             color_error="grey"):

    sns.barplot(x=importance, y=features, hue=category,
                palette=category_color_palette, dodge=False, ax=ax)
    ax.set_title("Top{0: .0f} Feature Importances".format(len(features)))
    ax.set_xlabel(r"permutation importance")
    if max_score_diff is not None:
        ax.set_xlabel(ax.get_xlabel() + " (100 = " + str(max_score_diff) + r" score-$\Delta$)")
    if importance_cum is not None:
        ax.plot(importance_cum, features, color="black", marker="o")
        ax.set_xlabel(ax.get_xlabel() + " /\n" + r"cumulative in % (-$\bullet$-)")
    if importance_se is not None:
        ax.errorbar(x=importance_mean if importance_mean is not None else importance, y=features, xerr=importance_se,
                    linestyle="none", marker="s", fillstyle="none", color=color_error)
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

    print("plot PD for feature " + feature_name)
    numeric_feature = pd.api.types.is_numeric_dtype(feature)
    # if yhat.ndim == 1:
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
            ax.axhline(refline, linestyle="dotted", color="black")  # priori line

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
            df_plot = df_plot.merge(helper_calc_barboxwidth(feature_ref, np.tile(1, len(feature_ref)), 
                                                            min_width=min_width),
                                    how="inner")
            '''
            df_plot = df_plot.merge(pd.DataFrame({feature_name: feature_ref}).assign(count=1)
                                    .groupby(feature_name, as_index=False)[["count"]].sum()
                                    .assign(pct=lambda x: x["count"] / x["count"].sum())
                                    .assign(width=lambda x: 0.9 * x["pct"] / x["pct"].max()), how="left")
            df_plot[feature_name] = df_plot[feature_name] + " (" + (df_plot["pct"] * 100).round(1).astype(str) + "%)"
            if min_width is not None:
                df_plot["width"] = np.where(df_plot["width"] < min_width, min_width, df_plot["width"])
            #ax2 = ax.twiny()
            #ax2.barh(df_plot[feature_name], df_plot["pct"], color="grey", edgecolor="grey", alpha=0.5, linewidth=0)
            ''' 
                       
        # Bar plot
        ax.barh(df_plot[feature_name] if feature_ref is None else df_plot[feature_name + "_fmt"],
                df_plot["yhat"],
                height=df_plot["w"] if feature_ref is not None else 0.8,
                color=color, edgecolor="black", alpha=0.5, linewidth=1)

        # Refline
        if refline is not None:
            ax.axvline(refline, linestyle="dotted", color="black")  # priori line
            
        # Inner barplot
        helper_inner_barplot(ax, x=df_plot[feature_name + "_fmt"], y=df_plot["pct"], inset_size=0.2)

        # Axis style
        ax.set_title(feature_name)
        ax.set_xlabel(r"$\^y$")
        if ylim is not None:
            ax.set_xlim(ylim)        
        
        # Crossvalidation
        if yhat_err is not None:
            ax.errorbar(df_plot["yhat"],
                        df_plot[feature_name] if feature_ref is None else df_plot[feature_name + "_fmt"],
                        xerr=df_plot["yhat_err"],
                        linestyle="none", marker="s", capsize=5, fillstyle="none", color="grey")


# Plot shap
def plot_shap(ax, shap_values, index, id,
              y_str=None, yhat_str=None,
              show_intercept=True, show_prediction=True,
              shap_lim=None,
              color=["blue", "red"], n_top=10, multiclass_index=None):

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
    if shap_lim is not None:
        x_min = shap_lim[0]
        x_max = shap_lim[1]
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
