########################################################################################################################
# Packages
########################################################################################################################

# General
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score, GridSearchCV, check_cv, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils import _safe_indexing
from sklearn.base import BaseEstimator, TransformerMixin, clone  # ClassifierMixin

# ML
import xgboost as xgb
import lightgbm as lgbm
from itertools import product  # for GridSearchCV_xlgb
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, roc_curve

# Other
from scipy.interpolate import splev, splrep

# Custom functions and classes
import up_utils as up


########################################################################################################################
# General bivariate plots
########################################################################################################################

def plot_bidistribution(ax, x, group, n_bins=20, xlim=None, xlabel=None, title=None, legend_title=None, inset_size=0.2):

    ax_act = ax

    # Distribution
    '''
    members = np.sort(df[target].unique())
    for m, member in enumerate(members):
        sns.distplot(df.loc[df[target] == member, feature_act].dropna(),
                        color=color[m],
                        bins=20,
                        label=member,
                        ax=ax_act)
    if varimp is not None:
        ax_act.set_title(feature_act + " (VI: " + str(varimp[feature_act]) + ")")
    else:
        ax_act.set_title(feature_act)
    '''
    sns.histplot(ax=ax_act, x=x, hue=group, stat="density", common_norm=False, kde=True, bins=n_bins)
    ax_act.set_ylabel("Density")
    if xlabel is not None:
        ax_act.set_xlabel(xlabel)
    #ax_act.legend(title=legend_title, loc = "best")
    if title is not None:
        ax_act.set_title(title)

    # Inner Boxplot
    ylim = ax_act.get_ylim()
    ax_act.set_ylim(ylim[0] - 1.5 * inset_size * (ylim[1] - ylim[0]))
    inset_ax = ax_act.inset_axes([0, 0, 1, 0.2])
    inset_ax.set_axis_off()
    ax_act.get_shared_x_axes().join(ax_act, inset_ax)
    sns.boxplot(ax=inset_ax, x=x, y=group, orient="h",
                showmeans=True, meanprops={"marker": "x", "markerfacecolor": "black", "markeredgecolor": "black"})


# Scatterplot as heatmap
def plot_biscatter(ax, x, y, xlabel=None, ylabel=None,
                   title=None, xlim=None, ylim=None,
                   regplot=False, smooth=0.5,
                   add_y_density=True, add_x_density=True, n_bins=20,
                   add_boxplot=True,
                   inset_size=0.2,
                   add_colorbar=True):

    ax_act = ax

    # Remove names
    if (xlabel is not None) and isinstance(x, pd.Series):
        x.name = xlabel
    if (ylabel is not None) and isinstance(y, pd.Series):
        y.name = ylabel

    '''
    # Helper for scaling of heat-points
    heat_scale = 1
    if ylim is not None:
        ax_act.set_ylim(ylim)
        heat_scale = heat_scale * (ylim[1] - ylim[0]) / (np.max(y) - np.min(y))
    if xlim is not None:
        ax_act.set_xlim(xlim)
        heat_scale = heat_scale * (xlim[1] - xlim[0]) / (np.max(x) - np.min(x))
    '''

    # Heatmap
    heat_cmap = LinearSegmentedColormap.from_list("bl_yl_rd", ["blue", "yellow", "red"])
    #p = ax_act.hexbin(x, y, gridsize=(int(50 * heat_scale), 50), mincnt=1, cmap=heat_cmap)
    p = ax_act.hexbin(x, y, mincnt=1, cmap=heat_cmap)
    if add_colorbar:
        plt.colorbar(p, ax=ax_act)

    # Spline
    if regplot:
        if len(x) < 1000:
            sns.regplot(x=x, y=y, lowess=True, scatter=False, color="black", ax=ax_act)
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
            ax_act.plot(x2, y2, color="black")

    # Set labels
    ax_act.set_ylabel(ylabel)
    ax_act.set_xlabel(xlabel)
    if title is not None:
        ax_act.set_title(title)

    # Get limits before any insetting
    if ylim is None:
        ylim = ax_act.get_ylim()
    if xlim is None:
        xlim = ax_act.get_xlim()

    # Add y density
    if add_y_density:
        # Inner Histogram on y
        inset_ax_y = ax_act.inset_axes([0, 0, inset_size, 1], zorder=10)
        inset_ax_y.get_xaxis().set_visible(False)
        ax_act.get_shared_y_axes().join(ax_act, inset_ax_y)
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
        inset_ax_x = ax_act.inset_axes([0, 0, 1, inset_size], zorder=10)
        inset_ax_x.get_yaxis().set_visible(False)
        ax_act.get_shared_x_axes().join(ax_act, inset_ax_x)
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
    ax_act.set_ylim(ylim[0] - 1.5 * inset_size * (ylim[1] - ylim[0]), ylim[1])
    ax_act.set_xlim(xlim[0] - 1.5 * inset_size * (xlim[1] - xlim[0]), xlim[1])

    # Hide intersection
    if add_y_density and add_x_density:
        inset_ax_over = ax_act.inset_axes([0, 0, inset_size, inset_size], zorder=20)
        inset_ax_over.set_facecolor("white")
        inset_ax_over.get_xaxis().set_visible(False)
        inset_ax_over.get_yaxis().set_visible(False)


def helper_calc_barboxwidth(x, y, min_width=0.2):
    df_hlp = pd.crosstab(x, y)
    df_barboxwidth = (df_hlp.div(df_hlp.sum(axis=1), axis=0)
               .assign(w=df_hlp.sum(axis=1))
               .reset_index()
               .assign(pct=lambda z: 100 * z["w"] / df_hlp.values.sum())
               .assign(w=lambda z: 0.9 * z["w"] / z["w"].max())
               .assign(x_fmt=lambda z: z["x"] + z["pct"].map("( {:,.1f} %)".format))
               .assign(w=lambda z: np.where(z["w"] < min_width, min_width, z["w"])))
    return df_barboxwidth


def plot_bibar(ax, x, y, xlabel=None, ylabel=None, label_yes=None,
               min_width=0.2, inset_size=0.2, ylim=None,
               refline=True, title=None,
               color=up.colorblind[1]):

    ax_act = ax

    # Adapt labels
    if (xlabel is None) and isinstance(x, pd.Series):
        xlabel = x.name
    if (ylabel is None) and isinstance(y, pd.Series):
        ylabel = y.name

    # Convert to series
    x = pd.Series(x).rename("x")
    y = pd.Series(y).rename("y")

    # Get "yes" class
    if label_yes is None:
        label_yes = y.value_counts().sort_values().index.values[0]

    # Define title
    if title is None:
        title = xlabel

    # Prepare data
    df_plot = helper_calc_barboxwidth(x, y, min_width=min_width)

    # Barplot
    ax_act.barh(df_plot["x_fmt"], df_plot[label_yes], height=df_plot["w"],
                color=color, edgecolor="black", alpha=0.5, linewidth=1)
    ax_act.set_xlabel("avg(" + ylabel + ")")
    ax_act.set_title(title)
    if ylim is not None:
        ax_act.set_xlim(ylim)

    # Refline
    if refline:
        ax_act.axvline((y == label_yes).sum() / len(y), ls="dotted", color="black")

    # Inner barplot
    xlim = ax_act.get_xlim()
    ax_act.set_xlim(xlim[0] - 1.5 * inset_size * (xlim[1] - xlim[0]))
    inset_ax = ax_act.inset_axes([0, 0, inset_size, 1], zorder=10)
    inset_ax.set_axis_off()
    ax_act.axvline(xlim[0], color="black")  # separation line
    ax_act.get_shared_y_axes().join(ax_act, inset_ax)
    inset_ax.barh(df_plot["x_fmt"], df_plot.w,
                  color="lightgrey", edgecolor="black", linewidth=1)

'''
var = "season"
df_plot = df[[var, target_name]].pivot(columns=var, values=target_name)
df_plot.columns

fig, ax = plt.subplots(1, 1)
values = df[var].unique()
color = uu.colorblind[1]
_ = plt.boxplot([df[target_name][df[var] == x] for x in values], labels=values, vert=False, widths=0.8,
                patch_artist=True,
                showmeans=True,
                boxprops=dict(facecolor=color, alpha=0.5),
                #capprops=dict(color=color),
                #whiskerprops=dict(color=color),
                medianprops=dict(color="black"),
                meanprops=dict(marker="x",
                               markeredgecolor="black"),
                flierprops=dict(marker="."))

fig, ax = plt.subplots(1, 1)
up.plot_bibar(ax, df["season"], df["cnt_CLASS_num"])
'''

########################################################################################################################
# Exploration plots
########################################################################################################################

# --- Classification ---------------------------------------------------------------------------------------


def plot_CLASS_cate(ax, target, feature, target_label=None, feature_label=None,
                    min_width=0.2, inset_size=0.2, ylim=None,
                    refline=True, varimp=None, varimp_fmt="0.2f",
                    color=up.colorblind[1]):

    # Adapt labels
    if not isinstance(target, pd.Series):
        target = pd.Series(target)
        target.name = target_label
    if not isinstance(feature, pd.Series):
        feature = pd.Series(feature)
        feature.name = feature_label

    # Add title
    title = feature.name
    if varimp is not None:
        title = title + " (VI: " + format(varimp, varimp_fmt) + ")"

    plot_bibar(ax, x=feature, y=target, title=title,
               min_width=min_width, inset_size=inset_size, ylim=ylim,
               refline=refline,
               color=color)

#def plot_CLASS_nume


# --- Regression ---------------------------------------------------------------------------------------

#def plot_REGR_cate
#def plot_REGR_nume

#def plot_target_nume
#def plot_target_cate

# --- Plot whole feature vector-------------------------------------------------------------------------------------
#def get_plotcalls_target_vs_features:
#
#    return d_calls


########################################################################################################################
# Interpretation plots
########################################################################################################################

# Plot ROC curve
def plot_roc(ax, y, yhat):
    #y = df_test[target_name]
    #yhat = yhat_test

    ax_act = ax

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
    # sns.lineplot(fpr, tpr, ax=ax_act, palette=sns.xkcd_palette(["red"]))
    ax_act.plot(fpr, tpr)
    props = {'xlabel': r"fpr: P($\^y$=1|$y$=0)",
             'ylabel': r"tpr: P($\^y$=1|$y$=1)",
             'title': "ROC (AUC = " + format(roc_auc, "0.2f") + ")"}
    _ = ax_act.set(**props)


# Plot calibration
def plot_calibration(ax, y, yhat, n_bins=5):
    #y = df_test[target_name]
    #yhat = yhat_test

    ax_act = ax

    # Calibration curve
    #true, predicted = calibration_curve(y, yhat, n_bins=n_bins)
    # sns.lineplot(predicted, true, ax=ax_act, marker="o")

    df_plot = (pd.DataFrame({"y": y, "yhat": yhat})
               .assign(bin=lambda x: pd.qcut(x["yhat"], n_bins, duplicates="drop").astype("str"))
               .groupby(["bin"], as_index=False).agg("mean")
               .sort_values("yhat"))
    #sns.lineplot("yhat", "y", data = df_calib, ax = ax_act, marker = "o")
    ax_act.plot(df_plot["yhat"], df_plot["y"], "o-")
    props = {'xlabel': r"$\bar{\^y}$ in $\^y$-bin",
             'ylabel': r"$\bar{y}$ in $\^y$-bin",
             'title': "Calibration"}
    _ = ax_act.set(**props)

    # Add diagonal line
    minmin = min(df_plot["y"].min(), df_plot["yhat"].min())
    maxmax = max(df_plot["y"].max(), df_plot["yhat"].max())
    ax_act.plot([minmin, maxmax], [minmin, maxmax], linestyle="--", color="grey")
    
    # Focus
    #yhat_max = df_plot["yhat"].max()
    #ax_act.set_xlim(None, yhat_max)
    #ax_act.set_ylim(None, yhat_max)


# PLot confusion matrix
def plot_confusion(ax, y, yhat, threshold=0.5, cmap="Blues"):

    ax_act = ax

    # binary label
    yhat_bin = np.where(yhat > threshold, 1, 0)

    # accuracy and confusion calculation
    acc = accuracy_score(y, yhat_bin)
    df_conf = pd.DataFrame(confusion_matrix(y, yhat_bin))

    # plot
    sns.heatmap(pd.DataFrame(confusion_matrix(y, yhat_bin)),
                annot=True, fmt=".5g", cmap=cmap, ax=ax_act)
    props = {'xlabel': "Predicted label",
             'ylabel': "True label",
             'title': "Confusion Matrix ($Acc_{" + format(threshold, "0.2f") + "}$ = " + format(acc, "0.2f") + ")"}
    ax_act.set(**props)


'''
def plot_pred_distribution(ax, y, yhat, xlim=None):

    ax_act = ax
        
    # plot distribution
    sns.histplot(x=yhat, hue=y, stat="density", common_norm=False, kde=True, bins=20, ax=ax_act)
    props = {'xlabel': r"Predictions ($\^y$)",
             'ylabel': "Density",
             'title': "Distribution of Predictions",
             'xlim': xlim}
    ax_act.set(**props)
    #ax_act.legend(title="Target", loc="best")
'''

# Plot precision-recall curve


def plot_precision_recall(ax, y, yhat, annotate=True, fontsize=10):

    ax_act = ax

    # precision recall calculation
    prec, rec, cutoff = precision_recall_curve(y, yhat)
    cutoff = np.append(cutoff, 1)
    prec_rec_auc = average_precision_score(y, yhat)

    # plot
    ax_act.plot(rec, prec)
    props = {'xlabel': r"recall=tpr: P($\^y$=1|$y$=1)",
             'ylabel': r"precision: P($y$=1|$\^y$=1)",
             'title': "Precision Recall Curve (AUC = " + format(prec_rec_auc, "0.2f") + ")"}
    ax_act.set(**props)

    # annotate text
    if annotate:
        for thres in np.arange(0.1, 1, 0.1):
            i_thres = np.argmax(cutoff > thres)
            ax_act.annotate(format(thres, "0.1f"), (rec[i_thres], prec[i_thres]), fontsize=fontsize)


# Plot precision curve
def plot_precision(ax, y, yhat, annotate=True, fontsize=10):

    ax_act = ax

    # precision calculation
    pct_tested = np.array([])
    prec, _, cutoff = precision_recall_curve(y, yhat)
    cutoff = np.append(cutoff, 1)
    for thres in cutoff:
        pct_tested = np.append(pct_tested, [np.sum(yhat >= thres) / len(yhat)])

    # plot
    #sns.lineplot(pct_tested, prec[:-1], ax=ax_act, palette=sns.xkcd_palette(["red"]))
    ax_act.plot(pct_tested, prec)
    props = {'xlabel': "% Samples Tested",
             'ylabel': r"precision: P($y$=1|$\^y$=1)",
             'title': "Precision Curve"}
    ax_act.set(**props)

    # annotate text
    if annotate:
        for thres in np.arange(0.1, 1, 0.1):
            i_thres = np.argmax(cutoff > thres)
            if i_thres:
                ax_act.annotate(format(thres, "0.1f"), (pct_tested[i_thres], prec[i_thres]),
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
    title = r"Observed vs. Fitted ($\rho_{Spearman}$ = " + format(up.spear(y, yhat), "0.2f") + ")"
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
