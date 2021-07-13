########################################################################################################################
# Packages
########################################################################################################################

# General
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

# Custom functions and classes
import my_utils as my


########################################################################################################################
# CLASS plots
########################################################################################################################

# Plot ROC curve
def plot_roc(ax, y, yhat):
    #y = df_test[target_name]
    #yhat = yhat_test

    ax_act = ax

    # also for regression
    if np.max(y) > 1:
        y = y / np.max(y)
    if np.max(yhat) > 1:
        yhat = yhat / np.max(yhat)

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
    true, predicted = calibration_curve(y, yhat, n_bins=n_bins)
    # sns.lineplot(predicted, true, ax=ax_act, marker="o")
    ax_act.plot(predicted, true, "o-")
    props = {'xlabel': r"$\bar{\^y}$ in $\^y$-bin",
             'ylabel': r"$\bar{y}$ in $\^y$-bin",
             'title': "Calibration"}
    _ = ax_act.set(**props)
    
    # Add diagonal line
    minmin = min(np.min(y), np.min(yhat))
    maxmax = max(np.max(y), np.max(yhat))
    ax_act.plot([minmin, maxmax], [minmin, maxmax], linestyle="--", color="grey")


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


def plot_pred_distribution(ax, y, yhat):

    ax_act = ax
        
    # plot distribution
    sns.histplot(x=yhat, hue=y, stat="density", common_norm=False, kde=True, bins=20, ax=ax_act)
    props = {'xlabel': r"Predictions ($\^y$)",
             'ylabel': "Density",
             'title': "Distribution of Predictions",
             'xlim': (0, 1)}
    ax_act.set(**props)
    #ax_act.legend(title="Target", loc="best")
    
    
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
def plot_model_performance_CLASS(y, yhat,
                                 n_bins=5, threshold=0.5, cmap="Blues", annotate=True, fontsize=10,
                                 l_plots=None,
                                 n_rows=2, n_cols=3, w=18, h=12, pdf_path=None):

    # yhat to 1-dim
    if ((yhat.ndim == 2) and (yhat.shape[1] == 2)):
        yhat = yhat[:, 1]
        
    # Define plot dict
    d_calls = dict()
    d_calls["roc"] = (plot_roc, dict(y=y, yhat=yhat))
    d_calls["confusion"] = (plot_confusion, dict(y=y, yhat=yhat, threshold=threshold, cmap=cmap))
    d_calls["distribution"] = (plot_pred_distribution, dict(y=y, yhat=yhat))
    d_calls["calibration"] = (plot_calibration, dict(y=y, yhat=yhat, n_bins=n_bins))
    d_calls["precision_recall"] = (plot_precision_recall, dict(y=y, yhat=yhat, annotate=annotate, fontsize=fontsize))
    d_calls["precision"] = (plot_precision, dict(y=y, yhat=yhat, annotate=annotate, fontsize=fontsize))

    # Filter plot dict
    if l_plots is not None:
        l_calls = [d_calls[x] for x in l_plots]
    else:
        l_calls = list(d_calls.values())
    
    # plot
    my.plot_function_calls(l_calls, n_rows=n_rows, n_cols=n_cols, figsize=(w, h), pdf_path=pdf_path)


# Wrapper for plot_model_performance_<target_type>
def plot_model_performance(y, yhat,
                           n_bins=5, threshold=0.5, cmap="Blues", annotate=True, fontsize=10,
                           l_plots=None,
                           n_rows=2, n_cols=3, w=18, h=12, pdf_path=None):
    # Derive target type
    target_type = dict(continuous="REGR", binary="CLASS",
                       multiclass="MULTICLASS")[type_of_target(y)]
    
    # Plot
    if target_type == "CLASS":
        plot_model_performance_CLASS(y, yhat, 
                                     n_bins, threshold, cmap, annotate, fontsize, 
                                     l_plots, n_rows, n_cols, w, h, pdf_path)
    elif target_type == "REGR":
        pass
        #plot_model_performance_REGR(y, yhat, n_bins, n_rows, n_cols, pdf_path)
    elif target_type == "MULTICLASS":
        pass
        #plot_model_performance_MULTICLASS(y, yhat, n_bins, n_rows, n_cols, pdf_path)
    else:
        warnings.warn("Target type cannot be determined")
    return
