# ######################################################################################################################
# Packages
# ######################################################################################################################

# Always
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import time
import pdb
import warnings

# ML
from category_encoders import target_encoder
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.model_selection import cross_validate
import xgboost as xgb
import lightgbm as lgbm
import shap
from sklearn.utils import _safe_indexing
from itertools import product  # for GridSearchCV_xlgb
from sklearn.base import BaseEstimator, TransformerMixin, clone  # , ClassifierMixin

# hmsPM specific
import hmsPM.calculation as hms_calc
import hmsPM.preprocessing as hms_preproc
import hmsPM.plotting as hms_plot
import hmsPM.metrics as hms_metrics


# ######################################################################################################################
# Parameters
# ######################################################################################################################

# Locations
dataloc = "../data/"
plotloc = "../output/"

# Util
sns.set(style="whitegrid")
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 20)

# Other
twocol = ["red", "green"]
threecol = ["green", "yellow", "red"]
'''
colors = pd.Series(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS))
colors = colors.iloc[np.setdiff1d(np.arange(len(colors)), [6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 26])]
sel = np.arange(50);  plt.bar(sel.astype("str"), 1, color=colors[sel])
'''

# Silent plotting (Overwrite to get default: plt.ion();  matplotlib.use('TkAgg'))
# plt.ion(); matplotlib.use('TkAgg')
#plt.ioff(); matplotlib.use('Agg')


# ######################################################################################################################
# Functions
# ######################################################################################################################

# --- General ----------------------------------------------------------------------------------------

def setdiff(a, b):
    return np.setdiff1d(a, b, True)


def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))


def calib(y_true, y_pred):
    return np.mean(y_true - y_pred)


def myrmse(y_true, y_pred):
    return np.sqrt(np.mean(np.power(y_true + 0.03 - y_pred, 2)))

# Scoring metrics
d_scoring = {"regr": {"spear": make_scorer(hms_metrics.spear, greater_is_better=True),
                      "rmse": make_scorer(hms_metrics.rmse, greater_is_better=False),
                      "myrmse": make_scorer(myrmse, greater_is_better=False),
                      "calib": make_scorer(calib, greater_is_better=False)},
             "class": {"auc": make_scorer(hms_metrics.auc, greater_is_better=True, needs_proba=True),
                       "acc": make_scorer(hms_metrics.acc, greater_is_better=True)},
             "multiclass": {"auc": make_scorer(hms_metrics.auc, greater_is_better=True, needs_proba=True),
                            "acc": make_scorer(hms_metrics.acc, greater_is_better=True)}}


# --- Explore -----------------------------------------------------------------------------------------------------

# Overview of values
def create_values_df(df, topn=5, dtypes=["object"]):
    df_tmp = df.select_dtypes(dtypes)
    return pd.concat([(df_tmp[catname].value_counts().iloc[: topn].reset_index()
                       .rename(columns={"index": catname, catname: catname + "_#"}))
                      for catname in df_tmp.columns.values],
                     axis=1)


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


# Variable importance
def calc_varimp_by_permutation(df, fit, fit_spm=None,
                               target="target", nume=None, cate=None, df_ref=None,
                               target_type="CLASS",
                               b_sample=None, b_all=None,
                               features=None,
                               random_seed=999,
                               n_jobs=4):

    # Define sparse matrix transformer if None, otherwise get information of it
    if fit_spm is None:
        fit_spm = hms_preproc.MatrixConverter(to_sparse=True).fit(df_ref[np.append(nume, cate)])
    else:
        nume = fit_spm.column_names_num
        cate = fit_spm.column_names_cat

    # pdb.set_trace()

    # df=df_train;  df_ref=df; target = "target"
    all_features = np.append(nume, cate)
    if features is None:
        features = all_features

    # Original performance
    if target_type in ["CLASS", "MULTICLASS"]:
        perf_orig = hms_metrics.auc(df[target], hms_calc.scale_predictions(fit.predict_proba(fit_spm.transform(df)),
                                                                           b_sample, b_all))
    else:
        perf_orig = hms_metrics.spear(df[target], fit.predict(fit_spm.transform(df)))

    # Performance per variable after permutation
    np.random.seed(random_seed)
    i_perm = np.random.permutation(np.arange(len(df)))  # permutation vector

    # TODO Arno: Solve Pep8 violation (add all variables used as parameter)
    def run_in_parallel(df, feature):
        df_perm = df.copy()
        df_perm[feature] = df_perm[feature].values[i_perm]
        if target_type in ["CLASS", "MULTICLASS"]:
            perf = hms_metrics.auc(df_perm[target],
                                   hms_calc.scale_predictions(fit.predict_proba(fit_spm.transform(df_perm)),
                                                              b_sample, b_all))
        else:
            perf = hms_metrics.spear(df_perm[target],
                                     fit.predict(fit_spm.transform(df_perm)))
        return perf

    perf = Parallel(n_jobs=n_jobs, max_nbytes='100M')(delayed(run_in_parallel)(df, feature)
                                                      for feature in features)

    # Collect performances and calculate importance
    df_varimp = pd.DataFrame({"feature": features, "perf_diff": np.maximum(0, perf_orig - perf)}) \
        .sort_values(["perf_diff"], ascending=False).reset_index(drop=False) \
        .assign(importance=lambda x: 100 * x["perf_diff"] / max(x["perf_diff"])) \
        .assign(importance_cum=lambda x: 100 * x["perf_diff"].cumsum() / sum(x["perf_diff"])) \
        .assign(importance_sumnormed=lambda x: 100 * x["perf_diff"] / sum(x["perf_diff"]))

    return df_varimp


# Partial dependence
def calc_partial_dependence(df, fit, df_ref, fit_spm=None,
                            nume=None, cate=None,
                            target_type="CLASS", target_labels=None,
                            b_sample=None, b_all=None,
                            features=None,
                            quantiles=np.arange(0, 1.1, 0.1),
                            n_jobs=4):
    # df=df_test;  df_ref=df_traintest; target = "target"; target_type=TARGET_TYPE; features=np.append(nume[0],cate[0]);
    # quantiles = np.arange(0, 1.1, 0.1);n_jobs=4

    # Define sparse matrix transformer if None, otherwise get information of it
    if fit_spm is None:
        fit_spm = hms_preproc.MatrixConverter(to_sparse=True).fit(df_ref[np.append(nume, cate)])

    else:
        nume = fit_spm.column_names_num
        cate = fit_spm.column_names_cat

    # Quantile and and values calculation
    d_quantiles = df[nume].quantile(quantiles).to_dict(orient="list")
    d_categories = fit_spm.categories_categorical_features

    # Set features to calculate importance for
    all_features = np.append(nume, cate)
    if features is None:
        features = all_features

    def run_in_parallel(feature):
        # feature = features[0]
        if feature in nume:
            values = np.array(d_quantiles[feature])
        else:
            values = d_categories[feature]

        df_tmp = df.copy()  # save original data

        df_pd_feature = pd.DataFrame()
        for value in values:
            # value=values[0]
            df_tmp[feature] = value
            if target_type == "CLASS":
                yhat_mean = np.mean(hms_calc.scale_predictions(fit.predict_proba(fit_spm.transform(df_tmp)),
                                                               b_sample, b_all), axis=0)
                df_pd_feature = pd.concat([df_pd_feature,
                                           pd.DataFrame({"feature": feature, "value": str(value),
                                                         "target": "target", "yhat_mean": yhat_mean[1]}, index=[0])])
            elif target_type == "MULTICLASS":
                yhat_mean = np.mean(hms_calc.scale_predictions(fit.predict_proba(fit_spm.transform(df_tmp)),
                                                               b_sample, b_all), axis=0)
                df_pd_feature = pd.concat([df_pd_feature,
                                           pd.DataFrame({"feature": feature, "value": str(value),
                                                         "target": target_labels, "yhat_mean": yhat_mean})])
            else:  # "REGR"
                yhat_mean = [np.mean(fit.predict(fit_spm.transform(df_tmp)))]
                df_pd_feature = pd.concat([df_pd_feature,
                                           pd.DataFrame({"feature": feature, "value": str(value),
                                                         "target": "target", "yhat_mean": yhat_mean}, index=[0])])
            # Append prediction of overwritten value

        return df_pd_feature

    # Run in parallel and append
    df_pd = pd.concat(Parallel(n_jobs=n_jobs, max_nbytes='100M')(delayed(run_in_parallel)(feature)
                                                                 for feature in features))
    df_pd = df_pd.reset_index(drop=True)
    return df_pd


# Calculate shapely values
# noinspection PyPep8Naming
def calc_shap(df_explain, fit, fit_spm=None, nume=None, cate=None, df_ref=None,
              target_type="CLASS", b_sample=None, b_all=None):
    # target_type = TARGET_TYPE;

    # Calc X_explain:
    if fit_spm is None:
        fit_spm = hms_preproc.MatrixConverter(to_sparse=True).fit(df_ref[np.append(nume, cate)])
    X_explain = fit_spm.transform(df_explain)

    # Calc mapper
    df_map = pd.DataFrame()
    if len(fit_spm.column_names_num) > 0:
        df_map = pd.concat([df_map, pd.DataFrame({"variable": fit_spm.column_names_num, "value": None})])
    if len(fit_spm.column_names_cat) > 0:
        df_map = pd.concat([df_map, (pd.DataFrame.from_dict(fit_spm.categories_categorical_features, orient='index')
                                     .T.melt().dropna().reset_index(drop=True))])
    df_map = df_map.reset_index(drop=True).reset_index().rename(columns={"index": "position"})

    # Get shap values
    # pdb.set_trace()
    explainer = shap.TreeExplainer(fit)
    shap_values = explainer.shap_values(X_explain)
    intercepts = explainer.expected_value

    # Make it iterable
    if target_type != "MULTICLASS":
        shap_values = [shap_values]
        intercepts = [intercepts]

    # Aggregate shap to variable and add intercept
    df_shap = pd.DataFrame()
    for i in range(len(shap_values)):
        df_shap = df_shap.append(
            pd.DataFrame(shap_values[i])
            .reset_index(drop=True)  # clear index
            .reset_index().rename(columns={"index": "row_id"})  # add row_id
            .melt(id_vars="row_id", var_name="position", value_name="shap_value")  # rotate
            .merge(df_map, how="left", on="position")  # add variable name to position
            .groupby(["row_id", "variable"])["shap_value"].sum().reset_index()  # aggregate cate features
            .merge(df_explain.reset_index()
                   .rename(columns={"index": "row_id"})
                   .melt(id_vars="row_id", var_name="variable", value_name="variable_value"),
                   how="left", on=["row_id", "variable"])  # add variable value
            .append(pd.DataFrame({"row_id": np.arange(len(df_explain)),
                                  "variable": "intercept",
                                  "shap_value": intercepts[i],
                                  "variable_value": None})).reset_index(drop=True)  # add intercept
            .assign(target=i)  # add target
            .assign(flag_intercept=lambda x: np.where(x["variable"] == "intercept", 1, 0),
                    abs_shap_value=lambda x: np.abs(x["shap_value"]))  # sorting columns
            .sort_values(["flag_intercept", "abs_shap_value"], ascending=False)  # sort
            .assign(shap_value_cum=lambda x: x.groupby(["row_id"])["shap_value"].transform("cumsum"))  # shap cum
            .sort_values(["row_id", "flag_intercept", "abs_shap_value"], ascending=[True, False, False])
            .assign(rank=lambda x: x.groupby(["row_id"]).cumcount()+1)).reset_index(drop=True)

    if target_type == "REGR":
        df_shap["yhat"] = df_shap["shap_value_cum"]
    elif target_type == "CLASS":
        df_shap["yhat"] = hms_calc.scale_predictions(inv_logit(df_shap["shap_value_cum"]), b_sample, b_all)
    else:  # MULTICLASS: apply "cumulated" softmax (exp(shap_value_cum) / sum(exp(shap_value_cum)) and rescale
        n_target = len(shap_values)
        df_shap_tmp = df_shap.eval("denominator = 0")
        for i in range(n_target):
            df_shap_tmp = (df_shap_tmp
                           .merge(df_shap.loc[df_shap["target"] == i, ["row_id", "variable", "shap_value"]]
                                         .rename(columns={"shap_value": "shap_value_" + str(i)}),
                                  how="left", on=["row_id", "variable"])  # add shap from "other" target
                           .sort_values("rank")  # sort by original rank
                           .assign(**{"nominator_" + str(i):
                                      lambda x: np.exp(x
                                                       .groupby(["row_id", "target"])["shap_value_" + str(i)]
                                                       .transform("cumsum"))})  # cumulate "other" targets and exp it
                           .assign(denominator=lambda x: x["denominator"] + x["nominator_" + str(i)])  # adapt denom
                           .drop(columns=["shap_value_" + str(i)])  # make shape original again for next loop
                           .reset_index(drop=True))

        # Rescale yhat
        df_shap_tmp = (df_shap_tmp.assign(**{"yhat_" + str(i):
                                             df_shap_tmp["nominator_" + str(i)] / df_shap_tmp["denominator"]
                                             for i in range(n_target)})
                       .drop(columns=["nominator_" + str(i) for i in range(n_target)]))
        yhat_cols = ["yhat_" + str(i) for i in range(n_target)]
        df_shap_tmp[yhat_cols] = hms_calc.scale_predictions(df_shap_tmp[yhat_cols], b_sample, b_all)

        # Select correct yhat
        df_shap_tmp2 = pd.DataFrame()
        for i in range(n_target):
            df_shap_tmp2 = df_shap_tmp2.append(
                (df_shap_tmp
                 .query("target == @i")
                 .assign(yhat=lambda x: x["yhat_" + str(i)])
                 .drop(columns=yhat_cols)))

        # Sort it to convenient shape
        df_shap = df_shap_tmp2.sort_values(["row_id", "target", "rank"]).reset_index(drop=True)

    return df_shap


# Check if shap values and yhat match
def check_shap(df_shap, yhat_shap, target_type="CLASS"):

    # Check
    # noinspection PyUnusedLocal
    max_rank = df_shap["rank"].max()
    if target_type == "CLASS":
        yhat_shap = yhat_shap[:, 1]
        close = np.isclose(df_shap.query("rank == @max_rank").yhat.values, yhat_shap)
    elif target_type == "MULTICLASS":
        close = np.isclose(df_shap.query("rank == @max_rank").pivot(index="row_id", columns="target",
                                                                    values="yhat"),
                           yhat_shap)
    else:
        close = np.isclose(df_shap.query("rank == @max_rank").yhat.values, yhat_shap)

    # Write warning
    if np.sum(close) != yhat_shap.size:
        warnings.warn("Warning: Shap values and yhat do not match! See following match array:")
        print(close)
    else:
        print("Info: Shap values and yhat match.")


# ######################################################################################################################
# Classes
# ######################################################################################################################

# Undersample
class Undersample(BaseEstimator, TransformerMixin):
    def __init__(self, n_max_per_level, random_state=42):
        self.n_max_per_level = n_max_per_level
        self.random_state = random_state
        self.b_sample = None
        self.b_all = None

    def fit(self, *_):
        return self

    # noinspection PyMethodMayBeStatic
    def transform(self, df):
        return df

    def fit_transform(self, df, y=None, target="target"):
        # pdb.set_trace()
        self.b_all = df[target].value_counts().values / len(df)
        df = df.groupby(target).apply(lambda x: x.sample(min(self.n_max_per_level, x.shape[0]),
                                                         random_state=self.random_state)) \
            .reset_index(drop=True) \
            .sample(frac=1).reset_index(drop=True)
        self.b_sample = df[target].value_counts().values / len(df)
        return df


# Special splitter: training fold only from training data, test fold only from test data
class TrainTestSep:
    def __init__(self, n_splits=1, sample_type="cv", fold_var="fold", random_state=42):
        self.n_splits = n_splits
        self.sample_type = sample_type
        self.fold_var = fold_var
        self.random_state = random_state

    def split(self, df, *args):
        i_df = np.arange(len(df))
        np.random.seed(self.random_state)
        np.random.shuffle(i_df)
        i_train = i_df[df[self.fold_var].values[i_df] == "train"]
        i_test = i_df[df[self.fold_var].values[i_df] == "test"]
        if self.sample_type == "cv":
            splits_train = np.array_split(i_train, self.n_splits)
            splits_test = np.array_split(i_test, self.n_splits)
        else:
            splits_train = None
            splits_test = None
        for i in range(self.n_splits):
            if self.sample_type == "cv":
                i_train_yield = np.concatenate(splits_train)
                if self.n_splits > 1:
                    i_train_yield = np.setdiff1d(i_train_yield, splits_train[i], assume_unique=True)
                i_test_yield = splits_test[i]
            elif self.sample_type == "bootstrap":
                np.random.seed(self.random_state * (i + 1))
                i_train_yield = np.random.choice(i_train, len(i_train))
                np.random.seed(self.random_state * (i + 1))
                i_test_yield = np.random.choice(i_test, len(i_test))
            else:
                i_train_yield = None
                i_test_yield = None
            yield i_train_yield, i_test_yield

    def get_n_splits(self, *args):
        return self.n_splits
    
# Special splitter: training fold only from training data, test fold only from test data


class InSample:
    def __init__(self, shuffle=True, random_state=42):
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, df, *args):
        i_df = np.arange(df.shape[0])
        print("blub")
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(i_df)
        i_train_yield = i_df
        i_test_yield = i_df
        yield i_train_yield, i_test_yield

    def get_n_splits(self, *args):
        return 1


# Incremental n_estimators GridSearch
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
                fit = (clone(self.estimator).set_params(**d_param,
                                                        n_estimators=int(max(n_estimators)))
                       .fit(_safe_indexing(X, i_train), _safe_indexing(y, i_train), **fit_params))

                # Score with all n_estimators
                for ntree_limit in n_estimators:
                    if isinstance(self.estimator, lgbm.sklearn.LGBMClassifier):
                        yhat_test = fit.predict_proba(_safe_indexing(X, i_test), num_iteration=ntree_limit)
                    elif isinstance(self.estimator, lgbm.sklearn.LGBMRegressor):
                        yhat_test = fit.predict(_safe_indexing(X, i_test), num_iteration=ntree_limit)
                    elif isinstance(self.estimator, xgb.sklearn.XGBClassifier):
                        yhat_test = fit.predict_proba(_safe_indexing(X, i_test), ntree_limit=ntree_limit)
                    else:
                        yhat_test = fit.predict(_safe_indexing(X, i_test), ntree_limit=ntree_limit)

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
        df_cv_results = df_cv_results.reset_index()
        self.cv_results_ = df_cv_results.to_dict(orient="list")

        # Refit
        if self.refit:
            self.scorer_ = self.scoring
            self.multimetric_ = True
            self.best_index_ = df_cv_results["mean_test_" + self.refit].idxmax()
            self.best_score_ = df_cv_results["mean_test_" + self.refit].loc[self.best_index_]
            self.best_params_ = (df_cv_results[param_names].loc[[self.best_index_]]
                                 .to_dict(orient="records")[0])
            self.best_estimator_ = (clone(self.estimator).set_params(**self.best_params_).fit(X, y, **fit_params))

        return self
