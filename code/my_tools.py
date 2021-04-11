########################################################################################################################
# Packages
########################################################################################################################

from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import warnings
from sklearn import model_selection

from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV, check_cv, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils import _safe_indexing
from sklearn.base import BaseEstimator, TransformerMixin, clone  # ClassifierMixin

import xgboost as xgb
import lightgbm as lgbm
from itertools import product  # for GridSearchCV_xlgb

'''
import os
import matplotlib
import time
from category_encoders import target_encoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate, check_cv
import shap
# hmsPM specific
import hmsPM.calculation as hms_calc
import hmsPM.preprocessing as hms_preproc
import hmsPM.plotting as hms_plot
import hmsPM.metrics as hms_metrics


########################################################################################################################
# Parameter
########################################################################################################################

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

colors = pd.Series(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS))
colors = colors.iloc[np.setdiff1d(np.arange(len(colors)), [6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 26])]
sel = np.arange(50);  plt.bar(sel.astype("str"), 1, color=colors[sel])


# Silent plotting (Overwrite to get default: plt.ion();  matplotlib.use('TkAgg'))
# plt.ion(); matplotlib.use('TkAgg')
#plt.ioff(); matplotlib.use('Agg')
'''


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


def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))


# Show closed figure again
def show_figure(fig):
    # create a dummy figure and use its manager to display "fig"
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)


# --- Metrics ----------------------------------------------------------------------------------------

# Regr

def spear(y_true, y_pred):
    return pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).corr(method="spearman").values[0, 1]


def pear(y_true, y_pred):
    return pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).corr(method="pearson").values[0, 1]


def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))


def me(y_true, y_pred):
    return np.mean(y_true - y_pred)


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

#def myrmse(y_true, y_pred):
#    return np.sqrt(np.mean(np.power(y_true + 0.03 - y_pred, 2)))


# Class + Multiclass

def auc(y_true, y_pred):
    if y_pred.ndim == 2:
        if y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
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
                      "me": make_scorer(me, greater_is_better=False),
                      "mae": make_scorer(mae, greater_is_better=False)},
             "CLASS": {"auc": make_scorer(auc, greater_is_better=True, needs_proba=True),
                       "acc": make_scorer(acc, greater_is_better=True)},
             "MULTICLASS": {"auc": make_scorer(auc, greater_is_better=True, needs_proba=True),
                            "acc": make_scorer(acc, greater_is_better=True)}}


########################################################################################################################
# Explore
#######################################################################################################################

# Overview of values
def value_counts(df, topn=5, dtypes=["object"]):
    df_tmp = df.select_dtypes(dtypes)
    return pd.concat([(df_tmp[catname].value_counts().iloc[: topn].reset_index()
                       .rename(columns={"index": catname, catname: "#"}))
                      for catname in df_tmp.columns.values],
                     axis=1).fillna("")


# Univariate model performance
# TODO: Remove
def variable_performance(features, target, splitter):

    target_type = dict(continuous="REGR", binary="CLASS", multiclass="MULTICLASS")[type_of_target(target)]
    print(target_type)
    metric = "spear" if target_type == "REGR" else "auc"
    print(metric)

    varimp = dict()
    for col in features.columns.values:
        print(col)
        df_hlp = features[[col]].assign(target=target).dropna().reset_index(drop=True)
        varimp[col] = np.mean(cross_val_score(
            estimator=(LinearRegression() if target_type == "REGR" else LogisticRegression()),
            X=(KBinsDiscretizer().fit_transform(df_hlp[[col]])
               if pd.api.types.is_numeric_dtype(df_hlp[col]) else OneHotEncoder().fit_transform(df_hlp[[col]])),
            y=df_hlp["target"], 
            cv=splitter, 
            scoring=d_scoring[target_type][metric]))
    return(pd.Series(varimp))

# TODO: Add scorer func.
def variable_performance_new(feature, target, splitter):

    target_type = dict(continuous="REGR", binary="CLASS", multiclass="MULTICLASS")[type_of_target(target)]
    print(target_type)
    metric = "spear" if target_type == "REGR" else "auc"
    print(metric)

    df_hlp = pd.DataFrame(feature).assign(target=target).dropna().reset_index(drop=True)
    numeric_feature = pd.api.types.is_numeric_dtype(df_hlp.iloc[:, [0]])
    perf = np.mean(cross_val_score(
        estimator=(LinearRegression() if target_type == "REGR" else LogisticRegression()),
        X=(KBinsDiscretizer().fit_transform(df_hlp.iloc[:, [0]]) if numeric_feature else 
           OneHotEncoder().fit_transform(df_hlp.iloc[:, [0]])),
        y=df_hlp["target"],
        cv=splitter,
        scoring=d_scoring[target_type][metric]))
    return perf


# Winsorize
class Winsorize(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=None, upper_quantile=None):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self._a_lower = None
        self._a_upper = None

    def fit(self, X, *_):
        X = pd.DataFrame(X)
        if self.lower_quantile is not None:
            self._a_lower = np.nanquantile(X, q=self.lower_quantile, axis=0)
        if self.upper_quantile is not None:
            self._a_upper = np.nanquantile(X, q=self.upper_quantile, axis=0)
        return self

    def transform(self, X, *_):
        if (self.lower_quantile is not None) or (self.upper_quantile is not None):
            X = np.clip(X, a_min=self._a_lower, a_max=self._a_upper)
        return X
    

# Map Non-topn frequent members of a string column to "other" label
class Collapse(BaseEstimator, TransformerMixin):
    def __init__(self, n_top=10, other_label="_OTHER_"):
        self.n_top = n_top
        self.other_label = other_label        
        #self._s_levinfo = None
        #self._toomany = None
        #self._d_top = None
        #self._statistics = None

    def fit(self, X, *_):
        #self._s_levinfo = pd.DataFrame(X).apply(lambda x: x.unique().size).sort_values(ascending = False)
        #self._toomany = self._s_levinfo[self._s_levinfo > self.n_top].index.values
        #self._d_top = {x: pd.DataFrame(X)[x].value_counts().index.values[:self.n_top] for x in self._toomany}
        #self._statistics = {"_s_levinfo": self._s_levinfo, "_toomany": self._toomany, "_d_top": self._d_top}
        self._d_top = pd.DataFrame(X).apply(lambda x: x.value_counts().index.values[:self.n_top])
        return self

    def transform(self, X):
        X = pd.DataFrame(X).apply(lambda x: x.where(np.in1d(x, self._d_top[x.name]),
                                                    other=self.other_label)).values
        return X 
    

# Impute Mode (simpleimputer is too slow)
class ImputeMode(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._impute_values = None

    def fit(self, X):
        self._impute_values = pd.DataFrame(X).mode().iloc[0]
        return self

    def transform(self, X):
        X = pd.DataFrame(X).fillna(self._impute_values).values
        return X
    

########################################################################################################################
# Model Comparison
#######################################################################################################################

# Undersample
def undersample(df, target, n_max_per_level, random_state=42):
    b_all = df[target].value_counts().values / len(df)
    df_under = (df.groupby(target).apply(lambda x: x.sample(min(n_max_per_level, x.shape[0]),
                                                            random_state=random_state))
                .sample(frac=1)
                .reset_index(drop=True))  # shuffle
    b_sample = df_under[target].value_counts().values / len(df_under)
    return df_under, b_sample, b_all


# Special splitter: training fold only from training data, test fold only from test data
class TrainTestSep:
    def __init__(self, n_splits=1, sample_type="cv", random_state=42):
        self.n_splits = n_splits
        self.sample_type = sample_type
        self.random_state = random_state

    def split(self, X, test_fold, *args):
        
        i_X = np.arange(len(X))
        i_train = i_X[test_fold == 0]
        i_test = i_X[test_fold == 1]
        np.random.seed(self.random_state)
        np.random.shuffle(i_train)
        np.random.seed(self.random_state)
        np.random.shuffle(i_test)
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
                i_train_yield = np.random.choice(i_train, len(i_train), replace=True)
                np.random.seed(self.random_state * (i + 1))
                i_test_yield = np.random.choice(i_test, len(i_test), replace=True)
            else:
                i_train_yield = None
                i_test_yield = None
            yield i_train_yield, i_test_yield

    def get_n_splits(self, *args):
        return self.n_splits
    

class KFoldSep(KFold):
    def __init__(self, features, *args, **kwargs):
        super().__init__(shuffle = True, *args, **kwargs)

    def split(self, X, y=None, groups=None, test_fold=None):
        i_test_fold = np.arange(len(X))[test_fold]
        for i_train, i_test in super().split(X, y, groups):
            yield i_train[~np.isin(i_train, i_test)], i_test[np.isin(i_test, i_test_fold)]
  
  
# Splitter: test==train fold, i.e. in-sample selection
class InSampleSplit:
    def __init__(self, shuffle=True, random_state=42):
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, df, *_):
        i_df = np.arange(df.shape[0])
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(i_df)
        i_train_yield = i_df
        i_test_yield = i_df
        yield i_train_yield, i_test_yield

    def get_n_splits(self, *_):
        return 1
    

# Column selector: Scikit's ColumnTransformer needs same columns for fit and transform (bug!)
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, *_):
        return self

    def transform(self, df, *_):
        return df[self.columns]


# Incremental n_estimators (warm start) GridSearch for XGBoost and Lightgbm
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
        yhat_rescaled = (tmp.T / tmp.sum(axis=1)).T  # transposing is needed for casting
        #yhat_rescaled = tmp / tmp.sum(axis=1).reshape(len(tmp),1)
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


# Convert result of scikit's variable importance to a dataframe
def varimp2df(varimp, features):
    df_varimp = (pd.DataFrame(dict(score_diff=varimp["importances_mean"], feature=features))
                 .sort_values(["score_diff"], ascending=False).reset_index(drop=True)
                 .assign(importance=lambda x: 100 * np.where(x["score_diff"] > 0,
                                                             x["score_diff"] / max(x["score_diff"]), 0),
                         importance_cum=lambda x: 100 * x["importance"].cumsum() / sum(x["importance"])))
    return df_varimp


# Dataframe based permutation importance which can select a subset of features for which to calculate VI
def variable_importance(estimator, df, y, features, scoring=None, n_jobs=None, random_state=None, **_):

    # Original performance
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


# Plot permutation base variable importance
def plot_variable_importance(features, importance,
                             importance_cum=None, importance_se=None, max_score_diff=None,
                             category=None,
                             category_label="Importance",
                             category_color_palette=sns.xkcd_palette(["blue", "orange", "red"]),
                             w=18, h=12, pdf=None):

    fig, ax = plt.subplots(1, 1)
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
    fig.tight_layout()
    fig.set_size_inches(w=w, h=h)
    if pdf:
        fig.savefig(pdf)


# Dataframe based patial dependence which can use a reference dataset for value-grid defintion
def partial_dependence(estimator, df, features,
                       df_ref=None, quantiles=np.arange(0.05, 1, 0.1),
                       n_jobs=4):
    #estimator=model; df=df_test[features]; features=features_top_test; df_ref=None; quantiles=np.arange(0.05, 1, 0.1)

    if df_ref is None:
        df_ref = df

    def run_in_parallel(feature):
        if pd.api.types.is_numeric_dtype(df[feature]):
            values = df_ref[feature].quantile(quantiles).values
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
        df_return.columns = "yhat" if estimator._estimator_type == "regressor" else estimator.classes_
        df_return["value"] = values

        return df_return
    
    # Run in parallel and append
    l_pd = (Parallel(n_jobs=n_jobs, max_nbytes='100M')(delayed(run_in_parallel)(feature)
                                                       for feature in features))
    d_pd = dict(zip(features, l_pd))
    return d_pd



    





    
    
