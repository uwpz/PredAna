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
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score
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


########################################################################################################################
# Parameter
########################################################################################################################

# Locations
dataloc = "../data/"
plotloc = "../output/"

# Number of cpus
n_jobs = 4

# Util
#sns.set(style="whitegrid")
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 20)

# Other
twocol = ["red", "green"]
threecol = ["green", "yellow", "red"]
manycol = np.delete(np.array(list(mcolors.BASE_COLORS.values()) + list(mcolors.CSS4_COLORS.values())),
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

# Overview of values
def value_counts(df, topn=5, dtypes=["object"]):
    df_tmp = df.select_dtypes(dtypes)
    return pd.concat([(df_tmp[catname].value_counts().iloc[: topn].reset_index()
                       .rename(columns={"index": catname, catname: "#"}))
                      for catname in df_tmp.columns.values],
                     axis=1).fillna("")


# Univariate model performance
def variable_performance(feature, target, scorer, splitter=KFold(5), groups=None):

    # Detect types
    target_type = dict(continuous="REGR", binary="CLASS", multiclass="MULTICLASS")[type_of_target(target)]
    numeric_feature = pd.api.types.is_numeric_dtype(feature)

    # Drop all missings
    df_hlp = pd.DataFrame().assign(feature=feature, target=target)
    if groups is not None:
        df_hlp["groups_for_split"] = groups
    df_hlp = df_hlp.dropna().reset_index(drop=True)
    
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
        if self.upper_quantile is not None:
            self.a_upper_ = np.nanquantile(X, q=self.upper_quantile, axis=0)
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
    

########################################################################################################################
# Model Comparison
#######################################################################################################################

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
            yield i_train[~np.isin(i_train, i_test)], i_test[np.isin(i_test, i_test_fold)]
  
  
# Splitter: test==train fold, i.e. in-sample selection, needed for quick change of cross-validation code to non-cv
class InSampleSplit:
    def __init__(self, shuffle=True, random_state=42):
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, *args):
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
   

# Plot list of tuples (plot_call, kwargs)
def plot_func(l_calls, n_row=2, n_col=3, figsize=(18, 12), pdf_path=None):

    # Open pdf
    if pdf_path is not None:
        pdf_pages = PdfPages(pdf_path)
    else:
        pdf_pages = None

    for i, (plot_func, kwargs) in enumerate(l_calls):
        # Init new page
        if i % (n_row * n_col) == 0:
            fig, ax = plt.subplots(n_row, n_col, figsize=figsize)
            i_ax = 0

        # Plot call
        plot_func(ax=ax.flat[i_ax], **kwargs)
        fig.tight_layout()
        i_ax += 1

        # "Close" page
        if (i_ax == n_row * n_col) or (i == len(l_calls) - 1):
            # Remove unused axes
            if (i == len(l_calls) - 1):
                for k in range(i_ax, n_row * n_col):
                    ax.flat[k].axis("off")

            # Write pdf
            if pdf_path is not None:
                pdf_pages.savefig(fig)

    # Close pdf
    if pdf_path is not None:
        pdf_pages.close()


# Plot partial dependence
def plot_pd(ax, feature_name, feature, yhat, feature_ref=None, yhat_err=None, refline=None, ylim=None,
            color="red", min_width=0.2):

    ax_act = ax
    numeric_feature = pd.api.types.is_numeric_dtype(feature)

    if numeric_feature:
        # Lineplot
        ax_act.plot(feature, yhat, marker=".", color=color)

        # Background density plot
        if feature_ref is not None:
            ax2 = ax_act.twinx()
            ax2.axis("off")
            sns.distplot(feature_ref, color="grey", hist=False,
                         kde=True, kde_kws={'shade': True, 'linewidth': 0},
                         ax=ax2)
        # Rugs
        sns.rugplot(feature, color="grey", ax=ax_act)

        # Refline
        if refline is not None:
            ax_act.axhline(refline, ls="dotted", color="black")  # priori line

        # Axis style
        ax_act.set_title(feature_name)
        ax_act.set_xlabel("")
        ax_act.set_ylabel(r"$\^y$")
        if ylim is not None:
            ax_act.set_ylim(ylim)

        # Crossvalidation
        if yhat_err is not None:
            #ax_act.plot(feature, yhat - yhat_se, linestyle="--", color=color)
            #ax_act.plot(feature,  yhat + yhat_se, linestyle="--", color=color)
            ax_act.fill_between(feature, yhat - yhat_err, yhat + yhat_err, color=color, alpha=0.2)

    else:
        # Use DataFrame for calculation
        df_plot = pd.DataFrame({feature_name: feature, "yhat": yhat})
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

            #ax2 = ax_act.twiny()
            #ax2.barh(df_plot[feature_name], df_plot["pct"], color="grey", edgecolor="grey", alpha=0.5, linewidth=0)

        # Refline
        if refline is not None:
            ax_act.axvline(refline, ls="dotted", color="black")  # priori line

        # Bar plot
        ax_act.barh(df_plot[feature_name], df_plot["yhat"],
                    height=df_plot["width"] if feature_ref is not None else 0.8,
                    color=color, edgecolor="black", alpha=0.5, linewidth=1)

        # Axis style
        ax_act.set_title(feature_name)
        ax_act.set_xlabel(r"$\^y$")
        if ylim is not None:
            ax_act.set_xlim(ylim)

        # Crossvalidation
        if yhat_err is not None:
            ax_act.errorbar(df_plot["yhat"], df_plot[feature_name], xerr=yhat_err,
                            fmt=".", marker="s", capsize=5, fillstyle="none", color="grey")



    
    
