########################################################################################################################
# Initialize: Packages, functions, parameter
########################################################################################################################

# --- Packages ------------------------------------------------------------------------------------

# General
from hmsPM.utils import select_features_by_scale
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt  # ,matplotlib
import seaborn as sns
import pickle
import hmsPM.plotting as hms_plot
import os  # sys.path.append(os.getcwd())
import importlib  # importlib.reload(my)
import time

# Special
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.inspection import permutation_importance
import xgboost as xgb
import shap

# Custom functions and classes
#from . import blub as bl
import my_tools as my


# --- Parameter --------------------------------------------------------------------------

# Main parameter
TARGET_TYPE = "CLASS"
target_name = "cnt_" + TARGET_TYPE

# Specific parameters
n_jobs = 4

# Locations
dataloc = "../data/"
plotloc = "../output/"

# Plot
plot = True
# Show directly or not
#%matplotlib Agg
plt.ioff(); matplotlib.use('Agg')  # stop standard

# Metric to use
metric = "spear" if TARGET_TYPE == "REGR" else "auc"

# Load results from exploration
df = nume_standard = cate_standard = cate_binned = nume_encoded = None
with open(dataloc + "1_explore.pkl", "rb") as file:
    d_pick = pickle.load(file)
for key, val in d_pick.items():
    exec(key + "= val")

# Tuning parameter to use (for xgb) and classifier definition
xgb_param = dict(n_estimators=1100, learning_rate=0.01,
                 max_depth=3, min_child_weight=10,
                 colsample_bytree=0.7, subsample=0.7,
                 gamma=0,
                 verbosity=0,
                 n_jobs=n_jobs)
clf = xgb.XGBRegressor(**xgb_param) if TARGET_TYPE == "REGR" else xgb.XGBClassifier(**xgb_param)


########################################################################################################################
# Prepare
########################################################################################################################

# --- Derive train/test ------------------------------------------------------------------------------------------------

# Undersample only training data (take all but n_maxpersample at most)
if TARGET_TYPE in ["CLASS", "MULTICLASS"]:
    df.query("fold == 'train'")[target_name].value_counts()
    df_train, b_sample, b_all = my.undersample(df.query("fold == 'train'"), target=target_name,
                                               n_max_per_level=3000)
    print(b_sample, b_all)
else:
    df_train = df.query("fold == 'train'").sample(n=3000, frac=None).reset_index(drop=True)

# Test data
df_test = df.query("fold == 'test'").reset_index(drop=True)  # .sample(300) #ATTENTION: Do not sample in final run!!!

# Combine again
df_traintest = pd.concat([df_train, df_test]).reset_index(drop=True)

# Folds for crossvalidation and check
cv_5foldsep = my.KFoldSep(5)
split_5foldsep = cv_5foldsep.split(df_traintest, test_fold=(df_traintest["fold"] == "test"))
i_train, i_test = next(split_5foldsep)
print("TRAIN-fold:", df_traintest["fold"].iloc[i_train].value_counts(), i_train[:5])
print("TEST-fold:", df_traintest["fold"].iloc[i_test].value_counts(), i_test[:5])

    
# ######################################################################################################################
# Performance
# ######################################################################################################################

# --- Do the full fit and predict on test data -------------------------------------------------------------------

# Fit
'''
pipe_old = Pipeline(
    [('matrix', (ColumnTransformer([('nume', MinMaxScaler(), nume_standard),
                                    ('cate', OneHotEncoder(sparse=True, handle_unknown="ignore"), cate_standard)]))),
     ('predictor', clf)])
'''
pipe = Pipeline(
    [('matrix', (ColumnTransformer([('nume', MinMaxScaler(), nume_standard),
                                    ('cate', OneHotEncoder(sparse=True, handle_unknown="ignore"), cate_standard)]))),
     ('predictor', my.ScalingEstimator(clf, b_sample=b_sample, b_all=b_all))])
features = np.append(nume_standard, cate_standard)
model = pipe.fit(df_train[features], df_train[target_name])

# Predict
if TARGET_TYPE in ["CLASS", "MULTICLASS"]:
    yhat_test = model.predict_proba(df_test[features])
    print(my.auc(df_test[target_name].values, yhat_test))
else:
    yhat_test = model.predict(df_test[features])
    print(my.spear(df_test[target_name].values, yhat_test))
print(pd.DataFrame(yhat_test).describe())

# Plot performance
if plot:
    perf_plot = (hms_plot.MultiPerformancePlotter(n_bins=5, w=18, h=12)
                 .plot(y=df_test[target_name], y_hat=yhat_test,
                       file_path=plotloc + "performance__" + TARGET_TYPE + ".pdf"))

# Check performance for crossvalidated fits
d_cv = cross_validate(model, df_traintest[features], df_traintest[target_name],
                      cv=cv_my5fold.split(df_traintest,
                                          test_fold=(df_traintest["fold"] == "test").values),  # special 5fold
                      scoring=my.d_scoring[TARGET_TYPE],
                      return_estimator=True,
                      n_jobs=n_jobs)
print(d_cv["test_" + metric], " \n", np.mean(d_cv["test_" + metric]), np.std(d_cv["test_" + metric]))


# --- Most important variables (importance_cum < 95) model fit ------------------------------------------------------

# Variable importance (on train data!)
df_varimp_train = my.variable_importance(model, df_train[features], df_train[target_name], features,
                                         scoring=my.d_scoring[TARGET_TYPE][metric],
                                         random_state=42, n_jobs=n_jobs)

# Top features (importances sum up to 95% of whole sum)
features_top_train = df_varimp_train.loc[df_varimp_train["importance_cum"] < 95, "feature"].values

# Fit again only on features_top
pipe_top = Pipeline([
    ('matrix', (ColumnTransformer([('nume', MinMaxScaler(), nume_standard[np.in1d(nume_standard, features_top_train)]),
                                   ('cate', OneHotEncoder(sparse=True, handle_unknown="ignore"),
                                    cate_standard[np.in1d(cate_standard, features_top_train)])]))),
    ('predictor', my.ScalingEstimator(clone(clf), b_sample=b_sample, b_all=b_all))])
model_top = pipe_top.fit(df_train[features_top_train], df_train[target_name])

# Plot performance of features_top model
if TARGET_TYPE in ["CLASS", "MULTICLASS"]:
    yhat_top = model_top.predict_proba(df_test[features_top_train])
    print(my.auc(df_test[target_name].values, yhat_top))
else:
    yhat_top = model_top.predict(df_test[features_top_train])
    print(my.spear(df_test[target_name].values, yhat_top))
if plot:
    perf_plot_top = (hms_plot.MultiPerformancePlotter(n_bins=5, w=18, h=12)
                     .plot(y=df_test[target_name], y_hat=yhat_top,
                           file_path=plotloc + "performance_top__" + TARGET_TYPE + ".pdf"))


########################################################################################################################
# Diagnosis
########################################################################################################################

# ---- Check residuals fot top features --------------------------------------------------------------------------------------

# Residuals
if TARGET_TYPE in ["CLASS", "MULTICLASS"]:
    # "1 - yhat_of_true_class"
    df_test["residual"] = 1 - yhat_test[np.arange(len(df_test[target_name])), df_test[target_name + "_num"]]
else:
    df_test["residual"] = df_test[target_name + "_num"] - yhat_test
df_test["abs_residual"] = df_test["residual"].abs()
df_test["residual"].describe()

# For non-regr tasks one might want to plot it for each target level (df_test.query("target == 0/1"))
if plot:
    (hms_plot.MultiFeatureDistributionPlotter(target_limits=None if TARGET_TYPE == "REGR" else (0, 0.5),
                                              n_rows=2, n_cols=3, w=18, h=12)
        .plot(features=df_test[features_top_train],
              target=df_test["residual"],
              file_path=plotloc + "diagnosis_residual__" + TARGET_TYPE + ".pdf"))

# Absolute residuals
if TARGET_TYPE == "REGR":
    if plot:
        (hms_plot.MultiFeatureDistributionPlotter(target_limits=None, n_rows=2, n_cols=3, w=18, h=12)
         .plot(features=df_test[features_top_train],
               target=df_test["abs_residual"],
               file_path=plotloc + "diagnosis_absolute_residual__" + TARGET_TYPE + ".pdf"))


# ---- Explain bad predictions ------------------------------------------------------------------------------------

def agg_onehot_shapley(X, d_map):
    df_cate = pd.DataFrame()
    start = 0
    for key, val in d_map.items():
        df_cate[key] = X[:, start:(start + len(val))].sum(axis=1)
        start = start + len(val)
    return df_cate


# Get shap for n_worst predicted records
n_worst = 10
df_explain = df_test.sort_values("abs_residual", ascending=False).iloc[:n_worst, :]
df_explain_display = df_explain[features].round(2)
explainer = shap.TreeExplainer(model[1].estimator)
shap_values = explainer(model[0].transform(X=df_explain[features]))

# Aggregate onehot shap_values
shap_values.feature_names = np.tile(features, (len(shap_values), 1))
shap_values.data = df_explain_display.values
df_shap_nume = pd.DataFrame(shap_values.values[:, 0:len(nume_standard)]).set_axis(nume_standard, axis=1)
df_shap_cate = agg_onehot_shapley(X=shap_values.values[:, len(nume_standard):],
                                  d_map=dict(zip(cate_standard, model[0].transformers_[1][1].categories_)))
df_shap = pd.concat([df_shap_nume, df_shap_cate], axis=1)
shap_values.values = df_shap.values

# Plot
%matplotlib inline
shap.plots.waterfall(shap_values[0], show=True)

# Check
np.isclose(my.scale_predictions(my.inv_logit(shap_values.values.sum(axis=1) + explainer.expected_value),
                                b_sample, b_all),
           model.predict_proba(df_explain[features])[:, 1])


########################################################################################################################
# Variable Importance
########################################################################################################################

# --- Default Variable Importance: uses gain sum of all trees ----------------------------------------------------------
xgb.plot_importance(model[1].estimator)


# --- Variable Importance by permuation argument ----------------------------------------------------------------------

# Importance (on test data!)
df_varimp_test = my.variable_importance(model, df_test[features], df_test[target_name], features,
                                        scoring=my.d_scoring[TARGET_TYPE][metric],
                                        random_state=42, n_jobs=n_jobs)
features_top_test = df_varimp_test.loc[df_varimp_test["importance_cum"] < 95, "feature"].values

# Compare variable importance for train and test (hints to variables prone to overfitting)
sns.barplot(x="score_diff", y="feature", hue="fold",
            data=pd.concat([df_varimp_train.assign(fold="train"),
                            df_varimp_test.assign(fold="test")], sort=False))

# Crossvalidate Importance (only for top features)
df_varimp_test_cv = pd.DataFrame()
for i, (i_train, i_test) in enumerate(cv_my5fold.split(df_traintest,
                                                       test_fold=(df_traintest["fold"] == "test").values)):
    df_tmp = df_traintest.iloc[i_train, :]
    df_varimp_test_cv = df_varimp_test_cv.append(
        my.variable_importance(d_cv["estimator"][i], df_tmp[features], df_tmp[target_name], features_top_test,
                               scoring=my.d_scoring[TARGET_TYPE][metric],
                               random_state=42, n_jobs=n_jobs).assign(run=i))
df_varimp_test_se = (df_varimp_test_cv.groupby("feature")["score_diff", "importance"].agg("sem")
                     .pipe(lambda x: x.set_axis([col + "_se" for col in x.columns], axis=1, inplace=False))
                     .reset_index())

# Add other information (e.g. special category)
df_varimp_test["category"] = pd.cut(df_varimp_test["importance"], [-np.inf, 10, 50, np.inf],
                                    labels=["low", "medium", "high"])

# Plot Importance
df_varimp_plot = (df_varimp_test.query("feature in @features_top_test")
                  .merge(df_varimp_test_se, how="left", on="feature"))
if plot:
    my.plot_variable_importance(df_varimp_plot["feature"], df_varimp_plot["importance"],
                                importance_cum=df_varimp_plot["importance_cum"],
                                importance_se=df_varimp_plot["importance_se"],
                                max_score_diff=df_varimp_plot["score_diff"][0].round(2),
                                category=df_varimp_plot["category"],
                                w=8, h=4, pdf=plotloc + "variable_importance__" + TARGET_TYPE + ".pdf")


########################################################################################################################
# Partial Dependance
########################################################################################################################

'''
from sklearn.inspection import partial_dependence
# cate
partial_dependence(model, df_test[features], features=features_top_test[0],
                   grid_resolution=np.inf,
                   kind="average")
# nume
from joblib import Parallel, delayed
Parallel(n_jobs=n_jobs, max_nbytes='100M')(
    delayed(partial_dependence)(model, df_test[features], feature,
                                grid_resolution=5,
                                kind="average")
    for feature in features_top_test[1:])
'''

# Calc PD
d_pd = my.partial_dependence(model, df_test[features], features_top_test, df_ref=df_train)

# Crossvalidate
d_pd_cv = {feature: pd.DataFrame() for feature in features_top_test}
for i, (i_train, i_test) in enumerate(cv_my5fold.split(df_traintest,
                                                       test_fold=(df_traintest["fold"] == "test").values)):
    d_pd_run = my.partial_dependence(model, df_traintest.iloc[i_test, :][features], features_top_test,
                                     df_ref=df_traintest.iloc[i_train, :])
    for feature in features_top_test:
        d_pd_cv[feature] = d_pd_cv[feature].append(d_pd_run[feature].assign(run=i)).reset_index(drop=True)

# Plot it
# TODO


########################################################################################################################
# Explanations
########################################################################################################################

# ---- Explain bad predictions ------------------------------------------------------------------------------------

# Filter data
n_select = 10
i_worst = df_test.sort_values("abs_residual", ascending=False).iloc[:n_select, :].index.values
i_best = df_test.sort_values("abs_residual", ascending=True).iloc[:n_select, :].index.values
i_random = df_test.sample(n=n_select).index.values
i_explain = np.unique(np.concatenate([i_worst, i_best, i_random]))
yhat_explain = yhat_test[i_explain]
df_explain = df_test.iloc[i_explain, :].reset_index(drop=True)

# Get shap
df_explain_display = df_explain[features].round(2)
explainer = shap.TreeExplainer(model[1].estimator)
shap_values = explainer.shap_values(model[0].transform(X=df_explain[features]))
intercepts = explainer.expected_value

'''
df_shap = calc_shap(df_explain, fit, fit_spm = fit_spm,
                    target_type = TARGET_TYPE, b_sample = b_sample, b_all = b_all)

# Check
check_shap(df_shap, yhat_explain, target_type = TARGET_TYPE)

# Plot: TODO
'''

plt.close("all")


# ######################################################################################################################
# Individual dependencies
# ######################################################################################################################

# TODO
