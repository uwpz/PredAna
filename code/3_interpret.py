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
from importlib import reload 
import time

# Special
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.inspection import permutation_importance, partial_dependence
import xgboost as xgb
import shap

# Custom functions and classes
#from . import blub as bl
import my_utils as my


# --- Parameter --------------------------------------------------------------------------

# Main parameter
TARGET_TYPE = "CLASS"
target_name = "cnt_" + TARGET_TYPE + "_num"
id_name = "instant"

# Plot
plot = True
#%matplotlib qt / %matplotlib inline  # activate standard/inline window
#plt.ioff() / plt.ion()  # stop/start standard window
#plt.plot(1, 1)

# Metric to use
metric = "spear" if TARGET_TYPE == "REGR" else "auc"

# Load results from exploration
df = nume_standard = cate_standard = cate_binned = nume_encoded = None
with open(my.dataloc + "1_explore.pkl", "rb") as file:
    d_pick = pickle.load(file)
for key, val in d_pick.items():
    exec(key + "= val")
nume = nume_standard
cate = cate_standard

# Tuning parameter to use (for xgb) and classifier definition
xgb_param = dict(n_estimators=1100, learning_rate=0.01,
                 max_depth=3, min_child_weight=10,
                 colsample_bytree=0.7, subsample=0.7,
                 gamma=0,
                 verbosity=0,
                 n_jobs=my.n_jobs)


########################################################################################################################
# Prepare
########################################################################################################################

# --- Derive train/test ------------------------------------------------------------------------------------------------

# Undersample only training data (take all but n_maxpersample at most)
if TARGET_TYPE in ["CLASS", "MULTICLASS"]:
    df.query("fold == 'train'")[target_name].value_counts()
    df_train, b_sample, b_all = my.undersample(df.query("fold == 'train'"), target=target_name,
                                               n_max_per_level=300)
    print(b_sample, b_all)
    if np.any(np.isclose(b_sample, b_all)):
        algo = xgb.XGBClassifier(**xgb_param)
    else:
        algo = my.ScalingEstimator(xgb.XGBClassifier(**xgb_param), b_sample=b_sample, b_all=b_all)
        #alternative: algo = XGBClassifier_rescale(**xgb_param, b_sample = b_sample, b_all = b_all)
else:
    df_train = df.query("fold == 'train'").sample(n=3000, frac=None).reset_index(drop=True)
    algo = xgb.XGBRegressor(**xgb_param)
    
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
pipe = Pipeline(
    [('matrix', (ColumnTransformer([('nume', MinMaxScaler(), nume),
                                    ('cate', OneHotEncoder(sparse=True, handle_unknown="ignore"), cate)]))),
     ('predictor', algo)])
features = np.append(nume, cate)
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
                       file_path=my.plotloc + "performance__" + TARGET_TYPE + ".pdf"))

# Check performance for crossvalidated fits
d_cv = cross_validate(model, df_traintest[features], df_traintest[target_name],
                      cv=cv_5foldsep.split(df_traintest, test_fold=(df_traintest["fold"] == "test")),  # special 5fold
                      scoring=my.d_scoring[TARGET_TYPE],
                      return_estimator=True,
                      n_jobs=my.n_jobs)
print(d_cv["test_" + metric], " \n", np.mean(d_cv["test_" + metric]), np.std(d_cv["test_" + metric]))


# --- Most important variables (importance_cum < 95) model fit ------------------------------------------------------

# Variable importance (on train data!)
df_varimp_train = my.variable_importance(model, df_train[features], df_train[target_name], features,
                                         scoring=my.d_scoring[TARGET_TYPE][metric],
                                         random_state=42, n_jobs=my.n_jobs)
# Scikit's VI: permuatation_importance("same parameter but remove features argument and add n_repeats=1")

# Top features (importances sum up to 95% of whole sum)
features_top_train = df_varimp_train.loc[df_varimp_train["importance_cum"] < 95, "feature"].values

# Fit again only on features_top
pipe_top = Pipeline([
    ('matrix', (ColumnTransformer([('nume', MinMaxScaler(), nume[np.in1d(nume, features_top_train)]),
                                   ('cate', OneHotEncoder(sparse=True, handle_unknown="ignore"),
                                    cate[np.in1d(cate, features_top_train)])]))),
    ('predictor', clone(algo))])
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
                           file_path=my.plotloc + "performance_top__" + TARGET_TYPE + ".pdf"))


########################################################################################################################
# Diagnosis
########################################################################################################################

# ---- Check residuals fot top features --------------------------------------------------------------------------------------

# Residuals
if TARGET_TYPE in ["CLASS", "MULTICLASS"]:
    # "1 - yhat_of_true_class"
    df_test["residual"] = 1 - yhat_test[np.arange(len(df_test[target_name])), df_test[target_name]]
else:
    df_test["residual"] = df_test[target_name] - yhat_test
df_test["abs_residual"] = df_test["residual"].abs()
df_test["residual"].describe()

# For non-regr tasks one might want to plot it for each target level (df_test.query("target == 0/1"))
if plot:
    (hms_plot.MultiFeatureDistributionPlotter(target_limits=None if TARGET_TYPE == "REGR" else (0, 0.5),
                                              n_rows=2, n_cols=3, w=18, h=12)
        .plot(features=df_test[features_top_train],
              target=df_test["residual"],
              file_path=my.plotloc + "diagnosis_residual__" + TARGET_TYPE + ".pdf"))

# Absolute residuals
if TARGET_TYPE == "REGR":
    if plot:
        (hms_plot.MultiFeatureDistributionPlotter(target_limits=None, n_rows=2, n_cols=3, w=18, h=12)
         .plot(features=df_test[features_top_train],
               target=df_test["abs_residual"],
               file_path=my.plotloc + "diagnosis_absolute_residual__" + TARGET_TYPE + ".pdf"))


# ---- Explain bad predictions ------------------------------------------------------------------------------------

# Get shap for n_worst predicted records
n_worst = 10
df_explain = df_test.sort_values("abs_residual", ascending=False).iloc[:n_worst, :]
explainer = shap.TreeExplainer(model[1].estimator if type(model[1]) is my.ScalingEstimator else model[1])
shap_values = my.agg_shap_values(explainer(model[0].transform(X=df_explain[features])),
                                 df_explain[features],
                                 len_nume=len(nume), l_map_onehot=model[0].transformers_[1][1].categories_, 
                                 round=2)

# Plot
fig, ax = plt.subplots(1, 1)
i = 0
ax.set_title("id = " + str(df_explain[id_name].iloc[i]) + " (y = " + str(df_explain[target_name].iloc[i]) + ")")
if TARGET_TYPE != "MULTICLASS":
    shap.plots.waterfall(shap_values[i], show=True)  # TDODO: replace "00"
else:
    shap.plots.waterfall(shap_values[i][:, df_explain[target_name + "_num"].iloc[i]], show=True)  

# Check
shaphat = shap_values.values.sum(axis=1) + shap_values.base_values
if TARGET_TYPE == "REGR":
    print(np.isclose(shaphat, model.predict(df_explain[features])))
elif TARGET_TYPE == "CLASS":
    print(np.isclose(my.scale_predictions(my.inv_logit(shaphat), b_sample, b_all),
                     model.predict_proba(df_explain[features])[:, 1]))
else:
    print(np.isclose(my.scale_predictions(np.exp(shaphat) / np.exp(shaphat).sum(axis=1, keepdims=True), 
                                          b_sample, b_all),
                     model.predict_proba(df_explain[features])))



########################################################################################################################
# Variable Importance
########################################################################################################################

# --- Default Variable Importance: uses gain sum of all trees ----------------------------------------------------------
xgb.plot_importance(model[1].estimator if type(model[1]) == my.ScalingEstimator else model[1])


# --- Variable Importance by permuation argument ----------------------------------------------------------------------

# Importance (on test data!)
df_varimp_test = my.variable_importance(model, df_test[features], df_test[target_name], features,
                                        scoring=my.d_scoring[TARGET_TYPE][metric],
                                        random_state=42, n_jobs=my.n_jobs)
features_top_test = df_varimp_test.loc[df_varimp_test["importance_cum"] < 95, "feature"].values

# Compare variable importance for train and test (hints to variables prone to overfitting)
fig, ax = plt.subplots(1, 1)
sns.barplot(x="score_diff", y="feature", hue="fold",
            data=pd.concat([df_varimp_train.assign(fold="train"),
                            df_varimp_test.assign(fold="test")], sort=False),
            ax=ax)

# Crossvalidate Importance (only for top features)
df_varimp_test_cv = pd.DataFrame()
for i, (i_train, i_test) in enumerate(cv_5foldsep.split(df_traintest, test_fold=(df_traintest["fold"] == "test"))):
    df_tmp = df_traintest.iloc[i_train, :]
    df_varimp_test_cv = df_varimp_test_cv.append(
        my.variable_importance(d_cv["estimator"][i], df_tmp[features], df_tmp[target_name], features_top_test,
                               scoring=my.d_scoring[TARGET_TYPE][metric],
                               random_state=42, n_jobs=my.n_jobs).assign(run=i))
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
                                w=8, h=4, pdf=my.plotloc + "vi__" + TARGET_TYPE + ".pdf")


########################################################################################################################
# Partial Dependance
########################################################################################################################

'''
# Scikit's partial dependence
# cate
cate_top_test = my.diff(features_top_test, nume)
partial_dependence(model, df_test[features],
                   features=cate_top_test[0],  # just one feature per call is possible!
                   grid_resolution=np.inf,  # workaround to take all members
                   kind="average")
# nume
nume_top_test = my.diff(features_top_test, cate)
from joblib import Parallel, delayed
Parallel(n_jobs=my.n_jobs, max_nbytes='100M')(
    delayed(partial_dependence)(model, df_test[features], feature,
                                grid_resolution=5,  # 5 quantiles
                                kind="average")
    for feature in nume_top_test)
'''

#features_top_test = nume

# Dataframe based patial dependence which can use a reference dataset for value-grid defintion
d_pd = my.partial_dependence(model, df_test[features], features_top_test, df_ref=df_train)

# Crossvalidate
d_pd_cv = {feature: pd.DataFrame() for feature in features_top_test}
for i, (i_train, i_test) in enumerate(cv_5foldsep.split(df_traintest,
                                                        test_fold=(df_traintest["fold"] == "test").values)):
    d_pd_run = my.partial_dependence(model, df_traintest.iloc[i_test, :][features], features_top_test,
                                     df_ref=df_train)
    for feature in features_top_test:
        d_pd_cv[feature] = d_pd_cv[feature].append(d_pd_run[feature].assign(run=i)).reset_index(drop=True)
d_pd_err = {feature: df_tmp.drop(columns="run").groupby("value").std() * 10  # TODO
            for feature, df_tmp in d_pd_cv.items()}

# Plot it
l_calls = list()
for i, feature in enumerate(list(d_pd.keys())):
    i_cols = {"CLASS": 1, "REGR": 0, "MULTICLASS": [0, 1, 2]}
    l_calls.append((my.plot_pd,
                    dict(feature_name=feature, feature=d_pd[feature]["value"],
                         yhat=d_pd[feature].iloc[:, i_cols[TARGET_TYPE]].values,
                         yhat_err=d_pd_err[feature].iloc[:, i_cols[TARGET_TYPE]].values,
                         feature_ref=df_test[feature],
                         reflines=df_test[target_name].mean(),
                         #reflines=[df_test[target_name + "_num"].mean()],
                         #reflines=(df_test.groupby(target_name + "_num")[id_name].count() / len(df_test)).values,
                         legend_labels=(None if TARGET_TYPE != "MULTICLASS" 
                                        else d_pd[feature].columns.values[:3]),
                         ylim=None, color=my.colorblind)))
#%%
my.plot_func(l_calls, pdf_path=my.plotloc + "pd__" + TARGET_TYPE + ".pdf")
#%%

# Shap based partial dependence
explainer = shap.TreeExplainer(model[1].estimator if type(model[1]) is my.ScalingEstimator else model[1])
shap_values = my.agg_shap_values(explainer(model[0].transform(X=df_test[features])),
                                 df_test[features],
                                 len_nume=len(nume), l_map_onehot=model[0].transformers_[1][1].categories_,
                                 round=2)
shap_values.values.shape

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
explainer = shap.TreeExplainer(model[1].estimator if type(model[1]) == my.ScalingEstimator else model[1])
shap_values = explainer(model[0].transform(X=df_explain[features]))
shap_values_agg = my.agg_shap_values(shap_values, df_explain[features],
                                     len_nume=len(nume), l_map_onehot=model[0].transformers_[1][1].categories_,
                                     round=2)

# Plot
fig, ax = plt.subplots(1, 1)
shap.plots.waterfall(shap_values_agg[0], show=True)

# Check
if TARGET_TYPE == "REGR":
    np.isclose(shap_values.values.sum(axis=1) + explainer.expected_value,
               model.predict(df_explain[features]))
else:
    np.isclose(my.scale_predictions(my.inv_logit(shap_values.values.sum(axis=1) + explainer.expected_value),
                                    b_sample, b_all),
               model.predict_proba(df_explain[features])[:, 1])

plt.close("all")


# ######################################################################################################################
# Individual dependencies / Counterfactuals
# ######################################################################################################################

# TODO

