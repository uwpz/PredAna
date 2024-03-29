########################################################################################################################
# Initialize: Packages, functions, parameter
########################################################################################################################

# --- Packages ---------------------------------------------------------------------------------------------------------

# General
from sklearn.inspection import permutation_importance, partial_dependence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
from importlib import reload   

# Special
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
import xgboost as xgb
import lightgbm as lgbm
import shap
from scipy.special import logit, expit

# Custom functions and classes
import utils_plots as up

# Settings
import settings as s


# --- Parameter --------------------------------------------------------------------------------------------------------

# Constants
TARGET_TYPE = "CLASS"
#for TARGET_TYPE in ["CLASS", "REGR", "MULTICLASS"]:
ID_NAME = "instant"
IMPORTANCE_CUM_THRESHOLD = 98

# Main parameter
target_name = "cnt_" + TARGET_TYPE + "_num"
target_labels = ["0_low", "1_high", "2_very_high"] if TARGET_TYPE == "MULTICLASS" else None
metric = "spear" if TARGET_TYPE == "REGR" else "auc"
scoring = up.D_SCORER[TARGET_TYPE]

# Plot
PLOT = True
%matplotlib
plt.ioff() 
# %matplotlib | %matplotlib qt | %matplotlib inline  # activate standard/inline window
# plt.ioff() | plt.ion()  # stop/start standard window
# plt.plot(range(10), range(10))

# Load results from exploration
df = nume_standard = cate_standard = features_binned = features_encoded = None
with open(s.DATALOC + "1_explore.pkl", "rb") as file:
    d_pick = pickle.load(file)
df = d_pick["df"]
nume = d_pick["nume_standard"]
cate = d_pick["cate_standard"]

# Tuning parameter to use (for xgb) and classifier definition
xgb_param = dict(n_estimators=1100, learning_rate=0.01,
                 max_depth=3, min_child_weight=10,
                 colsample_bytree=0.7, subsample=0.7,
                 gamma=0,
                 verbosity=0,
                 n_jobs=s.N_JOBS,
                 use_label_encoder=False)



########################################################################################################################
# Prepare
########################################################################################################################

# --- Define train/test, algorithm, CV ---------------------------------------------------------------------------------
'''
# OLD approch (without UndersampleEstimator)
# Undersample only training data (take all but n_maxpersample at most)
if TARGET_TYPE in ["CLASS", "MULTICLASS"]:
    df.query("fold == 'train'")[target_name].value_counts()
    df_train, b_sample, b_all = up.undersample(df.query("fold == 'train'"), target=target_name,
                                            n_max_per_level=300)
    print(b_sample, b_all)
    if np.any(np.isclose(b_sample, b_all)):
        algo = xgb.XGBClassifier(**xgb_param)
    else:
        algo = up.ScalingEstimator(xgb.XGBClassifier(**xgb_param), b_sample=b_sample, b_all=b_all)
        #alternative: algo = XGBClassifier_rescale(**xgb_param, b_sample = b_sample, b_all = b_all)
else:
    df_train = df.query("fold == 'train'").sample(n=3000, frac=None).reset_index(drop=True)
    algo = xgb.XGBRegressor(**xgb_param)

# Test data
df_test = df.query("fold == 'test'").reset_index(drop=True)  # .sample(300) #ATTENTION: Do not sample in final run!!!

# Combine again
df_traintest = pd.concat([df_train, df_test]).reset_index(drop=True)
'''
df_traintest = df
df_train = df.query("fold == 'train'").reset_index(drop=True)
df_test = df.query("fold == 'test'").reset_index(drop=True)
algo = up.UndersampleEstimator(xgb.XGBRegressor(**xgb_param) if TARGET_TYPE == "REGR"
                               else xgb.XGBClassifier(**xgb_param),
                               n_max_per_level=2000)
'''
lgbm_param = dict(n_estimators=1100, learning_rate=0.01,
                 num_leaves=8, min_child_samples=10,
                 colsample_bytree=0.7, subsample=0.7,
                 n_jobs=s.N_JOBS)
algo = up.UndersampleEstimator(lgbm.LGBMRegressor(**lgbm_param) if TARGET_TYPE == "REGR"
                               else lgbm.LGBMClassifier(**xgb_param),
                               n_max_per_level=2000)
'''

# CV strategy 
cv_5foldsep = up.KFoldSep(5)
split_5foldsep = cv_5foldsep.split(df_traintest, test_fold=(df_traintest["fold"] == "test"))
i_train, i_test = next(split_5foldsep)
print("TRAIN-fold:\n", df_traintest["fold"].iloc[i_train].value_counts(), i_train[:5])
print("TEST-fold:\n", df_traintest["fold"].iloc[i_test].value_counts(), i_test[:5])



# ######################################################################################################################
# Performance
# ######################################################################################################################

# --- Do the full fit and predict on test data -------------------------------------------------------------------------

# Fit
pipe = Pipeline(
    [('matrix', (ColumnTransformer([('nume', MinMaxScaler(), np.array(nume)),
                                    ('cate', OneHotEncoder(sparse=True, handle_unknown="ignore"), 
                                     np.array(cate))]))),
     ('predictor', algo)])
features = nume + cate
model = pipe.fit(df_train[features], df_train[target_name])

# Predict
#%%
if TARGET_TYPE in ["CLASS", "MULTICLASS"]:
    yhat_test = model.predict_proba(df_test[features])
    print(up.auc(df_test[target_name].values, yhat_test))
else:
    yhat_test = model.predict(df_test[features])
    print(up.spear(df_test[target_name].values, yhat_test))
print(pd.DataFrame(yhat_test).describe())
#%%
# Plot performance
if PLOT:
    d_calls = up.get_plotcalls_model_performance(y=df_test[target_name],
                                                 yhat=yhat_test, 
                                                 target_labels=target_labels)
    _ = up.plot_l_calls(l_calls=d_calls.values(), n_cols=3, n_rows=2,
                        constrained_layout=True if TARGET_TYPE == "MULTICLASS" else False,
                        pdf_path=f"{s.PLOTLOC}3__performance__{TARGET_TYPE}.pdf")

# Check performance for crossvalidated fits
d_cv = cross_validate(model, df_traintest[features], df_traintest[target_name],
                      cv=cv_5foldsep.split(df_traintest, test_fold=(df_traintest["fold"] == "test")),  # special 5fold
                      scoring=scoring,
                      return_estimator=True,
                      n_jobs=s.N_JOBS)
print(d_cv["test_" + metric], " \n", np.mean(d_cv["test_" + metric]), np.std(d_cv["test_" + metric]))


# --- Most important variables (on training data) model fit ------------------------------------------------------------

# Variable importance (on train data!)
df_varimp_train = up.variable_importance(model, df_train[features], df_train[target_name], 
                                         features=features,
                                         scorer=scoring[metric],
                                         random_state=42, n_jobs=s.N_JOBS)
# Scikit's VI: permuatation_importance("same parameter but remove features argument and add n_repeats=1")

# Top features (importances sum up to IMPORTANCE_CUM_THRESHOLD of whole sum)
features_top_train = df_varimp_train.query("importance_cum < @IMPORTANCE_CUM_THRESHOLD")["feature"].values

# Fit again only on features_top
pipe_top = Pipeline([
    ('matrix', (ColumnTransformer([('nume', MinMaxScaler(), 
                                    np.array([x for x in nume if x in features_top_train])),
                                   ('cate', OneHotEncoder(sparse=True, handle_unknown="ignore"),
                                    np.array([x for x in cate if x in features_top_train]))]))),
    ('predictor', clone(algo))])
model_top = pipe_top.fit(df_train[features_top_train], df_train[target_name])

# Plot performance of features_top_train model
if TARGET_TYPE in ["CLASS", "MULTICLASS"]:
    yhat_top = model_top.predict_proba(df_test[features_top_train])
    print(up.auc(df_test[target_name].values, yhat_top))
else:
    yhat_top = model_top.predict(df_test[features_top_train])
    print(up.spear(df_test[target_name].values, yhat_top))
if PLOT:
    d_calls = up.get_plotcalls_model_performance(y=df_test[target_name], yhat=yhat_top)
    _ = up.plot_l_calls(l_calls=d_calls.values(), n_cols=3, n_rows=2,
                        pdf_path=f"{s.PLOTLOC}3__performance_top__{TARGET_TYPE}.pdf")



########################################################################################################################
# Diagnosis
########################################################################################################################

# ---- Check residuals fot top features --------------------------------------------------------------------------------

# Residuals
if TARGET_TYPE in ["CLASS", "MULTICLASS"]:
    # "1 - yhat_of_true_class"
    df_test["residual"] = 1 - yhat_test[np.arange(len(df_test[target_name])), df_test[target_name]]
else:
    df_test["residual"] = df_test[target_name] - yhat_test
df_test["abs_residual"] = df_test["residual"].abs()
df_test["residual"].describe()

# For non-regr tasks you might want to plot it for each target level (df_test.query("target == 0/1"))
if PLOT:
    _ = up.plot_l_calls(pdf_path=f"{s.PLOTLOC}3__diagnosis_residual__{TARGET_TYPE}.pdf",
                        n_cols=3, n_rows=2, figsize=(18, 12),
                        l_calls=[(up.plot_feature_target,
                                  dict(feature=df_test[feature], target=df_test["residual"],
                                       add_miss_info=False, color=s.COLORBLIND[3]))
                                 for feature in up.diff(features_top_train, "day_of_month_ENCODED")])

# Absolute residuals
if TARGET_TYPE == "REGR":
    if PLOT:
        _ = up.plot_l_calls(pdf_path=f"{s.PLOTLOC}3__diagnosis_absolute_residual__{TARGET_TYPE}.pdf",
                            n_cols=3, n_rows=2, figsize=(18, 12),
                            l_calls=[(up.plot_feature_target,
                                      dict(feature=df_test[feature], target=df_test["abs_residual"],
                                           add_miss_info=False, color=s.COLORBLIND[3]))
                                     for feature in up.diff(features_top_train, "day_of_month_ENCODED")])



########################################################################################################################
# Variable Importance
########################################################################################################################

# --- Default Variable Importance: uses gain sum of all trees ----------------------------------------------------------

xgb.plot_importance(model[1].subestimator if hasattr(model[1], "subestimator") else model[1])


# --- Variable Importance by permuation argument -----------------------------------------------------------------------

# Importance (on test data!)
df_varimp_test = up.variable_importance(model, df_test[features], df_test[target_name], 
                                        features=features,
                                        scorer=scoring[metric],
                                        random_state=42, n_jobs=s.N_JOBS)
features_top_test = df_varimp_test.loc[df_varimp_test["importance_cum"] < IMPORTANCE_CUM_THRESHOLD, "feature"].values

# Compare variable importance for train and test (difference hints to variables prone to overfitting)
fig, ax = plt.subplots(1, 1)
sns.barplot(x="score_diff", y="feature", hue="fold",
            data=pd.concat([df_varimp_train.assign(fold="train"),
                            df_varimp_test.assign(fold="test")], sort=False),
            ax=ax)

# Crossvalidate Importance (only for top features)
df_varimp_test_cv = pd.DataFrame()
for i, (i_train, i_test) in enumerate(cv_5foldsep.split(df_traintest, test_fold=(df_traintest["fold"] == "test"))):
    df_test_cv = df_traintest.iloc[i_test, :]
    df_varimp_test_cv = df_varimp_test_cv.append(
        (up.variable_importance(d_cv["estimator"][i], df_test_cv[features], df_test_cv[target_name],
                                features=features_top_test,
                                scorer=scoring[metric],
                                random_state=42, n_jobs=s.N_JOBS)
         .assign(run=i)))
df_varimp_test_err = (df_varimp_test_cv.groupby("feature")["score_diff", "importance"].agg("std")
                      .pipe(lambda x: x.set_axis([col + "_error" for col in x.columns], axis=1, inplace=False))
                      .reset_index())

# Plot Importance
df_varimp_plot = (df_varimp_test.query("feature in @features_top_test")
                  .assign(category=lambda x: np.where(x["feature"].isin(nume), "nume", "cate"))  # for coloring
                  .merge(df_varimp_test_err, how="left", on="feature")
                  .merge(df_varimp_train[["feature", "importance", "importance_cum"]]
                         .rename(columns={"importance": "importance_train",
                                          "importance_cum": "importance_cum_train"}),
                         how="left", on="feature")
                  .sort_values("importance_train", ascending=False))
l_calls = [(up.plot_variable_importance,
            dict(features=df_varimp_plot["feature"],
                 importance=df_varimp_plot["importance_train"],  # train imp as bars
                 importance_cum=df_varimp_plot["importance_cum_train"],  # train imp for cumulative
                 importance_mean=df_varimp_plot["importance"],  # test imp for mean+error marker
                 importance_error=df_varimp_plot["importance_error"],
                 #max_score_diff=df_varimp_plot["score_diff"][0].round(2),  # no max_score_diff as diff for train/test
                 category=df_varimp_plot["category"],  # coloring
                 color_error="black"))]
if PLOT:
    _ = up.plot_l_calls(l_calls,
                        pdf_path=f"{s.PLOTLOC}3__vi__{TARGET_TYPE}.pdf",
                        n_rows=1, n_cols=1, figsize=(8, 4))



########################################################################################################################
# Partial Dependance
########################################################################################################################

'''
# Scikit's partial dependence example (has also plot functionality)

# cate
cate_top_test = up.diff(features_top_test, nume)
tmp = partial_dependence(model, df_test[features],
                         features=cate_top_test[[0]],  # just one feature per call is possible!
                         grid_resolution=np.inf,  # workaround to take all members
                         kind="individual")  # individual or average
                
# nume
nume_top_test = up.diff(features_top_test, cate)
from joblib import Parallel, delayed
Parallel(n_jobs=s.N_JOBS, max_nbytes='100M')(
    delayed(partial_dependence)(model, df_test[features], 
                                features=np.array([feature]),
                                grid_resolution=5,  # 5 quantiles
                                kind="average")
    for feature in nume_top_test)
'''


# --- Standard PD ------------------------------------------------------------------------------------------------------

# Dataframe based patial dependence which can use a reference dataset for value-grid defintion (for numeric features)
d_pd = up.partial_dependence(model, df_test[features], features_top_test, df_ref=df_train)

# Crossvalidate
d_pd_cv = {feature: pd.DataFrame() for feature in features_top_test}
for i, (i_train, i_test) in enumerate(cv_5foldsep.split(df_traintest,
                                                        test_fold=(df_traintest["fold"] == "test").values)):
    d_pd_run = up.partial_dependence(model, df_traintest.iloc[i_test, :][features], 
                                     features=features_top_test,
                                     df_ref=df_train)
    for feature in features_top_test:
        d_pd_cv[feature] = d_pd_cv[feature].append(d_pd_run[feature].assign(run=i)).reset_index(drop=True)
d_pd_err = {feature: df_tmp.drop(columns="run").groupby("value").std().reset_index(drop=True)
            for feature, df_tmp in d_pd_cv.items()}

# Plot it
l_calls = list()
for i, feature in enumerate(list(d_pd.keys())):
    i_col = {"REGR": 0, "CLASS": 1, "MULTICLASS": 2}
    l_calls.append(
        (up.plot_pd,
         dict(feature_name=feature, feature=d_pd[feature]["value"],
              yhat=d_pd[feature].iloc[:, i_col[TARGET_TYPE]],
              yhat_err=d_pd_err[feature].iloc[:, i_col[TARGET_TYPE]],
              feature_ref=df_test[feature],
              refline=yhat_test[:, i_col[TARGET_TYPE]].mean() if TARGET_TYPE != "REGR" else yhat_test.mean(),
              ylim=None, color=s.COLORBLIND[i_col[TARGET_TYPE]])))
if PLOT:
    up.plot_l_calls(l_calls, pdf_path=f"{s.PLOTLOC}3__pd__{TARGET_TYPE}.pdf")


# --- Shap-based PD ----------------------------------------------------------------------------------------------------

if TARGET_TYPE != "MULTICLASS":  # TODO: MULTICLASS

    # Get shap for test data
    explainer = shap.TreeExplainer(model[1].subestimator if hasattr(model[1], "subestimator") else model[1])
    shap_values = up.agg_shap_values(explainer(model[0].transform(X=df_test[features])),
                                     df_test[features],
                                     len_nume=len(nume), l_map_onehot=model[0].transformers_[1][1].categories_,
                                     round=2)

    # Rescale due to undersampling
    if TARGET_TYPE == "CLASS":
        shap_values.base_values = logit(up.scale_predictions(expit(shap_values.base_values),
                                                             model[1].b_sample_, model[1].b_all_))
    if TARGET_TYPE == "MULTICLASS":
        shap_values.base_values = np.log(up.scale_predictions(np.exp(shap_values.base_values) /
                                                              np.exp(shap_values.base_values).sum(axis=1, keepdims=True),
                                                              model[1].b_sample_, model[1].b_all_))
    # Aggregate shap
    d_pd_shap = up.shap2pd(shap_values, features_top_test, df_ref=df_train)  # TODO: MULTICLASS

    # Transform to response level
    if TARGET_TYPE == "CLASS":
        d_pd_shap = {key: df.assign(yhat=lambda x: expit(x["yhat"])) for key, df in d_pd_shap.items()}
    if TARGET_TYPE == "MULTICLASS":
        pass  # TODO: MULTICLASS

    # Plot it
    l_calls = list()
    for i, feature in enumerate(list(d_pd_shap.keys())):
        i_col = {"REGR": 0, "CLASS": 1, "MULTICLASS": 2}
        l_calls.append((up.plot_pd,
                        dict(feature_name=feature,
                             feature=d_pd_shap[feature]["value"],
                             yhat=d_pd_shap[feature]["yhat"],  
                             # feature_ref still does not work with numeric features due to non-bins in feature_ref
                             feature_ref=df_test[feature] if feature in cate else None,  
                             refline=(yhat_test[:, i_col[TARGET_TYPE]].mean() if TARGET_TYPE != "REGR"
                                      else yhat_test.mean()),
                             ylim=None, color=s.COLORBLIND[i_col[TARGET_TYPE]])))
    if PLOT:
        up.plot_l_calls(l_calls, pdf_path=f"{s.PLOTLOC}3__pd_shap__{TARGET_TYPE}.pdf")



########################################################################################################################
# Explanations
########################################################################################################################

# ---- Explain bad predictions -----------------------------------------------------------------------------------------

# Filter data
n_select = 6
i_worst = df_test.sort_values("abs_residual", ascending=False).iloc[:n_select, :].index.values
i_best = df_test.sort_values("abs_residual", ascending=True).iloc[:n_select, :].index.values
i_random = df_test.sample(n=n_select).index.values
i_explain = np.concatenate([i_worst, i_best, i_random])
df_explain = df_test.iloc[i_explain, :].reset_index(drop=True)
y_explain = df_explain[target_name]
yhat_explain = yhat_test[i_explain]

# Get shap
explainer = shap.TreeExplainer(model[1].subestimator if hasattr(model[1], "subestimator") else model[1])
X_explain = model[0].transform(X=df_explain[features])  # for lgbm might need: X_explain = X_explain.toarray()
shap_values = explainer(X_explain)

# Aggregate one-hot encodings to one shap value
shap_values = up.agg_shap_values(explainer(model[0].transform(X=df_explain[features])),
                                 df_explain[features],
                                 # TODO: make the following more robust as nume cols need to be at beginning of X
                                 len_nume=len(nume), 
                                 l_map_onehot=model[0].transformers_[1][1].categories_,
                                 round=2)  # aggregate onehot

# Rescale due to undersampling
if TARGET_TYPE == "CLASS":
    shap_values.base_values = logit(up.scale_predictions(expit(shap_values.base_values),
                                                         model[1].b_sample_, model[1].b_all_))
if TARGET_TYPE == "MULTICLASS":
    shap_values.base_values = np.log(
        up.scale_predictions(
            np.exp(shap_values.base_values) / np.exp(shap_values.base_values).sum(axis=1, keepdims=True),
            model[1].b_sample_, model[1].b_all_))

# Check
shaphat = shap_values.values.sum(axis=1) + shap_values.base_values
if TARGET_TYPE == "REGR":
    print(np.isclose(shaphat, model.predict(df_explain[features])))
elif TARGET_TYPE == "CLASS":
    print(np.isclose(expit(shaphat), model.predict_proba(df_explain[features])[:, 1]))
    #for lgbm might need: print(np.isclose(expit(shaphat)[:, 1], model.predict_proba(df_explain[features])[:, 1]))
else:
    print(np.isclose(np.exp(shaphat) / np.exp(shaphat).sum(axis=1, keepdims=True),
                     model.predict_proba(df_explain[features])))

'''
# Shap's default waterfall plot
fig, ax = plt.subplots(1, 1)
i = 1
i_col = {"CLASS": 1, "MULTICLASS": df_explain[target_name].iloc[i]}
y_str = (str(df_explain[target_name].iloc[i]) if TARGET_TYPE != "REGR" 
        else format(df_explain[target_name].iloc[i], ".2f"))
yhat_str = (format(yhat_explain[i, i_col[TARGET_TYPE]], ".3f") if TARGET_TYPE != "REGR" 
            else format(yhat_explain[i], ".2f"))
ax.set_title("id = " + str(df_explain[ID_NAME].iloc[i]) + " (y = " + y_str + ")" + r" ($\^y$ = " + yhat_str + ")")
if TARGET_TYPE != "MULTICLASS":
    shap.plots.waterfall(shap_values[i], show=True)  # TODO: replace "00" as it crashes otherwise
else:
    shap.plots.waterfall(shap_values[i][:, df_explain[target_name].iloc[i]], show=True)
'''

# Plot it
l_calls = list()
for i in range(len(df_explain)):
    y_str = (str(df_explain[target_name].iloc[i]) if TARGET_TYPE != "REGR"
             else format(df_explain[target_name].iloc[i], ".2f"))
    i_col = {"CLASS": 1, "MULTICLASS": df_explain[target_name].iloc[i]}
    yhat_str = (format(yhat_explain[i, i_col[TARGET_TYPE]], ".3f") if TARGET_TYPE != "REGR"
                else format(yhat_explain[i], ".2f"))
    l_calls.append((up.plot_shap,
                    dict(shap_values=(shap_values[:, :, i_col[TARGET_TYPE]] if TARGET_TYPE == "MULTICLASS" 
                                      else shap_values),
                         index=i,
                         id=df_explain[ID_NAME][i],
                         y_str=y_str,
                         yhat_str=yhat_str)))
if PLOT:
    up.plot_l_calls(l_calls, pdf_path=f"{s.PLOTLOC}3__shap__{TARGET_TYPE}.pdf")



# ######################################################################################################################
# Individual dependencies / Counterfactuals  / Ceteris Paribus
# ######################################################################################################################

# TODO

plt.close("all")
