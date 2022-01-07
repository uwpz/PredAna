########################################################################################################################
# Initialize: Packages, functions, parameters
########################################################################################################################

# --- Packages ---------------------------------------------------------------------------------------------------------

# General
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
from importlib import reload

# Special
from sklearn.model_selection import KFold, ShuffleSplit, PredefinedSplit, learning_curve, GridSearchCV, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # , GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, SGDRegressor, LogisticRegression, ElasticNet
import xgboost as xgb
import lightgbm as lgbm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras import optimizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
#  from sklearn.tree import DecisionTreeRegressor, plot_tree , export_graphviz

# Custom functions and classes
import utils_plots as up

# Settings
import settings as s


# --- Parameter --------------------------------------------------------------------------------------------------------

TARGET_TYPE = "MULTICLASS"
#for TARGET_TYPE in ["CLASS", "REGR", "MULTICLASS"]:

# Main parameter
target_name = "cnt_" + TARGET_TYPE + "_num"
metric = "spear" if TARGET_TYPE == "REGR" else "auc"
scoring = up.D_SCORER[TARGET_TYPE]

# Load results from exploration
df = nume_standard = cate_standard = features_binned = features_encoded = None
with open(s.DATALOC + "1_explore.pkl", "rb") as file:
    d_pick = pickle.load(file)
for key, val in d_pick.items():
    exec(key + "= val")



########################################################################################################################
# Prepare data
########################################################################################################################

# --- Sample data ------------------------------------------------------------------------------------------------------

# Undersample only training data (take all but n_max_per_level at most)
if TARGET_TYPE == "REGR":
    df_tmp = df.query("fold == 'train'").sample(n=2000, frac=None).reset_index(drop=True)
else:
    df.query("fold == 'train'")[target_name].value_counts()
    df_tmp, b_sample, b_all = up.undersample(df=df.query("fold == 'train'"), 
                                             target=target_name,
                                             n_max_per_level=2000)
    print(b_sample, b_all)  # base rates
df_tune = (pd.concat([df_tmp, df.query("fold == 'test'")], sort=False)  # take whole test data
           .sample(frac=1)  # shuffle
           .reset_index(drop=True))
df_tune.groupby("fold")[target_name].describe()

# Derive design matrices
X_standard = (ColumnTransformer([('nume', MinMaxScaler(), 
                                  np.array(nume_standard)),
                                 ('cate', OneHotEncoder(sparse=True, handle_unknown="ignore"), 
                                  np.array(cate_standard))])
              .fit_transform(df_tune[nume_standard + cate_standard]))
X_binned = OneHotEncoder(sparse=False, handle_unknown="ignore").fit_transform(df_tune[features_binned])
X_encoded = MinMaxScaler().fit_transform(df_tune[features_encoded])

'''
tmp = (ColumnTransformer([('nume', MinMaxScaler(), np.array(nume_standard)),
                          ('cate', OneHotEncoder(sparse=True, handle_unknown="ignore"), 
                                np.array(cate_standard))]))
tmp.named_transformers_["cate"].categories_
'''


# --- Define some possible CV strategies -------------------------------------------------------------------------------

cv_index = PredefinedSplit(df_tune["fold"].map({"train": -1, "test": 0}).values)
cv_5fold = KFold(5, shuffle=True, random_state=42)
cv_5foldsep = up.KFoldSep(5, random_state=42)

'''
# Test a split
split = cv_5fold.split(df_tune)
i_train, i_test = next(split)
df_tune["fold"].iloc[i_train].describe()
df_tune["fold"].iloc[i_test].describe()
df_tune["fold"].iloc[i_train].value_counts()
df_tune["fold"].iloc[i_test].value_counts()
print(np.sort(i_train))
print(np.sort(i_test))
'''



########################################################################################################################
# # Test an algorithm (and determine tuning parameter grid)
########################################################################################################################

# --- Lasso / Elastic Net ----------------------------------------------------------------------------------------------

fit = (GridSearchCV(SGDRegressor(penalty="ElasticNet", warm_start=True) if TARGET_TYPE == "REGR" else
                    SGDClassifier(loss="log", penalty="ElasticNet", warm_start=True),  # , tol=1e-2
                    {"alpha": [2 ** x for x in range(-8, -20, -2)],
                    "l1_ratio": [0, 0.5, 1]},
                    cv=cv_5fold.split(df_tune),
                    refit=False,
                    scoring=scoring,
                    return_train_score=True,
                    n_jobs=s.N_JOBS)
       .fit(X=X_binned,
            y=df_tune[target_name]))
# Plot: use metric="score" if scoring has only 1 metric
fig = up.plot_cvresults(fit.cv_results_, metric=metric, x_var="alpha", color_var="l1_ratio")
fig.savefig(f"{s.PLOTLOC}2__tune_sgd__{TARGET_TYPE}.pdf")

# Usually better alternative
if TARGET_TYPE in ["CLASS", "MULTICLASS"]:
    fit = (GridSearchCV(LogisticRegression(penalty="l1", fit_intercept=True, solver="liblinear"),
                        {"C": [2 ** x for x in range(2, -5, -1)]},
                        cv=cv_index.split(df_tune),
                        refit=False,
                        scoring=scoring,
                        return_train_score=True,
                        n_jobs=s.N_JOBS)
           .fit(X=X_binned,
                y=df_tune[target_name]))
    fig = up.plot_cvresults(fit.cv_results_, metric=metric, x_var="C")
    fig.savefig(f"{s.PLOTLOC}2__tune_lasso__{TARGET_TYPE}.pdf")
else:
    fit = (GridSearchCV(ElasticNet(),
                        {"alpha": [2 ** x for x in range(-8, -20, -2)],
                        "l1_ratio": [0, 0.5, 1]},
                        cv=cv_index.split(df_tune),
                        refit=False,
                        scoring=scoring,
                        return_train_score=True,
                        n_jobs=s.N_JOBS)
           .fit(X=X_binned,
                y=df_tune[target_name]))
    fig = up.plot_cvresults(fit.cv_results_, metric=metric, x_var="alpha", color_var="l1_ratio")
    fig.savefig(f"{s.PLOTLOC}2__tune_enet__{TARGET_TYPE}.pdf")


# --- Random Forest ----------------------------------------------------------------------------------------------------

fit = (GridSearchCV(RandomForestRegressor() if TARGET_TYPE == "REGR" else
                    RandomForestClassifier(),
                    {"n_estimators": [10, 20, 100],
                    "min_samples_leaf": [1, 10, 50]},
                    cv=cv_index.split(df_tune),
                    refit=False,
                    scoring=scoring,
                    return_train_score=True,
                    # use_warm_start=["n_estimators"],
                    n_jobs=s.N_JOBS)
       .fit(X=X_standard,
            y=df_tune[target_name]))
fig = up.plot_cvresults(fit.cv_results_, metric=metric,
                        x_var="n_estimators", color_var="min_samples_leaf")
fig.savefig(f"{s.PLOTLOC}2__tune_rforest__{TARGET_TYPE}.pdf")


# --- XGBoost ----------------------------------------------------------------------------------------------------------
start = time.time()
fit = (up.GridSearchCV_xlgb(xgb.XGBRegressor(verbosity=0) if TARGET_TYPE == "REGR" else
                            xgb.XGBClassifier(verbosity=0),
                            {"n_estimators": [x for x in range(100, 1100, 100)], "learning_rate": [0.01],
                            "max_depth": [3, 6], "min_child_weight": [5, 10]},
                            cv=cv_index.split(df_tune),
                            refit=False,
                            scoring=scoring,  # must be dict here
                            return_train_score=True,
                            n_jobs=s.N_JOBS)
       .fit(X=X_standard,
            y=df_tune[target_name]))
print(time.time() - start)
fig = up.plot_cvresults(fit.cv_results_, metric=metric,
                        x_var="n_estimators", color_var="max_depth", column_var="min_child_weight")
fig.savefig(f"{s.PLOTLOC}2__tune_xgb__{TARGET_TYPE}.pdf")


# --- LightGBM ---------------------------------------------------------------------------------------------------------

# Indices of categorical variables (for Lightgbm)
i_cate_standard = [i for i in range(len(nume_standard)) if nume_standard[i].endswith("_ENCODED")]

# Alterantive: pure encoded -> adapt fit call below
#i_cate_encoded = [i for i in range(len(features_encoded)) if features_encoded[i].endswith("_ENCODED")]

# Fit
start = time.time()
fit = (up.GridSearchCV_xlgb(lgbm.LGBMRegressor() if TARGET_TYPE == "REGR" else
                            lgbm.LGBMClassifier(),
                            {"n_estimators": [x for x in range(100, 3100, 500)], "learning_rate": [0.01],
                            "num_leaves": [8, 16, 32], "min_child_samples": [5]},
                            cv=cv_index.split(df_tune),
                            refit=False,
                            scoring=scoring,
                            return_train_score=True,
                            n_jobs=s.N_JOBS)
       .fit(X_standard,  # X_encoded
            categorical_feature=i_cate_standard,  # i_cate_encoded
            y=df_tune[target_name]))
print(time.time() - start)
fig = up.plot_cvresults(fit.cv_results_, metric=metric,
                        x_var="n_estimators", color_var="num_leaves", column_var="min_child_samples")
fig.savefig(f"{s.PLOTLOC}2__tune_lgbm__{TARGET_TYPE}.pdf")


# --- DeepL ------------------------------------------------------------------------------------------------------------

# Keras wrapper for Scikit
def keras_model(input_dim, output_dim, TARGET_TYPE,
                size="10",
                lambdah=None, dropout=None,
                lr=1e-5,
                batch_normalization=False,
                activation="relu"):
    model = Sequential()

    # Add dense layers
    for units in size.split("-"):
        model.add(Dense(units=int(units), activation=activation, input_dim=input_dim,
                        kernel_regularizer=l2(lambdah) if lambdah is not None else None,
                        kernel_initializer="glorot_uniform"))
        # Add additional layer
        if batch_normalization is not None:
            model.add(BatchNormalization())
        if dropout is not None:
            model.add(Dropout(dropout))

    # Output
    if TARGET_TYPE == "CLASS":
        model.add(Dense(1, activation='sigmoid',
                        kernel_regularizer=l2(lambdah) if lambdah is not None else None))
        model.compile(loss="binary_crossentropy", optimizer=optimizers.RMSprop(lr=lr),
                      metrics=["accuracy"])
    elif TARGET_TYPE == "MULTICLASS":
        model.add(Dense(output_dim, activation='softmax',
                        kernel_regularizer=l2(lambdah) if lambdah is not None else None))
        model.compile(loss="categorical_crossentropy", optimizer=optimizers.RMSprop(lr=lr),
                      metrics=["accuracy"])
    else:
        model.add(Dense(1, activation='linear',
                        kernel_regularizer=l2(lambdah) if lambdah is not None else None))
        model.compile(loss="mse", optimizer=optimizers.RMSprop(lr=lr),
                      metrics=["MeanSquaredError"])

    return model

# Fit
n_cols = X_standard.shape[1]
fit = (GridSearchCV(KerasRegressor(build_fn=keras_model,
                                   input_dim=n_cols,
                                   output_dim=1,
                                   TARGET_TYPE=TARGET_TYPE,
                                   verbose=0) if TARGET_TYPE == "REGR" else
                    KerasClassifier(build_fn=keras_model,
                                    input_dim=n_cols,
                                    output_dim=1 if TARGET_TYPE == "CLASS" else df_tune[target_name].nunique(),
                                    TARGET_TYPE=TARGET_TYPE,
                                    verbose=0),
                    {"size": ["10", "10-10", "20"],
                    "lambdah": [1e-8], "dropout": [None],
                     "batch_size": [40], "lr": [1e-3],
                     "batch_normalization": [False, True],
                     "activation": ["relu", "elu"],
                     "epochs": [2, 7, 15]},
                    cv=cv_index.split(df_tune),
                    refit=False,
                    scoring=scoring,
                    return_train_score=True,
                    n_jobs=s.N_JOBS)
       # TODO Bugcheck: Why not sparse for REGR???
       .fit(X_standard.todense() if TARGET_TYPE == "REGR" else X_standard,  # X_encoded as alternative
            y=(pd.get_dummies(df_tune[target_name]) if TARGET_TYPE == "MULTICLASS" else
            df_tune[target_name])))
fig = up.plot_cvresults(fit.cv_results_, metric=metric,
                        x_var="epochs", color_var="batch_normalization", column_var="activation", row_var="size")
fig.savefig(f"{s.PLOTLOC}2__tune_deepl__{TARGET_TYPE}.pdf")



########################################################################################################################
# Simulation: compare algorithms
########################################################################################################################

# --- Run methods ------------------------------------------------------------------------------------------------------

df_modelcomp_result = pd.DataFrame()  # intialize

# Elastic Net
cvresults = cross_validate(
    estimator=GridSearchCV(SGDRegressor(penalty="ElasticNet", warm_start=True) if TARGET_TYPE == "REGR" else
                           SGDClassifier(loss="log", penalty="ElasticNet", warm_start=True),  # , tol=1e-2
                           {"alpha": [2 ** x for x in range(-4, -12, -1)],
                            "l1_ratio": [1]},
                           cv=ShuffleSplit(1, test_size=0.2, random_state=999),  # just 1-fold for tuning
                           refit=metric,
                           scoring=scoring,
                           return_train_score=False,
                           n_jobs=s.N_JOBS),
    X=X_binned,
    y=df_tune[target_name],
    cv=cv_5foldsep.split(df_tune, test_fold=(df_tune["fold"] == "test")),
    scoring=scoring,
    return_train_score=False,
    n_jobs=s.N_JOBS)
df_modelcomp_result = df_modelcomp_result.append(pd.DataFrame.from_dict(cvresults).reset_index()
                                                 .assign(model="ElasticNet"),
                                                 ignore_index=True)

# Xgboost
cvresults = cross_validate(
    estimator=up.GridSearchCV(
        xgb.XGBRegressor(verbosity=0) if TARGET_TYPE == "REGR" else
        xgb.XGBClassifier(verbosity=0),
        {"n_estimators": [x for x in range(2000, 2001, 1)], "learning_rate": [0.01],
         "max_depth": [3], "min_child_weight": [5]},
        cv=ShuffleSplit(1, test_size=0.2, random_state=999),  # just 1-fold for tuning
        refit=metric,
        scoring=scoring,
        return_train_score=False,
        n_jobs=s.N_JOBS),
    X=X_standard,
    y=df_tune[target_name],
    cv=cv_5foldsep.split(df_tune, test_fold=(df_tune["fold"] == "test")),
    scoring=scoring,
    return_train_score=False,
    n_jobs=s.N_JOBS)
df_modelcomp_result = df_modelcomp_result.append(pd.DataFrame.from_dict(cvresults).reset_index()
                                                 .assign(model="XGBoost"),
                                                 ignore_index=True)

# Lgbm
cvresults = cross_validate(
    estimator=up.GridSearchCV(
        lgbm.LGBMRegressor(verbosity=0) if TARGET_TYPE == "REGR" else
        lgbm.LGBMClassifier(verbosity=0),
        {"n_estimators": [x for x in range(500, 501, 1)], "learning_rate": [0.01],
         "num_leaves": [32], "min_child_weight": [5]},
        cv=ShuffleSplit(1, test_size=0.2, random_state=999),  # just 1-fold for tuning
        refit=metric,
        scoring=scoring,
        return_train_score=False,
        n_jobs=s.N_JOBS),
    X=X_standard,
    y=df_tune[target_name],
    fit_params=dict(categorical_feature=i_cate_standard),
    cv=cv_5foldsep.split(df_tune, test_fold=(df_tune["fold"] == "test")),
    scoring=scoring,
    return_train_score=False,
    n_jobs=s.N_JOBS)
df_modelcomp_result = df_modelcomp_result.append(pd.DataFrame.from_dict(cvresults).reset_index()
                                                 .assign(model="Lgbm"),
                                                 ignore_index=True)


# --- Plot model comparison --------------------------------------------------------------------------------------------

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
up.plot_modelcomp(ax, df_modelcomp_result.rename(columns={"index": "run", "test_" + metric: metric}),
                  scorevar=metric)
fig.savefig(f"{s.PLOTLOC}2__model_comparison__{TARGET_TYPE}.pdf")



########################################################################################################################
# Learning curve for winner algorithm
########################################################################################################################

# Calc learning curve
n_train, score_train, score_test, time_train, time_test = learning_curve(
    estimator=GridSearchCV(
        xgb.XGBRegressor(verbosity=0) if TARGET_TYPE == "REGR" else
        xgb.XGBClassifier(verbosity=0),
        {"n_estimators": [2000], "learning_rate": [0.01],
         "max_depth": [3], "min_child_weight": [5]},
        cv=ShuffleSplit(1, test_size=0.2, random_state=999),  # just 1-fold for tuning
        refit=metric,
        scoring=scoring,
        return_train_score=False,
        n_jobs=s.N_JOBS),
    X=X_standard,
    y=df_tune[target_name],
    train_sizes=np.linspace(0.1, 1, 5),
    cv=cv_5fold.split(df_tune),
    scoring=scoring[metric],
    return_times=True,
    n_jobs=s.N_JOBS)

# Plot it
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
up.plot_learningcurve(ax, n_train, score_train, score_test, time_train)
fig.savefig(f"{s.PLOTLOC}2__learningCurve__{TARGET_TYPE}.pdf")
