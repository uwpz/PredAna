########################################################################################################################
# Initialize: Packages, functions, parameters
########################################################################################################################

# --- Packages ------------------------------------------------------------------------------------

# General
import os  # sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # ,matplotlib
import pickle
from importlib import reload
import time

# Special
import hmsPM.plotting as hms_plot
from sklearn.model_selection import KFold, ShuffleSplit, PredefinedSplit, learning_curve, GridSearchCV, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # , GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, SGDRegressor, LogisticRegression, ElasticNet
import xgboost as xgb
import lightgbm as lgbm
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.regularizers import l2
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
#  from sklearn.tree import DecisionTreeRegressor, plot_tree , export_graphviz

# Custom functions and classes
import my_utils as my


# --- Parameter --------------------------------------------------------------------------

# Main parameter
TARGET_TYPE = "REGR"

# Load results from exploration
df = nume_standard = cate_standard = cate_binned = nume_encoded = None
with open(my.dataloc + "1_explore.pkl", "rb") as file:
    d_pick = pickle.load(file)
for key, val in d_pick.items():
    exec(key + "= val")

# Adapt targets
df["cnt_CLASS"] = df["cnt_CLASS"].str.slice(0, 1).astype("int")
df["cnt_MULTICLASS"] = df["cnt_MULTICLASS"].str.slice(0, 1).astype("int")


########################################################################################################################
# Prepare data
########################################################################################################################

# --- Sample data ------------------------------------------------------------------------------------------------------

# Undersample only training data (take all but n_maxpersample at most)
if TARGET_TYPE == "REGR":
    df_tmp = df.query("fold == 'train'").sample(n=3000, frac=None).reset_index(drop=True)
else:
    df.query("fold == 'train'")["cnt_" + TARGET_TYPE].value_counts()
    df_tmp, b_sample, b_all = my.undersample(df.query("fold == 'train'"), target="cnt_" + TARGET_TYPE, 
                                             n_max_per_level=2000)
    print(b_sample, b_all)
df_tune = (pd.concat([df_tmp, df.query("fold == 'test'")], sort=False)
           .sample(frac=1)  # shuffle
           .reset_index(drop=True))
df_tune.groupby("fold")["cnt_" + TARGET_TYPE].describe()

# Derive design matrices
X_standard = (ColumnTransformer([('nume', MinMaxScaler(), nume_standard),
                                 ('cate', OneHotEncoder(sparse=True, handle_unknown="ignore"), cate_standard)])
              .fit_transform(df_tune[np.append(nume_standard, cate_standard)]))
X_binned = OneHotEncoder(sparse=False, handle_unknown="ignore").fit_transform(df_tune[cate_binned])
X_encoded = MinMaxScaler().fit_transform(df_tune[nume_encoded])



'''
tmp = (ColumnTransformer([('nume', MinMaxScaler(), nume_standard),
                                 ('cate', OneHotEncoder(sparse=True, handle_unknown="ignore"), cate_standard)]))
tmp.named_transformers_["cate"].categories_
'''

# --- Define some splits -------------------------------------------------------------------------------------------

cv_index = PredefinedSplit(df_tune["fold"].map({"train": -1, "test": 0}).values)
cv_5fold = KFold(5, shuffle=True, random_state=42)
cv_5foldsep = my.KFoldSep(5, random_state=42)

'''
# Test a split
df_tune["fold"].value_counts()
split = cv_my5fold.split(df_tune)
i_train, i_test = next(split)
df_tune["fold"].iloc[i_train].describe()
df_tune["fold"].iloc[i_test].describe()
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
                     "l1_ratio": [0, 1]},
                    cv=cv_index.split(df_tune),
                    refit=False,
                    scoring=my.d_scoring[TARGET_TYPE],
                    return_train_score=True,
                    n_jobs=my.n_jobs)
       .fit(X=X_binned,
            y=df_tune["cnt_" + TARGET_TYPE]))

# Plot: use metric="score" if scoring has only 1 metric
(hms_plot.ValidationPlotter(x_var="alpha", color_var="l1_ratio", show_gen_gap=True, w=8, h=6)
 .plot(fit.cv_results_, metric="rmse" if TARGET_TYPE == "REGR" else "auc"))  
# pd.DataFrame(fit.cv_results_)

# Usually better alternative
if TARGET_TYPE in ["CLASS", "MULTICLASS"]:
    fit = (GridSearchCV(LogisticRegression(penalty="l1", fit_intercept=True, solver="liblinear"),
                        {"C": [2 ** x for x in range(2, -5, -1)]},
                        cv=cv_index.split(df_tune),
                        refit=False,
                        scoring=my.d_scoring[TARGET_TYPE],
                        return_train_score=True,
                        n_jobs=my.n_jobs)
           .fit(X=X_binned,
                y=df_tune["cnt_" + TARGET_TYPE]))
    (hms_plot.ValidationPlotter(x_var="C", show_gen_gap=True)
     .plot(fit.cv_results_, metric="auc"))
else:
    fit = (GridSearchCV(ElasticNet(),
                        {"alpha": [2 ** x for x in range(-8, -20, -2)],
                         "l1_ratio": [0, 1]},
                        cv=cv_index.split(df_tune),
                        refit=False,
                        scoring=my.d_scoring[TARGET_TYPE],
                        return_train_score=True,
                        n_jobs=my.n_jobs)
           .fit(X=X_binned,
                y=df_tune["cnt_" + TARGET_TYPE]))
    (hms_plot.ValidationPlotter(x_var="alpha", color_var="l1_ratio", show_gen_gap=True)
     .plot(fit.cv_results_, metric="rmse"))


# --- Random Forest ----------------------------------------------------------------------------------------------------

fit = (GridSearchCV(RandomForestRegressor() if TARGET_TYPE == "REGR" else
                    RandomForestClassifier(),
                    {"n_estimators": [10, 20, 100],
                     "max_features": [x for x in range(1, nume_standard.size + cate_standard.size, 5)]},
                    cv=cv_index.split(df_tune),
                    refit=False,
                    scoring=my.d_scoring[TARGET_TYPE],
                    return_train_score=True,
                    # use_warm_start=["n_estimators"],
                    n_jobs=my.n_jobs)
       .fit(X=X_standard,
            y=df_tune["cnt_" + TARGET_TYPE]))
(hms_plot.ValidationPlotter(x_var="n_estimators", color_var="max_features", show_gen_gap=True)
 .plot(fit.cv_results_, metric="rmse" if TARGET_TYPE == "REGR" else "auc"))


# --- XGBoost ----------------------------------------------------------------------------------------------------------
#%%
start = time.time()
fit = (my.GridSearchCV_xlgb(xgb.XGBRegressor(verbosity=0) if TARGET_TYPE == "REGR" else
                            xgb.XGBClassifier(verbosity=0),
                            {"n_estimators": [x for x in range(600, 4100, 500)], "learning_rate": [0.01],
                             "max_depth": [3, 6], "min_child_weight": [5]},
                            cv=cv_index.split(df_tune),
                            refit=False,
                            scoring=my.d_scoring[TARGET_TYPE],
                            return_train_score=True,
                            n_jobs=1)
       .fit(X=X_standard,
            y=df_tune["cnt_" + TARGET_TYPE]))
print(time.time() - start)
(hms_plot.ValidationPlotter(x_var="n_estimators", color_var="max_depth", column_var="min_child_weight",
                            show_gen_gap=True)
 .plot(fit.cv_results_, metric="rmse" if TARGET_TYPE == "REGR" else "auc"))
#%%

# --- LightGBM ---------------------------------------------------------------------------------------------------------
 
# Indices of categorical variables (for Lightgbm)
i_cate_standard = [i for i in range(len(nume_standard)) if nume_standard[i].endswith("_ENCODED")]
i_cate_encoded = [i for i in range(len(nume_encoded)) if nume_encoded[i].endswith("_ENCODED")]

# Fit
start = time.time()
fit = (my.GridSearchCV_xlgb(lgbm.LGBMRegressor() if TARGET_TYPE == "REGR" else
                            lgbm.LGBMClassifier(),
                            {"n_estimators": [x for x in range(100, 3100, 500)], "learning_rate": [0.01],
                             "num_leaves": [8, 16, 32], "min_child_samples": [5]},
                            cv=cv_index.split(df_tune),
                            refit=False,
                            scoring=my.d_scoring[TARGET_TYPE],
                            return_train_score=True,
                            n_jobs=my.n_jobs)
       .fit(X_standard,
            categorical_feature=i_cate_standard,
            y=df_tune["cnt_" + TARGET_TYPE]))
print(time.time() - start)
(hms_plot.ValidationPlotter(x_var="n_estimators", color_var="num_leaves", column_var="min_child_samples",
                            show_gen_gap=True)
 .plot(fit.cv_results_, metric="rmse" if TARGET_TYPE == "REGR" else "auc"))


# --- DeepL ---------------------------------------------------------------------------------------------------------

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
        model.compile(loss="binary_crossentropy", optimizer=optimizers.RMSprop(lr=lr), metrics=["accuracy"])
    elif TARGET_TYPE == "MULTICLASS":
        model.add(Dense(output_dim, activation='softmax',
                        kernel_regularizer=l2(lambdah) if lambdah is not None else None))
        model.compile(loss="categorical_crossentropy", optimizer=optimizers.RMSprop(lr=lr),
                      metrics=["accuracy"])
    else:
        model.add(Dense(1, activation='linear',
                        kernel_regularizer=l2(lambdah) if lambdah is not None else None))
        model.compile(loss="mean_squared_error", optimizer=optimizers.RMSprop(lr=lr),
                      metrics=["mean_squared_error"])

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
                                    output_dim=1 if TARGET_TYPE == "CLASS" else df_tune["cnt_" + TARGET_TYPE].nunique(),
                                    TARGET_TYPE=TARGET_TYPE,
                                    verbose=0),
                    {"size": ["10", "10-10", "20"],
                     "lambdah": [1e-8], "dropout": [None],
                     "batch_size": [40], "lr": [1e-3],
                     "batch_normalization": [True],
                     "activation": ["relu", "elu"],
                     "epochs": [2, 7, 15]},
                    cv=cv_index.split(df_tune),
                    refit=False,
                    scoring=my.d_scoring[TARGET_TYPE],
                    return_train_score=True,
                    n_jobs=my.n_jobs)
       .fit(X_standard,
            y=(pd.get_dummies(df_tune["cnt_" + TARGET_TYPE]) if TARGET_TYPE == "MULTICLASS" else 
               df_tune["cnt_" + TARGET_TYPE])))
(hms_plot.ValidationPlotter(x_var="epochs", color_var="lambdah", column_var="activation", row_var="size",
                            show_gen_gap=True)
 .plot(fit.cv_results_, metric="rmse" if TARGET_TYPE == "REGR" else "auc"))



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
                           cv=ShuffleSplit(1, 0.2, random_state=999),  # just 1-fold for tuning
                           refit="spear" if TARGET_TYPE == "REGR" else "auc",
                           scoring=my.d_scoring[TARGET_TYPE],
                           return_train_score=False,
                           n_jobs=my.n_jobs),
    X=X_binned,
    y=df_tune["cnt_" + TARGET_TYPE],
    cv=cv_5foldsep.split(df_tune),
    scoring=my.d_scoring[TARGET_TYPE],
    return_train_score=False,
    n_jobs=my.n_jobs)
df_modelcomp_result = df_modelcomp_result.append(pd.DataFrame.from_dict(cvresults).reset_index()
                                                 .assign(model="ElasticNet"),
                                                 ignore_index=True)

# Xgboost
cvresults = cross_validate(
    estimator=my.GridSearchCV_xlgb(
        xgb.XGBRegressor(verbosity=0) if TARGET_TYPE == "REGR" else 
        xgb.XGBClassifier(verbosity=0),
        {"n_estimators": [x for x in range(2000, 2001, 1)], "learning_rate": [0.01],
         "max_depth": [3], "min_child_weight": [5]},
        cv=ShuffleSplit(1, 0.2, random_state=999),  # just 1-fold for tuning
        refit="spear" if TARGET_TYPE == "REGR" else "auc",
        scoring=my.d_scoring[TARGET_TYPE],
        return_train_score=False,
        n_jobs=my.n_jobs),
    X=X_standard,
    y=df_tune["cnt_" + TARGET_TYPE],
    cv=cv_5foldsep.split(df_tune),
    scoring=my.d_scoring[TARGET_TYPE],
    return_train_score=False,
    n_jobs=my.n_jobs)
df_modelcomp_result = df_modelcomp_result.append(pd.DataFrame.from_dict(cvresults).reset_index()
                                                 .assign(model="XGBoost"),
                                                 ignore_index=True)

# Lgbm
cvresults = cross_validate(
    estimator=my.GridSearchCV_xlgb(
        lgbm.LGBMRegressor(verbosity=0) if TARGET_TYPE == "REGR" else
        lgbm.LGBMClassifier(verbosity=0),
        {"n_estimators": [x for x in range(500, 501, 1)], "learning_rate": [0.01],
         "num_leaves": [32], "min_child_weight": [5]},
        cv=ShuffleSplit(1, 0.2, random_state=999),  # just 1-fold for tuning
        refit="spear" if TARGET_TYPE == "REGR" else "auc",
        scoring=my.d_scoring[TARGET_TYPE],
        return_train_score=False,
        n_jobs=my.n_jobs),
    X=X_standard,
    y=df_tune["cnt_" + TARGET_TYPE],
    fit_params=dict(categorical_feature=i_cate_standard),
    cv=cv_5foldsep.split(df_tune),
    scoring=my.d_scoring[TARGET_TYPE],
    return_train_score=False,
    n_jobs=my.n_jobs)
df_modelcomp_result = df_modelcomp_result.append(pd.DataFrame.from_dict(cvresults).reset_index()
                                                 .assign(model="Lgbm"),
                                                 ignore_index=True)


# --- Plot model comparison ------------------------------------------------------------------------------

metric = "rmse" if TARGET_TYPE == "REGR" else "auc"
my.plot_modelcomp(df_modelcomp_result.rename(columns={"index": "run", "test_" + metric: metric}),
                  scorevar=metric,
                  pdf=my.plotloc + "model_comparison.pdf")


########################################################################################################################
# Learning curve for winner algorithm
########################################################################################################################

# Calc learning curve
n_train, score_train, score_test, time_train, time_test = learning_curve(
    estimator=my.GridSearchCV_xlgb(
        xgb.XGBRegressor(verbosity=0) if TARGET_TYPE == "REGR" else
        xgb.XGBClassifier(verbosity=0),
        {"n_estimators": [2000], "learning_rate": [0.01],
         "max_depth": [3], "min_child_weight": [5]},
        cv=ShuffleSplit(1, 0.2, random_state=999),  # just 1-fold for tuning
        refit="spear" if TARGET_TYPE == "REGR" else "auc",
        scoring=my.d_scoring[TARGET_TYPE],
        return_train_score=False,
        n_jobs=my.n_jobs),
    X=X_standard,
    y=df_tune["cnt_" + TARGET_TYPE],
    train_sizes=np.arange(0.1, 1.1, 0.2),
    cv=cv_5fold.split(df_tune),
    scoring=my.d_scoring[TARGET_TYPE]["spear" if TARGET_TYPE == "REGR" else "auc"],
    return_times=True,
    n_jobs=my.n_jobs)

# Plot it
hms_plot.LearningPlotter().plot(n_train, score_train, score_test, time_train,
                                file_path=my.plotloc + "learningCurve.pdf")
