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
import importlib  # importlib.reload(my)
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
import my_tools as my


# --- Parameter --------------------------------------------------------------------------

# Main parameter
target_type = "class"

# Specific parameters
n_jobs = 4

# Locations
dataloc = "../data/"
plotloc = "../output/"

# Load results from exploration
df = nume_standard = cate_standard = cate_binned = nume_encoded = None
with open(dataloc + "1_explore.pkl", "rb") as file:
    d_pick = pickle.load(file)
for key, val in d_pick.items():
    exec(key + "= val")

# Adapt targets
df["cnt_class"] = df["cnt_class"].str.slice(0, 1).astype("int")
df["cnt_multiclass"] = df["cnt_multiclass"].str.slice(0, 1).astype("int")

# Scale "nume_enocded" features for DL (Tree-based are not influenced by this Trafo)
# df[nume_encoded] = MinMaxScaler().fit_transform(df[nume_encoded])
# df[nume_encoded].describe()



########################################################################################################################
# Prepare data
########################################################################################################################

# --- Sample data ------------------------------------------------------------------------------------------------------

# Undersample only training data (take all but n_maxpersample at most)
df.query("fold == 'train'")["cnt_" + target_type].value_counts()
if target_type == "regr":
    df_tmp = df.query("fold == 'train'").sample(n=3000)
else:
    b_all, b_sample, df_tmp = my.undersample(df.query("fold == 'train'"), target="cnt_" + target_type, 
                                             n_max_per_level=3000)
    print(b_all, b_sample)
df_tune = (pd.concat([df_tmp, df.query("fold == 'test'")], sort=False)
           .sample(frac=1)
           .reset_index(drop=True))
df_tune.groupby("fold")["cnt_" + target_type].describe()

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

split_index = PredefinedSplit(df_tune["fold"].map({"train": -1, "test": 0}).values)
split_5fold = KFold(5, shuffle=True, random_state=42)
split_my1fold_cv = my.TrainTestSep(1)
split_my5fold_cv = my.TrainTestSep(5)

'''
# Test a split
df_tune["fold"].value_counts()
split = split_my5fold_cv.split(df_tune)
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

fit = (GridSearchCV(SGDRegressor(penalty="ElasticNet", warm_start=True) if target_type == "regr" else
                    SGDClassifier(loss="log", penalty="ElasticNet", warm_start=True),  # , tol=1e-2
                    {"alpha": [2 ** x for x in range(-8, -20, -2)],
                     "l1_ratio": [0, 1]},
                    cv=split_my1fold_cv.split(df_tune),
                    refit=False,
                    scoring=my.d_scoring[target_type],
                    return_train_score=True,
                    n_jobs=1)
       .fit(X=X_binned,
            y=df_tune["cnt_" + target_type]))

# Plot: use metric="score" if scoring has only 1 metric
(hms_plot.ValidationPlotter(x_var="alpha", color_var="l1_ratio", show_gen_gap=True)
 .plot(fit.cv_results_, metric="spear" if target_type == "regr" else "auc"))  
# pd.DataFrame(fit.cv_results_)

# Usually better alternative
if target_type in ["class", "multiclass"]:
    fit = (GridSearchCV(LogisticRegression(penalty="l1", fit_intercept=True, solver="liblinear"),
                        {"C": [2 ** x for x in range(2, -5, -1)]},
                        cv=split_my1fold_cv.split(df_tune),
                        refit=False,
                        scoring=my.d_scoring[target_type],
                        return_train_score=True,
                        n_jobs=n_jobs)
           .fit(X=X_binned,
                y=df_tune["cnt_" + target_type]))
    (hms_plot.ValidationPlotter(x_var="C", show_gen_gap=True)
     .plot(fit.cv_results_, metric="auc"))
else:
    fit = (GridSearchCV(ElasticNet(),
                        {"alpha": [2 ** x for x in range(-8, -20, -2)],
                         "l1_ratio": [0, 1]},
                        cv=split_my1fold_cv.split(df_tune),
                        refit=False,
                        scoring=my.d_scoring[target_type],
                        return_train_score=True,
                        n_jobs=n_jobs)
           .fit(X=X_binned,
                y=df_tune["cnt_" + target_type]))
    (hms_plot.ValidationPlotter(x_var="alpha", color_var="l1_ratio", show_gen_gap=True)
     .plot(fit.cv_results_, metric="spear"))


# --- Random Forest ----------------------------------------------------------------------------------------------------

fit = (GridSearchCV(RandomForestRegressor() if target_type == "regr" else
                    RandomForestClassifier(),
                    {"n_estimators": [10, 20, 100],
                     "max_features": [x for x in range(1, nume_standard.size + cate_standard.size, 5)]},
                    cv=split_my1fold_cv.split(df_tune),
                    refit=False,
                    scoring=my.d_scoring[target_type],
                    return_train_score=True,
                    # use_warm_start=["n_estimators"],
                    n_jobs=n_jobs)
       .fit(X=X_standard,
            y=df_tune["cnt_" + target_type]))
(hms_plot.ValidationPlotter(x_var="n_estimators", color_var="max_features", show_gen_gap=True)
 .plot(fit.cv_results_, metric="spear" if target_type == "regr" else "auc"))


# --- XGBoost ----------------------------------------------------------------------------------------------------------

start = time.time()
fit = (my.GridSearchCV_xlgb(xgb.XGBRegressor(verbosity=0) if target_type == "regr" else
                            xgb.XGBClassifier(verbosity=0),
                            {"n_estimators": [x for x in range(600, 4100, 500)], "learning_rate": [0.01],
                             "max_depth": [3, 6], "min_child_weight": [5]},
                            cv=split_my1fold_cv.split(df_tune),
                            refit=False,
                            scoring=my.d_scoring[target_type],
                            return_train_score=True,
                            n_jobs=n_jobs)
       .fit(X=X_standard,
            y=df_tune["cnt_" + target_type]))
print(time.time() - start)
(hms_plot.ValidationPlotter(x_var="n_estimators", color_var="max_depth", column_var="min_child_weight",
                            show_gen_gap=True)
 .plot(fit.cv_results_, metric="spear" if target_type == "regr" else "auc"))


# --- LightGBM ---------------------------------------------------------------------------------------------------------
 
# Indices of categorical variables (for Lightgbm)
i_cate_standard = [i for i in range(len(nume_standard)) if nume_standard[i].endswith("_ENCODED")]
i_cate_encoded = [i for i in range(len(nume_encoded)) if nume_encoded[i].endswith("_ENCODED")]

# Fit
start = time.time()
fit = (my.GridSearchCV_xlgb(lgbm.LGBMRegressor() if target_type == "regr" else
                            lgbm.LGBMClassifier(),
                            {"n_estimators": [x for x in range(100, 3100, 500)], "learning_rate": [0.01],
                             "num_leaves": [8, 16, 32], "min_child_samples": [5]},
                            cv=split_my1fold_cv.split(df_tune),
                            refit=False,
                            scoring=my.d_scoring[target_type],
                            return_train_score=True,
                            n_jobs=n_jobs)
       .fit(X_standard,
            categorical_feature=i_cate_standard,
            y=df_tune["cnt_" + target_type]))
print(time.time() - start)
(hms_plot.ValidationPlotter(x_var="n_estimators", color_var="num_leaves", column_var="min_child_samples",
                            show_gen_gap=True)
 .plot(fit.cv_results_, metric="spear" if target_type == "regr" else "auc"))


# --- DeepL ---------------------------------------------------------------------------------------------------------

# Keras wrapper for Scikit
def keras_model(input_dim, output_dim, target_type,
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
    if target_type == "class":
        model.add(Dense(1, activation='sigmoid',
                        kernel_regularizer=l2(lambdah) if lambdah is not None else None))
        model.compile(loss="binary_crossentropy", optimizer=optimizers.RMSprop(lr=lr), metrics=["accuracy"])
    elif target_type == "multiclass":
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
                                   target_type=target_type,
                                   verbose=0) if target_type == "regr" else
                    KerasClassifier(build_fn=keras_model,
                                    input_dim=n_cols,
                                    output_dim=1 if target_type == "class" else df_tune["cnt_" + target_type].nunique(),
                                    target_type=target_type,
                                    verbose=0),
                    {"size": ["10", "10-10", "20"],
                     "lambdah": [1e-8], "dropout": [None],
                     "batch_size": [40], "lr": [1e-3],
                     "batch_normalization": [True],
                     "activation": ["relu", "elu"],
                     "epochs": [2, 7, 15]},
                    cv=split_my1fold_cv.split(df_tune),
                    refit=False,
                    scoring=my.d_scoring[target_type],
                    return_train_score=True,
                    n_jobs=1)
       .fit(X_standard,
            y=(pd.get_dummies(df_tune["cnt_" + target_type]) if target_type == "multiclass" else 
               df_tune["cnt_" + target_type])))
(hms_plot.ValidationPlotter(x_var="epochs", color_var="lambdah", column_var="activation", row_var="size",
                            show_gen_gap=True)
 .plot(fit.cv_results_, metric="spear" if target_type == "regr" else "auc"))



########################################################################################################################
# Simulation: compare algorithms
########################################################################################################################

# Placeholder for data sampling
df_modelcomp = df_tune


# --- Run methods ------------------------------------------------------------------------------------------------------

df_modelcomp_result = pd.DataFrame()  # intialize

# Elastic Net
cvresults = cross_validate(
    estimator=GridSearchCV(SGDRegressor(penalty="ElasticNet", warm_start=True) if target_type == "regr" else
                           SGDClassifier(loss="log", penalty="ElasticNet", warm_start=True),  # , tol=1e-2
                           {"alpha": [2 ** x for x in range(-4, -12, -1)],
                            "l1_ratio": [1]},
                           cv=ShuffleSplit(1, 0.2, random_state=999),  # just 1-fold for tuning
                           refit="spear" if target_type == "regr" else "auc",
                           scoring=my.d_scoring[target_type],
                           return_train_score=False,
                           n_jobs=n_jobs),
    X=X_binned,
    y=df_modelcomp["cnt_" + target_type],
    cv=split_my5fold_cv.split(df_modelcomp),
    scoring=my.d_scoring[target_type],
    return_train_score=False,
    n_jobs=n_jobs)
df_modelcomp_result = df_modelcomp_result.append(pd.DataFrame.from_dict(cvresults).reset_index()
                                                 .assign(model="ElasticNet"),
                                                 ignore_index=True)

# Xgboost
cvresults = cross_validate(
    estimator=my.GridSearchCV_xlgb(
        xgb.XGBRegressor(verbosity=0) if target_type == "regr" else 
        xgb.XGBClassifier(verbosity=0),
        {"n_estimators": [x for x in range(2000, 2001, 1)], "learning_rate": [0.01],
         "max_depth": [3], "min_child_weight": [5]},
        cv=ShuffleSplit(1, 0.2, random_state=999),  # just 1-fold for tuning
        refit="spear" if target_type == "regr" else "auc",
        scoring=my.d_scoring[target_type],
        return_train_score=False,
        n_jobs=n_jobs),
    X=X_standard,
    y=df_modelcomp["cnt_" + target_type],
    cv=split_my5fold_cv.split(df_modelcomp),
    scoring=my.d_scoring[target_type],
    return_train_score=False,
    n_jobs=n_jobs)
df_modelcomp_result = df_modelcomp_result.append(pd.DataFrame.from_dict(cvresults).reset_index()
                                                 .assign(model="XGBoost"),
                                                 ignore_index=True)


# --- Plot model comparison ------------------------------------------------------------------------------

metric = "spear" if target_type == "regr" else "auc"
my.plot_modelcomp(df_modelcomp_result.rename(columns={"index": "run", "test_" + metric: metric}),
                  scorevar=metric,
                  pdf=plotloc + "model_comparison.pdf")


# ######################################################################################################################
# Learning curve for winner algorithm
# ######################################################################################################################

# Basic data sampling
df_lc = df_tune
'''
X_standard = (ColumnTransformer([('nume', MinMaxScaler(), nume_standard),
                                 ('cate', OneHotEncoder(sparse=True, handle_unknown="ignore"), cate_standard)])
              .fit_transform(df_lc[np.append(nume_standard, cate_standard)]))
'''

# Calc learning curve
n_train, score_train, score_test, time_train, time_test = learning_curve(
    estimator=my.GridSearchCV_xlgb(
        xgb.XGBRegressor(verbosity=0) if target_type == "regr" else
        xgb.XGBClassifier(verbosity=0),
        {"n_estimators": [2000], "learning_rate": [0.01],
         "max_depth": [3], "min_child_weight": [5]},
        cv=ShuffleSplit(1, 0.2, random_state=999),  # just 1-fold for tuning
        refit="spear" if target_type == "regr" else "auc",
        scoring=my.d_scoring[target_type],
        return_train_score=False,
        n_jobs=n_jobs),
    X=X_standard,
    y=df_lc["cnt_" + target_type],
    train_sizes=np.arange(0.1, 1.1, 0.2),
    cv=split_5fold.split(df_lc),
    scoring=my.d_scoring[target_type]["spear" if target_type == "regr" else "auc"],
    return_times=True,
    n_jobs=n_jobs)

# Plot it
hms_plot.LearningPlotter().plot(n_train, score_train, score_test, time_train,
                                file_path=plotloc + "learningCurve.pdf")
