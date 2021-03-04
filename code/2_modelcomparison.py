# ######################################################################################################################
#  Initialize: Packages, functions, parameters, data-loading
# ######################################################################################################################

# General libraries, parameters and functions
from initialize import *
# sys.path.append(os.getcwd() + "\\code")  # not needed if code is marked as "source" in pycharm

# Specific libraries
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # , GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, SGDRegressor, LogisticRegression  # , ElasticNet
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.regularizers import l2
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
#  from sklearn.tree import DecisionTreeRegressor, plot_tree , export_graphviz

# Main parameter
TYPE = "class"

# Specific parameters
n_jobs = 4

# Load results from exploration
df = nume_standard = cate_standard = nume_binned = cate_binned = nume_encoded = cate_encoded = None
with open(dataloc + "1_explore.pkl", "rb") as file:
    d_pick = pickle.load(file)
for key, val in d_pick.items():
    exec(key + "= val")
    
# Adapt targets
df["cnt_class"] = df["cnt_class"].str.slice(0,1).astype("int")

# Scale "nume_enocded" features for DL (Tree-based are not influenced by this Trafo)
df[nume_encoded] = MinMaxScaler().fit_transform(df[nume_encoded])
df[nume_encoded].describe()


# ######################################################################################################################
# # Test an algorithm (and determine parameter grid)
# ######################################################################################################################

# --- Sample data ------------------------------------------------------------------------------------------------------

# Undersample only training data (take all but n_maxpersample at most)
under_samp = Undersample(n_max_per_level=3000)
df_tmp = under_samp.fit_transform(df.query("fold == 'train'").reset_index(drop=True),
                                  target="cnt_class")
b_all = under_samp.b_all
b_sample = under_samp.b_sample
print(b_all, b_sample)
df_tune = (pd.concat([df_tmp, df.query("fold == 'test'").reset_index(drop=True)], sort=False)
           .reset_index(drop=True))
df_tune.groupby("fold")["cnt_class"].describe()


# --- Define some splits -------------------------------------------------------------------------------------------

split_index = PredefinedSplit(df_tune["fold"].map({"train": -1, "test": 0}).values)
split_5fold = KFold(5, shuffle=True, random_state=42)
split_my1fold_cv = TrainTestSep(1)
split_my5fold_cv = TrainTestSep(5)
split_my5fold_boot = TrainTestSep(5, "bootstrap")

'''
# Test a split
df_tune["fold"].value_counts()
split = split_my5fold_boot.split(df_tune)
i_train, i_test = next(split)
df_tune["fold"].iloc[i_train].describe()
df_tune["fold"].iloc[i_test].describe()
i_test.sort()
i_test
'''


# --- Fits -----------------------------------------------------------------------------------------------------------

# Lasso / Elastic Net
fit = (GridSearchCV(SGDRegressor(penalty="ElasticNet", warm_start=True) if TYPE == "regr" else
                    SGDClassifier(loss="log", penalty="ElasticNet", warm_start=True),  # , tol=1e-2
                    {"alpha": [2 ** x for x in range(-8, -20, -2)],
                     "l1_ratio": [0, 1]},
                    cv=split_my1fold_cv.split(df_tune),
                    refit=False,
                    scoring=d_scoring[TYPE],
                    return_train_score=True,
                    n_jobs=n_jobs)
       .fit(hms_preproc.MatrixConverter(to_sparse=True)
            .fit_transform(df_tune[np.append(nume_binned, cate_binned)]),
            #.fit_transform(df_tune[np.append(nume_standard, cate_standard)]),
            #.fit_transform(df_tune[np.append(nume_encoded, cate_encoded)]),
            df_tune["cnt_" + TYPE]))
(hms_plot.ValidationPlotter(x_var="alpha", color_var="l1_ratio", show_gen_gap=True)
 .plot(fit.cv_results_, metric="spear" if TYPE == "regr" else "auc"))
#pd.DataFrame(fit.cv_results_)

if TYPE in ["class", "multiclass"]:
    fit = (GridSearchCV(LogisticRegression(penalty="l1", fit_intercept=True, solver="liblinear"),
                        {"C": [2 ** x for x in range(2, -5, -1)]},
                        cv=split_my1fold_cv.split(df_tune),
                        refit=False,
                        scoring=d_scoring[TYPE],
                        return_train_score=True,
                        n_jobs=n_jobs)
           .fit(hms_preproc.MatrixConverter(to_sparse=True)
                .fit_transform(df_tune[np.append(nume_binned, cate_binned)]),
                df_tune["cnt_" + TYPE]))
    (hms_plot.ValidationPlotter(x_var="C", show_gen_gap=True)
     .plot(fit.cv_results_, metric="spear" if TYPE=="regr" else "auc"))
# -> keep l1_ratio=1 to have a full Lasso


# Random Forest
fit = (GridSearchCV(RandomForestRegressor() if TYPE == "regr" else 
                    RandomForestClassifier(),
                    {"n_estimators": [10, 20],
                     "max_features": [x for x in range(1, nume_standard.size + cate_standard.size, 5)]},
                    cv=split_my1fold_cv.split(df_tune),
                    refit=False,
                    scoring=d_scoring[TYPE],
                    return_train_score=True,
                    # use_warm_start=["n_estimators"],
                    n_jobs=n_jobs)
       .fit(hms_preproc.MatrixConverter(to_sparse=True)
            .fit_transform(df_tune[np.append(nume_standard, cate_standard)]),
            df_tune["cnt_" + TYPE]))
(hms_plot.ValidationPlotter(x_var="n_estimators", color_var="max_features",
                            show_gen_gap=True)
 .plot(fit.cv_results_, metric="spear" if TYPE == "regr" else "auc"))
# -> keep around the recommended values: max_features = floor(sqrt(length(features)))


# XGBoost
start = time.time()
fit = (GridSearchCV_xlgb(xgb.XGBRegressor(verbosity=0) if TYPE == "regr" else 
                         xgb.XGBClassifier(verbosity=0),
                         {"n_estimators": [x for x in range(100, 4100, 500)], "learning_rate": [0.01],
                          "max_depth": [3, 6], "min_child_weight": [5]},
                         cv=split_my1fold_cv.split(df_tune),
                         refit=False,
                         scoring=d_scoring[TYPE],
                         return_train_score=True,
                         n_jobs=n_jobs)
       .fit((hms_preproc.MatrixConverter(to_sparse=True)
            #.fit_transform(df_tune[np.append(nume_standard, cate_standard)])),          
            #.fit_transform(df_tune[np.append(nume_binned, cate_binned)])),          
            .fit_transform(df_tune[np.append(nume_encoded, cate_encoded)])),
            df_tune["cnt_" + TYPE]))
print(time.time() - start)
(hms_plot.ValidationPlotter(x_var="n_estimators", color_var="max_depth", column_var="min_child_weight",
                            show_gen_gap=True)
 .plot(fit.cv_results_, metric="spear" if TYPE == "regr" else "auc"))
# -> keep around the recommended values: max_depth = 6, shrinkage = 0.01, n.minobsinnode = 10


# LightGBM
start = time.time()
fit = (GridSearchCV_xlgb(lgbm.LGBMRegressor() if TYPE == "regr" else 
                         lgbm.LGBMClassifier(),
                         {"n_estimators": [x for x in range(100, 3100, 500)], "learning_rate": [0.01],
                          "num_leaves": [8, 16, 32], "min_child_samples": [5]},
                         cv=split_my1fold_cv.split(df_tune),
                         refit=False,
                         scoring=d_scoring[TYPE],
                         return_train_score=True,
                         n_jobs=1)
       .fit(df_tune[nume_encoded], 
            categorical_feature=[x for x in nume_encoded.tolist() if "_ENCODED" in x],
       #.fit((hms_preproc.MatrixConverter(to_sparse=True)
            #.fit_transform(df_tune[np.append(nume_standard, cate_standard)])),
            #.fit_transform(df_tune[np.append(nume_encoded, cate_encoded)])),
            y = df_tune["cnt_" + TYPE]))
print(time.time() - start)
(hms_plot.ValidationPlotter(x_var="n_estimators", color_var="num_leaves", column_var="min_child_samples",
                            show_gen_gap=True)
 .plot(fit.cv_results_, metric="spear" if TYPE == "regr" else "auc"))


# DeepL

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

fit = (GridSearchCV(KerasRegressor(build_fn=keras_model,
                                   #input_dim=nume_encoded.size,
                                   input_dim=69,
                                   output_dim=1,
                                   target_type=TYPE,
                                   verbose=0) if TYPE == "regr" else
                    KerasClassifier(build_fn=keras_model,
                                    #input_dim=nume_encoded.size,
                                    input_dim=69,
                                    output_dim=1 if TYPE == "class" else target_labels.size,
                                    target_type=TYPE,
                                    verbose=0),
                    {"size": ["10", "10-10", "20"],
                     "lambdah": [1e-8], "dropout": [None],
                     "batch_size": [40], "lr": [1e-3],
                     "batch_normalization": [True],
                     "activation": ["relu", "elu"],
                     "epochs": [2, 5, 10, 15, 20]},
                    cv=split_my1fold_cv.split(df_tune),
                    refit=False,
                    scoring=d_scoring[TYPE],
                    return_train_score=True,
                    n_jobs=1)
       .fit((hms_preproc.MatrixConverter(to_sparse=True)
            # .fit_transform(df_tune[nume_encoded])),
             .fit_transform(df_tune[np.append(nume_standard, cate_standard)])),
             y = pd.get_dummies(df_tune["cnt_" + TYPE]) if TYPE == "multiclass" else df_tune["cnt_" + TYPE]))
(hms_plot.ValidationPlotter(x_var="epochs", color_var="lambdah", column_var="activation", row_var="size",
                            show_gen_gap=True)
 .plot(fit.cv_results_, metric="spear" if TYPE == "regr" else "auc"))



# ######################################################################################################################
# Simulation: compare algorithms
# ######################################################################################################################

# Basic data sampling
df_modelcomp = df_tune.copy()


# --- Run methods ------------------------------------------------------------------------------------------------------

df_modelcomp_result = pd.DataFrame()  # intialize

# Elastic Net
cvresults = cross_validate(
    estimator=GridSearchCV(SGDRegressor(penalty="ElasticNet", warm_start=True) if TYPE == "regr" else
                           SGDClassifier(loss="log", penalty="ElasticNet", warm_start=True),  # , tol=1e-2
                           {"alpha": [2 ** x for x in range(-4, -12, -1)],
                            "l1_ratio": [1]},
                           cv=ShuffleSplit(1, 0.2, random_state=999),  # just 1-fold for tuning
                           refit="spear" if TYPE == "regr" else "auc",
                           scoring=d_scoring[TYPE],
                           return_train_score=False,
                           n_jobs=n_jobs),
    X=hms_preproc.MatrixConverter(to_sparse=True).fit_transform(df_modelcomp[np.append(nume_binned, cate_binned)]),
    y=df_modelcomp["cnt_" + TYPE],
    cv=split_my5fold_cv.split(df_modelcomp),
    scoring=d_scoring[TYPE],
    return_train_score=False,
    n_jobs=n_jobs)
df_modelcomp_result = df_modelcomp_result.append(pd.DataFrame.from_dict(cvresults).reset_index()
                                                 .assign(model="ElasticNet"),
                                                 ignore_index=True)

# Xgboost
cvresults = cross_validate(
    estimator=GridSearchCV_xlgb(
        xgb.XGBRegressor(verbosity=0) if TYPE == "regr" else 
        xgb.XGBClassifier(verbosity=0),
        {"n_estimators": [x for x in range(2000, 2001, 1)], "learning_rate": [0.01],
         "max_depth": [3], "min_child_weight": [5]},
        cv=ShuffleSplit(1, 0.2, random_state=999),  # just 1-fold for tuning
        refit="spear" if TYPE == "regr" else "auc",
        scoring=d_scoring[TYPE],
        return_train_score=False,
        n_jobs=n_jobs),
    X=hms_preproc.MatrixConverter(to_sparse=True)
    .fit_transform(df_modelcomp[np.append(nume_standard, cate_standard)]),
    y=df_modelcomp["cnt_" + TYPE],
    cv=split_my5fold_cv.split(df_modelcomp),
    scoring=d_scoring[TYPE],
    return_train_score=False,
    n_jobs=n_jobs)
df_modelcomp_result = df_modelcomp_result.append(pd.DataFrame.from_dict(cvresults).reset_index()
                                                 .assign(model="XGBoost"),
                                                 ignore_index=True)


# --- Plot model comparison ------------------------------------------------------------------------------
metric = "spear" if TYPE == "regr" else "auc"
plot_modelcomp(df_modelcomp_result.rename(columns={"index": "run", "test_" + metric: metric}),
               scorevar=metric,
               pdf=plotloc + "model_comparison.pdf")


# ######################################################################################################################
# Learning curve for winner algorithm
# ######################################################################################################################

# Basic data sampling
df_lc = df_tune.copy()

# Calc learning curve
n_train, score_train, score_test, time_train, time_test = learning_curve(
    estimator=GridSearchCV_xlgb(
        xgb.XGBRegressor(verbosity=0) if TYPE == "regr" else 
        xgb.XGBClassifier(verbosity=0),
        {"n_estimators": [x for x in range(2000, 2001, 1)], "learning_rate": [0.01],
         "max_depth": [3], "min_child_weight": [5]},
        cv=ShuffleSplit(1, 0.2, random_state=999),  # just 1-fold for tuning
        refit="spear" if TYPE == "regr" else "auc",
        scoring=d_scoring[TYPE],
        return_train_score=False,
        n_jobs=n_jobs),
    X=hms_preproc.MatrixConverter(to_sparse=True).fit_transform(df_lc[np.append(nume_standard, cate_standard)]),
    y=df_lc["cnt_" + TYPE],
    train_sizes=np.arange(0.1, 1.1, 0.2),
    cv=split_my1fold_cv.split(df_lc),
    scoring=d_scoring[TYPE]["spear" if TYPE == "regr" else "auc"],
    return_times=True,
    n_jobs=n_jobs)

# Plot it
hms_plot.LearningPlotter().plot(n_train, score_train, score_test, time_train,
                                file_path=plotloc + "learningCurve.pdf")


