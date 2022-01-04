########################################################################################################################
# Initialize: Packages, functions, parameter
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
from swifter import swifter  # ATTENTION: remove swifter calls below if running into error when using
from category_encoders import target_encoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, ShuffleSplit, PredefinedSplit

# Custom functions and classes
import utils_plots as up

# Settings
import settings as s


# --- Parameter --------------------------------------------------------------------------------------------------------

# Plot
PLOT = False
%matplotlib
plt.ioff() 
# %matplotlib | %matplotlib qt | %matplotlib inline  # activate standard/inline window
# plt.ioff() | plt.ion()  # stop/start standard window
# plt.plot(range(10), range(10))

# Constants
TARGET_TYPES = ["REGR", "CLASS", "MULTICLASS"]
MISSPCT_THRESHOLD = 0.95
VARPERF_THRESHOLD_DATADRIFT = 0.53
TOOMANY_THRESHOLD = 5



########################################################################################################################
# ETL
########################################################################################################################

# --- Read data and adapt  ---------------------------------------------------------------------------------------------

# Read data and adapt to be more readable
df_orig = (pd.read_csv(s.DATALOC + "hour.csv", parse_dates=["dteday"])
           .replace({"season": {1: "1_winter", 2: "2_spring", 3: "3_summer", 4: "4_fall"},
                     "yr": {0: "2011", 1: "2012"},
                     "holiday": {0: "No", 1: "Yes"},
                     "workingday": {0: "No", 1: "Yes"},
                     "weathersit": {1: "1_clear", 2: "2_misty", 3: "3_light rain", 4: "4_heavy rain"}})
           .assign(weekday=lambda x: x["weekday"].astype("str") + "_" + x["dteday"].dt.day_name().str.slice(0, 3),
                   mnth=lambda x: x["mnth"].astype("str").str.zfill(2),
                   hr=lambda x: x["hr"].astype("str").str.zfill(2))
           .assign(temp=lambda x: x["temp"] * 47 - 8,  # original data is scaled
                   atemp=lambda x: x["atemp"] * 66 - 16,  # dito
                   windspeed=lambda x: x["windspeed"] * 67)  # original data is scaled to max
           .assign(kaggle_fold=lambda x: np.where(x["dteday"].dt.day >= 20, "test", "train")))

# Create some artifacts helping to illustrate important concepts
df_orig["high_card"] = df_orig["hum"].astype(str)  # high cardinality categorical variable
df_orig["weathersit"] = df_orig["weathersit"].where(df_orig["weathersit"] != "4_heavy rain", np.nan)  # some missings
df_orig["holiday"] = np.where(np.random.choice(range(10), len(df_orig)) == 0, np.nan, df_orig["holiday"])
df_orig["windspeed"] = df_orig["windspeed"].where(df_orig["windspeed"] != 0, other=np.nan)  # some missings

# Create artificial targets
df_orig["cnt_REGR"] = np.log(df_orig["cnt"] + 1)
df_orig["cnt_CLASS"] = pd.qcut(df_orig["cnt"], q=[0, 0.8, 1], labels=["0_low", "1_high"]).astype(str)
df_orig["cnt_MULTICLASS"] = pd.qcut(df_orig["cnt"], q=[0, 0.8, 0.95, 1],
                                    labels=["0_low", "1_high", "2_very_high"]).astype(str)

'''
# Check some stuff
df_orig.dtypes
df_orig.describe()
up.value_counts(df_orig, dtypes=["object"]).T

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
df_orig["cnt"].plot.hist(bins=50, ax=ax[0])
df_orig["cnt"].hist(density=True, cumulative=True, bins=50, histtype="step", ax=ax[1])
np.log(df_orig["cnt"]).plot.hist(bins=50, ax=ax[2])

fig, ax = plt.subplots(1,1)
up.plot_feature_target(ax, df_orig["windspeed"], df_orig["cnt_CLASS"])
'''

# "Save" original data
df_orig.to_csv(s.DATALOC + "df_orig.csv", index=False)  # save as "original" for productive train&score example
df = df_orig.copy()


# --- Get metadata information -----------------------------------------------------------------------------------------

df_meta = pd.read_excel(s.DATALOC + "datamodel_bikeshare.xlsx", header=1, engine='openpyxl')

# Check difference of metainfo to data
print(up.diff(df.columns, df_meta["variable"]))
print(up.diff(df_meta.query("category == 'orig'").variable, df.columns))

# Subset on data that is "ready" to get processed
df_meta_sub = df_meta.query("status in ['ready']").reset_index()


# --- Feature engineering ----------------------------------------------------------------------------------------------

df["day_of_month"] = df['dteday'].dt.day.astype("str").str.zfill(2)

# Check metadata again
print(up.diff(df_meta_sub["variable"], df.columns))


# --- Define train/test/util-fold --------------------------------------------------------------------------------------

df["fold"] = np.where(df.index.isin(df.query("kaggle_fold == 'train'")
                                    .sample(frac=0.1, random_state=42).index.values),
                      "util", df["kaggle_fold"])
# df["fold_num"] = df["fold"].replace({"train": 0, "util": 0, "test": 1})  # Used for pedicting test data


########################################################################################################################
# Numeric features: Explore and adapt
########################################################################################################################

# --- Define numeric features ------------------------------------------------------------------------------------------

nume = df_meta_sub.query("type == 'nume'")["variable"].values.tolist()
df[nume] = df[nume].apply(lambda x: pd.to_numeric(x))
df[nume].describe()


# --- Missings + Outliers + Skewness -----------------------------------------------------------------------------------

# Remove features with too many missings
misspct = df[nume].isnull().mean().round(3)
print("misspct:\n", misspct.sort_values(ascending=False))
remove = misspct[misspct > MISSPCT_THRESHOLD].index.values.tolist()
nume = up.diff(nume, remove)

# Plot untransformed features -> check for outliers and skewness
df[nume].describe()
start = time.time()
for TARGET_TYPE in TARGET_TYPES:
    if PLOT:
        _ = up.plot_l_calls(pdf_path=f"{s.PLOTLOC}1__distr_nume_orig__{TARGET_TYPE}.pdf",
                            l_calls=[(up.plot_feature_target,
                                      dict(feature=df[feature],
                                           target=df["cnt_" + TARGET_TYPE]))
                                     for feature in nume])
print(time.time() - start)

# Winsorize
df[nume] = up.Winsorize(lower_quantile=None, upper_quantile=0.99).fit_transform(df[nume])

# Log-Transform
tolog = ["hum"]
if len(tolog):
    df[up.add(tolog, "_LOG")] = df[tolog].apply(lambda x: np.log(x - min(0, np.min(x)) + 1))  # start always at 0
    nume = [x + "_LOG" if (x in tolog) else (x) for x in nume]  # adapt metadata (keep order)


# --- Create categorical (binned) equivalents for all numeric features (for linear models to mimic non-linearity) ------

nume_BINNED = up.add(nume, "_BINNED")
df[nume_BINNED] = df[nume].apply(lambda x: up.bin(x, precision=1))

# Convert missings to own level ("(Missing)")
df[nume_BINNED] = df[nume_BINNED].fillna("(Missing)")
print(up.value_counts(df[nume_BINNED], 6))

# Get binned variables with just 1 bin (removed later)
onebin = (df[nume_BINNED].nunique() == 1)[lambda x: x].index.values.tolist()
print(onebin)


# --- Final feature information ----------------------------------------------------------------------------------------

for TARGET_TYPE in TARGET_TYPES:
    #TARGET_TYPE = "REGR"

    # Univariate variable performances
    varperf_nume = df[nume + nume_BINNED].swifter.progress_bar(False).apply(
        lambda x: (up.variable_performance(feature=x, 
                                           target=df["cnt_" + TARGET_TYPE],
                                           splitter=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
                                           scorer=up.D_SCORER[TARGET_TYPE]["spear" if TARGET_TYPE == "REGR" 
                                                                           else "auc"])))
    print(varperf_nume.sort_values(ascending=False))

    # Plot
    if PLOT:
        _ = up.plot_l_calls(pdf_path=f"{s.PLOTLOC}1__distr_nume__{TARGET_TYPE}.pdf",
                            l_calls=[(up.plot_feature_target,
                                      dict(feature=df[feature], target=df["cnt_" + TARGET_TYPE],
                                           title=f"{feature} (VI:{varperf_nume[feature]: 0.2f})",
                                           regplot_type="lowess",
                                           add_miss_info=True if feature in nume else False))
                                     for feature in up.interleave(nume, nume_BINNED)])


# --- Removing variables (inlcuding correlation analysis) --------------------------------------------------------------

# Remove leakage features
remove = ["xxx", "xxx"]
nume = up.diff(nume, remove)

# Remove highly/perfectly correlated (the ones with less NA!)
df[nume].describe()
_ = up.plot_l_calls(pdf_path=s.PLOTLOC + "1__corr_nume.pdf", n_rows=1, n_cols=1, figsize=(6, 6),
                    l_calls=[(up.plot_corr,
                              dict(df=df[nume], method="spearman", cutoff=0))])
remove = ["atemp"]
nume = up.diff(nume, remove)
''' Alternative
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
up.plot_corr(ax, df=df[nume], method="spearman", cutoff=0)
fig.tight_layout()
fig.savefig(s.PLOTLOC + "1__corr_nume.pdf")
'''


# --- Detect data drift (time/fold depedency of features) --------------------------------------------------------------

# HINT: In case of having a detailed date variable, this can be used as regression target here as well!

# Univariate variable importance 
varperf_nume_fold = df[nume].swifter.progress_bar(False).apply(
    lambda x: up.variable_performance(feature=x, 
                                      target=df["fold"],
                                      splitter=up.InSampleSplit(),
                                      scorer=up.D_SCORER["CLASS"]["auc"]))
print(varperf_nume_fold.sort_values(ascending=False))

# Plot only variables with highest importance
nume_toplot = varperf_nume_fold[varperf_nume_fold > VARPERF_THRESHOLD_DATADRIFT].index.values
if len(nume_toplot):
    if PLOT:
        _ = up.plot_l_calls(pdf_path=f"{s.PLOTLOC}1__distr_nume_folddep__{TARGET_TYPE}.pdf",
                            l_calls=[(up.plot_feature_target,
                                      dict(feature=df[feature], target=df["fold"],
                                           title=f"{feature} (VI:{varperf_nume_fold[feature]: 0.2f})",))
                                     for feature in nume_toplot])


# --- Create missing indicator and impute feature missings--------------------------------------------------------------

miss = df[nume].isnull().any()[lambda x: x].index.values.tolist()
df[up.add("MISS_", miss)] = pd.DataFrame(np.where(df[miss].isnull(), "No", "Yes"))
df[up.add("MISS_", miss)].describe()

# Impute missings
df[miss] = SimpleImputer(strategy="median").fit_transform(df[miss])
df[miss].isnull().sum()


########################################################################################################################
# Categorical features: Explore and adapt
########################################################################################################################

# --- Define categorical features --------------------------------------------------------------------------------------

cate = df_meta_sub.query("type == 'cate'")["variable"].values.tolist()
df[cate] = df[cate].astype("str")
df[cate].describe()


# --- Handling factor values -------------------------------------------------------------------------------------------

# Map missings to own level
df[cate] = df[cate].fillna("(Missing)").replace("nan", "(Missing)")
df[cate].describe()

# Create ordinal and binary-encoded features
# ATTENTION: Usually this processing needs special adaption depending on the data
ordi = ["hr", "day_of_month", "mnth", "yr"]
df[up.add(ordi, "_ENCODED")] = df[ordi].apply(lambda x: pd.to_numeric(x))
yesno = ["workingday"] + ["MISS_" + x for x in miss]  # binary features
df[up.add(yesno, "_ENCODED")] = df[yesno].apply(lambda x: x.map({"No": 0, "Yes": 1}))

# Create target-encoded features for nominal variables
nomi = up.diff(cate, ordi + yesno)
df_util = df.query("fold == 'util'").reset_index(drop=True)
df[up.add(nomi, "_ENCODED")] = target_encoder.TargetEncoder().fit(df_util[nomi],
                                                                  df_util["cnt_REGR"]).transform(df[nomi])

# Get "too many members" columns and lump levels
levinfo = df[cate].nunique().sort_values(ascending=False)  # number of levels
print(levinfo)
toomany = levinfo[levinfo > TOOMANY_THRESHOLD].index.values
print(toomany)
toomany = up.diff(toomany, ["hr", "mnth", "weekday"])  # set exception for important variables
if len(toomany):
    df[toomany] = up.Collapse(n_top=TOOMANY_THRESHOLD).fit_transform(df[toomany])


# --- Final variable information ---------------------------------------------------------------------------------------

for TARGET_TYPE in TARGET_TYPES:
    #TARGET_TYPE = "CLASS"

    # Univariate variable importance
    varperf_cate = df[cate + up.add("MISS_", miss)].swifter.progress_bar(False).apply(
        lambda x: (up.variable_performance(feature=x, 
                                           target=df["cnt_" + TARGET_TYPE],
                                           splitter=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
                                           scorer=up.D_SCORER[TARGET_TYPE]["spear" if TARGET_TYPE == "REGR" 
                                                                           else "auc"])))
    print(varperf_cate.sort_values(ascending=False))

    # Check
    if PLOT:
        _ = up.plot_l_calls(pdf_path=f"{s.PLOTLOC}1__distr_cate__{TARGET_TYPE}.pdf",
                            l_calls=[(up.plot_feature_target,
                                      dict(feature=df[feature],
                                           target=df["cnt_" + TARGET_TYPE],
                                           title=f"{feature} (VI:{varperf_cate[feature]: 0.2f})",
                                           add_miss_info=False))
                                     for feature in cate + up.add("MISS_", miss)])


# --- Removing variables (inlcuding correlation analysis) --------------------------------------------------------------

# Remove leakage variables
cate = up.diff(cate, ["xxx"])
toomany = up.diff(toomany, ["xxx"])

# Remove highly/perfectly correlated (the ones with less levels!)
_ = up.plot_l_calls(pdf_path=s.PLOTLOC + "1__corr_cate.pdf", n_rows=1, n_cols=1, figsize=(8, 6),
                    l_calls=[(up.plot_corr,
                              dict(df=df[cate + up.add("MISS_", miss)], method="cramersv", cutoff=0))])
'''
# Interscale correlation!
_ = up.plot_l_calls(pdf_path=s.PLOTLOC + "1__corr.pdf", n_rows=1, n_cols=1, figsize=(12, 12),
                    l_calls=[(up.plot_corr,
                              dict(df=df[cate + up.add("MISS_", miss) + up.add(nume,"_BINNED")], 
                                   method="cramersv", cutoff=0))])
'''


# --- Time/fold depedency ----------------------------------------------------------------------------------------------

# Hint: In case of having a detailed date variable it can be used as regression target here as well!
# Univariate variable importance
varperf_cate_fold = df[cate + up.add("MISS_", miss)].swifter.progress_bar(False).apply(
    lambda x: (up.variable_performance(feature=x,
                                       target=df["fold"],
                                       splitter=up.InSampleSplit(),
                                       scorer=up.D_SCORER["CLASS"]["auc"])))

# Plot: only variables with with highest importance
cate_toplot = varperf_cate_fold[varperf_cate_fold > VARPERF_THRESHOLD_DATADRIFT].index.values
if len(cate_toplot):
    if PLOT:
        _ = up.plot_l_calls(pdf_path=f"{s.PLOTLOC}1__distr_cate_folddep__{TARGET_TYPE}.pdf",
                            l_calls=[(up.plot_feature_target,
                                      dict(feature=df[feature],
                                           target=df["fold"],
                                           title=f"{feature} (VI:{varperf_cate_fold[feature]: 0.2f})"))
                                     for feature in cate_toplot])


########################################################################################################################
# Prepare final data
########################################################################################################################

# --- Add numeric target -----------------------------------------------------------------------------------------------

# Target mostly needs to be numeric
df["cnt_REGR_num"] = df["cnt_REGR"]
df["cnt_CLASS_num"] = df["cnt_CLASS"].str.slice(0, 1).astype("int")
df["cnt_MULTICLASS_num"] = df["cnt_MULTICLASS"].str.slice(0, 1).astype("int")


# --- Define final features --------------------------------------------------------------------------------------------

# Standard: can be used together by all algorithms
nume_standard = nume + up.add(toomany, "_ENCODED")
cate_standard = cate + up.add("MISS_", miss)

# Binned: can be used by elasticnet (without any numeric features) to mimic non-linear numeric effects
features_binned = up.diff(up.add(nume, "_BINNED"), onebin) + cate

# Encoded: can be used as complete feature-set for deeplearning (as bad with one-hot encoding)
# or lightgbm (with additionally denoting encoded features as "categorical")
features_encoded = np.unique(nume + up.add(cate, "_ENCODED") + ["MISS_" + x + "_ENCODED" for x in miss]).tolist()

# Check again
all_features = np.unique(nume_standard + cate_standard + features_binned + features_encoded).tolist()
up.diff(all_features, df.columns.values.tolist())
up.diff(df.columns.values.tolist(), all_features)


# --- Remove "burned" data -----------------------------------------------------------------------------------------------

df = df.query("fold != 'util'").reset_index(drop=True)


# --- Save image -------------------------------------------------------------------------------------------------------

# Clean up
plt.close(fig="all")  # plt.close(plt.gcf())
del df_orig

# Serialize
with open(s.DATALOC + "1_explore.pkl", "wb") as file:
    pickle.dump({"df": df,
                 "nume_standard": nume_standard,
                 "cate_standard": cate_standard,
                 "features_binned": features_binned,
                 "features_encoded": features_encoded},
                file)
