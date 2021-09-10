########################################################################################################################
# Initialize: Packages, functions, parameter
########################################################################################################################

# --- Packages --------------------------------------------------------------------------

# General
import numpy as np 
import pandas as pd
from pandas.core.indexes.api import all_indexes_same 
import swifter
import matplotlib.pyplot as plt
import pickle
from importlib import reload
import time

# Special
from category_encoders import target_encoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, ShuffleSplit, PredefinedSplit

# Custom functions and classes
import utils_plots as up

# Setting
import settings as sett


# --- Parameter --------------------------------------------------------------------------

# Plot 
plot = True
%matplotlib 
#%matplotlib qt / %matplotlib inline  # activate standard/inline window
plt.ioff()  # / plt.ion()  # stop/start standard window
#plt.plot(1, 1)

# Specific parameters 
TARGET_TYPES = ["REGR", "CLASS", "MULTICLASS"]


########################################################################################################################
# ETL
########################################################################################################################

# --- Read data and adapt to be more readable --------------------------------------------------------------------------

# Read and adapt

df_orig = (pd.read_csv(sett.dataloc + "hour.csv", parse_dates=["dteday"])
           .replace({"season": {1: "1_winter", 2: "2_spring", 3: "3_summer", 4: "4_fall"},
                     "yr": {0: "2011", 1: "2012"},
                     "holiday": {0: "No", 1: "Yes"},
                     "workingday": {0: "No", 1: "Yes"},
                     "weathersit": {1: "1_clear", 2: "2_misty", 3: "3_light rain", 4: "4_heavy rain"}})
           .assign(weekday=lambda x: x["weekday"].astype("str") + "_" + x["dteday"].dt.day_name().str.slice(0, 3),
                   mnth=lambda x: x["mnth"].astype("str").str.zfill(2),
                   hr=lambda x: x["hr"].astype("str").str.zfill(2))
           .assign(temp=lambda x: x["temp"] * 47 - 8,
                   atemp=lambda x: x["atemp"] * 66 - 16,
                   windspeed=lambda x: x["windspeed"] * 67)
           .assign(kaggle_fold=lambda x: np.where(x["dteday"].dt.day >= 20, "test", "train")))

# Create some artifacts
df_orig["high_card"] = df_orig["hum"].astype('str')  # high cardinality categorical variable
df_orig["weathersit"] = df_orig["weathersit"].where(df_orig["weathersit"] != "heavy rain", np.nan)  # some missings
df_orig["windspeed"] = df_orig["windspeed"].where(df_orig["windspeed"] != 0, other=np.nan)  # some missings

# Create artificial targets
df_orig["cnt_REGR"] = np.log(df_orig["cnt"] + 1)
df_orig["cnt_CLASS"] = pd.qcut(df_orig["cnt"], q=[0, 0.8, 1], labels=["0_low", "1_high"]).astype("object")
df_orig["cnt_MULTICLASS"] = pd.qcut(df_orig["cnt"], q=[0, 0.8, 0.95, 1],
                                    labels=["0_low", "1_high", "2_very_high"]).astype("object")



'''
# Check some stuff
df_orig.dtypes
df_orig.describe()
up.value_counts(df_orig, dtypes=["object"]).T
catname = "holiday"
(df_orig[catname].value_counts().astype("int").iloc[: 5].reset_index()
                       .rename(columns={"index": catname, catname: "#"})).T

fig, ax = plt.subplots(1, 3, figsize=(15,5))
df_orig["cnt"].plot.hist(bins=50, ax=ax[0])
df_orig["cnt"].hist(density=True, cumulative=True, bins=50, histtype="step", ax=ax[1])
np.log(df_orig["cnt"]).plot.hist(bins=50, ax=ax[2])
'''

# "Save" original data
df = df_orig.copy()


# --- Read metadata (Project specific) ---------------------------------------------------------------------------------

df_meta = pd.read_excel(sett.dataloc + "datamodel_bikeshare.xlsx", header=1, engine='openpyxl')

# Check
print(up.diff(df.columns, df_meta["variable"]))
print(up.diff(df_meta.query("category == 'orig'").variable, df.columns))

# Filter on "ready"
df_meta_sub = df_meta.query("status in ['ready']").reset_index()


# --- Feature engineering ----------------------------------------------------------------------------------------------

df["day_of_month"] = df['dteday'].dt.day.astype("str").str.zfill(2)

# Check again
print(up.diff(df_meta_sub["variable"], df.columns))


# --- Define train/test/util-fold --------------------------------------------------------------------------------------

df["fold"] = np.where(df.index.isin(df.query("kaggle_fold == 'train'")
                                    .sample(frac=0.1, random_state=42).index.values),
                      "util", df["kaggle_fold"])
#df["fold_num"] = df["fold"].replace({"train": 0, "util": 0, "test": 1})  # Used for pedicting test data



########################################################################################################################
# Numeric variables: Explore and adapt
########################################################################################################################

# --- Define numeric covariates ----------------------------------------------------------------------------------------

nume = df_meta_sub.loc[df_meta_sub["type"] == "nume", "variable"].values
df[nume] = df[nume].apply(lambda x: pd.to_numeric(x))
df[nume].describe()


# --- Create nominal variables for all numeric variables (for linear models)  -----------------------------------------

#df[nume + "_BINNED"] = (df[nume].swifter.apply(lambda x: (pd.qcut(x, 5)))
#                        .apply(lambda x: (("q" + x.cat.codes.astype("str") + " " + x.astype("str")))))
df[nume + "_BINNED"] = df[nume].apply(lambda x: up.bin(x, precision=1))


# Convert missings to own level ("(Missing)")
df[nume + "_BINNED"] = df[nume + "_BINNED"].fillna("(Missing)")
print(up.value_counts(df[nume + "_BINNED"], 6))

# Get binned variables with just 1 bin (removed later)
onebin = (nume + "_BINNED")[(df[nume + "_BINNED"].nunique() == 1).values]
print(onebin)


# --- Missings + Outliers + Skewness -----------------------------------------------------------------------------------

# Remove covariates with too many missings
misspct = df[nume].isnull().mean().round(3)  # missing percentage
print("misspct:\n", misspct.sort_values(ascending=False))  # view in descending order
remove = misspct[misspct > 0.95].index.values  # vars to remove
nume = up.diff(nume, remove)  # adapt metadata

# Check for outliers and skewness
df[nume].describe()
start = time.time()
for TARGET_TYPE in TARGET_TYPES:
    if plot:
        _ = up.plot_l_calls(pdf_path=sett.plotloc + "1__distr_nume_orig__" + TARGET_TYPE + ".pdf",
                            l_calls=[(up.plot_feature_target,
                                      dict(feature=df[feature], target=df["cnt_" + TARGET_TYPE], 
                                           smooth=30)) 
                                     for feature in nume])        
print(time.time() - start)
    
# Winsorize (hint: plot again before deciding for log-trafo)
df[nume] = up.Winsorize(lower_quantile=None, upper_quantile=0.99).fit_transform(df[nume])

# Log-Transform
tolog = np.array([], dtype="object")
if len(tolog):
    df[tolog + "_LOG_"] = df[tolog].apply(lambda x: np.log(x - min(0, np.min(x)) + 1))
    nume = np.where(np.isin(nume, tolog), nume + "_LOG_", nume)  # adapt metadata (keep order)
    df.rename(columns=dict(zip(tolog + "_BINNED", tolog + "_LOG_" + "_BINNED")), inplace=True)  # adapt binned version


# --- Final variable information ---------------------------------------------------------------------------------------

for TARGET_TYPE in TARGET_TYPES:
    #TARGET_TYPE = "CLASS"
    
    # Univariate variable performances
    varperf_nume = df[np.append(nume, nume + "_BINNED")].swifter.progress_bar(False).apply(lambda x: (
        up.variable_performance(x, df["cnt_" + TARGET_TYPE],
                                splitter=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
                                scorer=up.d_scoring[TARGET_TYPE]["spear" if TARGET_TYPE == "REGR" else "auc"])))
    print(varperf_nume.sort_values(ascending=False))
    
    # Plot
    if plot:
        _ = up.plot_l_calls(pdf_path=sett.plotloc + "1__distr_nume__" + TARGET_TYPE + ".pdf", 
                            l_calls=[(up.plot_feature_target,
                                      dict(feature=df[feature], target=df["cnt_" + TARGET_TYPE], 
                                           title=feature + " (VI: " + format(varperf_nume[feature], "0.2f") + ")",
                                           smooth=30)) 
                                     for feature in np.column_stack((nume, nume + "_BINNED")).ravel()])


# --- Removing variables -----------------------------------------------------------------------------------------------

# Remove leakage features
remove = ["xxx", "xxx"]
nume = up.diff(nume, remove)

# Remove highly/perfectly (>=98%) correlated (the ones with less NA!)
df[nume].describe()
_ = up.plot_l_calls(pdf_path=sett.plotloc + "1__corr_nume.pdf", n_rows=1, n_cols=1, figsize=(6, 6),
                    l_calls=[(up.plot_corr,
                              dict(df=df[nume], method="spearman", cutoff=0))])
remove = ["atemp"]
nume = up.diff(nume, remove)


# --- Time/fold depedency ----------------------------------------------------------------------------------------------

# Hint: In case of having a detailed date variable this can be used as regression target here as well!

# Univariate variable importance (again ONLY for non-missing observations!)
varperf_nume_fold = df[nume].swifter.apply(lambda x: up.variable_performance(x, df["fold"],
                                                                             splitter=up.InSampleSplit(),
                                                                             scorer=up.d_scoring["CLASS"]["auc"]))


# Plot: only variables with with highest importance
nume_toprint = varperf_nume_fold[varperf_nume_fold > 0.53].index.values
if len(nume_toprint):
    if plot:
        _ = up.plot_l_calls(pdf_path=sett.plotloc + "1__distr_nume_folddep" + TARGET_TYPE + ".pdf",
                            l_calls=[(up.plot_feature_target,
                                      dict(feature=df[feature], target=df["fold"],
                                           title=feature + " (VI: " + format(varperf_nume_fold[feature], "0.2f") + ")",
                                           smooth=30))
                                     for feature in nume_toprint])


# --- Missing indicator and imputation (must be done at the end of all processing)--------------------------------------

miss = nume[df[nume].isnull().any().values]
df["MISS_" + miss] = pd.DataFrame(np.where(df[miss].isnull(), "No", "Yes"))
df["MISS_" + miss].describe()

# Impute missings with randomly sampled value (or median, see below)
np.random.seed(123)
df[miss] = SimpleImputer(strategy="median").fit_transform(df[miss])
df[miss].isnull().sum()



########################################################################################################################
# Categorical  variables: Explore and adapt
########################################################################################################################

# --- Define categorical covariates ------------------------------------------------------------------------------------

# Categorical variables
cate = df_meta_sub.loc[df_meta_sub.type.isin(["cate"]), "variable"].values
df[cate] = df[cate].astype("str")
df[cate].describe()


# --- Handling factor values -------------------------------------------------------------------------------------------

# Convert "standard" features: map missings to own level
df[cate] = df[cate].fillna("(Missing)").replace("nan", "(Missing)")
df[cate].describe()

# Create ordinal and binary-encoded features
ordi = np.array(["hr", "mnth", "yr"], dtype="object")
df[ordi + "_ENCODED"] = df[ordi].apply(lambda x: pd.to_numeric(x))  # ordinal
yesno = np.concatenate([np.array(["holiday", "workingday"], dtype="object"), "MISS_" + miss])
df[yesno + "_ENCODED"] = df[yesno].apply(lambda x: x.map({"No": 0, "Yes": 1}))  # binary

# Create target-encoded features for nominal variables
nomi = up.diff(cate, np.concatenate([ordi, yesno]))
df_util = df.query("fold == 'util'").reset_index(drop=True)
df[nomi + "_ENCODED"] = target_encoder.TargetEncoder().fit(df_util[nomi], df_util["cnt_REGR"]).transform(df[nomi])
#df = df.query("fold != 'util'").reset_index(drop=True)  # remove utility data

# Get "too many members" columns and lump levels
topn_toomany = 5
levinfo = df[cate].nunique().sort_values(ascending=False)  # number of levels
print(levinfo)
toomany = levinfo[levinfo > topn_toomany].index.values
print(toomany)
toomany = up.diff(toomany, ["hr", "mnth", "weekday"])  # set exception for important variables
if len(toomany):
    df[toomany] = up.Collapse(n_top=topn_toomany).fit_transform(df[toomany])


# --- Final variable information ---------------------------------------------------------------------------------------

for TARGET_TYPE in TARGET_TYPES:
    #TARGET_TYPE = "CLASS"

    # Univariate variable importance
    varperf_cate = df[np.append(cate, ["MISS_" + miss])].swifter.apply(lambda x: (
        up.variable_performance(x, df["cnt_" + TARGET_TYPE],
                                splitter=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
                                scorer=up.d_scoring[TARGET_TYPE]["spear" if TARGET_TYPE == "REGR" else "auc"])))
    print(varperf_cate.sort_values(ascending=False))

    # Check
    if plot:
        _ = up.plot_l_calls(pdf_path=sett.plotloc + "1__distr_cate__" + TARGET_TYPE + ".pdf",
                            l_calls=[(up.plot_feature_target,
                                      dict(feature=df[feature],
                                           target=df["cnt_" + TARGET_TYPE],
                                           title=feature + " (VI: " + format(varperf_cate[feature], "0.2f") + ")",
                                           smooth=30))
                                     for feature in np.append(cate, "MISS_" + miss)])
        


# --- Removing variables -----------------------------------------------------------------------------------------------

# Remove leakage variables
cate = up.diff(cate, ["xxx"])
toomany = up.diff(toomany, ["xxx"])

# Remove highly/perfectly (>=99%) correlated (the ones with less levels!)
_ = up.plot_l_calls(pdf_path=sett.plotloc + "1__corr_cate.pdf", n_rows=1, n_cols=1, figsize=(8, 6),
                    l_calls=[(up.plot_corr,
                              dict(df=df[np.append(cate, "MISS_" + miss)], method="cramersv", cutoff=0))])
'''
_ = up.plot_l_calls(pdf_path=sett.plotloc + "1__corr.pdf", n_rows=1, n_cols=1, figsize=(8, 6),
                    l_calls=[(up.plot_corr,
                              dict(df=df[np.concatenate([cate, "MISS_" + miss, nume + "_BINNED"])], 
                                   method="cramersv", cutoff=0))])
'''


# --- Time/fold depedency ----------------------------------------------------------------------------------------------

# Hint: In case of having a detailed date variable this can be used as regression target here as well!
# Univariate variable importance (again ONLY for non-missing observations!)
varperf_cate_fold = df[np.append(cate, ["MISS_" + miss])].swifter.apply(lambda x: (
    up.variable_performance(x, df["fold"],
                            splitter=up.InSampleSplit(),
                            scorer=up.d_scoring["CLASS"]["auc"])))

# Plot: only variables with with highest importance
cate_toprint = varperf_cate_fold[varperf_cate_fold > 0.52].index.values
if len(nume_toprint):
    if plot:
        _ = up.plot_l_calls(pdf_path=sett.plotloc + "1__distr_cate_folddep" + TARGET_TYPE + ".pdf",
                            l_calls=[(up.plot_feature_target,
                                      dict(feature=df[feature],
                                           target=df["fold"],
                                           title=feature + " (VI: " + format(varperf_cate_fold[feature], "0.2f") + ")",
                                           smooth=30))
                                     for feature in cate_toprint])


########################################################################################################################
# Prepare final data
########################################################################################################################

# --- Add numeric target -----------------------------------------------------------------------------------------------------
df["cnt_REGR_num"] = df["cnt_REGR"]
df["cnt_CLASS_num"] = df["cnt_CLASS"].str.slice(0, 1).astype("int")
df["cnt_MULTICLASS_num"] = df["cnt_MULTICLASS"].str.slice(0, 1).astype("int")


# --- Define final features --------------------------------------------------------------------------------------------

# Standard: for all algorithms
nume_standard = np.append(nume, toomany + "_ENCODED")
cate_standard = np.append(cate, "MISS_" + miss)

# Binned: for Lasso
cate_binned = np.append(up.diff(nume + "_BINNED", onebin), cate)

# Encoded: for Lightgbm or DeepLearning
nume_encoded = np.concatenate([nume, cate + "_ENCODED", "MISS_" + miss + "_ENCODED"])

# Check
all_features = np.unique(np.concatenate([nume_standard, cate_standard, cate_binned, nume_encoded]))
up.diff(all_features, df.columns.values.tolist())
up.diff(df.columns.values.tolist(), all_features)


# --- Remove burned data -----------------------------------------------------------------------------------------------

df = df.query("fold != 'util'").reset_index(drop=True)


# --- Save image -------------------------------------------------------------------------------------------------------

# Clean up
plt.close(fig="all")  # plt.close(plt.gcf())
del df_orig

# Serialize
with open(sett.dataloc + "1_explore.pkl", "wb") as file:
    pickle.dump({"df": df,
                 "nume_standard": nume_standard,
                 "cate_standard": cate_standard,
                 "cate_binned": cate_binned,
                 "nume_encoded": nume_encoded},
                file)
