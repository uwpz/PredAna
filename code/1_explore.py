# ######################################################################################################################
#  Initialize: Packages, functions, parameters
# ######################################################################################################################

# General libraries, parameters and functions
import initialize as my
#my.create_values_df
exec(open('initialize.py').read())
# import os; sys.path.append(os.getcwd() + "\\code")  # not needed if code is marked as "source" in pycharm

# Plot and show directly or not
plot = False
%matplotlib Agg
#plt.ioff(); matplotlib.use('Agg')
#%matplotlib inline
#plt.ion(); matplotlib.use('TkAgg')
plt.plot(1,1)

# Specific parameters (CLASS is default)
types = ["regr", "class", "multiclass"]

'''
target_name = "survived"
ylim = (0, 1)
min_width = 0
cutoff_corr = 0.1
cutoff_varimp = 0.52
if TARGET_TYPE == "MULTICLASS":
    target_name = "SalePrice_Category"
    ylim = None
    min_width = 0.2
    cutoff_corr = 0.1
    cutoff_varimp = 0.52
if TARGET_TYPE == "REGR":
    target_name = "SalePrice"
    ylim = (0, 300e3)
    cutoff_corr = 0.8
    cutoff_varimp = 0.52
'''


# ######################################################################################################################
# ETL
# ######################################################################################################################

# --- Read data and adapt to be more readable --------------------------------------------------------------------------

# Read
#df_orig_old = pd.concat([pd.read_csv(dataloc + "train.csv", parse_dates=["datetime"]).assign(fold_orig="train"),
#                     pd.read_csv(dataloc + "test.csv", parse_dates=["datetime"]).assign(fold_orig="test")])
#df_orig_old.describe()

# Adapt to be more readable
df_orig = (pd.read_csv(dataloc + "hour.csv", parse_dates=["dteday"])
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
#df_orig["hum"] = df_orig["hum"].where(np.random.random_sample(len(df_orig)) > 0.1, other=np.nan)  # some missings
df_orig["weathersit"] = df_orig["weathersit"].where(df_orig["weathersit"] != "heavy rain", np.nan)
df_orig["windspeed"] = df_orig["windspeed"].where(df_orig["windspeed"] != 0, other=np.nan)  # some missings

# Create artificial targets
df_orig["cnt_regr"] = np.log(df_orig["cnt"] + 1)
df_orig["cnt_class"] = pd.qcut(df_orig["cnt"], q=[0, 0.8, 1], labels=["0_low", "1_high"]).astype("object")
df_orig["cnt_multiclass"] = pd.qcut(df_orig["cnt"], q=[0, 0.8, 0.95, 1],
                                    labels=["0_low", "1_high", "2_very_high"]).astype("object")


'''
# Check some stuff
df_orig.dtypes
df_orig.describe()
create_values_df(df_orig, dtypes=["object"]).T

fig, ax = plt.subplots(1, 3, figsize=(15,5))
df_orig["cnt"].plot.hist(bins=50, ax=ax[0])
df_orig["cnt"].hist(density=True, cumulative=True, bins=50, histtype="step", ax=ax[1])
np.log(df_orig["cnt"]).plot.hist(bins=50, ax=ax[2])
'''

# "Save" original data
df = df_orig.copy()


# --- Read metadata (Project specific) ---------------------------------------------------------------------------------

df_meta = pd.read_excel(dataloc + "datamodel_bikeshare.xlsx", header=1, engine='openpyxl')

# Check
print(setdiff(df.columns.values, df_meta["variable"].values))
print(setdiff(df_meta.query("category == 'orig'").variable.values, df.columns.values))

# Filter on "ready"
df_meta_sub = df_meta.query("status in ['ready']").reset_index()


# --- Feature engineering ----------------------------------------------------------------------------------------------

df["day_of_month"] = df['dteday'].dt.day.astype("str").str.zfill(2)

# Check again
print(setdiff(df_meta_sub["variable"].values, df.columns.values))


# --- Define train/test/util-fold --------------------------------------------------------------------------------------

df["fold"] = np.where(df.index.isin(df.query("kaggle_fold == 'train'")
                                    .sample(frac=0.1, random_state=42).index.values),
                      "util", df["kaggle_fold"])
#df["fold_num"] = df["fold"].replace({"train": 0, "util": 0, "test": 1})  # Used for pedicting test data


# ######################################################################################################################
# Numeric variables: Explore and adapt
# ######################################################################################################################

# --- Define numeric covariates ----------------------------------------------------------------------------------------

nume = df_meta_sub.loc[df_meta_sub["type"] == "nume", "variable"].values
df[nume] = df[nume].apply(lambda x: pd.to_numeric(x))


# --- Create nominal variables for all numeric variables (for linear models) before imputing ---------------------------

df[nume + "_BINNED"] = (df[nume].apply(lambda x: (pd.qcut(x, 5)  # alternative: sklearns KBinsDiscretizer
                                                  .astype("str").replace("nan", np.nan))))

# Convert missings to own level ("(Missing)")
df[nume + "_BINNED"] = df[nume + "_BINNED"].fillna("(Missing)")
print(create_values_df(df[nume + "_BINNED"], 6))

# Get binned variables with just 1 bin (removed later)
onebin = (nume + "_BINNED")[df[nume + "_BINNED"].nunique() == 1]
print(onebin)


# --- Missings + Outliers + Skewness -----------------------------------------------------------------------------------

# Remove covariates with too many missings from nume
misspct = df[nume].isnull().mean().round(3)  # missing percentage
print("misspct:\n", misspct.sort_values(ascending=False))  # view in descending order
remove = misspct[misspct > 0.95].index.values  # vars to remove
nume = setdiff(nume, remove)  # adapt metadata

# Check for outliers and skewness
df[nume].describe()
start = time.time()
for type in types:
    if plot:
        distr_nume_plots = (hms_plot.MultiFeatureDistributionPlotter(n_rows=2, n_cols=3, w=18, h=12)
                            .plot(features=df[nume],
                                  target=df["cnt_" + type],
                                  file_path=plotloc + "distr_nume__" + type + ".pdf"))
    print(time.time() - start)

# Winsorize (hint: plot again before deciding for log-trafo)
df = hms_preproc.Winsorizer(column_names=nume, quantiles=(0.01, 0.99)).fit_transform(df)

# Log-Transform
tolog = np.array([], dtype="object")
if len(tolog):
    df[tolog + "_LOG_"] = df[tolog].apply(lambda x: np.log(x - min(0, np.min(x)) + 1))
    nume = np.where(np.isin(nume, tolog), nume + "_LOG_", nume)  # adapt metadata (keep order)
    df.rename(columns=dict(zip(tolog + "_BINNED", tolog + "_LOG_" + "_BINNED")), inplace=True)  # adapt binned version


# --- Final variable information ---------------------------------------------------------------------------------------

#import importlib
#importlib.reload(my)


from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import cross_val_score
from pandas.api.types import is_numeric_dtype
split_index = PredefinedSplit(df["fold"].map({"train": -1,"util": -1, "test": 0}).values)
split_shuffle = ShuffleSplit(1, 0.2)
split_my1fold_cv = TrainTestSep(1)
#%%
split_insample = my.InSample()
split = split_insample.split(df)
i_train, i_test = next(split)
#%%



#%%

    features = df[nume]
    target = df["cnt_" + type]
    split = split_insample
    
def calc_varimp(features, target, splitter):    

    target_type = dict(continuous="regr", binary="class",
                       multiclass="multiclass")[type_of_target(target)]
    print(target_type)
    metric = "spear" if target_type == "regr" else "auc"
    print(metric)
    
    varimp = dict()
    for col in features.columns.values:
        varimp[col] = cross_val_score(estimator=(LinearRegression() if target_type == "regr" else
                                                 LogisticRegression()),
                            X=(KBinsDiscretizer().fit_transform(features[[col]])if is_numeric_dtype(features[col]) else
                               OneHotEncoder().fit_transform(features[[col]])),
                            y=target,
                            cv=splitter,
                            scoring=d_scoring[target_type][metric])
    return(varimp)
 

calc_varimp(df[nume + "_BINNED"], df["cnt_" + type], split_shuffle)
        
#%%


for type in types:

    # Univariate variable importance
    varimps_nume = (hms_calc.UnivariateFeatureImportanceCalculator(n_bins=5, n_digits=2)
                    .calculate(features=df[np.append(nume, nume + "_BINNED")], target=df["cnt_" + type]))
    print(varimps_nume)

    # Plot
    if plot:
        distr_nume_plots = (hms_plot.MultiFeatureDistributionPlotter(show_regplot=True,
                                                                     n_rows=2, n_cols=2, w=12, h=8)
                            .plot(features=df[np.column_stack((nume, nume + "_BINNED")).ravel()],
                                  target=df["cnt_" + type],
                                  varimps=varimps_nume,
                                  file_path=plotloc + "distr_nume_final__" + type + ".pdf"))


# --- Removing variables -----------------------------------------------------------------------------------------------

# Remove leakage features
remove = ["xxx", "xxx"]
nume = setdiff(nume, remove)

# Remove highly/perfectly (>=98%) correlated (the ones with less NA!)
df[nume].describe()
corr_plot = (hms_plot.CorrelationPlotter(cutoff=0, w=8, h=6)
             .plot(features=df[nume], file_path=plotloc + "corr_nume.pdf"))
remove = ["atemp"]
nume = setdiff(nume, remove)


# --- Time/fold depedency ----------------------------------------------------------------------------------------------

# Hint: In case of having a detailed date variable this can be used as regression target here as well!

# Univariate variable importance (again ONLY for non-missing observations!)
varimps_nume_fold = (hms_calc.UnivariateFeatureImportanceCalculator(n_bins=5, n_digits=2)
                     .calculate(features=df[nume], target=df["fold"]))

# Plot: only variables with with highest importance
nume_toprint = varimps_nume_fold[varimps_nume_fold > 0.52].index.values
if plot:
    distr_nume_folddep_plots = (hms_plot.MultiFeatureDistributionPlotter(n_rows=2, n_cols=3, w=18, h=12)
                                .plot(features=df[nume_toprint],
                                      target=df["fold"],
                                      varimps=varimps_nume_fold,
                                      file_path=plotloc + "distr_nume_folddep.pdf"))


# --- Missing indicator and imputation (must be done at the end of all processing)--------------------------------------

miss = nume[df[nume].isnull().any().values]
df["MISS_" + miss] = pd.DataFrame(np.where(df[miss].isnull(), "No", "Yes"))
df["MISS_" + miss].describe()

# Impute missings with randomly sampled value (or median, see below)
np.random.seed(123)
df = hms_preproc.Imputer(strategy="median", column_names=miss).fit_transform(df)
df[miss].isnull().sum()


# ######################################################################################################################
# Categorical  variables: Explore and adapt
# ######################################################################################################################

# --- Define categorical covariates ------------------------------------------------------------------------------------

# Categorical variables
cate = df_meta_sub.loc[df_meta_sub.type.isin(["cate"]), "variable"].values
df[cate] = df[cate].astype("object")
df[cate].describe()


# --- Handling factor values -------------------------------------------------------------------------------------------

# Convert "standard" features: map missings to own level
df[cate] = df[cate].fillna("(Missing)")
df[cate].describe()

# Create ordinal/binary-encoded features
ordi = np.array(["hr", "mnth", "yr"], dtype="object")
df[ordi + "_ENCODED"] = df[ordi].apply(lambda x: pd.to_numeric(x))  # ordinal
yesno = np.concatenate([np.array(["holiday", "workingday"], dtype="object"), "MISS_" + miss])
df[yesno + "_ENCODED"] = df[yesno].apply(lambda x: x.map({"No": 0, "Yes": 1}))  # binary

# Create target-encoded features for nominal variables
nomi = setdiff(cate, np.concatenate([ordi, yesno]))
df_util = df.query("fold == 'util'").reset_index(drop=True)
df[nomi + "_ENCODED"] = target_encoder.TargetEncoder().fit(df_util[nomi], df_util["cnt_regr"]).transform(df[nomi])
#df = df.query("fold != 'util'").reset_index(drop=True)  # remove utility data

# Get "too many members" columns and lump levels
topn_toomany = 5
levinfo = df[cate].nunique().sort_values(ascending=False)  # number of levels
print(levinfo)
toomany = levinfo[levinfo > topn_toomany].index.values
print(toomany)
toomany = setdiff(toomany, ["hr", "mnth", "weekday"])  # set exception for important variables
if len(toomany):
    df[toomany] = hms_preproc.CategoryCollapser(n_top=5).fit_transform(df[toomany])


# --- Final variable information ---------------------------------------------------------------------------------------

for type in types:

    # Univariate variable importance
    varimps_cate = (hms_calc.UnivariateFeatureImportanceCalculator(n_digits=2)
                    .calculate(features=df[np.append(cate, ["MISS_" + miss])], target=df["cnt_" + type]))
    print(varimps_cate)

    # Check
    if plot:
        distr_cate_plots = (hms_plot.MultiFeatureDistributionPlotter(n_rows=2, n_cols=3, w=18, h=12)
                            .plot(features=df[np.append(cate, ["MISS_" + miss])],
                                  target=df["cnt_" + type],
                                  varimps=varimps_cate,
                                  file_path=plotloc + "distr_cate__" + type + ".pdf"))


# --- Removing variables -----------------------------------------------------------------------------------------------

# Remove leakage variables
cate = setdiff(cate, ["xxx"])
toomany = setdiff(toomany, ["xxx"])

# Remove highly/perfectly (>=99%) correlated (the ones with less levels!)
corr_cate_plot = (hms_plot.CorrelationPlotter(cutoff=0, w=8, h=6)
                  .plot(features=df[np.append(cate, ["MISS_" + miss])],
                        file_path=plotloc + "corr_cate.pdf"))


# --- Time/fold depedency ----------------------------------------------------------------------------------------------

# Hint: In case of having a detailed date variable this can be used as regression target here as well!
# Univariate variable importance (again ONLY for non-missing observations!)
varimps_cate_fold = (hms_calc.UnivariateFeatureImportanceCalculator(n_digits=2)
                     .calculate(features=df[np.append(cate, ["MISS_" + miss])], target=df["fold"]))

# Plot: only variables with with highest importance
cate_toprint = varimps_cate_fold[varimps_cate_fold > 0.51].index.values
distr_cate_folddep_plots = (hms_plot.MultiFeatureDistributionPlotter(n_rows=2, n_cols=3, w=18, h=12)
                            .plot(features=df[cate_toprint],
                                  target=df["fold"],
                                  varimps=varimps_cate_fold,
                                  file_path=plotloc + "distr_cate_folddep.pdf"))


'''
# Fancy!

from hmsPM.plotting.output import save_plot_grids_to_pdf
from hmsPM.plotting.distribution import FeatureDistributionPlotter
from hmsPM.plotting.grid import PlotGridBuilder
from hmsPM.datatypes import PlotFunctionCall
plot_calls = []
features = np.concatenate([nume, cate[[0, 4, 5, 6, 7]]])
for row in features:
    for col in features:
        if row == col:
            plot_calls.append(PlotFunctionCall(FeatureDistributionPlotter().plot,
                                               kwargs=dict(feature=df[row], target=df["cnt_class"])))
        else:
            plot_calls.append(PlotFunctionCall(FeatureDistributionPlotter().plot,
                                               kwargs=dict(feature=df[row], target=df[col])))
plot_grids = PlotGridBuilder(n_rows=len(features), n_cols=len(features), h=60, w=60).build(plot_calls=plot_calls)
save_plot_grids_to_pdf(plot_grids, plotloc + "tmp.pdf")
'''


########################################################################################################################
# Prepare final data
########################################################################################################################

# --- Adapt target -----------------------------------------------------------------------------------------------------

# Switch target to numeric in case of multiclass
#tmp = LabelEncoder()
#df["cnt_multiclass"] = tmp.fit_transform(df["cnt_multiclass"])
#target_labels = tmp.classes_
#CLASS:    target_labels = target_name


# --- Define final features --------------------------------------------------------------------------------------------

# Standard: for xgboost or Lasso
nume_standard = np.append(nume, toomany + "_ENCODED")
cate_standard = np.append(cate, "MISS_" + miss)

# Binned: for Lasso
nume_binned = np.array([])
cate_binned = np.append(setdiff(nume + "_BINNED", onebin), cate)

# Encoded: for Lightgbm or DeepLearning
nume_encoded = np.concatenate([nume, cate + "_ENCODED", "MISS_" + miss + "_ENCODED"])
cate_encoded = np.array([])

# Check
all_features = np.unique(np.concatenate([nume_standard, cate_standard, nume_binned, cate_binned, nume_encoded]))
setdiff(all_features, df.columns.values.tolist())
setdiff(df.columns.values.tolist(), all_features)


# --- Remove burned data -----------------------------------------------------------------------------------------------

#df = df.query("fold != 'util'").reset_index(drop=True)


# --- Save image -------------------------------------------------------------------------------------------------------

# Clean up
plt.close(fig="all")  # plt.close(plt.gcf())
del df_orig

# Serialize
with open(dataloc + "1_explore.pkl", "wb") as file:
    pickle.dump({"df": df,
                 "nume_standard": nume_standard,
                 "cate_standard": cate_standard,
                 "nume_binned": nume_binned,
                 "cate_binned": cate_binned,
                 "nume_encoded": nume_encoded,
                 "cate_encoded": cate_encoded},
                file)
