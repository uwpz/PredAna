# ######################################################################################################################
#  Initialize: Packages, functions, parameters
# ######################################################################################################################

# General libraries, parameters and functions
from initialize import *
# import os; sys.path.append(os.getcwd() + "\\code")  # not needed if code is marked as "source" in pycharm


%matplotlib
plt.ioff(); matplotlib.use('Agg')
#%matplotlib inline
#plt.ion(); matplotlib.use('TkAgg')


'''
# Main parameter
TARGET_TYPE = "CLASS"
'''
types = ["regr", "class", "multiclass"]

# Specific parameters (CLASS is default)
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
df_orig["hum"] = df_orig["hum"].where(np.random.random_sample(len(df_orig)) > 0.1, other=np.nan)  # some missings
df_orig["weathersit"] = df_orig["weathersit"].where(df_orig["weathersit"] != "heavy rain", np.nan)

# Create artificial targets
df_orig["cnt_regr"] = np.log(df_orig["cnt"])
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


# --- Read metadata (Project specific) -----------------------------------------------------------------------------

df_meta = pd.read_excel(dataloc + "datamodel_bikeshare.xlsx", header=1, engine='openpyxl')

# Check
print(setdiff(df.columns.values, df_meta["variable"].values))
print(setdiff(df_meta.query("category == 'orig'").variable.values, df.columns.values))

# Filter on "ready"
df_meta_sub = df_meta.query("status in ['ready']").reset_index()


# --- Feature engineering -----------------------------------------------------------------------------------------

df["day_of_month"] = df['dteday'].dt.day.astype("str").str.zfill(2)

# Check again
print(setdiff(df_meta_sub["variable"].values, df.columns.values))


# --- Define train/test/util-fold ----------------------------------------------------------------------------

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

df[nume + "_BINNED"] = (df[nume].apply(lambda x: (pd.qcut(x, 5)
                                                  .astype("str").replace("nan",np.nan))))  # alternative: sklearns KBinsDiscretizer

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

for type in types:
 
    # Univariate variable importance
    varimps_nume = (hms_calc.UnivariateFeatureImportanceCalculator(n_bins=5, n_digits=2)
                    .calculate(features=df[np.append(nume, nume + "_BINNED")], target=df["cnt_" + type]))
    print(varimps_nume)

    # Plot
    distr_nume_plots = (hms_plot.MultiFeatureDistributionPlotter(show_regplot=True,
                                                                 n_rows=2, n_cols=2, w=12, h=8)
                        .plot(features=df[np.column_stack((nume, nume + "_BINNED")).ravel()],
                              target=df["cnt_" + type],
                              varimps=varimps_nume,
                              file_path=plotloc + "distr_nume_final__" + type + ".pdf"))

'''
%matplotlib inline
def show_figure(fig):
    # create a dummy figure and use its manager to display "fig"
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
from matplotlib.backends.backend_pdf import PdfPages

pdf_pages = PdfPages(plotloc + TARGET_TYPE + "_deleteme.pdf")
for page in range(len(distr_nume_plots)):
    ax = distr_nume_plots[page][1][0, 0]
    ax.set_title("blub")
    leg = ax.legend()
    leg.set_text(leg.get_texts()[0])
    
    show_figure(distr_nume_plots[page][0])
    pdf_pages.savefig(distr_nume_plots[page][0])
pdf_pages.close()


plt.ion(); matplotlib.use('TkAgg')

page = 0

fig, ax = plt.subplots(2,3)
fig.set_size_inches(w = 12, h = 8)
fig.tight_layout()

# Remove empty ax
new_ax = ax[1,1]
#new_ax.remove()

# Get old_ax and assign it to the figure, move it to the position of new_ax and add it to figure
old_ax = distr_nume_plots[page][1][0, 0]
old_ax.set_title("blub")
#old_ax = ax1[0,1]
#old_ax.__dict__
type(old_ax)
old_ax._position = new_ax._position
old_ax._originalPosition = new_ax._originalPosition
old_ax.reset_position()
old_ax.figure = fig
#old_ax.figbox = new_ax.figbox
old_ax.change_geometry(*(new_ax.get_geometry()))
old_ax.pchanged()
#old_ax.set_position(new_ax.get_position())

fig.add_axes(old_ax)
'''


# --- Removing variables -------------------------------------------------------------------------------------------

# Remove leakage features
remove = ["xxx", "xxx"]
nume = setdiff(nume, remove)

# Remove highly/perfectly (>=98%) correlated (the ones with less NA!)
df[nume].describe()
corr_plot = (hms_plot.CorrelationPlotter(cutoff=0, w=8, h=6)
             .plot(features=df[nume], file_path=plotloc + "corr_nume.pdf"))
remove = ["atemp"]
nume = setdiff(nume, remove)


# --- Time/fold depedency --------------------------------------------------------------------------------------------

# Hint: In case of having a detailed date variable this can be used as regression target here as well!

# Univariate variable importance (again ONLY for non-missing observations!)
varimps_nume_fold = (hms_calc.UnivariateFeatureImportanceCalculator(n_bins=5, n_digits=2)
                     .calculate(features=df[nume], target=df["fold"]))

# Plot: only variables with with highest importance
nume_toprint = varimps_nume_fold[varimps_nume_fold > 0.52].index.values
distr_nume_folddep_plots = (hms_plot.MultiFeatureDistributionPlotter(n_rows=2, n_cols=3, w=18, h=12)
                            .plot(features=df[nume_toprint],
                                  target=df["fold"],
                                  varimps=varimps_nume_fold,
                                  file_path=plotloc + "distr_nume_folddep.pdf"))


# --- Missing indicator and imputation (must be done at the end of all processing)------------------------------------

miss = nume[df[nume].isnull().any().values]  
df["MISS_" + miss] = pd.DataFrame(np.where(df[miss].isnull(), "miss", "no_miss"))
df["MISS_" + miss].describe()

# Impute missings with randomly sampled value (or median, see below)
np.random.seed(123)
df = hms_preproc.Imputer(strategy="median", column_names=miss).fit_transform(df)
df[miss].isnull().sum()


# ######################################################################################################################
# Categorical  variables: Explore and adapt
# ######################################################################################################################

# --- Define categorical covariates -----------------------------------------------------------------------------------

# Categorical variables
cate = df_meta_sub.loc[df_meta_sub.type.isin(["cate"]), "variable"].values
df[cate] = df[cate].astype("object")
df[cate].describe()


# --- Handling factor values ----------------------------------------------------------------------------------------

# Convert "standard" features: map missings to own level
df[cate] = df[cate].fillna("(Missing)")
df[cate].describe()

# Get "too many members" columns and copy these for additional encoded features (for tree based models)
topn_toomany = 50
levinfo = df[cate].nunique().sort_values(ascending=False)  # number of levels
print(levinfo)
toomany = levinfo[levinfo > topn_toomany].index.values
print(toomany)
toomany = setdiff(toomany, ["xxx", "xxx"])  # set exception for important variables

'''
# Create encoded features (for tree based models), i.e. numeric representation
df[cate + "_ENCODED"] = (hms_preproc.TargetEncoder(subset_index=df[df["fold"] == "util"].index.values)
                         .fit_transform(df[cate], df[target_name]))
df["MISS_" + miss + "_ENCODED"] = df["MISS_" + miss].apply(lambda x: x.map({"no_miss": 0, "miss": 1}))

# BUG: Some non-exist even they do exist
i = 4
print(df[[cate[i],cate[i] + "_ENCODED"]].drop_duplicates())
print(df[df["fold"] == "util"][cate[i]].value_counts())
'''

# Convert toomany features: lump levels and map missings to own level
if len(toomany):
    df[toomany] = hms_preproc.CategoryCollapser(n_top=10).fit_transform(df[toomany])


# --- Final variable information ---------------------------------------------------------------------------------------

for type in types:
    
    # Univariate variable importance
    varimps_cate = (hms_calc.UnivariateFeatureImportanceCalculator(n_digits=2)
                    .calculate(features=df[np.append(cate, ["MISS_" + miss])], target=df["cnt_" + type]))
    print(varimps_cate)

    # Check
    distr_cate_plots = (hms_plot.MultiFeatureDistributionPlotter(n_rows=2, n_cols=3, w=18, h=12)
                        .plot(features=df[np.append(cate, ["MISS_" + miss])],
                              target=df["cnt_" + type],
                              varimps=varimps_cate,
                              file_path=plotloc + "distr_cate__" + type + ".pdf"))

'''
from hmsPM.datatypes import PlotFunctionCall
from hmsPM.plotting.grid import PlotGridBuilder
from hmsPM.plotting.distribution import FeatureDistributionPlotter
plot_calls = [
    PlotFunctionCall(FeatureDistributionPlotter().plot, kwargs = dict(feature = df[cate[1]], target = df["target"])),
    PlotFunctionCall(sns.distplot, kwargs = dict(a = np.random.randn(100)))
]
tmp = PlotGridBuilder(n_rows=2, n_cols=2, h=6, w=6).build(plot_calls=plot_calls)
'''

# --- Removing variables ---------------------------------------------------------------------------------------------

# Remove leakage variables
cate = setdiff(cate, ["xxx"])
toomany = setdiff(toomany, ["xxx"])

# Remove highly/perfectly (>=99%) correlated (the ones with less levels!)
corr_cate_plot = (hms_plot.CorrelationPlotter(cutoff=0, w=8, h=6)
                  .plot(features=df[np.append(cate, ["MISS_" + miss])],
                        file_path=plotloc + "corr_cate.pdf"))


# --- Time/fold depedency --------------------------------------------------------------------------------------------

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


########################################################################################################################
# Prepare final data
########################################################################################################################

# --- Adapt target ----------------------------------------------------------------------------------------

# Switch target to numeric in case of multiclass
tmp = LabelEncoder()
df["cnt_multiclass"] = tmp.fit_transform(df["cnt_multiclass"])
target_labels = tmp.classes_
#CLASS:    target_labels = target_name


# --- Define final features ----------------------------------------------------------------------------------------

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


# --- Remove burned data ----------------------------------------------------------------------------------------

df = df.query("fold != 'util'").reset_index(drop=True)


# --- Save image ----------------------------------------------------------------------------------------------------

# Clean up
plt.close(fig="all")  # plt.close(plt.gcf())
del df_orig

# Serialize
with open(TARGET_TYPE + "_1_explore_HMS.pkl", "wb") as file:
    pickle.dump({"df": df,
                 "target_name": target_name,
                 "target_labels": target_labels,
                 "nume_standard": nume_standard,
                 "cate_standard": cate_standard,
                 "nume_binned": nume_binned,
                 "cate_binned": cate_binned,
                 "nume_encoded": nume_encoded,
                 "cate_encoded": cate_encoded},
                file)
