# ######################################################################################################################
#  Initialize: Packages, functions, parameters
# ######################################################################################################################

# General libraries, parameters and functions
from HMS_initialize import *
# import os; sys.path.append(os.getcwd() + "\\code")  # not needed if code is marked as "source" in pycharm

# Main parameter
TARGET_TYPE = "CLASS"

# Specific parameters (CLASS is default)
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


# ######################################################################################################################
# ETL
# ######################################################################################################################

# --- Read data ------------------------------------------------------------------------------------------------------

# noinspection PyUnresolvedReferences
if TARGET_TYPE == "CLASS":
    df_orig = pd.read_csv(dataloc + "titanic.csv")
else:
    df_orig = pd.read_csv(dataloc + "AmesHousing.txt", delimiter = "\t")
    df_orig.columns = df_orig.columns.str.replace(" ", "_")
    # alternatively: df_orig.rename(columns = lambda x: x.replace(" ", "_"), inplace=True)
df_orig.describe()
df_orig.describe(include = ["object"])

"""
# Check some stuff
df_values = create_values_df(df_orig, 10)
print(df_values)
if TARGET_TYPE == "CLASS":
    df_orig["survived"].value_counts() / df_orig.shape[0]
if TARGET_TYPE =="REGR":
    fig, ax = plt.subplots(1, 2)
    df_orig["SalePrice"].plot.hist(bins=20, ax = ax[0])
    np.log(df_orig["SalePrice"]).plot.hist(bins=20, ax = ax[1]);
"""

# "Save" original data
df = df_orig.copy()


# --- Read metadata (Project specific) -----------------------------------------------------------------------------

if TARGET_TYPE == "CLASS":
    df_meta = pd.read_excel(dataloc + "datamodel_titanic.xlsx", header = 1)
else:
    df_meta = pd.read_excel(dataloc + "datamodel_AmesHousing.xlsx", header = 1)

# Check
print(setdiff(df.columns.values, df_meta["variable"].values))
print(setdiff(df_meta.loc[df_meta["category"] == "orig", "variable"].values, df.columns.values))

# Filter on "ready"
df_meta_sub = df_meta.loc[df_meta["status"].isin(["ready", "derive"])].reset_index()


# --- Feature engineering -----------------------------------------------------------------------------------------

if TARGET_TYPE == "CLASS":
    df["deck"] = df["cabin"].str[:1]
    df["familysize"] = df["sibsp"] + df["parch"] + 1
    df["fare_pp"] = df["fare"] / df.groupby("ticket")["fare"].transform("count")
    df[["deck", "familysize", "fare_pp"]].describe(include = "all")
if TARGET_TYPE in ["REGR", "MULTICLASS"]:
    pass  # number of rooms, sqm_per_room, ...

# Check
print(setdiff(df_meta["variable"].values, df.columns.values))


# --- Define target and train/test-fold ----------------------------------------------------------------------------
'''
# Target
if TARGET_TYPE == "CLASS":
    df["target"] = df["survived"]
if TARGET_TYPE == "REGR":
    df["target"] = df["SalePrice"]
if TARGET_TYPE == "MULTICLASS":
    df["target"] = hms_preproc.QuantileBinner(n_bins = 3, output_format = "quantiles").fit_transform(df["SalePrice"])
df["target"].describe()
'''
if TARGET_TYPE == "MULTICLASS":
    df["SalePrice_Category"] = hms_preproc.QuantileBinner(n_bins = 3, output_format = "quantiles").fit_transform(df["SalePrice"])


# Train/Test fold: usually split by time
np.random.seed(123)
# noinspection PyTypeChecker
df["fold"] = np.random.permutation(
    pd.qcut(np.arange(len(df)), q = [0, 0.1, 0.8, 1], labels = ["util", "train", "test"]))
print(df.fold.value_counts())
df["fold_num"] = df["fold"].replace({"train": 0, "util": 0, "test": 1})  # Used for pedicting test data

# Define the id
df["id"] = np.arange(len(df)) + 1


# ######################################################################################################################
# Numeric variables: Explore and adapt
# ######################################################################################################################

# --- Define numeric covariates -------------------------------------------------------------------------------------

nume = df_meta_sub.loc[df_meta_sub["type"] == "metr", "variable"].values
df = hms_preproc.ScaleConverter(column_names = nume, scale = "numerical").fit_transform(df)
if TARGET_TYPE in ["REGR", "MULTICLASS"]:
    # Zeros are missings in AmesHousing
    df[nume] = df[nume].replace(0, np.nan)
df[nume].describe()

# --- Create nominal variables for all numeric variables (for linear models) before imputing -------------------------
df[nume + "_BINNED"] = hms_preproc.QuantileBinner(n_bins = 10, output_format = "quantiles").fit_transform(df[nume])

# Convert missings to own level ("(Missing)")
df[nume + "_BINNED"] = df[nume + "_BINNED"].fillna("(Missing)")
print(create_values_df(df[nume + "_BINNED"], 11))

# Get binned variables with just 1 bin (removed later)
onebin = (nume + "_BINNED")[df[nume + "_BINNED"].nunique() == 1]
print(onebin)


# --- Missings + Outliers + Skewness ---------------------------------------------------------------------------------

# Remove covariates with too many missings from nume
misspct = df[nume].isnull().mean().round(3)  # missing percentage
print("misspct:\n", misspct.sort_values(ascending = False))  # view in descending order
remove = misspct[misspct > 0.95].index.values  # vars to remove
nume = setdiff(nume, remove)  # adapt metadata

# Check for outliers and skewness
df[nume].describe()
start = time.time()
distr_nume_plots = (hms_plot.MultiFeatureDistributionPlotter(n_rows = 2, n_cols = 3, w = 18, h = 12)
                    .plot(features = df[nume],
                          target = df[target_name],
                          file_path = plotloc + TARGET_TYPE + "_distr_nume.pdf"))
print(time.time()-start)

# Winsorize (hint: plot again before deciding for log-trafo)
df = hms_preproc.Winsorizer(column_names = nume, quantiles = (0.02, 0.98)).fit_transform(df)

# Log-Transform
if TARGET_TYPE == "CLASS":
    tolog = np.array(["fare"], dtype = "object")
else:
    tolog = np.array(["Lot_Area"], dtype = "object")
df[tolog + "_LOG_"] = df[tolog].apply(lambda x: np.log(x - min(0, np.min(x)) + 1))
nume = np.where(np.isin(nume, tolog), nume + "_LOG_", nume)  # adapt metadata (keep order)
df.rename(columns = dict(zip(tolog + "_BINNED", tolog + "_LOG_" + "_BINNED")), inplace = True)  # adapt binned version


# --- Final variable information ------------------------------------------------------------------------------------

# Univariate variable importance
varimps_nume = (hms_calc.UnivariateFeatureImportanceCalculator(n_bins = 10, n_digits = 2)
                .calculate(features = df[np.append(nume, nume + "_BINNED")], target = df[target_name]))
print(varimps_nume)

# Plot
distr_nume_plots = (hms_plot.MultiFeatureDistributionPlotter(target_limits = ylim, show_regplot = True,
                                                             n_rows = 2, n_cols = 2, w = 12, h = 8)
                    .plot(features = df[np.column_stack((nume, nume + "_BINNED")).ravel()],
                          target = df[target_name],
                          varimps = varimps_nume,
                          file_path = plotloc + TARGET_TYPE + "_distr_nume_final.pdf"))
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
'''
'''
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
corr_plot = (hms_plot.CorrelationPlotter(cutoff = cutoff_corr, w = 8, h = 6)
             .plot(features = df[nume], file_path = plotloc + TARGET_TYPE + "_corr_nume.pdf"))
remove = ["xxx", "xxx"]
nume = setdiff(nume, remove)


# --- Time/fold depedency --------------------------------------------------------------------------------------------

# Hint: In case of having a detailed date variable this can be used as regression target here as well!

# Univariate variable importance (again ONLY for non-missing observations!)
varimps_nume_fold = (hms_calc.UnivariateFeatureImportanceCalculator(n_bins = 10, n_digits = 2)
                     .calculate(features = df[nume], target = df["fold_num"]))

# Plot: only variables with with highest importance
nume_toprint = varimps_nume_fold[varimps_nume_fold > cutoff_varimp].index.values
distr_nume_folddep_plots = (hms_plot.MultiFeatureDistributionPlotter(n_rows = 2, n_cols = 3, w = 18, h = 12)
                            .plot(features = df[nume_toprint],
                                  target = df["fold_num"],
                                  varimps = varimps_nume_fold,
                                  file_path = plotloc + TARGET_TYPE + "_distr_nume_folddep.pdf"))


# --- Missing indicator and imputation (must be done at the end of all processing)------------------------------------

miss = nume[df[nume].isnull().any().values]  # alternative: [x for x in nume if df[x].isnull().any()]
df["MISS_" + miss] = pd.DataFrame(np.where(df[miss].isnull(), "miss", "no_miss"))
df["MISS_" + miss].describe()

# Impute missings with randomly sampled value (or median, see below)
np.random.seed(123)
df = hms_preproc.Imputer(strategy = "median", column_names = miss).fit_transform(df)
df[miss].isnull().sum()


# ######################################################################################################################
# Categorical  variables: Explore and adapt
# ######################################################################################################################

# --- Define categorical covariates -----------------------------------------------------------------------------------

# Categorical variables
cate = df_meta_sub.loc[df_meta_sub.type.isin(["nomi", "ordi"]), "variable"].values
df = hms_preproc.ScaleConverter(column_names = cate, scale = "categorical").fit_transform(df)
df[cate].describe()

# Convert ordinal features to make it "alphanumerically sorted"
if TARGET_TYPE == "CLASS":
    df["familysize"] = df["familysize"].str.zfill(2)
else:
    tmp = ["Overall_Qual", "Overall_Cond", "Bsmt_Full_Bath", "Full_Bath", "Half_Bath", "Bedroom_AbvGr",
           "TotRms_AbvGrd", "Fireplaces", "Garage_Cars"]
    df[tmp] = df[tmp].apply(lambda x: x.str.zfill(2))


# --- Handling factor values ----------------------------------------------------------------------------------------

# Convert "standard" features: map missings to own level
df[cate] = df[cate].fillna("(Missing)")
df[cate].describe()

# Get "too many members" columns and copy these for additional encoded features (for tree based models)
topn_toomany = 10
levinfo = df[cate].nunique().sort_values(ascending = False)  # number of levels
print(levinfo)
toomany = levinfo[levinfo > topn_toomany].index.values
print(toomany)
toomany = setdiff(toomany, ["xxx", "xxx"])  # set exception for important variables

# Create encoded features (for tree based models), i.e. numeric representation
df[cate + "_ENCODED"] = (hms_preproc.TargetEncoder(subset_index = df[df["fold"] == "util"].index.values)
                         .fit_transform(df[cate], df[target_name]))
df["MISS_" + miss + "_ENCODED"] = df["MISS_" + miss].apply(lambda x: x.map({"no_miss": 0, "miss": 1}))
''' 
# BUG: Some non-exist even they do exist
i = 4
print(df[[cate[i],cate[i] + "_ENCODED"]].drop_duplicates())
print(df[df["fold"] == "util"][cate[i]].value_counts())
'''

# Convert toomany features: lump levels and map missings to own level
df[toomany] = hms_preproc.CategoryCollapser(n_top = 10).fit_transform(df[toomany])

# Univariate variable importance
varimps_cate = (hms_calc.UnivariateFeatureImportanceCalculator(n_digits = 2)
                .calculate(features = df[np.append(cate, ["MISS_" + miss])], target = df[target_name]))
print(varimps_cate)

# Check
distr_cate_plots = (hms_plot.MultiFeatureDistributionPlotter(target_limits = ylim, min_bar_width = min_width,
                                                             n_rows = 2, n_cols = 3, w = 18, h = 12)
                    .plot(features = df[np.append(cate, ["MISS_" + miss])],
                          target = df[target_name],
                          varimps = varimps_cate,
                          file_path = plotloc + TARGET_TYPE + "_distr_cate.pdf"))

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
if TARGET_TYPE == "CLASS":
    cate = setdiff(cate, ["boat", "xxx"])
    toomany = setdiff(toomany, ["boat", "xxx"])

# Remove highly/perfectly (>=99%) correlated (the ones with less levels!)
corr_cate_plot = (hms_plot.CorrelationPlotter(cutoff = cutoff_corr, w = 8, h = 6)
                  .plot(features = df[np.append(cate, ["MISS_" + miss])],
                        file_path = plotloc + TARGET_TYPE + "_corr_cate.pdf"))


# --- Time/fold depedency --------------------------------------------------------------------------------------------

# Hint: In case of having a detailed date variable this can be used as regression target here as well!
# Univariate variable importance (again ONLY for non-missing observations!)
varimps_cate_fold = (hms_calc.UnivariateFeatureImportanceCalculator(n_digits = 2)
                     .calculate(features = df[np.append(cate, ["MISS_" + miss])], target = df["fold_num"]))

# Plot: only variables with with highest importance
cate_toprint = varimps_cate_fold[varimps_cate_fold > cutoff_varimp].index.values
distr_cate_folddep_plots = (hms_plot.MultiFeatureDistributionPlotter(n_rows = 2, n_cols = 3, w = 18, h = 12)
                            .plot(features = df[cate_toprint],
                                  target = df["fold_num"],
                                  varimps = varimps_cate_fold,
                                  file_path = plotloc + TARGET_TYPE + "_distr_cate_folddep.pdf"))


########################################################################################################################
# Prepare final data
########################################################################################################################

# --- Adapt target ----------------------------------------------------------------------------------------

# Switch target to numeric in case of multiclass
if TARGET_TYPE == "MULTICLASS":
    tmp = LabelEncoder()
    df[target_name] = tmp.fit_transform(df[target_name])
    target_labels = tmp.classes_
else:
    target_labels = target_name


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

df = df.query("fold != 'util'").reset_index(drop = True)


# --- Save image ----------------------------------------------------------------------------------------------------

# Clean up
plt.close(fig = "all")  # plt.close(plt.gcf())
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
