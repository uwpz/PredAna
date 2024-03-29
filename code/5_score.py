########################################################################################################################
# Initialize: Packages, functions, parameter
########################################################################################################################

# --- Packages ---------------------------------------------------------------------------------------------------------

# General
import numpy as np
import pandas as pd
import dill

# Custom functions and classes
import utils_plots as up

# Settings
import settings as s


########################################################################################################################
# Read data and score
########################################################################################################################

# --- Read data  -------------------------------------------------------------------------------------------------------

# Read data with predefined dtypes
df_meta = (pd.read_excel(s.DATALOC + "datamodel_bikeshare.xlsx", header=1,
                         engine='openpyxl').query("status in ['ready']"))
nume = df_meta.query("type == 'nume'")["variable"].values.tolist()
cate = df_meta.query("type == 'cate'")["variable"].values.tolist()
df = (pd.read_csv(s.DATALOC + "df_orig.csv", parse_dates=["dteday"],
                  dtype={**{x: np.float64 for x in nume}, **{x: object for x in cate}})
      .query("kaggle_fold == 'test'").reset_index(drop=True))


# --- Score  -----------------------------------------------------------------------------------------------------------

with open(s.DATALOC + "4_train.pkl", "rb") as file:
    pipeline = dill.load(file)["pipeline"]
score = pipeline.predict(df)
print("spear: ", up.spear(score, df["cnt_REGR"]))
print("rmse:",  up.rmse(score, df["cnt_REGR"]))
# too bad? -> overfit: adapt n_max_per_level=np.inf, n_estimators=2100, max_depth=9

