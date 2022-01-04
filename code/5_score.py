########################################################################################################################
# Initialize: Packages, functions, parameter
########################################################################################################################

# --- Packages ---------------------------------------------------------------------------------------------------------

# General
import numpy as np
import pandas as pd

import pickle
import time
from importlib import reload

# Special
'''
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import target_encoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb
'''
# Custom functions and classes
#import utils_plots as up

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
      .sample(n=100).reset_index(drop=True))


# --- Score  -----------------------------------------------------------------------------------------------------------

with open(s.DATALOC + "4_train.pkl", "rb") as file:
    pipeline = pickle.load(file)["pipeline"]
score = pipeline.predict_proba(df)
print(score)
