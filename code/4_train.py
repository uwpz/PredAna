########################################################################################################################
# Initialize: Packages, functions, parameter
########################################################################################################################

# --- Packages ---------------------------------------------------------------------------------------------------------

# General
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
from importlib import reload

# Special
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import target_encoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb

# Custom functions and classes
import utils_plots as up

# Settings
import settings as s



########################################################################################################################
# Read data and fit pipeline
########################################################################################################################

# --- Read data  -------------------------------------------------------------------------------------------------------

# Read data with predefined dtypes
df_meta = (pd.read_excel(s.DATALOC + "datamodel_bikeshare.xlsx", header=1,
                         engine='openpyxl').query("status in ['ready']"))
nume = df_meta.query("type == 'nume'")["variable"].values.tolist()
cate = df_meta.query("type == 'cate'")["variable"].values.tolist()
df = pd.read_csv(s.DATALOC + "df_orig.csv", parse_dates=["dteday"], 
                 dtype={**{x: np.float64 for x in nume}, **{x: object for x in cate}})

# Define target
df["target"] = df["cnt_CLASS"].str.slice(0, 1).astype("int")

# Split in train and util
df["fold"] = np.where(df.index.isin(df.query("kaggle_fold == 'train'")
                                    .sample(frac=0.1, random_state=42).index.values),
                      "util", df["kaggle_fold"])
df_train = df.query("fold != 'util'").reset_index(drop=True)
df_util = df.query("fold == 'util'").reset_index(drop=True)



# --- Fit --------------------------------------------------------------------------------------------------------------

# Feature lists
miss = ["windspeed"]  # create missing indicator for 
ordi = ["day_of_month", "mnth", "yr"]  # no "hr" as most important variable -> more information by 1-hot-encoding
yesno = ["workingday"] + ["MISS_" + x for x in miss]  # no "holiday" as this contains also "(Missing)"
nomi = [x for x in cate if x not in ordi + yesno]  # treated as pure categorical
toomany = ["high_card"]  # duplicate them (which gets target_encoding) and collapse
#nume_standard = nume + up.add(toomany, "_ENCODED")
#cate_standard = cate + up.add("MISS_", miss)

# Etl pipeline (! Etl class must be imported for pickle to get loaded during scoring)
pipe_etl = Pipeline(steps=[("etl", s.Etl(derive_day_of_month=True, 
                                         miss=miss, cate_fill_na=cate, toomany=toomany))])

# Numerical pipeline including ordinal encoding for categorical features on ordinal scale 
# All custom functions must be imported for pickle to get loaded during scoring
pipe_nume = Pipeline(steps=[
    ("column_transform", ColumnTransformer(transformers=[        
        ("nume_log", FunctionTransformer(func=s.nume_log), ["hum"]),  # cannot pickle lambda functions
        ("ordinal_encoding", OrdinalEncoder(), ordi + yesno)
    ], remainder="passthrough")),
    ("impute", SimpleImputer(strategy="median"))  # might add additional winsorizing or scaling in case of elasticnet
])

# Categorical pipeline
pipe_cate = Pipeline(steps=[
    ("column_transform", ColumnTransformer(transformers=[
        ("collapse", up.Collapse(n_top=5), toomany)
    ], remainder="passthrough")),
    ("one_hot", OneHotEncoder(sparse=True, handle_unknown="ignore"))
])  

# Complete pipeline
pipeline = Pipeline([
    ('etl', pipe_etl),
    ('fe', ColumnTransformer(transformers=[
        ('nume', pipe_nume, nume + ordi + yesno + up.add(toomany, "_ENCODED")),
        ('cate', pipe_cate, nomi)
    ])),
    ('algo', up.UndersampleEstimator(xgb.XGBClassifier(**dict(n_estimators=1100, learning_rate=0.01,
                                                              max_depth=3, min_child_weight=10,
                                                              colsample_bytree=0.7, subsample=0.7,
                                                              gamma=0,
                                                              verbosity=0,
                                                              n_jobs=s.N_JOBS,
                                                              use_label_encoder=False)),
                                     n_max_per_level=2000))
])

# Fit
pipeline_fit = pipeline.fit(df_train, df_train["target"])

'''
# Test some stuff
pipeline_fit.predict_proba(df)[:, 1].mean()
up.diff(df_train.columns.values, nume + ordi + yesno + nomi)
up.diff(nume + ordi + yesno + nomi, df_train.columns.values)
df_etl = pipeline.named_steps["etl"].fit_transform(df_train)
check = (pipeline.named_steps["fe"].named_transformers_["nume"].named_steps["column_transform"]
         .named_transformers_["toomany_encoding"])  # only for fitted transformers
check = pipe_nume.named_steps["column_transform"].get_params()["transformers"][2][1]  # also for unfitted transformers
check.fit_transform(df_etl["high_card_ENCODED"], df_train["target"])
'''


# --- Save -------------------------------------------------------------------------------------------------------------

with open(s.DATALOC + "4_train.pkl", "wb") as file:
    pickle.dump({"pipeline": pipeline_fit}, file)


