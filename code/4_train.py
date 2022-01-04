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
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

# Custom functions and classes
import utils_plots as up

# Settings
import settings as s


########################################################################################################################
# XXX
########################################################################################################################

# --- Read data  ---------------------------------------------------------------------------------------------

# Read data with predefined dtypes
df_meta = (pd.read_excel(s.DATALOC + "datamodel_bikeshare.xlsx", header=1,
                         engine='openpyxl').query("status in ['ready']"))
nume = df_meta.query("type == 'nume'")["variable"].values.tolist()
cate = df_meta.query("type == 'cate'")["variable"].values.tolist()
df = pd.read_csv(s.DATALOC + "df_orig.csv", parse_dates=["dteday"], 
                 dtype={**{x: np.float64 for x in nume}, **{x: object for x in cate}})

# Split in train and util
df["fold"] = np.where(df.index.isin(df.query("kaggle_fold == 'train'")
                                    .sample(frac=0.1, random_state=42).index.values),
                      "util", df["kaggle_fold"])
df_train = df.query("fold != 'util'").reset_index(drop=True)
df_util = df.query("fold == 'util'").reset_index(drop=True).assign()


# --- FE  ---------------------------------------------------------------------------------------------

class Etl(BaseEstimator, TransformerMixin):
    def __init__(self, derive_day_of_month=True, miss=None, 
                 cate_fill_na=None, toomany=None):
        self.derive_day_of_month = derive_day_of_month
        self.miss = miss
        self.cate_fill_na = cate_fill_na
        self.toomany = toomany
      
    def fit(self, *_):
        return self
    
    def transform(self, df, *_):
        if self.derive_day_of_month:
            df["day_of_month"] = df["dteday"].dt.day.astype("str").str.zfill(2)
        if self.miss is not None:
            df[up.add("MISS_", miss)] = pd.DataFrame(np.where(df[miss].isnull(), "No", "Yes"))
        if self.cate_fill_na is not None:
            df[self.cate_fill_na] = df[self.cate_fill_na].fillna("(Missing)").replace("nan", "(Missing)")
        if self.toomany is not None:
            df[up.add(toomany, "_ENCODED")] = df[toomany]
        return df
        
        
#etl: day_of_month, missing indicators, fillna_for_cate, duplicate toomany into encoded
miss = ["windspeed"]
ordi = ["day_of_month", "mnth", "yr"]  # no "hr" as most important variable -> more information by 1-hot-encoding
yesno = ["workingday"] + ["MISS_" + x for x in miss]  # no "holiday" as it contains also "(Missing)"
toomany = ["high_card"]

pipeline = Pipeline([
    ('etl', Etl(derive_day_of_month=True, miss=miss, cate_fill_na=cate, toomany=toomany))
])
df_etl = pipeline.fit_transform(df_train)

#nume + ordi_and_yesno: (split into log_trafo, toomany_target_encoded capable of using external data,
#                        ordinal_encoding_for_ordi_yesno, identity_for_the_rest) + impute + commenting: wins and minmax
#nomi + miss: (split into toomany_collapse, identity_for_the_rest) + onehot

pipe_nume = ColumnTransformer(
    transformers=[
        ('nume_identity', FunctionTransformer(func=lambda x: x), ["temp", "windspeed"]),
        ('nume_log', FunctionTransformer(func=lambda x: np.log(x+1)), ["hum"]),
        ('ordinal_encoding', OrdinalEncoder(), ordi + yesno)#,
        #('toomany_encoding', target_encoder.TargetEncoder(), up.add(toomany, "_ENCODED"))
    ])
a_nume = pipe_nume.fit_transform(df_etl)
a_nume.shape

pipe_cate = ColumnTransformer(
    transformers=[
        ('cate_identity', FunctionTransformer(func=lambda x: x), cate)
    ])
a_cate = pipe_cate.fit_transform(df_etl)
a_cate.shape

pipe_features = ColumnTransformer(
    transformers=[
        ('nume', pipe_nume, nume + ordi + yesno),
        ('cate', pipe_cate, cate)
    ])
a_features = pipe_features.fit_transform(df_etl)
a_features.shape




# TODO: adapt log_hum to function transformer
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, log_hum=True, miss=None, toomany=None, df_util=None):
        self.log_hum = log_hum
        self.miss = miss
        self.toomany = toomany
        self.df_util = df_util

    def fit(self, *_):
        if self.toomany is not None:
            self._toomany_encoder = (target_encoder.TargetEncoder(cols=self.toomany)
                                     .fit(df_util[self.toomany],
                                          df_util["cnt_CLASS"].map({"0_low": 0, "1_high": 1})))
            self._toomany_collapser = up.Collapse().fit(df[self.toomany])
        return self

    def transform(self, df, *_):
        df["day_of_month"] = df["dteday"].dt.day.astype("str").str.zfill(2)
        if self.log_hum:
            df["hum"] = np.log(df["hum"] + 1)           
        if self.miss is not None:
            df[up.add("MISS_", self.miss)] = pd.DataFrame(np.where(df[self.miss].isnull(), "No", "Yes"))
        if self.toomany is not None:
            df[up.add(self.toomany, "_ENCODED")] = self._toomany_encoder.transform(df[self.toomany])
            df[self.toomany] = self._toomany_collapser.transform(df[self.toomany])
        return df

pipeline = Pipeline([
    ('feature_engineering', FeatureEngineering(miss=miss, toomany=toomany, df_util=df_util))
])
df_train = pipeline.fit_transform(df_train)

# Nume pipeline
pipe_nume = Pipeline(steps=[
    ('to_float', FunctionTransformer(func=lambda x: x)),
    ('winsorize', up.Winsorize(lower_quantile=None, upper_quantile=0.99))
   # ('impute', SimpleImputer(strategy="median"))
])

# Cate pipeline
pipe_cate = Pipeline(steps=[
    ('to_string', FunctionTransformer(func=lambda x: x))#,
   # ('impute', FunctionTransformer(func=lambda x: pd.DataFrame(x).fillna("(Missing)").replace("nan", "(Missing)")))
   # ('onehot', OneHotEncoder(sparse=True, handle_unknown="ignore"))
])

# Features (nume + cate) pipeline
pipe_features = ColumnTransformer(
    transformers=[
        ('nume', pipe_nume, nume),
        ('cate', pipe_cate, cate[:3])
])

#%%
X_features = pipe_features.fit_transform(df)
#%%
pd.DataFrame(X_features).dtypes
pd.DataFrame(X_features).apply(lambda x: x == "(Missing)").sum()
pd.DataFrame(X_features).describe()


pipe_cate.fit_transform(df[cate])
pipe_nume.fit_transform(df[nume])