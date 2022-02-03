########################################################################################################################
# Initialize: Packages, functions, parameter
########################################################################################################################

# --- Packages ---------------------------------------------------------------------------------------------------------

# General
import numpy as np
import pandas as pd
import dill

# Special
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import target_encoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
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
#df["target"] = df["cnt_CLASS"].str.slice(0, 1).astype("int")
df["target"] = df["cnt_REGR"]

# Split in train and util
df["fold"] = np.where(df.index.isin(df.query("kaggle_fold == 'train'")
                                    .sample(frac=0.1, random_state=42).index.values),
                      "util", df["kaggle_fold"])
df_train = df.query("fold == 'train'").reset_index(drop=True)
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


# Etl pipeline
class Etl(BaseEstimator, TransformerMixin):
    def __init__(self, derive_day_of_month=True, miss=None,
                 cate_fill_na=None, toomany=None, df_util=None, target_name="target"):
        self.derive_day_of_month = derive_day_of_month
        self.miss = miss
        self.cate_fill_na = cate_fill_na
        self.toomany = toomany
        self.df_util = df_util
        self.target_name = target_name

    def fit(self, df, *_):
        if self.toomany is not None:
            self._toomany_encoder = (target_encoder.TargetEncoder(cols=self.toomany)
                                     .fit(self.df_util[self.toomany] if self.df_util is not None
                                          else df[self.toomany],
                                          self.df_util[self.target_name] if self.df_util is not None
                                          else df[self.target_name]))
        return self

    def transform(self, df, *_):
        if self.derive_day_of_month:
            df["day_of_month"] = df["dteday"].dt.day.astype("str").str.zfill(2)
        if self.miss is not None:
            df[up.add("MISS_", self.miss)] = pd.DataFrame(np.where(df[self.miss].isnull(), "No", "Yes"))
        if self.cate_fill_na is not None:
            df[self.cate_fill_na] = df[self.cate_fill_na].fillna("(Missing)").replace("nan", "(Missing)")
        if self.toomany is not None:
            df[up.add(self.toomany, "_ENCODED")] = self._toomany_encoder.transform(df[self.toomany])
        return df
    
pipe_etl = Pipeline(steps=[("etl", Etl(derive_day_of_month=True, 
                                       miss=miss, cate_fill_na=cate, toomany=toomany))])

'''
# Example with just a simple preparing function which is called at begin of training and scoring 
# and target encoding done later on whole training data
def etl(df, miss, cate, toomany):
    df["day_of_month"] = df["dteday"].dt.day.astype("str").str.zfill(2)
    df[up.add("MISS_", miss)] = pd.DataFrame(np.where(df[miss].isnull(), "No", "Yes"))
    df[cate] = df[cate].fillna("(Missing)").replace("nan", "(Missing)")
    df[up.add(toomany, "_ENCODED")] = df[toomany]
    return df

# Example with default pipeline and "pandarizer"
class TargetEncoderUtil(target_encoder.TargetEncoder):
    def __init__(self, X_util=None, y_util=None, **kwargs):
        super().__init__(**kwargs)
        self.X_util = X_util
        self.y_util = y_util
    def fit(self, X, y, **kwargs):
        super().fit(self.X_util, self.y_util, **kwargs)
        return self
pipe_etl2 = Pipeline(steps=[
    ("column_transform_1", ColumnTransformer([
        ("day_of_month", FunctionTransformer(lambda x: x.apply(lambda y: y.dt.day.astype("str").str.zfill(2))),
         ["dteday"]),
        ("miss_indicator", FunctionTransformer(lambda x: np.where(x.isnull(), "No", "Yes")),
         miss),
        ("copy_miss", FunctionTransformer(lambda x: x),
         miss),
        ("cate_fill_na", FunctionTransformer(lambda x: x.fillna("(Missing)").replace("nan", "(Missing)")),
         [x for x in cate if x not in ["day_of_month"]])
    ])),
    ("to_df_1", FunctionTransformer(lambda x: pd.DataFrame(x, columns=(["dteday"] +
                                                                     up.add("MISS_", miss) + miss +
                                                                     up.diff(cate, ["day_of_month"]))))),
    ("column_transform_2", ColumnTransformer([
        ("target_encoding", TargetEncoderUtil(X_util=df_util[toomany], y_util=df_util["target"]),
         toomany),
        ("copy_target_encoding", FunctionTransformer(lambda x: x),
         toomany)],
        remainder="passthrough")
     ),
    ("to_df_2", FunctionTransformer(lambda x: pd.DataFrame(x, columns=(up.add(toomany, "_ENCODED") + toomany +
                                                                       ["dteday"] +
                                                                       up.add("MISS_", miss) + miss +
                                                                       up.diff(cate, toomany + ["day_of_month"])))))
])
pipe_etl2.fit_transform(df_train, df_train["target"])
'''

# Numerical pipeline including ordinal encoding for categorical features on ordinal scale 
# All custom functions must be imported for pickle to get loaded during scoring
pipe_numerical = Pipeline(steps=[
    ("column_transform", ColumnTransformer(transformers=[        
        ("log_trafo", FunctionTransformer(func=lambda x: np.log(x + 1)), 
         ["hum"]),  
        ("ordinal_encoding", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), 
         yesno)  # add ordi (and move to pipe_numerical below)?
        #("target_encoding", target_encoding.TargetEncoder()), up.add(toomany, "_ENCODED"))
    ], remainder="passthrough")),
    ("impute", SimpleImputer(strategy="median"))  # might add additional winsorizing or scaling in case of elasticnet
])

# Categorical pipeline
pipe_categorical = Pipeline(steps=[
    ("column_transform", ColumnTransformer(transformers=[
        ("collapse_toomany", up.Collapse(n_top=5), 
         toomany)
    ], remainder="passthrough")),
    ("one_hot", OneHotEncoder(sparse=True, handle_unknown="ignore"))
])  

# Complete pipeline
pipeline = Pipeline([
    ('etl', pipe_etl),
    ('fe', ColumnTransformer(transformers=[
        ('nume', pipe_numerical, nume + yesno + up.add(toomany, "_ENCODED")),
        ('cate', pipe_categorical, nomi + ordi)
    ])),
    ('algo', up.UndersampleEstimator(xgb.XGBRegressor(**dict(n_estimators=1100, learning_rate=0.01,
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

from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
cv = KFold()
cross_val_score(pipeline, df_train, df_train["target"],
                cv=cv, scoring=up.D_SCORER["CLASS"]["auc"], n_jobs=3)
                
fit = (GridSearchCV(pipeline,
                            {"algo__n_estimators": [x for x in range(100, 500, 100)]},
                            cv=cv.split(df_train),
                            refit=False,
                            scoring=up.D_SCORER["CLASS"],
                            return_train_score=True,
                            n_jobs=1)
       .fit(df_train, df_train["target"]))
up.plot_cvresults(fit.cv_results_, metric="auc", x_var="algo__n_estimators")
'''

# --- Save -------------------------------------------------------------------------------------------------------------

with open(s.DATALOC + "4_train.pkl", "wb") as file:
    dill.dump({"pipeline": pipeline_fit}, file)  # cannot pickle lambda functions, so need dill


