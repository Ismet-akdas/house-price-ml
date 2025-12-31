import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# -------------------- DATA LOAD --------------------

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')


#print(train.info())
print(train.shape)



# -------------------- MISSING VALUES --------------------

missing = train.isnull().sum()
missinga = missing[missing > 0].sort_values(ascending=False)

missing_ratio = missinga/ len(train)
missing_ratio = missing_ratio[missing_ratio > 0.06].sort_values(ascending=False)


for i in missing_ratio.index:

    testing = train.groupby(train[i].isnull())["SalePrice"].median()

drop_cols = ["MiscFeature", "Alley", "Fence" , "LotFrontage"]



train = train.drop(columns=drop_cols)
test = test.drop(columns=drop_cols) 



# -------------------- TARGET TRANSFORM --------------------

target = train["SalePrice"].describe()
train["SalePrice"] = np.log1p(train["SalePrice"])




# -------------------- FEATURE SELECTION --------------------

numeric_features = train.select_dtypes(include=[np.number])
categorical_features = train.select_dtypes(include=[object])

print(numeric_features.columns)
corr = numeric_features.corr()["SalePrice"].sort_values(ascending=False) 
corr_features = corr[abs(corr) > 0.5].index

drop_list = numeric_features.columns.difference(corr_features)

train = train.drop(columns=drop_list)
test = test.drop(columns=drop_list)



# -------------------- SPLIT --------------------

X= train.drop("SalePrice" , axis =1)
y=train["SalePrice"]

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)





from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -------------------- PIPELINE --------------------

numeric_features = X.select_dtypes(include=[np.number]).columns
categorical_features = X.select_dtypes(include=[object]).columns

num_pip= Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pip= Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pip, numeric_features),
        ('cat', cat_pip, categorical_features)
    ])

from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])


num_features = X.select_dtypes(include=[np.number]).columns
cat_features = X.select_dtypes(include=[object]).columns



preprocess = ColumnTransformer(
    transformers=[
        ('num', num_pip, num_features),
        ('cat', cat_pip, cat_features)
    ])

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    
    n_jobs=-1
)
from sklearn.pipeline import Pipeline

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("rf", RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_features=0.7,
        n_jobs=-1
    ))
])
# -------------------- TRAIN --------------------


model.fit(X_train, y_train)

y_pred = model.predict(X_val)
tryit = y_pred[0]
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

rmse = root_mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)

print("RMSE:", rmse)
print("MAE:", mae)


# -------------------- BASELINE --------------------

from sklearn.dummy import DummyRegressor

dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)
pred_dummy = dummy.predict(X_val)
rmae_dummy = root_mean_squared_error(y_val, pred_dummy)
print("RMSE Dummy:", rmae_dummy)
mae_dummy = mean_absolute_error(y_val, pred_dummy)
print("MAE Dummy:", mae_dummy)







y_pred = model.predict(X_val)
errors = y_val - y_pred  
abs_errors = np.abs(errors) 
print(abs_errors.describe())

# -------------------- FINAL TRAIN --------------------


X_full = pd.concat([X_train, X_val])
y_full = pd.concat([y_train, y_val])

model.fit(X_full, y_full)



# -------------------- SAVE MODEL --------------------


import joblib
joblib.dump(model, "../models/model.joblib")

