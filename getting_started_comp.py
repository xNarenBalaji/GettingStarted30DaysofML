# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 11:23:51 2021

@author: NarenBalaji
"""

# =============================================================================
# Importing libraries
# =============================================================================
import numpy as np
import pandas as pd
import seaborn as sns

import os
import sys
from datetime import datetime

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# =============================================================================
# Reading Data
# =============================================================================

pathlocation = 'Not Chosen'
pathlocation = input("Location of dataset (home or away): ")
if pathlocation.lower() == 'home':
    base_path = ""
elif pathlocation.lower() == 'away':
    base_path ="D:\\Naren Balaji\\External Data\\Kaggle\\ml_30days"
else:
    print("Please choose correct location")
    sys.exit()
print(base_path)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

train = pd.read_csv("train.csv",index_col='id')
test = pd.read_csv("test.csv",index_col='id')

# =============================================================================
# Segregating data
# Training target values in 'y'
# Training feature values in 'features'
# =============================================================================
pd.set_option('display.max_columns', None)
train.head()

y = train['target']
features = train.drop(['target'],axis=1)
features.head()

# Checking for missing values
cols_with_missing = [col for col in features.columns
                     if features[col].isnull().any()]
# No missing values

# Separating categorical and numeric features
object_cols = [col for col in features.columns if 'cat' in col]
numeric_cols = [col for col in features.columns if 'cont' in col]


# =============================================================================
# Seaborn EDA
# =============================================================================
#sns.lineplot(data=features[numeric_cols[1]])
#features[numeric_cols].describe()
#sns.scatterplot(x=features.cont9,y=features.cont10)


# =============================================================================
# Train_test_split
# =============================================================================
#X_train, X_valid, y_train, y_valid = train_test_split(features, y, train_size)



# =============================================================================
# Imputing for missing values, however train/valid/test do not have any
# =============================================================================
numerical_transformer = SimpleImputer(strategy='mean')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numeric_cols),
        ('cat', categorical_transformer, object_cols)
    ])

model = RandomForestRegressor(n_estimators=100, random_state=1)

my_pipeline = Pipeline(steps=[('preprocessor',preprocessor),('model',model)])

print('reached here')
scores = -1 * cross_val_score(my_pipeline, features, y, cv=5, 
                              scoring='neg_root_mean_squared_error')

print(scores)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)