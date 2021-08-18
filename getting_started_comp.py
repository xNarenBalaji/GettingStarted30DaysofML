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

from xgboost import XGBRegressor
# =============================================================================
# Reading Data
# =============================================================================
def TrainLocation():
    
    pathlocation = 'Not Chosen'
    pathlocation = input("Location of dataset (home or away): ")
    if pathlocation.lower() == 'home':
        base_path = "C:\\Users\\NarenBalaji\\JB\\ML\\Kaggle\\GettingStarted30DaysofML"
    elif pathlocation.lower() == 'away':
        base_path ="D:\\Naren Balaji\\External Data\\Kaggle\\ml_30days"
    elif pathlocation.lower() == 'kaggle':
        base_path = ""
    else:
        print("Please choose correct location")
        sys.exit()
    print(base_path)
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    
    train = pd.read_csv(base_path+"\\"+"train.csv",index_col='id')
    test = pd.read_csv(base_path+"\\"+"test.csv",index_col='id')
    
    return [train,test]

# =============================================================================
# Segregating data
# Training target values in 'y'
# Training feature values in 'features'
# =============================================================================
def DataSegregation(train,test):
    #train.isna().any() #Checking for missing values
    train = train.dropna(axis=0) 
    y_train = train['target']
    X_full = train.drop(['target'],axis=1)
    X_full.head()
    X_test = test

    # Checking for missing values
    # cols_with_missing = [col for col in X_full.columns if X_full[col].isnull().any()]
    # No missing values
    
    # Separating categorical and numeric features
    obj_cols = [col for col in X_full.columns if 'cat' in col]
    num_cols = [col for col in X_full.columns if 'cont' in col]
    
    return [X_full, y_train, X_test, obj_cols, num_cols]
    

def DataProcessorChoice(num_cols, obj_cols, choice=0):
    
    if choice == 1:
        num_transformer = SimpleImputer(strategy='mean')
        cat_transformer = Pipeline(
            steps=[('imputer', SimpleImputer(strategy='most_frequent'))
                   ,('ordinal',OrdinalEncoder(handle_unknown='error'))
                   ])
        preprocessor = ColumnTransformer(transformers=
                                         [('cont', num_transformer, num_cols)
                                          ,('cat',cat_transformer, obj_cols)
                                          ])
    else: 
        num_transformer = SimpleImputer(strategy='mean')
        cat_transformer = Pipeline(
            steps=[('imputer', SimpleImputer(strategy='most_frequent'))
                   ,('onehot',OneHotEncoder(handle_unknown='ignore'))
                   ])
        preprocessor = ColumnTransformer(transformers=
                                         [('cont', num_transformer, num_cols)
                                          ,('cat',cat_transformer, obj_cols)
                                          ])
        
    return preprocessor
    

def ModelChoice(choice=0):
    
    if choice==1:
        print("do what?")
    else:
        model = XGBRegressor(n_estimators=10000,
                             learning_rate = 0.25, random_state=1)
        
    return model
# =============================================================================
# Imputing for missing values, however train/valid/test do not have any
# =============================================================================

def model_choice(i,n_estimators=10):
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    
    if i == 1:
        model = RandomForestRegressor(n_estimators=100, random_state=1)
        my_pipeline = Pipeline(steps=[('preprocessor',preprocessor),
                                      ('model',model)])
        scores = -1 * cross_val_score(my_pipeline, X_full, y_train, 
                                      cv=5, n_jobs = -1, 
                                      scoring='neg_root_mean_squared_error')
    
    elif i == 2:
        model = XGBRegressor(n_estimators=n_estimators,random_state=1,
                             early_stopping_rounds=10)
        my_pipeline = Pipeline(steps=[('preprocessor',preprocessor),
                                      ('model',model)])
        scores = -1 * cross_val_score(my_pipeline, X_full, y_train, 
                                      cv=5, n_jobs = -1,
                                      scoring='neg_root_mean_squared_error')
    else:
        model = RandomForestRegressor(n_estimators=50, random_state=1)
        my_pipeline = Pipeline(steps=[('preprocessor',preprocessor),
                                      ('model',model)])
        scores = -1 * cross_val_score(my_pipeline, X_full, y_train, 
                                      cv=5, n_jobs = -1, 
                                      scoring='neg_root_mean_squared_error')
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    
    print(scores.mean())
    return scores.mean()

#def ModelFit(model, preprocessor):
    
    
# =============================================================================
# Training model
# =============================================================================


if __name__ == "__main__":
    [train,test] = TrainLocation()
    [X_full, y_train, X_test, 
     obj_cols, num_cols] =  DataSegregation(train, test)
    preprocessor = DataProcessorChoice(num_cols, obj_cols, choice=1)
    model = ModelChoice(choice=0)
    
    pipe = Pipeline([
        ('preprocessor',preprocessor), 
        ('model', model)
        ])
    
    pipe.fit(X_full,y_train)
    
    y_train_pred = pipe.predict(X_full)
    
    print(f"Predictions on training data: {y_train_pred}")
    
    mean_squared_error(y_train, y_train_pred, squared=False)
    
    y_test_pred = pipe.predict(X_test)
    
    print(f"Predictions on test data: {y_test_pred}")
    

# =============================================================================

# Use the model to generate predictions
# predictions = my_pipeline.predict(test)


#### To submit predictions

# =============================================================================
# # Save the predictions to a CSV file
# output = pd.DataFrame({'Id': test.index,
#                        'target': y_test_pred})
# output.to_csv('submission.csv', index=False)
# =============================================================================
    
#### Unused code

# =============================================================================
# Seaborn EDA
# =============================================================================
#sns.lineplot(data=features[numeric_cols[1]])
#features[numeric_cols].describe()
#sns.scatterplot(x=features.cont9,y=features.cont10)



