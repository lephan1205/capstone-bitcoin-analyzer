#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 23:39:29 2019

@author: lephan
"""

import numpy as np
import pandas as pd
from datetime import datetime
from fastparquet import write
import time

import finml as fn


## --- Read csv files and reformat ---
start = time.time()
data = pd.read_csv('data/20190722.csv')
data = data[data.symbol == 'XBTUSD']
paths = ['data/20190723.csv', 'data/20190724.csv',
         'data/20190725.csv', 'data/20190726.csv']
for path in paths:
    df = pd.read_csv(path)
    df = df[df.symbol == 'XBTUSD']
    data = data.append(df)
print(time.time() - start)
#timestamp parsing
start = time.time()
data['timestamp'] = data.timestamp.map(lambda t: datetime.strptime(t[:-3], 
                                        "%Y-%m-%dD%H:%M:%S.%f")) 
print(time.time() - start)
data.set_index('timestamp', inplace=True)
data.sort_index(inplace=True)

# sampling data
sampled_df = fn.dollar_sampling(data)

# labeling
labeled_df = fn.labeling(sampled_df)

# save to file
write('labeled_data.pq', labeled_df)


##======= Modeling =============

# columns to keep and features
cols = ['volume','open', 'high', 'low', 'close', 'retp1', 'threshold']

# split data into training and test sets
X = data_ohlc[cols].values
y = np.squeeze(data_ohlc[['label']].values)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
X.shape, X_train.shape, X_test.shape

# classifier
rf = RandomForestClassifier()

## Grid search for tuning parameters
#rf_grid = GridSearchCV(
#    rf,
#    {"n_estimators": [100, 125, 150],
#     "max_depth" : [8, 10, 12],
#     "criterion" : ['gini', 'entropy']},
#    cv=5,     # 5-fold cross-validation
#    n_jobs=2, # run each hyperparameter in one of two parallel jobs
#)
#rf_grid.fit(X_train, y_train)
#print(rf_grid.best_params_)

# Predict with random forest using balanced resampling
rf = RandomForestClassifier(n_estimators=150, max_depth=12, class_weight='balanced', 
                            criterion='gini', random_state=42)

# fit and predict
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# prediction accuracy
rf.score(X_test, y_test)

pd.Series(y_pred).value_counts()


