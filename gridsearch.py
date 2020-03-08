# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:55:48 2020

@author: Marek
"""

import pandas as pd
import numpy as np
import preprocess as p
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

name = 'gridsearch_xgb{0}_some_dropped.csv'.format(np.random.rand(1,1)[0][0])
train = pd.read_csv(r'data/train.csv')
#test = pd.read_csv(r'data/test.csv')
#model = SVC(tol=1e-5, max_iter=10000)
#model = RandomForestClassifier(n_jobs=4)
model = XGBClassifier()

## SVC
#params = {
#        'C': [0.3, 0.5, 1.3, 3],
#        'class_weight':['balanced', {0:1.05, 1:1.5}, {0:1.1, 1:1.8}, None],
#        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
#        'degree':[2,3,5,8],
#        'coef0':[0,0.01,0.1,1],
#        'gamma':['scale', 'auto'],
#        }

## RF
#params = {
#        'n_estimators':[50,100,300],
#        'max_depth':[2, 4, 6],
##        'criterion':['gini', 'entropy'],
##        'min_samples_split': [1,2,3],
##        'min_samples_leaf':[1,3],
##        'min_weight_fraction_leaf':[0.0,0.1],
##        'max_features':["auto", 'log2', None],
##        'max_leaf_nodes':[1,3,None],
#        'bootstrap':[True, False],
#        'class_weight':['balanced', {0:1.05, 1:1.5}, {0:1.1, 1:1.8}, None]
#        }
## XGB
params = {
        'learning_rate': [0.01, 0.05],
#        'gamma':[0,0.3,1],
        'n_estimators':[100, 300, 500],
        'max_depth':[2,4,6],
        'min_child_weight':[0,1,3],
        'reg_lambda': [0, 1, 3],
        'reg_alpha':[0, 1, 3],
        'class_weight':['balanced', {0:1.05, 1:1.5}, None]
        }

proc = p.Preprocess(train, mode='train')
x, y = proc.get_data_central_tendency()

scaler = StandardScaler().fit(x)
x = scaler.transform(x)

gs = GridSearchCV(model, params, n_jobs=4, scoring='accuracy', verbose=10,
                  cv=StratifiedKFold(10, shuffle=True))
gs.fit(x, y)
df = pd.DataFrame(gs.cv_results_)
df.to_csv(name)





