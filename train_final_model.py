# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:01:16 2020

@author: Marek
"""

import preprocess as p
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

pth = r'models\\'
data = pd.read_csv(r'data/train.csv')

model = RandomForestClassifier(bootstrap= True, max_depth=6, 
                               n_estimators= 300, class_weight=None)
proc = p.Preprocess(data, mode='train')
x, y = proc.get_data_central_tendency()

cols = list(x.columns)

scaler = StandardScaler().fit(x)
x = scaler.transform(x)

model.fit(x, y)
pred = model.predict(x)
        
train_results = {
        'f1_score': f1_score(y, pred),
        'acc': accuracy_score(y, pred),
        }

result = {
        'model': model,
        'scaler': scaler,
        'results': train_results,
        'cols': cols
        }

pickle.dump(result, open(pth+'final_rf_300_ct'+'.pkl', 'wb'))
print(roc_auc_score(y, model.predict(x)))
