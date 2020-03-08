# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:11:42 2020

@author: Marek
"""

import pandas as pd
import preprocess as p

mod_name = 'final_rf_50_ct.pkl'
test = pd.read_csv(r'Data/test.csv')
submission = pd.read_csv(r'Data/sample_submission.csv')
model = pd.read_pickle(r'models/{}'.format(mod_name))

proc = p.Preprocess(test, mode='test')
x = proc.get_data_central_tendency()

x = proc.dummy_handler(model['cols'], list(x.columns), x)

X = model['scaler'].transform(x)
pred = model['model'].predict(X)

submission['Loan_Status'] = pred
submission['Loan_Status'] = submission['Loan_Status'].map({1:'Y', 0:'N'})

ts = pd.Timestamp.now().strftime('%d-%m-%H%M')
submission.to_csv(r'submissions/submission'+ts+'.csv', index=False)

