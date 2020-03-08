import preprocess as p
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier

from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error, r2_score

pth = r'models\\'
train_models = ['Married', 'Loan_Amount_Term', 'LoanAmount']
data = pd.read_csv(r'data/train.csv')

for target_var in train_models:
    proc = p.Preprocess(data, mode='train')
    x, y = proc.get_helper_data(target_var)
    
    if target_var=='Married':
        y = y.map({
                'Yes': 1,
                'No': 0
                })
        model = RandomForestClassifier(n_estimators=500, max_depth=4, class_weight='balanced')
        model.fit(x, y)
        pred = model.predict(x)
        
        train_results = {
                'f1_score': f1_score(y, pred),
                'acc': accuracy_score(y, pred),
                }
        result = {
                'model': model,
                'results': train_results,
                'cols': list(x.columns)
                }
        pickle.dump(result, open(pth+target_var+'.pkl', 'wb'))
        
    if target_var=='Loan_Amount_Term':
        model = XGBClassifier(n_estimators=500, max_depth=6, reg_lambda=4.5,
                              reg_alpha=0.3, class_weight='balanced', n_jobs=-1)
        model.fit(x, y)
        pred = model.predict(x)
        
        train_results = {
                'f1_score': f1_score(y, pred, average='weighted'),
                'MAE': mean_absolute_error(y, pred)
                }
        result = {
                'model': model,
                'results': train_results,
                'cols': list(x.columns)
                }
        pickle.dump(result, open(pth+target_var+'.pkl', 'wb'))
        
    if target_var=='LoanAmount':
        model = RandomForestRegressor(n_estimators=300, max_depth=6)
        model.fit(x, y)
        pred = model.predict(x)
        
        train_results = {
                'r2': r2_score(y, pred),
                'MAE': mean_absolute_error(y, pred)
                }
        result = {
                'model': model,
                'results': train_results,
                'cols': list(x.columns)
                }
        pickle.dump(result, open(pth+target_var+'.pkl', 'wb'))
    