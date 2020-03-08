import pandas as pd
import numpy as np
# =============================================================================
# This class is based on:
#   * data_exploration_analysis.ipnb
#   * ML_nan.ipnb
# =============================================================================

class Preprocess():

    filling = {
            'Self_Employed': 'No',
            'Gender': 'Male',
            'Loan_Amount_Term': 341.91,
            'Credit_History': 1,
            'Dependents': '0'
            }
    medians = pd.DataFrame([[130,113], [157.5,130]],
                           columns=['Graduate', 'Not Graduate'],
                           index=['No', 'Yes'])
    try:
        loan_amount_term_model = pd.read_pickle(r'models/Loan_Amount_Term.pkl')
        loan_amount_model = pd.read_pickle(r'models/LoanAmount.pkl')
        married_model = pd.read_pickle(r'models/Married.pkl')
    except:
        pass
    
    def __init__(self, data, mode):
        self.df = data.drop(['Loan_ID'], axis=1)
        self.mode = mode
    
    def dummy_handler(self, train_cols, pred_cols, data):
        # Get missing columns in the training test
        missing_cols = set(train_cols) - set(pred_cols)
        # Add a missing column in test set with default value equal to 0
        for c in missing_cols:
            data[c] = 0
        # Ensure the order of column in the test set is in the same order than in train set
        data = data[train_cols]
        return data
    
    def process_data(self):
        data = self.df
        data['TotalIncome'] = np.log(data['ApplicantIncome'] + data['CoapplicantIncome'])

        data['Credit_History'] = data['Credit_History'].fillna(2)
        data['Credit_History'] = data['Credit_History'].astype(str)
        self.df = data


    def get_helper_data(self, target_var):
        self.process_data()
        df = self.df
        # I dont want helper models to have some bias from filling NA
        df.dropna(inplace=True)
        df.drop('Loan_Status', axis=1, inplace=True)
        
        y = df[target_var]
        x = df.drop([target_var], axis=1)
        X = pd.get_dummies(x)
        return X, y
    
    def get_data_central_tendency(self):
        def fill_median(x): 
            return self.medians.loc[x['Self_Employed'],x['Education']]
        
        mode = self.mode
        self.process_data()
        df = self.df
        
        df['Self_Employed'].fillna(self.filling.get('Self_Employed'), inplace=True)
        df['Gender'].fillna(self.filling.get('Gender'), inplace=True)
        df['Loan_Amount_Term'].fillna(self.filling.get('Loan_Amount_Term'), inplace=True)
        df['Dependents'].fillna(self.filling.get('Dependents'), inplace=True)
        
        df['LoanAmount'][df['LoanAmount'].isnull()] = df[df['LoanAmount'].isnull()].apply(fill_median, axis=1)
        df['loan_ratio'] = df['LoanAmount']/df['Loan_Amount_Term']
        df['loan-income_ratio'] = df['TotalIncome']/df['LoanAmount']
        df['balance_income'] = df['TotalIncome'] - (df['loan_ratio']*1000)
        df.drop(['ApplicantIncome', 'CoapplicantIncome',], axis=1, inplace=True)
        
        if ( (mode=='train') ):
            df.dropna(inplace=True)
            
            y = df['Loan_Status'].map({'Y':1, 'N':0})
            x = df.drop('Loan_Status', axis=1)
            X = pd.get_dummies(x)
            return X, y
        else:
            x = pd.get_dummies(df)
            return x
        
    def pred_missing(self, col, data, model):
        p_model = model['model']
        df = data[data[col].isna()==True]
        #if there are no missing values
        if len(df)==0:
            return data
        idx = df.index
        x = df.drop(col, axis=1)
        try:
            x = x.drop('Loan_Status', axis=1)
        except:
            pass
        x = pd.get_dummies(x)
        x = self.dummy_handler(model['cols'], list(x.columns), x)
        
        pred = pd.Series(p_model.predict(x), index=idx)
        if col=='Married':
            pred = pred.map({1:'Yes', 0:'No'})
        data[col] = data[col].combine_first(pred)
        return data
        
    def get_data_ml(self):
        mode = self.mode
        self.process_data()
        df = self.df
        
        df['Self_Employed'].fillna(self.filling.get('Self_Employed'), inplace=True)
        df['Gender'].fillna(self.filling.get('Gender'), inplace=True)
        df['Dependents'].fillna(self.filling.get('Dependents'), inplace=True)
        
        df = self.pred_missing('Loan_Amount_Term', df, self.loan_amount_term_model)
        df = self.pred_missing('LoanAmount', df, self.loan_amount_model)
        df = self.pred_missing('Married', df, self.married_model)
        
        df['loan_ratio'] = df['LoanAmount']/df['Loan_Amount_Term']
        df['loan-income_ratio'] = df['TotalIncome']/df['LoanAmount']
        df['balance_income'] = np.exp(df['TotalIncome']) - (df['loan_ratio']*1000)
        df.drop(['ApplicantIncome', 'CoapplicantIncome',], axis=1, inplace=True)
        
        if mode=='train':
            y = df['Loan_Status'].map({'Y':1, 'N':0})
            x = df.drop('Loan_Status', axis=1)
            X = pd.get_dummies(x)
            return X, y
        else:
            X = pd.get_dummies(df)
            return X
        
        
