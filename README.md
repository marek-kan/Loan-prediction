# Loan-prediction
Hackathon at Analytics Vidhya

## Description
These data are great for excercise how to handle missing values in training dataset and also in testing one. My first step was exploration of the data, data_exploration_analysis.ipynb contains notes. 
Then I preformed quick check of what kind of task it is and which type of estimator could perform the best. For this task I created model_selection.ipynb. 
When I had some idea what could work I tried which treatment of missing values is more promissing. I was deciding between filling NAs with central tendencies or ML approach (centralTendency_vs_ML.ipynb). Even if this notebook shows that taking ML approach is probably waste of time I implemented it in preprocess.py and train_helper_models.py for sake of excercise. 

Script preprocess.py contains class which is designed to deal with preprocessing steps which are necessary. After that I performed gridsearch to confirm in bigger scale that ML approach performs worse. The best estimator was Random Forest with mean test accuracy 0.8143. The difference between ML and central tendency method is marginal but ML approach suffers with higher StDev of test accuracy.

train_*.py are scripts for training and exporting models which can be used by predict.py and preprocess.py
