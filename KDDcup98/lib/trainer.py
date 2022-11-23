'''
Contains all methods to do train multiple models for this project.
This notebooks splits the Data into train and test
performs PCA on the data
Balances the dataset using simple Undersampling, ADASYN Oversampling, SMOTE Oversampling
Train models on 4 different methods, Logistic Regression, Decision Trees, Random Forest, XGBoost
'''

import numpy as np
import pandas as pd
import operator

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from imblearn import over_sampling
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

#preprocessing libraries
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler




from lib.utils import Performance

class Trainer:

    @staticmethod
    def train_model(df):
        
        #split dat into X,y
        #Removing the target variable from the dataset
        X=df.drop(columns=['TARGET_B','TARGET_D'],axis=1)
        y=df['TARGET_B']

       
        #Lets normalize the data before PCA as a prerequisite
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        #Lets create PCA from the scaled data
        pca = PCA(n_components=50)
        principalComponents = pca.fit_transform(X)
        pca_X = pd.DataFrame(data = principalComponents)
               
       #Split the data into training and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                           random_state = 42,
                                           stratify=y)
                                           
        
        #Defining 4 models for training
       

        #Model 1 - Logistic Regression
        
      
        # Creating KFold object with 5 splits
        folds = KFold(n_splits=5, shuffle=True, random_state=42)

        # Specify params
        params = {"C": [0.01, 0.1, 1, 10, 100, 1000]}

        # Specifing score as roc-auc
        lr_grid_search = GridSearchCV(estimator = LogisticRegression(),
                                param_grid = params, 
                                scoring= 'roc_auc', 
                                cv = folds, 
                                verbose = 1,
                                return_train_score=True) 

        #Model 2 - Decision Trees

        # Create the parameter grid 
        param_grid = {
            'max_depth': range(5, 15, 5),
            'min_samples_leaf': range(50, 150, 50),
            'min_samples_split': range(50, 150, 50),
        }


        # Instantiate the grid search model
        dtree = DecisionTreeClassifier()

        dt_grid_search = GridSearchCV(estimator = dtree, 
                                   param_grid = param_grid, 
                                   scoring= 'roc_auc',
                                   cv = 5, 
                                   n_jobs=-1,
                                   verbose = 1)

        #Model 3 - Random Forest

        # Create the parameter grid 
        param_grid = {
            'max_depth': range(5, 15, 5),
            'min_samples_leaf': range(50, 150, 50),
            'min_samples_split': range(50, 150, 50),
        }


        # Instantiate the grid search model
        model = RandomForestClassifier()

        rf_grid_search = GridSearchCV(estimator = model, 
                                   param_grid = param_grid, 
                                   scoring= 'roc_auc',
                                   cv = 5, 
                                   n_jobs=-1,
                                   verbose = 1)

        #Model 4 - XGBoost

        # creating a KFold object 
        folds = 5

        # specify range of hyperparameters
        param_grid = {'learning_rate': [0.2, 0.6], 
                     'subsample': [0.3, 0.6, 0.9]}          


        # specify model
        xgb_model = XGBClassifier(max_depth=2, n_estimators=200)

        # set up GridSearchCV()
        xgb_grid_search = GridSearchCV(estimator = xgb_model, 
                                param_grid = param_grid, 
                                scoring= 'roc_auc', 
                                cv = folds, 
                                verbose = 1,
                                n_jobs=-1,
                                return_train_score=True) 

        # Creates a balanced trainset
        # In classification, some methods perform better with bal datasets,
        # particularly tree-based methods like decision trees and random forests.

        print("########## Balancing method - RANDOM Undersampling ##########")

        # define undersample strategy
        undersample = RandomUnderSampler(sampling_strategy='majority')
        
    
        # fit and apply the transform
        train_bal, y_train_bal = undersample.fit_resample(X, y)



        #### Model 1 | Logistic Regression Model ####

        print("Model 1 | Logistic Regression Model executing...")

        # Fit the model on data balanced using adasyn
        lr_grid_search.fit(train_bal, y_train_bal)
        #print the evaluation result by choosing a evaluation metric
        #print('Best ROC AUC score: ', lr_grid_search.best_score_)

        # Instantiating the model
        lr_model = lr_grid_search.best_estimator_

        # Fitting the model 
        lr_model.fit(train_bal, y_train_bal)

        # Evaluating on test data
        y_test_pred = lr_model.predict(X_test)
        perf_model1 = Performance.get_perf(y_test, y_test_pred)

        # Confusion Matrix
        #print(pd.crosstab(y_test, y_test_pred, rownames = ['actual'], colnames = ['preds']))

        # Gets performance
        perf_model1 = Performance.get_perf(y_test.values, y_test_pred)

        print("Model 2 | Decision Tree Model executing...")
        #### Model 2 | Decision Tree Model ####


        # Fit the grid search to the data
        dt_grid_search.fit(train_bal, y_train_bal)

        # Model with optimal hyperparameters
        dt_model =dt_grid_search.best_estimator_
        dt_model.fit(train_bal, y_train_bal)

        # Evaluating on test data
        y_test_pred = dt_model.predict(X_test)
        perf_model2 = Performance.get_perf(y_test, y_test_pred)



        #### Model 3 | Random Forest Model  ####

        print("Model 3 | Random Forest Model  executing...")

        # Fit the grid search to the data
        rf_grid_search.fit(train_bal, y_train_bal)

        # Model with optimal hyperparameters
        rf_model =rf_grid_search.best_estimator_
        rf_model.fit(train_bal, y_train_bal)

        # Evaluating on test data
        y_test_pred = rf_model.predict(X_test)
        perf_model3 = Performance.get_perf(y_test, y_test_pred)


        #### Model 4 | XG Boost Model  ####
        print("Model 4 | XG Boost Model  executing...")

        # fit the model
        xgb_grid_search.fit(train_bal, y_train_bal)

        # Model with optimal hyperparameters
        xgb_model =xgb_grid_search.best_estimator_
        xgb_model.fit(train_bal, y_train_bal)

        # Evaluating on test data
        y_test_pred = xgb_model.predict(X_test)
        perf_model4 = Performance.get_perf(y_test, y_test_pred)


        print("########## Balancing method - ADASYN Oversampling ##########")

        ada = over_sampling.ADASYN(random_state=42)
        X_train_adasyn, y_train_adasyn = ada.fit_resample(X_train, y_train)



        #### Model 5 | Logistic Regression | ADASYN ####
        print("Model 5 | Logistic Regression | ADASYN   executing...") 

        # Fit the model on data balanced using adasyn
        lr_grid_search.fit(X_train_adasyn, y_train_adasyn)
        #print the evaluation result by choosing a evaluation metric
        #print('Best ROC AUC score: ', model_cv.best_score_)

        # Instantiating the model
        logreg_adasyn_model = lr_grid_search.best_estimator_

        # Fitting the model 
        logreg_adasyn_model.fit(X_train_adasyn, y_train_adasyn)

        # Evaluating on test data
        y_test_pred = logreg_adasyn_model.predict(X_test)
        perf_model5 = Performance.get_perf(y_test, y_test_pred)


        #### Model 6 | Decision Tree | ADASYN ####
        print("Model 6 | Decision Tree | ADASYN  executing...") 
        # # Decision Tree on balanced data with ADASYN

        # Fit the grid search to the data
        dt_grid_search.fit(X_train_adasyn,y_train_adasyn)

        # Model with optimal hyperparameters
        dt_adasyn_model =dt_grid_search.best_estimator_
        dt_adasyn_model.fit(X_train_adasyn, y_train_adasyn)

        # Evaluating on test data
        y_test_pred = dt_adasyn_model.predict(X_test)
        perf_model6 = Performance.get_perf(y_test, y_test_pred)



        #### Model 7 |Random Forest| ADASYN ####

        print("Model 7 |Random Forest| ADASYN  executing...") 
        # # RandomForest on balanced data with ADASYN

        # Fit the grid search to the data
        rf_grid_search.fit(X_train_adasyn,y_train_adasyn)

        # Model with optimal hyperparameters
        rf_adasyn_model =rf_grid_search.best_estimator_
        rf_adasyn_model.fit(X_train_adasyn, y_train_adasyn)

        # Evaluating on test data
        y_test_pred = rf_adasyn_model.predict(X_test)
        perf_model7 = Performance.get_perf(y_test, y_test_pred)




        #### Model 8 | XG Boost | ADASYN ####

        print("Model 8 | XG Boost | ADASYN  executing...") 
        # # XGBoost on balanced data with ADASYN

        # fit the model
        xgb_grid_search.fit(X_train_adasyn, y_train_adasyn)

        # Model with optimal hyperparameters
        xgb_adasyn_model =xgb_grid_search.best_estimator_
        xgb_adasyn_model.fit(X_train_adasyn, y_train_adasyn)

        # Evaluating on test data
        y_test_pred = xgb_adasyn_model.predict(X_test)
        perf_model8 = Performance.get_perf(y_test, y_test_pred)




        print("########## Balancing method - SMOTE Oversampling ########## ")

        sm = SMOTE(random_state=42)
        X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)



        #### Model 9 | Logistic Regression | SMOTE ####
        print("Model 9 | Logistic Regression | SMOTE  executing...") 

        # Fit the model on data balanced using adasyn
        lr_grid_search.fit(X_train_smote, y_train_smote)
        #print the evaluation result by choosing a evaluation metric
        #print('Best ROC AUC score: ', model_cv.best_score_)

        # Instantiating the model
        logreg_smote_model = lr_grid_search.best_estimator_

        # Fitting the model 
        logreg_smote_model.fit(X_train_smote, y_train_smote)

        # Evaluating on test data
        y_test_pred = logreg_smote_model.predict(X_test)
        perf_model9 = Performance.get_perf(y_test, y_test_pred)


        #### Model 10 | Decision Tree | SMOTE ####
        print("Model 10 | Decision Tree | SMOTE  executing...") 
        # # Decision Tree on balanced data with ADASYN

        # Fit the grid search to the data
        dt_grid_search.fit(X_train_smote,y_train_smote)

        # Model with optimal hyperparameters
        dt_smote_model =dt_grid_search.best_estimator_
        dt_smote_model.fit(X_train_smote, y_train_smote)

        # Evaluating on test data
        y_test_pred = dt_smote_model.predict(X_test)
        perf_model10 = Performance.get_perf(y_test, y_test_pred)



        #### Model 11 |Random Forest| SMOTE ####

        print("Model 11 |Random Forest| SMOTE  executing...") 
        # # RandomForest on balanced data with ADASYN

        # Fit the grid search to the data
        rf_grid_search.fit(X_train_smote,y_train_smote)

        # Model with optimal hyperparameters
        rf_smote_model =rf_grid_search.best_estimator_
        rf_smote_model.fit(X_train_smote,y_train_smote)

        # Evaluating on test data
        y_test_pred = rf_smote_model.predict(X_test)
        perf_model11 = Performance.get_perf(y_test, y_test_pred)




        #### Model 12 | XG Boost | SMOTE ####

        print("Model 12 | XG Boost | SMOTE  executing...") 
        # # XGBoost on balanced data with smote

        # fit the model
        xgb_grid_search.fit(X_train_smote, y_train_smote)

        # Model with optimal hyperparameters
        xgb_smote_model =xgb_grid_search.best_estimator_
        xgb_smote_model.fit(X_train_smote, y_train_smote)

        # Evaluating on test data
        y_test_pred = xgb_smote_model.predict(X_test)
        perf_model12 = Performance.get_perf(y_test, y_test_pred)




        #### Model comparison ####

        all_models = {'UNDERSAMPLE Decision Trees Model': perf_model1,
                      'UNDERSAMPLE Random Forest Model': perf_model2,
                      'UNDERSAMPLE Logistic Regression Model': perf_model3,
                      'UNDERSAMPLE XGBoost':perf_model4,
                      'ADASYN Decision Trees Model': perf_model6,
                      'ADASYN Random Forest Model': perf_model7,
                      'ADASYN Logistic Regression Model': perf_model5,
                      'ADASYN XGBoost':perf_model8,
                      'SMOTE Decision Trees Model': perf_model10,
                      'SMOTE Random Forest Model': perf_model11,
                      'SMOTE Logistic Regression Model': perf_model9,
                      'SMOTE XGBoost':perf_model12,}

        perf_all_models = pd.DataFrame([[col1, col2, col3 * 100] for col1, d in
            all_models.items() for col2, col3 in d.items()], index = None,
            columns = ['Model Name', 'Performance Metric', 'Value'])

        return (perf_all_models)