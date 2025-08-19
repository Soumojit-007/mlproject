# will have the common functionalities
import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path= os.path.dirname(file_path)
        os.makedirs(dir_path , exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            param = params[model_name]   # get that model’s params

            # ✅ Hyperparameter tuning with GridSearchCV
            gs = GridSearchCV(model, param, cv=3, n_jobs=-1, verbose=1)
            gs.fit(X_train, y_train)

            # Best model from grid search
            best_model = gs.best_estimator_

            # Train best model on training data
            best_model.fit(X_train, y_train)

            # Predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Save best test score
            report[model_name] = test_model_score

        return report   

    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj) 
    except Exception as e:
        raise CustomException(e,sys)