import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)
       
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        
        report=[]
        models.fit(X_train,y_train)
        
        y_train_pred = models.predict(X_train)

        y_test_pred = models.predict(X_test)

        train_model_score = r2_score(y_train, y_train_pred)

        test_model_score = r2_score(y_test, y_test_pred)
        
        report.append(test_model_score)
        
        return report
    except Exception as e:
        raise CustomException(e,sys)
    
        
      
