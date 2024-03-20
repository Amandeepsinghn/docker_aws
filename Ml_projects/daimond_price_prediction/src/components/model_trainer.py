import os 
import sys

from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.metrics import r2_score

from src.exception import CustomException

from src.logger import logging

from src.utils import save_object,evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifact','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_training_config=ModelTrainerConfig()
        
    def intiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('split training and test input data')
            
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            
            model_report=evaluate_model(X_train,y_train,X_test,y_test,models=RandomForestRegressor())
            
            model=RandomForestRegressor()
            
            save_object(file_path=self.model_training_config.trained_model_file_path,
                        obj=model)
            
            
            
            return model_report
            
        except Exception as e:
            raise CustomException(e,sys)
        