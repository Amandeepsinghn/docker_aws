import sys 
from dataclasses import dataclass

import numpy as np 
import pandas as pd 

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging 
import os 

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifact','preprocessor.pkl')
    

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    
    def get_data_transformer_object(self):
        logging.info('transformation has started')
        try:
            numerical_columns=['carat','depth','table','x','y','z']
            categorical_columns=['cut','color','clarity']
            
            num_pipeline=Pipeline(steps=[
               ("imputer",SimpleImputer(strategy='median')),
               ('scaler',StandardScaler()) 
                
                
                ]
            )
            
            
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f'cateforical_columns are {categorical_columns}')
            logging.info(f'numerical_columns are {numerical_columns}')
            
            
            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )           
            return preprocessor
        

        except Exception as e:
            raise CustomException(e,sys)
        
        
    def intiate_transformation(self,train_path,test_path):
        
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("reading train and test dataset is completed")
            
            logging.info('obtaining preprocessing object')
            
            preprocessor_obj=self.get_data_transformer_object()
            
            target_column_name='price'
            
            numerical_columns=['carat','depth','table','x','y','z']
            
            input_feature_train=train_df.drop(target_column_name,axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test=test_df.drop(target_column_name,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            logging.info('applying precoressing on training and testing dataframe')
            
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test)
            
            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
        
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
        
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
        
