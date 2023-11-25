import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer # Used to create pipeline for transforming cat and num features
from sklearn.impute import  SimpleImputer # TO handle missing values
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformConfig:
    preprocessor_obj_path = os.path.join('Artifacts','preprocessor.pkl')


class DataTransform:
    def __init__(self) -> None:
        self.data_transform_config = DataTransformConfig()
    
    def get_data_transformer_object(self):
        try:
            numerical_features = ['Ever_Married', 'Age', 'Graduated', 'Work_Experience', 'Family_Size', 'Ismale']
            categorial_features = ['Profession', 'Spending_Score', 'Var_1']
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')), # to handle Missing Values
                    ('scaler',StandardScaler()) # for data transformation
                ]
            )
            logging.info(f'Numerical columns:{numerical_features}')
            cat_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='most_frequent')),# to handle Missing Values
                    ('OneHotEncoder',OneHotEncoder()),  #Doing One hot encoding
                    ('scaler',StandardScaler(with_mean=False)) # to transform data not much needed but still used
                ]
            )
            logging.info(f'Categorial columns {categorial_features}')
            # combining both pipelines together
            preprocessor = ColumnTransformer(
                [
                    ('Num pipeline',num_pipeline,numerical_features),
                    ('Cat pipeline', cat_pipeline, categorial_features)
                ]
            )
            return (
                preprocessor
            )
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading Train and Test data Completed')
            logging.info('Obtaining Preprocessing Object')

            preprocessing_obj = self.get_data_transformer_object()
            target_column = 'Segmentation'
            numerical_features = ['Ever_Married', 'Age', 'Graduated', 'Work_Experience', 'Family_Size', 'Ismale']
            categorial_features = ['Profession', 'Spending_Score', 'Var_1']
            #making df for as x_train, x_test, y_train, y_test
            input_feature_train_df = train_df.drop(columns = [target_column],axis = 1)
            target_feature_train_df = train_df[target_column]
            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info('Applying preprocessing object to training and testing DataFrame')
                # calling the saved pickle file as preprocessing_obj and doing fit_tranform on training dataset.
                # and transform on test dataset
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
                # combining the dataset and the transformed data of training and test set as array
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info('Saved Preprocessed Objects')
            save_object(file_path =  self.data_transform_config.preprocessor_obj_path,obj = preprocessing_obj)

            return(
                train_arr,
                test_arr,
                self.data_transform_config.preprocessor_obj_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)