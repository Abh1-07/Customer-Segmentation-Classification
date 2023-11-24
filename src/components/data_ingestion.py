import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransform


@dataclass
class DataIngestionConfig:
    train_path = os.path.join('Artifacts',"train.csv")
    test_path = os.path.join('Artifacts',"test.csv")
    raw_path = os.path.join('Artifacts',"raw.csv")

class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Enter the Data INgestion Method')
        try:
            df = pd.read_csv("notebooks\\final.csv")
            logging.info("Data extracted and Read as DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_path,index = False,header = True)

            logging.info('Train Test Split Initiated')
            train_test, test_test = train_test_split(df, test_size=0.2, random_state=42)
            train_test.to_csv(self.ingestion_config.train_path, index = False, header = True)
            test_test.to_csv(self.ingestion_config.test_path, index = False, header = True)
            logging.info('DataIngestion Completed')
            return (
                self.ingestion_config.train_path,
                self.ingestion_config.test_path
                )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__main__':

    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transform = DataTransform()
    train_arr, test_arr,_= data_transform.initiate_data_transformation(train_data, test_data)