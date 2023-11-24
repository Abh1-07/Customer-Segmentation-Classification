import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.metrics import r2_score
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        # saving the pickle model to desired place
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
        logging.info('Pickle file made and dumped with data')
    except Exception as e:
        raise CustomException(e,sys)