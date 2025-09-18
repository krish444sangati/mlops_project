import pandas as pd
import numpy as np
import os
import joblib
from src.logger import get_logger
from sklearn.model_selection import train_test_split
from src.custom_exception import CustomException

logger= get_logger(__name__)

class Data_Processing():
    def __init__(self,filepath):
        self.filepath=filepath
        self.df=pd.read_csv("artifact/raw/data.csv")
        self.processed_path="artifact/processed"
        os.makedirs(self.processed_path,exist_ok=True)
    def load_data(self):
        try:
            self.df=pd.read_csv("artifact/raw/data.csv")
            logger.info("Read data successfully")
        except Exception as e:
            logger.info(f"Error while reading the file {e}")
            raise CustomException("Reading file is unsuccessful",e)
        
    def handle_outliers(self,column):
        try:
            logger.info("Started handled ouliers")
            self.Q1=self.df[column].quantile(0.25)
            self.Q3=self.df[column].quantile(0.75)
            IQR=self.Q3-self.Q1
            upper_bound=self.Q3 + 1.5*IQR
            lower_bound=self.Q1 - 1.5*IQR
            column_median=np.median(self.df[column])

            for i in self.df[column]:
                if i>upper_bound or i<lower_bound:
                    self.df[column]=self.df[column].replace(i,column_median)
            logger.info("Handled ouliers successfully")
        except Exception as e:
            logger.info(f"Error while handling outliers {e}")
            raise CustomException("Failed to handle outliers" , e)
        
    def split_data(self):
        try:
            x = self.df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
            y = self.df["Species"]

            x_train,x_test,y_train,y_test = train_test_split(x,y , test_size=0.2 , random_state=42)

            logger.info("Data Splitted sucesfullyy....")

            joblib.dump(x_train , os.path.join(self.processed_path , "x_train.pkl"))
            joblib.dump(x_test , os.path.join(self.processed_path , "x_test.pkl"))
            joblib.dump(y_train , os.path.join(self.processed_path , "y_train.pkl"))
            joblib.dump(y_test , os.path.join(self.processed_path , "y_test.pkl"))
            # print(x_train.head())  
            # print(x_test.shape)   

            logger.info("Files saved sucesfully for Data processing step..")
        
        except Exception as e:
            logger.error(f"Error while splitting data {e}")
            raise CustomException("Failed to split data" , e)
    def run(self):
        self.load_data()
        self.handle_outliers("SepalWidthCm")
        self.split_data()
if __name__=="__main__":
    data_processor = Data_Processing("artifacts/raw/data.csv")
    data_processor.run()


