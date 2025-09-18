import joblib
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self):
        self.processed_path="artifact/processed"
        self.model_path="artifact/model"
        os.makedirs(self.model_path,exist_ok=True)
        self.model=DecisionTreeClassifier(criterion="gini",max_depth=30,random_state=42)
        logger.info("Model trained successfully..")
    def load_data(self):
        try:
            x_train=joblib.load(os.path.join(self.processed_path,"x_train.pkl"))
            x_test=joblib.load(os.path.join(self.processed_path,"x_test.pkl"))
            y_train=joblib.load(os.path.join(self.processed_path,"y_train.pkl"))
            y_test=joblib.load(os.path.join(self.processed_path,"y_test.pkl"))
            logger.info("data loaded successfully..")
            return x_train,x_test,y_train,y_test
        
        except Exception as e:
            logger.error(f"Error while loading data into the paths : {e}")
            raise CustomException("Error while loading data " , e)
        
    def train_model(self,x_train,y_train):
        try: 
            self.model.fit(x_train,y_train)
            joblib.dump(self.model,os.path.join(self.model_path,"model.pkl"))
            logger.info("Model Trained successfully...")
            
        except Exception as e:
            logger.error(f"Error while training model : {e}")
            raise CustomException("Error while training model",e)
    def evaluate_model(self,x_test,y_test):
        try:
            y_pred=self.model.predict(x_test)
            accuracy=accuracy_score(y_test,y_pred)
            precision=precision_score(y_test,y_pred,average="weighted")
            recall=recall_score(y_test,y_pred,average="weighted")
            f1=f1_score(y_test,y_pred,average="weighted")

            logger.info(f"Accuracy : {accuracy}")
            logger.info(f"Precision : {precision}")
            logger.info(f"Recall : {recall}")
            logger.info(f"F1 : {f1}")
            
            cm=confusion_matrix(y_test,y_pred)
            sns.heatmap(cm,annot=True,xticklabels = np.unique(y_test),yticklabels = np.unique(y_test)) # type: ignore
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted Label")
            plt.ylabel("Actual Label")
            cm_path=os.path.join(self.model_path,"confusion_matrix.png")
            plt.savefig(cm_path)
            plt.close()
            logger.info("Confusion matrix implemented succesfully..")

        except Exception as e:
            logger.error(f"Error while implementing Confusion matrix : {e}")
            raise CustomException("Error while implementing Confusion matrix", e)
    def run(self):
        x_train,x_test,y_train,y_test = self.load_data()
        self.train_model(x_train,y_train)
        self.evaluate_model(x_test,y_test)
    
if __name__=="__main__":
    trainer=ModelTraining()
    trainer.run()




            
            
