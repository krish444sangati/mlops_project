from src.data_processing import Data_Processing
from src.model_training import ModelTraining

if __name__=="__main__":
    data_processor = Data_Processing("artifacts/raw/data.csv")
    data_processor.run()

    trainer=ModelTraining()
    trainer.run()
