import os
import sys
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logging
from src.data_access.proj1_data import Proj1Data


class DataIngestion:
    def __init__(self,data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e,sys)
        
    def export_data_into_feature_store(self)->DataFrame:           #mongoDB se data lao
        try:
            logging.info(f'Exporting data from MongoDB')
            my_data = Proj1Data()                              # Teri MongoDB connection wali class
            dataframe = my_data.export_collection_as_dataframe(collection_name = self.data_ingestion_config.collection_name)     #MongoDB collection ko dataframe me convert karta hai
            logging.info(f'Shape of dataframe: {dataframe.shape}')     #MongoDB collection ko dataframe me convert karta hai

            
            # path banata hai jha .csv save hoga || agr ni bani toh bana dega 
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True) 

            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path,index=False,header = True)
            return dataframe
        
        except Exception as e:
            raise MyException(e,sys)
        
    def split_data_as_train_test(self,dataframe: DataFrame) ->None:
        """
        Method Name :   split_data_as_train_test
        Description :   This method splits the dataframe into train set and test set based on split ratio 
        
        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")

        try:
            train_set,test_set = train_test_split(dataframe,test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info('performed train test split on the dataframe')
            logging.info(
                "Exited split_data_as_train_test method of Data_ingestion class"
            )
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)          #Folder pehle banaya jaata hai agar exist nahi karta
            os.makedirs(dir_path,exist_ok=True)

            logging.info(f'Exporting train test file path')
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            logging.info(f'Exporting train test file path')
        
        except Exception as e:
            raise MyException(e,sys) from e
        
    def initiate_data_ingestion(self) ->DataIngestionArtifact:                     ##Yeh function pura data ingestion trigger karta hai – kind of run() method for ingestion step
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion components of training pipeline 
        
        Output      :   train set and test set are returned as the artifacts of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

        try:
            dataframe = self.export_data_into_feature_store()           #MongoDB se data uthao aur feature store me daalo

            logging.info("Got the data from mongodb")

            self.split_data_as_train_test(dataframe)        #Train/test split

            logging.info("Performed train test split on the dataset")

            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )

            data_ingestion_artifact = DataIngestionArtifact(                                                  #DataIngestionArtifact ek object hai jisme train aur test file path store hote hain  || Ye object next component (jaise DataValidation) me pass hota hai
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
                )
            
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")

            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys) from e




