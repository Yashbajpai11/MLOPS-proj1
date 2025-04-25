import json
import sys
import os
import pandas as pd
from pandas import DataFrame

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file, write_yaml_file
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(self,data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        :data_ingestion_artifact : data is taken from the ingestion step 
        :data_validation_config : config file se path lene ka step
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e,sys)
        
    def vaildation_number_of_columns(self,dataframe: DataFrame) ->bool:                #Yeh function check karta hai ki dataframe mein jitne columns hain, woh schema ke columns se match karte hain ya nahi.
        try:
            status = len(dataframe.columns) == len(self._schema_config['columns'])          #Agar column count match karta hai, toh True return karta hai, warna False.
            logging.info(f'Is required columns present:[{status}]')
            return status
        except Exception as e:
            raise MyException(e,sys)
        
    def is_column_exist(self,df:DataFrame) -> bool:                                   #Yeh function check karta hai ki schema mein defined numerical aur categorical columns dataframe mein actually exist karte hain ya nahi.
        try:
            dataframe_columns = df.columns
            missing_numerical_columns =[]
            missing_categorical_columns = []

            for column in self._schema_config['numerical_columns']:      #Numerical columns ke liye check karta hai ki woh dataframe mein hain ya nahi. Nahi mile toh list mein add karta hai.
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns)>0:                                                               #Agar missing columns mile, toh log mein info likhta hai. || Same kaam categorical columns ke liye bhi karta hai.|| Agar dono missing column lists empty hain, tabhi True return karta hai, warna False.
                logging.info(f'missing numerical column: {missing_numerical_columns}')

            for column in self._schema_config['categorical_columns']:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns)>0:
                logging.info(f'Missing categorical column: {missing_categorical_columns}')

            return False if len(missing_categorical_columns)>0 or len(missing_numerical_columns)>0 else True
        except Exception as e:
            raise MyException(e,sys) from e
        
    @staticmethod
    def read_data(file_path) -> DataFrame:                   #Yeh ek helper static method hai jo given file_path se CSV file read karke pandas DataFrame return karta hai.
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e,sys)
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            validation_error_msg =""
            logging.info('Starting data validation')
            train_df, test_df = (DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path),              #Training aur testing data ko read karta hai file se.
                                 DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path))
            
            status = self.vaildation_number_of_columns(dataframe=train_df)                                              #Training aur testing dono ke columns count check karta hai.
            if not status:
                validation_error_msg += f"column are missing in training dataframe."
            else:
                logging.info(f"All required columns present in training dataframe:{status}")

            status = self.vaildation_number_of_columns(dataframe=test_df)
            if not status:
                validation_error_msg += f"column are missing in test dataframe."
            else:
                logging.info(f"all required columns present in testing dataframe: {status}")

            
            status = self.is_column_exist(df=train_df)                                          #Training aur testing data ke andar required columns exist karte hain ya nahi check karta hai.
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe. "
            else:
                logging.info(f"All categorical/int columns present in training dataframe: {status}")

            status = self.is_column_exist(df=test_df)
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe."
            else:
                logging.info(f"All categorical/int columns present in testing dataframe: {status}")


            validation_status = len(validation_error_msg) == 0         #Agar koi error nahi aayi, toh True return karega, warna False.

            data_validation_artifact = DataValidationArtifact(
               validation_status=validation_status,
               messages=validation_error_msg, 
               validation_report_file_path=self.data_validation_config.validation_report_file_path
)


            # Ensure the directory for validation_report_file_path exists
            report_dir = os.path.dirname(self.data_validation_config.validation_report_file_path)
            os.makedirs(report_dir, exist_ok=True)

            # Save validation status and message to a JSON file
            validation_report = {
                "validation_status": validation_status,
                "message": validation_error_msg.strip()
            }

            with open(self.data_validation_config.validation_report_file_path, "w") as report_file:                           #Validation ka result (success ya failure) ek JSON file mein save hota hai.
                json.dump(validation_report, report_file, indent=4)

            logging.info("Data validation artifact created and saved to JSON file.")
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise MyException(e, sys) from e



               
    
        



