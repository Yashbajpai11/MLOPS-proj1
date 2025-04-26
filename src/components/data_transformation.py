import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN                                             #Data ko balance karne ke liye (imbalanced datasets mein zyada useful).
from sklearn.pipeline import Pipeline                                                  #Multiple steps ek sequence mein run karwane ke liye
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer                                     #Different columns ke liye different transformations lagane ke liye.

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(self,data_ingestion_artifact: DataIngestionArtifact,              #Jisme train/test file ka path hoga.
                 data_transformation_config: DataTransformationConfig,             # Jisme save karne ka path/config hoga.
                 data_validation_artifact: DataValidationArtifact):                 # Validation status hoga (valid ya invalid).
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)                    # Yeh ek YAML file padh kar load karega — jo batayegi kaunse columns numeric hain, kaunse min-max scale karne hain, etc.
        except Exception as e:
            raise MyException (e,sys)
    
    @staticmethod
    def read_data(file_path)-> pd.DataFrame:                         #ye function = CSV file ko load karke DataFrame mein return karta hai.
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e,sys)
        
    def get_data_transformer_object(self)-> Pipeline:                    # Yeh method ek data transformer (scaler) banata hai.

        logging.info('Entered get_data_transformed_object method of DataTransformation class')
        try:
            numeric_transformer = StandardScaler()
            Min_Max_Scaler = MinMaxScaler()
            logging.info('Transformed initialized: standardScaler-MinMaxScaler')

            # Load schema configuration
            num_features = self._schema_config['num_features']
            mm_columns = self._schema_config['mm_columns']
            logging.info('cols loaded from schema.')

            # Creating preprocessor pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('StandardScaler', numeric_transformer,num_features),
                    ('MinMaxScaler', Min_Max_Scaler,mm_columns)
                ],
                remainder='passthrough'     # Leaves other columns as they are 
            )

            final_pipeline = Pipeline(steps=[('Preprocessor', preprocessor)])
            logging.info('Final pipeline Ready!!')
            logging.info('Exited get_data_transformer_object method of Data transformation class')
            return final_pipeline
        
        except Exception as e:
            logging.exception('Exception occured in get_data_transformer_object method of DataTransformation class')
            raise MyException(e,sys) from e
        

    def _map_gender_column(self, df):                  # Gender column ko Female → 0, Male → 1 mein convert karta hai.
        logging.info("Mapping 'Gender' column to binary values")
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
        return df
    
    def _create_dummy_columns(self, df):                 # Categorical columns ke liye One Hot Encoding karta hai — matlab multiple columns mein todta hai jaise:
        """Create dummy variables for categorical features."""
        logging.info("Creating dummy variables for categorical features")
        df = pd.get_dummies(df, drop_first=True)
        return df
    
    def _rename_columns(self, df):
        """Rename specific columns and ensure integer types for dummy columns."""
        logging.info("Renaming specific columns and casting to int")
        df = df.rename(columns={                                                      #Kuch specific columns ke naam change karta hai, jaise: ||Vehicle_Age_<1 Year → Vehicle_Age_lt_1_Year
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })
        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype('int')                                                      #Aur kuch columns ko integer type mein convert karta hai.
        return df
    
    def _drop_id_column(self, df):
        """Drop the 'id' column if it exists."""
        logging.info("Dropping 'id' column")
        drop_col = self._schema_config['drop_columns']
        if drop_col in df.columns:
            df = df.drop(drop_col, axis=1)
        return df
    

    def initiate_data_transformation(self) -> DataTransformationArtifact:                              #Yeh sabse important function hai. || Ismein pura process step-by-step chalta hai:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:                   #Agar data validation fail ho gaya, to process stop.
                raise Exception(self.data_validation_artifact.message)

            # Load train and test data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)          # Train aur Test data load karte hain.
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)                    #Features aur Target column alag karte hain.
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info("Input and Target cols defined for both train and test df.")

            # Apply custom transformations in specified sequence
            input_feature_train_df = self._map_gender_column(input_feature_train_df)
            input_feature_train_df = self._drop_id_column(input_feature_train_df)
            input_feature_train_df = self._create_dummy_columns(input_feature_train_df)
            input_feature_train_df = self._rename_columns(input_feature_train_df)

            input_feature_test_df = self._map_gender_column(input_feature_test_df)
            input_feature_test_df = self._drop_id_column(input_feature_test_df)
            input_feature_test_df = self._create_dummy_columns(input_feature_test_df)
            input_feature_test_df = self._rename_columns(input_feature_test_df)
            logging.info("Custom transformations applied to train and test data")

            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformer_object()                                      #Ab columns pe scaling apply hoti hai — ek standard format mein data aa jata hai.
            logging.info("Got the preprocessor object")

            logging.info("Initializing transformation for Training-data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            logging.info("Initializing transformation for Testing-data")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Transformation done end to end to train-test df.")

            logging.info("Applying SMOTEENN for handling imbalanced dataset.")
            smt = SMOTEENN(sampling_strategy="minority")                                          # Jab hamara data imbalanced hota hai (jaise 90% No aur 10% Yes samples) to SMOTEENN help karta hai — synthetic samples banakar aur cleaning karke.
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                input_feature_test_arr, target_feature_test_df
            )
            logging.info("SMOTEENN applied to train-test df.")

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]                # Final training/testing arrays bana lete hain — features ke saath target bhi attach karte hain.
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]
            logging.info("feature-target concatenation done for train-test df.")

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)                     #Apna preprocessor (pipeline object) aur transformed data files save karte hain.
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info("Saving transformation object and transformed files.")

            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(                                                                                      #Final output ek Artifact object ke form mein return karte hain.
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
        
        except Exception as e:
            raise MyException(e, sys) from e





        


