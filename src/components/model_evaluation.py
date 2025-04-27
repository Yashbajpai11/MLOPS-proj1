from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score
from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.logger import logging
from src.utils.main_utils import load_object
import sys
import pandas as pd
from typing import Optional
from dataclasses import dataclass

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    is_model_accepted: bool


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,                                                       #Constructor hai. Jab object banega to: ||   Tumhara config (ModelEvaluationConfig) ||   Tumhara data ingestion ka result (DataIngestionArtifact)    ||        Tumhara trained model ka result (ModelTrainerArtifact)    ||       ye sab yahan store ho jayega self ke andar.       
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys) from e
        
    def _map_gender_column(self, df):                                                                       #Gender column ko numeric mein convert karna:
        """Map Gender column to 0 for Female and 1 for Male."""
        logging.info("Mapping 'Gender' column to binary values")
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
        return df

    def _create_dummy_columns(self, df):                                                                              #Jo columns categorical hain (jaise 'Vehicle_Age', 'Vehicle_Damage'), unko dummy variables mein convert karna (One-Hot Encoding).
        """Create dummy variables for categorical features."""
        logging.info("Creating dummy variables for categorical features")
        df = pd.get_dummies(df, drop_first=True)
        return df

    def _rename_columns(self, df):                                                                                          #Dummy columns ke naam thoda sahi karna:
        """Rename specific columns and ensure integer types for dummy columns."""
        logging.info("Renaming specific columns and casting to int")
        df = df.rename(columns={
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })
        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype('int')
        return df
    
    def _drop_id_column(self, df):                                                                         #MongoDB se data aaya hoga to "_id" field hoti hai — woh drop kar dena.
        """Drop the 'id' column if it exists."""
        logging.info("Dropping 'id' column")
        if "_id" in df.columns:
            df = df.drop("_id", axis=1)
        return df

    def evaluate_model(self) -> EvaluateModelResponse:                                                                              #Ye naye trained model aur purane model ko compare karta hai aur decide karta hai kaunsa better hai.
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)                                                    #Test Data load karo (jo ingestion mein split hua tha).
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]                                                           #x features aur y target ko separate karo.

            logging.info("Test data loaded and now transforming it for prediction...")

            x = self._map_gender_column(x)
            x = self._drop_id_column(x)
            x = self._create_dummy_columns(x)
            x = self._rename_columns(x)

            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            logging.info("Loaded trained model successfully.")

            y_pred = trained_model.predict(x)
            from sklearn.metrics import f1_score
            trained_model_f1_score = f1_score(y, y_pred)

            logging.info(f"Trained model F1 Score: {trained_model_f1_score}")

            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                is_model_accepted=True
            )
            return result

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:                                                                       #Ye ek wrapper method hai — jo evaluation start karta hai.
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            logging.info("Starting Model Evaluation.")
            evaluation_response = self.evaluate_model()

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluation_response.is_model_accepted,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluation_response.trained_model_f1_score
            )
            logging.info(f"Model Evaluation Artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            raise MyException(e, sys) from e