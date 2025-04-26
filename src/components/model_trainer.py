import sys
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.entity.estimator import MyModel

class ModelTrainer:                                                                                                                         #Yeh ek Python class hai jiska kaam Model Training ka pura process handle karna hai.
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,                                                           #Jab ModelTrainer ka object banega, toh yeh do cheezein input lega: || Data Transformation phase ka output (transformed training aur testing data ka path etc.) || Configuration file jisme model training ke liye settings hain (n_estimators, max_depth, file paths, etc.)
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        self.data_transformation_artifact = data_transformation_artifact                                                 #Ye dono ko apni class ke andar save kar liya:
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:                             #Ye method || Model ko train karega || Performance metrics nikaalega || Trained model aur metrics dono return karega.
        """
        Method Name :   get_model_object_and_report
        Description :   This function trains a RandomForestClassifier with specified parameters
        
        Output      :   Returns metric artifact object and trained model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Training RandomForestClassifier with specified parameters")

            # Splitting the train and test data into features and target variables
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]                                        #Features (X) aur Labels (Y) alag kiye hain. || :-1 means saare columns except last (features) || -1 means sirf last column (target label)
            logging.info("train-test split done.")

            # Initialize RandomForestClassifier with specified parameters
            model = RandomForestClassifier(                                                                                                          # Yeh ModelTrainerConfig ke andar jo parameters diye the, unse RandomForestClassifier banaya gaya.
                n_estimators = self.model_trainer_config._n_estimators,
                min_samples_split = self.model_trainer_config._min_samples_split,
                min_samples_leaf = self.model_trainer_config._min_samples_leaf,
                max_depth = self.model_trainer_config._max_depth,
                criterion = self.model_trainer_config._criterion,
                random_state = self.model_trainer_config._random_state
            )

            # Fit the model
            logging.info("Model training going on...")
            model.fit(x_train, y_train)                                                                                                   #Model ko training data pe fit kar diya.
            logging.info("Model training done.")

            # Predictions and evaluation metrics
            y_pred = model.predict(x_test)                                                  #Test data pe prediction kiya aur accuracy, f1, precision, recall calculate kiya.
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            # Creating metric artifact
            metric_artifact = ClassificationMetricArtifact(f1_score=f1, precision_score=precision, recall_score=recall)                               #Saare metrics ko ek chhoti basket mein store kar diya (dataclass ClassificationMetricArtifact).
            return model, metric_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:                                                                                          #Yeh pura model training pipeline ko start karta hai aur last mein ModelTrainerArtifact return karta hai.
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates the model training steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            print("------------------------------------------------------------------------------------------------")                           
            print("Starting Model Trainer Component")                                                                                                 # user ko batane ky liye ki ab training st ho rhe hai 
            # Load transformed train and test data
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)                                                # Pehle DataTransformation Artifact se transformed numpy arrays ko load kiya (training and testing ke liye).
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("train-test data loaded")
            
            # Train model and get metrics
            trained_model, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)                                                                   #get_model_object_and_report() ko call kiya jisme model train hoke trained model aur metric artifact dono mil gaya.
            logging.info("Model object and artifact loaded.")
            
            # Load preprocessing object
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)                                                                        #Data ko preprocess karne wala object (Scaler, Encoder, etc.) load kiya jo pehle data transformation mein save kiya tha.
            logging.info("Preprocessing obj loaded.")

            # Check if the model's accuracy meets the expected threshold
            if accuracy_score(train_arr[:, -1], trained_model.predict(train_arr[:, :-1])) < self.model_trainer_config.expected_accuracy:                                                                  #Training data pe prediction karke check kiya ke model ki accuracy expected_accuracy se zyada hai ya nahi. || Agar nahi hai ➔ Toh exception raise kar diya.
                logging.info("No model found with score above the base score")
                raise Exception("No model found with score above the base score")

            # Save the final model object that includes both preprocessing and the trained model
            logging.info("Saving new model as performace is better than previous one.")
            my_model = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=trained_model)                                                                        # MyModel naam ka ek object banaya jisme:|| Preprocessing ka object || Trained model ka object || Dono ko save kiya ek file mein.
            save_object(self.model_trainer_config.trained_model_file_path, my_model)
            logging.info("Saved final model object that includes both preprocessing and the trained model")

            # Create and return the ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(                                                                                         # Ab ek final artifact banaya jisme:|| Trained model ka file path || Aur uske performance metrics save kiye.
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e