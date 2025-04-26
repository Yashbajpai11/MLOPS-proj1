import os
from src.constants import *
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name:str = DATA_INGESTION_COLLECTION_NAME


@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir,DATA_VALIDATION_DIR_NAME)
    validation_report_file_path: str = os.path.join(data_validation_dir,DATA_VALIDATION_REPORT_FILE_NAME)


@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)                       #Har experiment ke liye ek artifact folder hota hai (training_pipeline_config.artifact_dir se mila).|| Uske andar ek data_transformation folder banega.
    transformed_train_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,                   #data_transformation_dir ke andar ek transformed folder hoga ||   Uske andar train file save hogi in .npy format (kuki model .npy array chahta hai, CSV nahi).||   "train.csv" ko "train.npy" banaya gaya.
                                                    TRAIN_FILE_NAME.replace("csv", "npy"))
    transformed_test_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                   TEST_FILE_NAME.replace("csv", "npy"))
    transformed_object_file_path: str = os.path.join(data_transformation_dir,                                                                      #Aapka preprocessing object (jaise StandardScaler, ColumnTransformer, etc.) ko pickle file ke form me save kiya jaa raha hai.||Ye future me model inference ke waqt use hoga (transform karne ke liye).
                                                     DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                     PREPROCSSING_OBJECT_FILE_NAME)
    

@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)                      #model_trainer_dir ek path (folder ka address) banata hai jaha model training se related sab cheezein save hongi.
    trained_model_file_path: str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_FILE_NAME)             #trained_model_file_path ka matlab hai trained model ka exact file address.|| Pehle jo model_trainer_dir banaya tha, uske andar:|| MODEL_TRAINER_TRAINED_MODEL_DIR naam ka folder hoga || aur uske andar MODEL_FILE_NAME naam ki file hogi.
    expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE                                                               #expected_accuracy batata hai ki hum kitni minimum accuracy chahte hain model se. || MODEL_TRAINER_EXPECTED_SCORE ek predefined value hai (jaise 0.8 ya 80%).|| Agar model itni accuracy nahi deta, toh hum kehte hain — model reject, try again!
    model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH                                 #model_config_file_path ka matlab hai — ek file ka address jisme model ke parameters (kaise train karna hai) likhe hain.

    #Yeh hyperparameters hain model ke liye, jaise RandomForest ke andar settings hoti hain.
    _n_estimators = MODEL_TRAINER_N_ESTIMATORS                                         #Kitne trees banana hai forest mein (jaise 100 trees, 200 trees)
    _min_samples_split = MODEL_TRAINER_MIN_SAMPLES_SPLIT                                  
    _min_samples_leaf = MODEL_TRAINER_MIN_SAMPLES_LEAF
    _max_depth = MIN_SAMPLES_SPLIT_MAX_DEPTH
    _criterion = MIN_SAMPLES_SPLIT_CRITERION
    _random_state = MIN_SAMPLES_SPLIT_RANDOM_STATE
