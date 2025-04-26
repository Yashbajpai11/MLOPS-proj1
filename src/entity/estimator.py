import sys

import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.exception import MyException
from src.logger import logging

#Ye file (estimator.py) tumhare final model object (MyModel) aur target value mapping (TargetValueMapping) ko handle karne ke liye hai.|| Jab tumhara ML model ready hota hai, tab sirf model ka object kaafi nahi hota.Uske saath preprocessing steps (scaling, encoding) bhi save karne padte hain, taki prediction ke waqt input data pe wahi steps apply ho.||


class TargetValueMapping:                                                               #Jab model banate hain, hume labels ko numeric format mein convert karna padta hai      ||       Ye class automatically mapping set kar rahi hai.
    def __init__(self): 
        self.yes:int = 0
        self.no:int = 1
    def _asdict(self):                                                                  #Ye function pura mapping dictionary format mein return karega.
        return self.__dict__
    def reverse_mapping(self):                                                                        #Agar hume reverse mapping chahiye, matlab:   ||      Jab prediction ke baad numeric output aata hai (0, 1), aur hume wapas original label chahiye.
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(),mapping_response.keys()))

class MyModel:                                                                                                          #preprocessing_object: Ye tumhara preprocessor (Scaler, Encoder, etc.) hai.       ||       trained_model_object: Ye tumhara actual trained model hai (jaise RandomForest, XGBoost, etc.).    ||        Ye class dono cheezein ek saath save aur manage karegi.
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model 
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: pd.DataFrame) -> DataFrame:                                                     #Ye method prediction karne ke liye bana hai.
        """
        Function accepts preprocessed inputs (with all custom transformations already applied),
        applies scaling using preprocessing_object, and performs prediction on transformed features.
        """
        try:
            logging.info("Starting prediction process.")

            # Step 1: Apply scaling transformations using the pre-trained preprocessing object
            transformed_feature = self.preprocessing_object.transform(dataframe)                                                 #Matlab: Jo bhi transformations (Scaler, Encoding) model training ke waqt ki thi,wahi transformations prediction ke waqt bhi apply karo.

            # Step 2: Perform prediction using the trained model
            logging.info("Using the trained model to get predictions")
            predictions = self.trained_model_object.predict(transformed_feature)                                #Ab transformed data pe trained model se predict karo. ||     Isse guarantee hoti hai ki training ke preprocessing steps aur prediction ke preprocessing steps same rahenge.

            return predictions

        except Exception as e:
            logging.error("Error occurred in predict method", exc_info=True)
            raise MyException(e, sys) from e


    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"                                                         # Jab tum print(MyModel()) ya console pe object ko print karoge, to model ka naam dikhayega, jaise: ||        RandomForestClassifier()       ||        Ye readability ke liye important hota hai, debugging ke time helpful.

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"