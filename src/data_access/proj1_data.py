import sys
import pandas as pd
import numpy as np
from typing import Optional

from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import DATABASE_NAME
from src.exception import MyException

class Proj1Data:
    """ ye class MongoDB ke data ko pandas dataframe mai convert krta hai"""

    def __init__(self) -> None:
        """
        initializes the MongoDB connections
        """
        try:
            self.mongo_client =  MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise MyException(e,sys)

    #Mongo Collection ko DataFrame me convert karta hai    
    def export_collection_as_dataframe(self,collection_name: str,database_name:Optional[str]=None)-> pd.DataFrame:

        try:
            if database_name is None:                           # Agar tu database ka naam nahi dega to:||Ye default DB (DATABASE_NAME) ka collection_name access karega.
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name] #Agar tu custom database ka naam dega to:||Us database ke andar se collection fetch karega.

            print('Fetching data from MongoDB')
            df = pd.DataFrame(list(collection.find()))   #collection.find() se data aata hai (cursor form me)|||list(...) us cursor ko list me convert karta hai||| pd.DataFrame(...) se poora data DataFrame ban jata hai
            print(f'Data fetched with len:{len(df)}') #print karega ki kitne row aye h
            if "id" in df.columns.to_list():
                df = df.drop(columns=["id"], axis=1)             #Agar id column hai to usko hata diya jaata hai (shayad wo duplicate ho _id se
            df.replace({"na":np.nan}, inplace=True)
            return df 
        except Exception as e:
            raise MyException(e, sys)

        

