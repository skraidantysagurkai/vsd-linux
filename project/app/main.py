from fastapi import FastAPI
from typing import List
import numpy as np

from models import Log
from ml.pipeline import MlPipeline
from storage.database import db
from utils.handy_functions import *
from storage.pipelines import embedded_pipeline, thirthy_sec_pipeline, five_min_pipeline


class PredictionAPI:
    def __init__(self):
        self.app = FastAPI()
        self.ml_pipeline = MlPipeline()
        self.db = db
        
        self.setup_routes()
        
    def setup_routes(self):
        @self.app.post("/predict")
        def predict(log: Log):
            """
            construct full command -> add original log json to db -> embed log command -> 
            -> get log's user command history features -> construct feature list -> predict 
            """
            log = log.model_dump_json()
            
            log = add_full_command_to_log(log)
            
            self.db.insert_into_db(log, 'regular')
            
            current_embed = self.ml_pipeline.feature_extractor.embed_command([log['full_command']])
            current_embed = self.ml_pipeline.pca_model.transform(current_embed)
            
            features = self.get_features_history(log)
            
            full_feature_dict = add_current_embeds_to_features(features, current_embed)
            
            features_list = self.unpack_features_to_list(full_feature_dict)
            
            prediction = self.ml_pipeline.predict(features_list)

            print(prediction)
            
            # if malicious command predicted call llm for further examination
            
            return
    
    def get_features_history(self, log: dict) -> dict:
        result_raw_30 = [np.array(doc['features']) for doc in self.embedded_collection.aggregate(embedded_pipeline(log, 30))]
        result_raw_300 = [np.array(doc['features']) for doc in self.embedded_collection.aggregate(embedded_pipeline(log, 300))]
        
        # get thirty sec and five min aggregated features
        thirty_sec_features = self.log_collection.aggregate(thirthy_sec_pipeline(log))
        five_min_features = self.log_collection.aggregate(five_min_pipeline(log))
        
        # compute averages of embedded features
        average_features_30 = np.mean(result_raw_30, axis=0)
        average_features_300 = np.mean(result_raw_300, axis=0)
        
        # we need all thirty sec embeds
        thirty_sec_avg_embeds = self.ml_pipeline.pca_model.transform([average_features_30])
        # we only need number 5 and 6
        five_min_avg_embeds = self.ml_pipeline.pca_model.transform([average_features_300])[4:6]
        
        features_dict = self.construct_features(thirty_sec_features, five_min_features,
                                               thirty_sec_avg_embeds, five_min_avg_embeds)
        
        return features_dict
    
        
prediction_api = PredictionAPI()
app = prediction_api.app