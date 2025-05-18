from fastapi import FastAPI
import numpy as np
import json
import logging

from project.logger_setup import setup_logger
from project.app.models import Log
from project.ml.pipeline import MlPipeline
from project.storage.database import Database
from project.utils.handy_functions import *
from project.storage.pipelines import embedded_pipeline, thirthy_sec_pipeline, five_min_pipeline

setup_logger()
logger = logging.getLogger(__name__)

class PredictionAPI:
    def __init__(self):
        self.app = FastAPI()
        self.ml_pipeline = MlPipeline()
        self.db = Database()
        
        self.setup_routes()
        
    def setup_routes(self):
        @self.app.post("/predict")
        def predict(log: Log):
            """
            construct full command -> add original log json to db -> embed log command -> 
            -> get log's user command history features -> construct feature list -> predict 
            """
            log_dict = json.loads(log.model_dump_json())
            logger.info("Recieved log")
            log_dict = add_full_command_to_log(log_dict)
            
            # self.db.insert_into_db(log_dict, 'regular')
            logger.info("Embedding recieved logs command")
            current_embed = self.ml_pipeline.feature_extractor.embed_command([log_dict['full_command']])
            current_embed = self.ml_pipeline.transform_features(current_embed[0])
            
            features = self.get_features_history(log_dict)
            
            full_feature_dict = add_current_embeds_to_features(features, current_embed)
            
            features_list = unpack_features_to_list(full_feature_dict)
            
            prediction = self.ml_pipeline.predict(features_list)

            print(prediction)
            
            # if malicious command predicted call llm for further examination
            
            return
    
    def get_features_history(self, log: dict) -> dict:
        try:    
            result_raw_30 = self.db.get_embedded_features(embedded_pipeline(log, 30))
            result_raw_300 = self.db.get_embedded_features(embedded_pipeline(log, 300))

            # get thirty sec and five min aggregated features
            thirty_sec_features = self.db.get_regular_features(thirthy_sec_pipeline(log), 30)[0]
            
            five_min_features = self.db.get_regular_features(five_min_pipeline(log), 300)[0]
            
            # compute averages of embedded features
            average_features_30 = np.mean(result_raw_30, axis=0)
            average_features_300 = np.mean(result_raw_300, axis=0)

            # we need all thirty sec embeds
            thirty_sec_avg_embeds = self.ml_pipeline.transform_features(average_features_30)
            # we only need number 5 and 6
            five_min_avg_embeds = self.ml_pipeline.transform_features(average_features_300)[4:6]
            
            features_dict = construct_features(thirty_sec_features, five_min_features,
                                                thirty_sec_avg_embeds, five_min_avg_embeds)
        except Exception as e:
            logger.error(f'Failed fetching history from DB \n {e}')
            raise LookupError("Error idk")
        
        logger.info('Fetched history from database')
        return features_dict
    
        
prediction_api = PredictionAPI()
app = prediction_api.app