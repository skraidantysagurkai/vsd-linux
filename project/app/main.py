from fastapi import FastAPI
import numpy as np
import json
import logging

from project.logger_setup import setup_logger
from project.app.models import Log, TestingLog
from project.ml.pipeline import MlPipeline
from project.storage.database import Database
from project.utils.handy_functions import FeatureManager as fm
from project.storage.pipelines import embedded_pipeline, thirthy_sec_pipeline, five_min_pipeline
from project.llm.pipeline import LLM

setup_logger()
logger = logging.getLogger(__name__)

class PredictionAPI:
    def __init__(self):
        self.app = FastAPI()
        self.ml_pipeline = MlPipeline()
        self.db = Database()
        self.llm = LLM()
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
            log_dict = fm.add_full_command_to_log(log_dict)
            
            # for testing purposes
            self.db.insert_into_db(log_dict, 'regular')
            # logger.info("Embedding recieved logs command")
            current_embed = self.ml_pipeline.feature_extractor.embed_command([log_dict['full_command']])
            
            self.db.insert_into_db({'uid':log_dict['uid'], 'timestamp':log_dict['timestamp'], 'features':[float(x) for x in current_embed[0]]}, 'embedded') 
            current_embed = self.ml_pipeline.transform_features(current_embed[0])
            
            # logger.info('Getting features')
            features = self.get_features_history(log_dict)
            
            full_feature_dict = fm.add_current_embeds_to_features(features, current_embed)
            
            features_list = fm.unpack_features_to_list(full_feature_dict)
            
            # logger.info('Predicting log maliciousness')
            prediction = self.ml_pipeline.predict(features_list)

            logger.info(f'XGBOOST PREDICTION {prediction}')
            
            if prediction == 1:
                full_user_history = self.db.get_user_complete_history(log_dict)
                llm_response = self.llm.classify_log(log=log_dict, user_history=full_user_history)
                if llm_response == 1:
                    logger.warning("MALICIOUS COMMAND DETECTED!")
            
            return
        
        
        @self.app.post("/debug")
        def predict(item: TestingLog):
            
            item_dict = json.loads(item.model_dump_json())
            
            
            log_dict = item_dict['content']
            # logger.info("Recieved log")
            log_dict = fm.add_full_command_to_log(log_dict)
            
            # for testing purposes
            self.db.insert_into_db(log_dict, 'regular')
            # logger.info("Embedding recieved logs command")
            current_embed = self.ml_pipeline.feature_extractor.embed_command([log_dict['full_command']])
            
            self.db.insert_into_db({'uid':log_dict['uid'], 'timestamp':log_dict['timestamp'], 'features':[float(x) for x in current_embed[0]]}, 'embedded') 
            current_embed = self.ml_pipeline.transform_features(current_embed[0])
            
            # logger.info('Getting features')
            features = self.get_features_history(log_dict)
            
            full_feature_dict = fm.add_current_embeds_to_features(features, current_embed)
            
            features_list = fm.unpack_features_to_list(full_feature_dict)
            
            # logger.info('Predicting log maliciousness')
            prediction = self.ml_pipeline.predict(features_list)
            logger.info(f'TARGET: {item_dict["target"]}')
            logger.info(f'XGBOOST PREDICTION {prediction}')
            
            if prediction == 1:
                full_user_history = self.db.get_user_complete_history(log_dict)
                llm_response = self.llm.classify_log(log=log_dict, user_history=full_user_history)
                if llm_response == 1:
                    logger.warning("MALICIOUS COMMAND DETECTED!")
            
            return
    
    def get_features_history(self, log: dict) -> dict:
        try:    
            result_raw_30 = self.db.get_embedded_features(embedded_pipeline(log, 30))
            result_raw_300 = self.db.get_embedded_features(embedded_pipeline(log, 300))

            # get thirty sec and five min aggregated features
            thirty_sec_features = self.db.get_regular_features(thirthy_sec_pipeline(log), 30)
            five_min_features = self.db.get_regular_features(five_min_pipeline(log), 300)

            # compute averages of embedded features
            average_features_30 = np.mean(result_raw_30, axis=0)
            average_features_300 = np.mean(result_raw_300, axis=0)

            # we need all thirty sec embeds
            thirty_sec_avg_embeds = self.ml_pipeline.transform_features(average_features_30)
            # we only need number 5 and 6
            five_min_avg_embeds = self.ml_pipeline.transform_features(average_features_300)[4:6]
            
            features_dict = fm.construct_features(thirty_sec_features, five_min_features,
                                                thirty_sec_avg_embeds, five_min_avg_embeds)
        except Exception as e:
            logger.error(f'Failed fetching history from DB \n {e}', exc_info=True)
            return
        
        logger.info('Fetched history from database')
        return features_dict
    
        
prediction_api = PredictionAPI()
app = prediction_api.app