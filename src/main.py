from fastapi import FastAPI
from typing import List
from pymongo import MongoClient
from lelemas import LLM 
import numpy as np

from src.shared.log_check import Log
from src.ml_pipeline import MlPipeline
from src.utils.mongo_pipelines import embeded_pipeline, thirthy_sec_pipeline, five_min_pipeline

class IdentificationPipeline:
    def __init__(self):
        self.app = FastAPI()
        self.client = MongoClient("mongodb://localhost:27017/")
        self.ml_pipeline = MlPipeline()
        self.llm = LLM(model_name="openai/gpt-4o")
        db = self.client["ml_logs"]
        self.log_collection = db["logs"]
        self.embedded_collection = db["embedded_logs"]
        
        self.accept_request()
        
    def accept_request(self):
        @self.app.post("/predict")
        def predict(log: Log):
            log = log.model_dump_json()
            
            current_embed = self.save_log_to_db(log)
            current_embed = self.ml_pipeline.pca_model.transform(current_embed)
            features = self.get_features_history(log)
            features['cur_event_avg_embedded_command_0'] = current_embed[0]
            features['cur_event_avg_embedded_command_6'] = current_embed[6]
            features['cur_event_avg_embedded_command_9'] = current_embed[9]
            features['cur_event_avg_embedded_command_7'] = current_embed[7]
            features['cur_event_avg_embedded_command_4'] = current_embed[4]
            features['cur_event_avg_embedded_command_2'] = current_embed[2]
            features['cur_event_avg_embedded_command_5'] = current_embed[5]
            features['cur_event_avg_embedded_command_3'] = current_embed[3]
            features['cur_event_avg_embedded_command_1'] = current_embed[1]
            features['cur_event_avg_embedded_command_8'] = current_embed[8]
            
            features_list = self.unpack_features_to_list(features)
            prediction = self.ml_pipeline.predict(features_list)
            
            # implement logic for prediction
            if prediction == 1:
                user_history = self.get_user_history(log)
            
                # llm_response = self.llm.classify_log(log.model_dump_json())
                # return {
                #     "response": response,
                # }
                
                # call SLACK ADMIN!
            return
            
    def get_features_history(self, log: dict):
        # 30 seconds window
        pipeline_30 = embeded_pipeline(log, 30)
        # 300 seconds window
        pipeline_300 = embeded_pipeline(log, 300)

        result_raw_30 = [np.array(doc['features']) for doc in self.embedded_collection.aggregate(pipeline_30)]
        result_raw_300 = [np.array(doc['features']) for doc in self.embedded_collection.aggregate(pipeline_300)]
        
        # get thirty sec and five min aggregated features
        thirty_sec_features = self.log_collection.aggregate(thirthy_sec_pipeline(log))
        five_min_features = self.log_collection.aggregate(five_min_pipeline(log))
        
        # compute averages
        average_features_30 = np.mean(result_raw_30, axis=0)
        average_features_300 = np.mean(result_raw_300, axis=0)
        
        # we need all thirty sec embeds
        thirty_sec_avg_embeds = self.ml_pipeline.pca_model.transform([average_features_30])
        # we only need number 5 and 6
        five_min_avg_embeds = self.ml_pipeline.pca_model.transform([average_features_300])[4:6]
        features_dict = self.construct_features(thirty_sec_features, five_min_features,
                                               thirty_sec_avg_embeds, five_min_avg_embeds)
        
        return features_dict
                
    def get_user_history(self, log: dict) -> List[dict]:
        pipeline = [
            {"$match": {
                "uid": log["uid"],
                "timestamp": {"$gte": log['timestamp'] - 300, "$lte": log['timestamp']}
            }},
            {"$sort": {"timestamp": -1}}
        ]
        
        result = self.embedded_collection.aggregate(pipeline)
        
        return result
                        

    def save_log_to_db(self, log: dict) -> List[int]:
        # save not embeded log
        log["arg_count"] = len(log["arguments"]) if log["arguments"] else 0
        self.log_collection.insert_one(log)
        
        embedded_command = self.ml_pipeline.feature_extractor.embed_log([log['command']])
        
        self.embedded_collection.insert_one(
            {
                "uid": log["uid"],
                "timestamp": log["timestamp"],
                "features": embedded_command
            }
        )
        
        return embedded_command
        
    @staticmethod
    def construct_features(thirty_sec_features, five_min_features,
                            thirty_sec_avg_embeds, five_min_avg_embeds) -> dict:
        
        features = {
            'thirty_sec_bash_count_rate':thirty_sec_features['bash_ratio'],
            'thirty_sec_avg_embedded_command_4': thirty_sec_avg_embeds[4],
            'thirty_sec_avg_embedded_command_0': thirty_sec_avg_embeds[0],
            'thirty_sec_avg_embedded_command_7': thirty_sec_avg_embeds[7],
            'thirty_sec_avg_embedded_command_8': thirty_sec_avg_embeds[8],
            'thirty_sec_avg_embedded_command_9': thirty_sec_avg_embeds[9],
            'thirty_sec_log_count': thirty_sec_features['log_count'],
            'thirty_sec_avg_embedded_command_5': thirty_sec_avg_embeds[5],
            'thirty_sec_avg_embedded_command_3': thirty_sec_avg_embeds[3],
            'thirty_sec_avg_embedded_command_2': thirty_sec_avg_embeds[2],
            'thirty_sec_avg_embedded_command_1': thirty_sec_avg_embeds[1],
            'thirty_sec_success_rate': thirty_sec_features['success_rate'],
            'thirty_sec_avg_embedded_command_6': thirty_sec_avg_embeds[6],
            'thirty_sec_unique_pids': thirty_sec_features['unique_pids'],
            'five_min_avg_embedded_command_4': five_min_features[4],
            'five_min_bash_count_rate': five_min_features['bash_ratio'],
            'five_min_avg_embedded_command_5': five_min_avg_embeds[5],
            'five_min_success_rate': five_min_features['success_rate'],
            'five_min_log_count': five_min_features['log_count']
        }
    
        return features


    @staticmethod
    def unpack_features_to_list(features: dict) -> List[float]:
        """
        Maximum LOL function
        """
        return [
            features['thirty_sec_bash_count_rate'],
            features['thirty_sec_avg_embedded_command_4'],
            features['thirty_sec_avg_embedded_command_0'],
            features['thirty_sec_avg_embedded_command_7'],
            features['thirty_sec_avg_embedded_command_8'],
            features['thirty_sec_avg_embedded_command_9'],
            features['thirty_sec_log_count'],
            features['thirty_sec_avg_embedded_command_5'],
            features['thirty_sec_avg_embedded_command_3'],
            features['thirty_sec_avg_embedded_command_2'],
            features['thirty_sec_avg_embedded_command_1'],
            features['thirty_sec_success_rate'],
            features['thirty_sec_avg_embedded_command_6'],
            features['cur_event_avg_embedded_command_0'],
            features['cur_event_avg_embedded_command_6'],
            features['thirty_sec_unique_pids'],
            features['five_min_avg_embedded_command_4'],
            features['five_min_bash_count_rate'],
            features['five_min_avg_embedded_command_5'],
            features['cur_event_avg_embedded_command_9'],
            features['cur_event_avg_embedded_command_7'],
            features['cur_event_avg_embedded_command_4'],
            features['cur_event_avg_embedded_command_2'],
            features['five_min_success_rate'],
            features['cur_event_avg_embedded_command_5'],
            features['five_min_log_count'],
            features['cur_event_avg_embedded_command_3'],
            features['cur_event_avg_embedded_command_1'],
            features['cur_event_avg_embedded_command_8']
        ]
    # regular log example
    # {"timestamp": 1234567890, "success": 1, "uid": "1234", pid: "5678", "command": "ls", "arguments": ["-l"], "CWD": "/home/user"}
    
    # embedded log example
    # {"uid": "1234", "timestamp": 1234567890, "features": [0.2, ... , 0.3]}}
    
    
    
                
ml_api = IdentificationPipeline()
app = ml_api.app