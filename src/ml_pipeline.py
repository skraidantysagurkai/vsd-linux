import pickle
import torch
import joblib
from src.data.features.event_feature_extractor import EventFeatureExtractor
from src.shared.log_check import Log
from paths import MODEL_DIR
import os


class MlPipeline:
    def __init__(self, 
                 pca_path=os.path.join(MODEL_DIR,'pca_model.pkl'), 
                 xgboost_path=os.path.join(MODEL_DIR, 'xgboost_model.pkl')):

        self.feature_extractor = EventFeatureExtractor()
        try:
            with open(pca_path, 'rb') as f:
                self.pca_model = joblib.load(f)
        except Exception as e:
            import traceback
            print("Exceotion while loading pca model occured", e)
            traceback.print_exc()

        with open(xgboost_path, 'rb') as f:
            self.xgboost_model = pickle.load(f)
    
    
    def predict(self, features):
        prediction = self.xgboost_model.predict(features)
        
        return prediction[0]

