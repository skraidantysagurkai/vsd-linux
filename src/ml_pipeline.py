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
        
        with open(pca_path, 'rb') as f:
            self.pca_model = joblib.load(f)

        with open(xgboost_path, 'rb') as f:
            self.xgboost_model = pickle.load(f)
    
    
    def predict(self, code_snippet):
        codebert_features = self.feature_extractor.embed_command([code_snippet])
        
        pca_features = self.pca_model.transform(codebert_features)
        
        prediction = self.xgboost_model.predict(pca_features)
        
        return prediction[0]

