import os
import pickle
import joblib
import numpy as np
from typing import List
from config.settings import settings
from src.data.features.event_feature_extractor import EventFeatureExtractor

class MlPipeline:
    def __init__(self, 
                 pca_path: str = settings.PCA_MODEL_PATH,
                 xgboost_path: str = settings.XGBOOST_MODEL_PATH):
        self.feature_extractor = EventFeatureExtractor()
        self.pca_model = None
        self.xgboost_model = None
        
        self._load_models(pca_path, xgboost_path)
        
        
    def _load_models(self, pca_path: str, xgboost_path: str) -> None:
        if not os.path.exists(pca_path):
            raise FileNotFoundError(f"PCA model file not found at: {pca_path}")
            
        with open(pca_path, 'rb') as f:
            self.pca_model = joblib.load(f)
            # logger.info("PCA model loaded successfully")
            
        if not os.path.exists(xgboost_path):
            raise FileNotFoundError(f"XGBoost model file not found at: {xgboost_path}")
            
        with open(xgboost_path, 'rb') as f:
            self.xgboost_model = pickle.load(f)
            # logger.info("XGBoost model loaded successfully")
            
            
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        if self.pca_model is None:
            raise RuntimeError("PCA model not loaded")
            
        try:
            transformed_features = self.pca_model.transform([features])
            return transformed_features[0]
        except Exception as e:
            # logger.error(f"Error transforming features: {str(e)}")
            raise
        
    def predict(self, features: List[float]) -> int:
        if self.xgboost_model is None:
            raise RuntimeError("XGBoost model not loaded")
            
        try:
            # Make prediction
            features_array = np.array([features])
            prediction = self.xgboost_model.predict(features_array)
            return int(prediction[0])
        except Exception as e:
            # logger.error(f"Prediction error: {str(e)}")
            raise RuntimeError(f"Error making prediction: {str(e)}")
