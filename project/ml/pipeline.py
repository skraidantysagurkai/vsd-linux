import os
import pickle
import joblib
import numpy as np
from typing import List
import logging

from project.config.settings import settings
from src.data.features.event_feature_extractor import EventFeatureExtractor
import torch
import cupy as cp
import cuml

logger = logging.getLogger(__name__)

class MlPipeline:
    def __init__(self, 
                 pca_path: str = settings.PCA_MODEL_PATH,
                 xgboost_path: str = settings.XGBOOST_MODEL_PATH):
        self._check_cuda_availability()
        self.feature_extractor = EventFeatureExtractor()
        self.pca_model = None
        self.xgboost_model = None
        
        self._load_models(pca_path, xgboost_path)

    @staticmethod
    def _check_cuda_availability():
        assert torch.cuda.is_available(), "CUDA is not available. Please check your PyTorch installation or GPU setup."
        
        
    def _load_models(self, pca_path: str, xgboost_path: str) -> None:
        if not os.path.exists(pca_path):
            raise FileNotFoundError(f"PCA model file not found at: {pca_path}")
            
        with open(pca_path, 'rb') as f:
            self.pca_model = pickle.load(f)
            logger.info("PCA model loaded successfully")
            
        if not os.path.exists(xgboost_path):
            raise FileNotFoundError(f"XGBoost model file not found at: {xgboost_path}")
            
        with open(xgboost_path, 'rb') as f:
            self.xgboost_model = pickle.load(f)
            logger.info("XGBoost model loaded successfully")
            
            
    def transform_features(self, features: cp.array) -> cp.array:
        if self.pca_model is None:
            raise RuntimeError("PCA model not loaded")
            
        try:
            transformed_features = self.pca_model.transform(features)
            return transformed_features
        except Exception as e:
            logger.error(f"Error transforming features: {str(e)}")
            raise
        
    def predict(self, features: cp.array) -> np.array:
        if self.xgboost_model is None:
            raise RuntimeError("XGBoost model not loaded")
            
        try:
            prediction = self.xgboost_model.predict(features)
            return int(prediction[0])
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise RuntimeError(f"Error making prediction: {str(e)}")
