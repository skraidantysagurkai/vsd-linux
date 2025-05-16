from pydantic_settings import BaseSettings
import os
from paths import MODEL_DIR

class Settings(BaseSettings):
    MONGODB_URL: str = "mongodb://localhost:27017/"
    MONGODB_DB_NAME: str = "kursinis"
    MONGODB_LOG_COLLECTION: str = "logs"
    MONGODB_EMBEDDED_COLLECTION: str = "embedded_logs"
    
    EMBEDDING_DIM: int = 
    
    PCA_MODEL_PATH: str = os.path.join(MODEL_DIR, 'pca_model.pkl')
    XGBOOST_MODEL_PATH: str = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
    
    LOG_LEVEL: str = "INFO"
    
settings = Settings()