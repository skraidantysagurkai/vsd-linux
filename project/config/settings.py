from pydantic_settings import BaseSettings
import os
from paths import MODEL_DIR, KEY_PATH

class Settings(BaseSettings):
    MONGODB_URL: str = 'mongodb://localhost:27017/'
    MONGODB_DB_NAME: str = 'kursinis'
    # MONGODB_LOG_COLLECTION: str = 'logs'
    # MONGODB_EMBEDDED_COLLECTION: str = 'embedded_logs'
    # for testing
    MONGODB_LOG_COLLECTION: str = "logs_empty"
    MONGODB_EMBEDDED_COLLECTION: str = "embeds_empty"
    
    EMBEDDING_DIM: int = 768
    
    PCA_MODEL_PATH: str = os.path.join(MODEL_DIR, 'pca_model.pkl')
    XGBOOST_MODEL_PATH: str = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
    
    LOG_LEVEL: str = 'INFO'
    DOCUMENTATION_PATH: str = ''
    
    LLM_MODEL: str = 'openai/gpt-4o'
    LLM_ENDPOINT: str =  'https://models.github.ai/inference'
    
    with open(KEY_PATH, 'r') as f:
        GITHUB_TOKEN: str = f.read().strip()
    
settings = Settings()