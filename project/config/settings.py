from pydantic_settings import BaseSettings
import os
from paths import MODEL_DIR

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
    # GITHUB_TOKEN: str = str(os.environ.get('GITHUB_MODELS_TOKEN'))
    GITHUB_TOKEN: str = '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'
    
settings = Settings()