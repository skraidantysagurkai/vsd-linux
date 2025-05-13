import pickle
from transformers import AutoTokenizer, AutoModel
import torch
import joblib
from src.data.features.event_feature_extractor import EventFeatureExtractor
from src.shared.log_check import Log

class MlPipeline:
    def __init__(self, 
                 pca_path='pca_model.pkl', 
                 xgboost_path='xgboost_model.pkl'):

        self.feature_extractor = EventFeatureExtractor()
        
        with open(pca_path, 'rb') as f:
            self.pca_model = joblib.load(f)

        with open(xgboost_path, 'rb') as f:
            self.xgboost_model = pickle.load(f)
    
    def extract_codebert_features(self, code_snippet):
        inputs = self.tokenizer(code_snippet, return_tensors='pt', 
                                max_length=512, truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.codebert_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        
        return embeddings
    
    def predict(self, code_snippet):
        codebert_features = self.extract_codebert_features(code_snippet)
        
        pca_features = self.pca_model.transform(codebert_features)
        
        prediction = self.xgboost_model.predict(pca_features)
        
        return prediction[0]
    
    def embed_log(self, log: Log):
        self.feature_extractor.embed_command(log['command'])

if __name__ == "__main__":
    main()
