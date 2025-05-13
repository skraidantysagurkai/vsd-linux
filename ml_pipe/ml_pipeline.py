import pickle
import numpy as np
import xgboost as xgb
from transformers import AutoTokenizer, AutoModel
import torch
import joblib
from typing import List

class MLFilterPipeline:
    def __init__(self, 
                 codebert_model_name='microsoft/codebert-base',
                 pca_path='pca_model.pkl', 
                 xgboost_path='xgboost_model.pkl'):

        self.tokenizer = AutoTokenizer.from_pretrained(codebert_model_name)
        self.codebert_model = AutoModel.from_pretrained(codebert_model_name)
        
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

    
    
def main():

    ml_filter = MLFilterPipeline(
        codebert_model_name='microsoft/codebert-base',
        pca_path= 'models/pca_model.pkl',
        xgboost_path= 'models/xgboost_model.pkl'
    )
    print(ml_filter.pca_model.n_components_)
    # print(f"Prediction result: {result}")

if __name__ == "__main__":
    main()
