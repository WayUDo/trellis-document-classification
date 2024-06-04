from pydantic import BaseModel, Field, validator
from typing import Any, Dict, Type, Callable
from abc import ABC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from joblib import load
import asyncio
from transformers import AutoTokenizer, TFAutoModel
import numpy as np

#Defines schema for valid configuration for model contents loaded from pickle
class DocumentClassifierConfig(BaseModel):
    label_encoder: Any
    fit_model: Any
    prob_threshold: float

#loads and validates the contents according to the schema defined in the DocumentClassifierConfig
class DocumentClassifierConfigLoader:
    def __init__(self, filepath, config_type: DocumentClassifierConfig):
        self.filepath = filepath
        self.config = None
        self.config_type = config_type

    def load_and_validate(self):
        parameters = load(self.filepath)
        self.config = self.config_type(**parameters)
        print("Data loaded and validated successfully.")

    def get_config(self):
        if self.config is None:
            raise ValueError("Configuration has not been loaded or validated.")
        return self.config

#Valid Document Request Object
class DocumentRequest(BaseModel):
    text: str

    @validator('text')
    def check_text_length(cls, text):
        if len(text.split(' ')) > 15000:
            raise ValueError('text has too many tokens!')
        if len(text) > 60000:
            raise ValueError('text has too many characters!')
        return text

#Valid Document Response Object
class PredictedDocumentClass(BaseModel):
    class_label: str

#Main object that handles the classification. When it is loaded it loads the embedding
#model into itself just once, and uses those loaded models for processing async requests to different users.
class DocumentClassifier():
    def __init__(self, config_loader: DocumentClassifierConfigLoader):
        self.config_loader = config_loader
        self.config = None

    def load(self):
        self.config_loader.load_and_validate()
        self.config = self.config_loader.get_config()
        self.embedding_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.embedding_model = TFAutoModel.from_pretrained('bert-base-uncased')
    
    async def get_embeddings(self, text):
        return await asyncio.to_thread(self._sync_embeddings, text)

    def _sync_embeddings(self, text: str) -> Any:
        inputs = self.embedding_tokenizer(text, return_tensors='tf', truncation=True, padding='max_length', max_length=512)
        outputs = self.embedding_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy()
    
    def run_log_threshold_model(self, embedding: list):
        model = self.config.fit_model
        threshold = self.config.prob_threshold
        label_encoder = self.config.label_encoder

        probabilities = model.predict_proba(embedding)
        
        for prob in probabilities:
            max_prob = np.max(prob)
            if max_prob > threshold:
                return label_encoder.inverse_transform([np.argmax(prob)])[0]
            else:
                return 'other'
    
    async def predict(self, text: str) -> str:
        embeddings = await self.get_embeddings(text)

        # Run prediction in a separate thread if it's a CPU-bound task
        predicted_class = await asyncio.to_thread(self.run_log_threshold_model, embeddings)

        return predicted_class