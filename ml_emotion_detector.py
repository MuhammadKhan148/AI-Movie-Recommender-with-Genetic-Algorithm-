"""
Optional ML Enhancement for Emotion Detection

This module provides an OPTIONAL ML-based emotion detector that can replace
the simple keyword-based detector IF you want to demonstrate ML capabilities.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import os
from typing import Tuple

class MLEmotionDetectorOptional:
    """Optional ML-based emotion detector - can be used instead of keyword matching"""
    
    def __init__(self):
        self.model = None
        self.create_simple_model()
    
    def create_simple_model(self):
        """Create a simple model for demonstration"""
        # Training data
        texts = [
            'I am so happy', 'This is wonderful', 'Great job',
            'I feel sad', 'This is depressing', 'Very sad',
            'I am angry', 'This is frustrating', 'Mad about this',
            'I am scared', 'This is terrifying', 'Feeling anxious',
            'I am surprised', 'Cant believe it', 'Shocking',
            'The weather today', 'Please review', 'As expected'
        ]
        labels = ['joy']*3 + ['sadness']*3 + ['anger']*3 + ['fear']*3 + ['surprise']*3 + ['neutral']*3
        
        # Train simple model
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=100)),
            ('nb', MultinomialNB())
        ])
        self.model.fit(texts, labels)
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict emotion using ML"""
        if self.model is None:
            return "neutral", 0.5
            
        emotion = self.model.predict([text])[0]
        probs = self.model.predict_proba([text])[0]
        confidence = float(max(probs))
        return emotion, confidence