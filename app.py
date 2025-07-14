from flask import Flask, request, jsonify, render_template
import pickle
import os
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Category-specific keywords
category_keywords = {
    'Food': ['restaurant', 'dinner', 'lunch', 'breakfast', 'cafe', 'food', 'meal', 'snack', 'coffee'],
    'Transport': ['gas', 'fuel', 'car', 'uber', 'taxi', 'bus', 'train', 'transport', 'ride', 'travel'],
    'Utilities': ['bill', 'electricity', 'water', 'internet', 'utility', 'phone', 'service'],
    'Electronics': ['laptop', 'phone', 'computer', 'gadget', 'device', 'electronic'],
    'Entertainment': ['movie', 'concert', 'netflix', 'subscription', 'game', 'entertainment'],
    'Health': ['medicine', 'doctor', 'gym', 'fitness', 'health', 'medical', 'pharmacy'],
    'Groceries': ['grocery', 'supermarket', 'walmart', 'food', 'supplies'],
    'Shopping': ['mall', 'clothes', 'shopping', 'store', 'buy', 'purchase'],
    'Housing': ['rent', 'apartment', 'house', 'housing', 'mortgage'],
    'Insurance': ['insurance', 'coverage', 'policy', 'protection']
}

class KeywordFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, keywords=None):
        self.keywords = keywords if keywords is not None else category_keywords

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            text = text.lower()
            feature_vector = []
            for category, keywords in self.keywords.items():
                matches = sum(1 for keyword in keywords if keyword in text)
                feature_vector.append(matches)
            features.append(feature_vector)
        return np.array(features)

    def get_params(self, deep=True):
        return {"keywords": self.keywords}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

class AmountExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.amount_ranges = {
            'very_low': (0, 20),
            'low': (20, 50),
            'medium': (50, 100),
            'high': (100, 500),
            'very_high': (500, float('inf'))
        }
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.array([self._get_amount_features(text) for text in X])
    
    def _extract_amount(self, text):
        amount_match = re.search(r'\$?(\d+(?:\.\d{2})?)', text)
        if amount_match:
            return float(amount_match.group(1))
        return 0.0
    
    def _get_amount_features(self, text):
        amount = self._extract_amount(text)
        features = []
        for min_val, max_val in self.amount_ranges.values():
            features.append(1 if min_val <= amount < max_val else 0)
        features.append(amount / 1000)  # Normalized amount
        return features

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english')) - {'on', 'for', 'at', 'in', 'to'}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.preprocess_text(text) for text in X]

    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Extract amount if present
        amount = 0
        amount_match = re.search(r'\$?\d+', text)
        if amount_match:
            amount = float(amount_match.group().replace('$', ''))
        
        # Enhance text with keywords
        text = self.enhance_text_with_keywords(text, amount)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)

    def enhance_text_with_keywords(self, text, amount):
        enhanced_text = text.lower()
        
        # Add amount-based context
        if amount <= 20:
            enhanced_text += ' small_purchase minor_expense'
        elif amount <= 50:
            enhanced_text += ' medium_purchase regular_expense'
        elif amount <= 100:
            enhanced_text += ' significant_purchase'
        elif amount <= 500:
            enhanced_text += ' major_purchase big_expense'
        else:
            enhanced_text += ' large_purchase significant_expense'
        
        # Add category-specific keywords
        for category, keywords in category_keywords.items():
            if any(keyword in enhanced_text for keyword in keywords):
                enhanced_text += f' {category.lower()}_related'
        
        return enhanced_text

app = Flask(__name__)

# Load the model
try:
    logger.info("Loading model from models/expense_model.pkl")
    with open('models/expense_model.pkl', 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
    
    # Ensure the model's components have the correct state
    if hasattr(model, 'named_steps'):
        if 'keyword_extractor' in model.named_steps:
            logger.info("Updating keyword extractor parameters")
            model.named_steps['keyword_extractor'].keywords = category_keywords
    
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.error(traceback.format_exc())
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log the incoming request
        logger.info("Received prediction request")
        
        # Get the text from request
        text = request.json.get('text', '')
        logger.info(f"Input text: {text}")
        
        if not text:
            logger.warning("Empty text received")
            return jsonify({
                'error': 'No text provided',
                'category': 'Unknown',
                'confidence': 0.0
            }), 400
        
        # Format text similar to training data
        formatted_text = f"Spent ${text}" if '$' not in text else text
        logger.info(f"Formatted text: {formatted_text}")
        
        # Make prediction
        logger.info("Making prediction")
        prediction = model.predict([formatted_text])[0]
        logger.info(f"Predicted category: {prediction}")
        
        # Get prediction probability
        proba = model.predict_proba([formatted_text])[0]
        confidence = float(max(proba))
        logger.info(f"Confidence: {confidence}")
        
        return jsonify({
            'category': prediction,
            'confidence': confidence
        })
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'category': 'Error',
            'confidence': 0.0
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 