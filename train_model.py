import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from sklearn.base import BaseEstimator, TransformerMixin

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Enhanced category-specific keywords with more context
category_keywords = {
    'Food': ['restaurant', 'dinner', 'lunch', 'breakfast', 'cafe', 'food', 'meal', 'snack', 'coffee', 'dining', 'eat', 'cuisine', 'menu', 'takeout', 'delivery', 'burger', 'pizza', 'sandwich', 'drink', 'beverage'],
    'Transport': ['gas', 'fuel', 'car', 'uber', 'taxi', 'bus', 'train', 'transport', 'ride', 'travel', 'commute', 'transportation', 'vehicle', 'driving', 'metro', 'fare', 'ticket', 'station', 'parking', 'toll'],
    'Utilities': ['bill', 'electricity', 'water', 'internet', 'utility', 'phone', 'service', 'power', 'gas', 'energy', 'connection', 'network', 'monthly', 'subscription', 'provider', 'usage', 'meter', 'supply', 'payment', 'due'],
    'Electronics': ['laptop', 'phone', 'computer', 'gadget', 'device', 'electronic', 'technology', 'tech', 'hardware', 'software', 'digital', 'equipment', 'battery', 'charger', 'accessory', 'screen', 'camera', 'printer', 'wireless', 'smart'],
    'Entertainment': ['movie', 'concert', 'netflix', 'subscription', 'game', 'entertainment', 'show', 'theater', 'music', 'streaming', 'play', 'fun', 'leisure', 'ticket', 'event', 'performance', 'festival', 'cinema', 'amusement', 'recreation'],
    'Health': ['medicine', 'doctor', 'gym', 'fitness', 'health', 'medical', 'pharmacy', 'healthcare', 'prescription', 'wellness', 'exercise', 'treatment', 'clinic', 'hospital', 'checkup', 'appointment', 'therapy', 'dental', 'vitamin', 'consultation'],
    'Groceries': ['grocery', 'supermarket', 'walmart', 'food', 'supplies', 'market', 'store', 'fresh', 'produce', 'ingredients', 'shopping', 'cart', 'basket', 'fruit', 'vegetable', 'meat', 'dairy', 'household', 'pantry', 'organic'],
    'Shopping': ['mall', 'clothes', 'shopping', 'store', 'buy', 'purchase', 'retail', 'shop', 'merchandise', 'goods', 'items', 'products', 'online', 'fashion', 'apparel', 'accessories', 'brand', 'sale', 'discount', 'boutique'],
    'Housing': ['rent', 'apartment', 'house', 'housing', 'mortgage', 'lease', 'property', 'home', 'accommodation', 'residence', 'living', 'tenant', 'landlord', 'maintenance', 'repair', 'utility', 'deposit', 'fee', 'monthly', 'contract'],
    'Insurance': ['insurance', 'coverage', 'policy', 'protection', 'premium', 'claim', 'insure', 'risk', 'safety', 'security', 'plan', 'benefit', 'health', 'life', 'auto', 'property', 'liability', 'deductible', 'provider', 'monthly']
}

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english')) - {'on', 'for', 'at', 'in', 'to', 'from', 'by'}
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return [self._preprocess_text(text) for text in X]
    
    def _preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^a-z0-9\s$\.%-]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)

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

# Create directories if they don't exist
if not os.path.exists('static'):
    os.makedirs('static')
if not os.path.exists('models'):
    os.makedirs('models')

# Load and prepare data
print("Loading and preparing data...")
df = pd.read_csv('expense_data.csv')

# Feature engineering
df['text_with_amount'] = df.apply(
    lambda x: f"Spent ${x['Amount']} on {x['Description']}", axis=1
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    df['text_with_amount'],
    df['Category'],
    test_size=0.2,
    random_state=42,
    stratify=df['Category']
)

# Create feature union pipeline
feature_union = FeatureUnion([
    ('text_features', Pipeline([
        ('preprocessor', TextPreprocessor()),
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95
        ))
    ])),
    ('amount_features', AmountExtractor()),
    ('keyword_features', KeywordFeatureExtractor())
])

# Create main pipeline
pipeline = Pipeline([
    ('features', feature_union),
    ('scaler', StandardScaler(with_mean=False)),  # sparse matrices don't support with_mean=True
    ('clf', RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

print("Training model...")
pipeline.fit(X_train, y_train)

# Make predictions
print("Evaluating model performance...")
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Create confusion matrix visualization
plt.figure(figsize=(12, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(df['Category'].unique()),
            yticklabels=sorted(df['Category'].unique()))
plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.2f}')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('static/confusion_matrix.png')
plt.close()

# Create category distribution visualization
plt.figure(figsize=(12, 6))
df['Category'].value_counts().plot(kind='bar')
plt.title('Distribution of Expense Categories')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('static/category_distribution.png')
plt.close()

# Save the model
with open('models/expense_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Print detailed evaluation metrics
print("\n=== Model Evaluation ===")
print(f"Test Set Accuracy: {accuracy:.2f}")

print("\nCross-validation scores:")
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
for i, score in enumerate(cv_scores, 1):
    print(f"Fold {i}: {score:.2f}")
print(f"Average CV Score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Print confidence distribution
confidences = np.max(y_pred_proba, axis=1)
plt.figure(figsize=(10, 6))
plt.hist(confidences, bins=20, edgecolor='black')
plt.title('Distribution of Prediction Confidences')
plt.xlabel('Confidence')
plt.ylabel('Count')
plt.savefig('static/confidence_distribution.png')
plt.close()

print("\nModel and visualizations have been saved.")