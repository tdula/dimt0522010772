import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

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
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.array([[self._count_category_keywords(text, cat) 
                         for cat in category_keywords.keys()] 
                        for text in X])
    
    def _count_category_keywords(self, text, category):
        text = text.lower()
        count = sum(1 for keyword in category_keywords[category] if keyword in text)
        return count / len(category_keywords[category])  # Normalize by category size

# Load the trained model
print("Loading model...")
with open('models/expense_model.pkl', 'rb') as f:
    model = pickle.load(f)

# New test cases with known categories
test_cases = [
    # Format: (text, amount, expected_category)
    ("Bought groceries at Walmart", 50, "Groceries"),
    ("Monthly Netflix subscription", 15, "Entertainment"),
    ("Electricity bill payment", 100, "Utilities"),
    ("Uber ride to work", 25, "Transport"),
    ("New laptop from Amazon", 800, "Electronics"),
    ("Dinner at Italian restaurant", 45, "Food"),
    ("Health insurance premium", 200, "Insurance"),
    ("Gym membership monthly", 50, "Health"),
    ("Movie tickets for family", 60, "Entertainment"),
    ("Gas station fill up", 40, "Transport"),
    ("Internet bill", 60, "Utilities"),
    ("Coffee and sandwich", 12, "Food"),
    ("Monthly rent payment", 1200, "Housing"),
    ("New phone case", 20, "Electronics"),
    ("Grocery shopping at local store", 75, "Groceries"),
    ("Doctor's appointment", 100, "Health"),
    ("Bus ticket", 5, "Transport"),
    ("Shopping mall clothes", 120, "Shopping"),
    ("House insurance", 150, "Insurance"),
    ("Water bill", 40, "Utilities")
]

# Evaluate model on test cases
print("\n=== Model Evaluation on New Test Cases ===")
correct = 0
predictions = []
true_categories = []
confidences = []

for text, amount, expected in test_cases:
    # Format input text
    input_text = f"Spent ${amount} on {text}"
    
    # Make prediction
    prediction = model.predict([input_text])[0]
    confidence = max(model.predict_proba([input_text])[0])
    
    predictions.append(prediction)
    true_categories.append(expected)
    confidences.append(confidence)
    
    if prediction == expected:
        correct += 1
    
    print(f"\nInput: {text} (${amount})")
    print(f"Expected: {expected}")
    print(f"Predicted: {prediction}")
    print(f"Confidence: {confidence:.2%}")
    print("✓" if prediction == expected else "✗")

# Calculate overall accuracy
accuracy = correct / len(test_cases)
print(f"\nOverall Accuracy on New Test Cases: {accuracy:.2%}")

# Print detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(true_categories, predictions))

# Create confusion matrix
cm = confusion_matrix(true_categories, predictions)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(set(true_categories)),
            yticklabels=sorted(set(true_categories)))
plt.title('Confusion Matrix on New Test Cases')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('static/evaluation_confusion_matrix.png')
plt.close()

# Analyze confidence distribution
plt.figure(figsize=(10, 6))
plt.hist(confidences, bins=20, edgecolor='black')
plt.title('Distribution of Prediction Confidences')
plt.xlabel('Confidence')
plt.ylabel('Count')
plt.savefig('static/confidence_distribution.png')
plt.close()

# Analyze performance by confidence threshold
print("\nPerformance by Confidence Threshold:")
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
for threshold in thresholds:
    high_conf_mask = np.array(confidences) >= threshold
    if sum(high_conf_mask) > 0:
        high_conf_acc = accuracy_score(
            np.array(true_categories)[high_conf_mask],
            np.array(predictions)[high_conf_mask]
        )
        print(f"Accuracy for confidence >= {threshold:.1%}: {high_conf_acc:.2%} "
              f"({sum(high_conf_mask)} predictions)")

# Analyze common misclassifications
print("\nCommon Misclassifications:")
misclassified = [(text, amount, true, pred) 
                 for (text, amount, true), pred 
                 in zip(test_cases, predictions) 
                 if true != pred]
if misclassified:
    print("\nMisclassified Examples:")
    for text, amount, true, pred in misclassified:
        print(f"Text: {text} (${amount})")
        print(f"True: {true}, Predicted: {pred}")
        print("-" * 50)
else:
    print("No misclassifications found in test cases.")

# Save evaluation results
results = {
    'accuracy': accuracy,
    'test_cases': len(test_cases),
    'correct_predictions': correct,
    'misclassifications': len(misclassified),
    'confidence_mean': np.mean(confidences),
    'confidence_std': np.std(confidences)
}

print("\nSummary:")
print(f"Total test cases: {results['test_cases']}")
print(f"Correct predictions: {results['correct_predictions']}")
print(f"Misclassifications: {results['misclassifications']}")
print(f"Average confidence: {results['confidence_mean']:.2%}")
print(f"Confidence std dev: {results['confidence_std']:.2%}")

# Save results to file
with open('evaluation_results.txt', 'w') as f:
    for key, value in results.items():
        f.write(f"{key}: {value}\n") 