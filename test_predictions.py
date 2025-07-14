import pickle
import pandas as pd
from nltk.stem import WordNetLemmatizer
import nltk
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def get_amount_category(amount):
    if amount <= 20:
        return 'very_low'
    elif amount <= 50:
        return 'low'
    elif amount <= 100:
        return 'medium'
    elif amount <= 500:
        return 'high'
    else:
        return 'very_high'

# Load the model
with open('models/expense_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Test cases
test_cases = [
    "Spent $50 on groceries at Walmart",
    "Bought coffee for $5",
    "Paid $800 for rent",
    "Dinner at restaurant $30",
    "Netflix subscription $15",
    "Bought new laptop for $900",
    "Gas for car $40",
    "Monthly internet bill $50",
    "Concert tickets $75",
    "Medicine from pharmacy $20"
]

print("\n=== Testing Model Predictions ===\n")
print("Format: Input Text -> Predicted Category (Confidence)")
print("-" * 60)

for text in test_cases:
    # Extract amount
    amount_match = re.search(r'\$?\d+', text)
    amount = float(amount_match.group().replace('$', '')) if amount_match else 0
    
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Add amount context
    amount_cat = get_amount_category(amount)
    processed_text = f"spent {amount_cat} amount {processed_text}"
    
    # Get prediction and confidence
    prediction = model.predict([processed_text])[0]
    confidence = max(model.predict_proba([processed_text])[0])
    
    print(f"\nInput: {text}")
    print(f"Predicted Category: {prediction}")
    print(f"Confidence: {confidence:.2%}")

# Load original data to compare predictions
df = pd.read_csv('expense_data.csv')
print("\n\n=== Model Performance on Different Categories ===\n")
print("Category-wise examples from training data:")
print("-" * 60)

for category in df['Category'].unique():
    examples = df[df['Category'] == category].sample(min(2, len(df[df['Category'] == category])))
    print(f"\nCategory: {category}")
    for _, row in examples.iterrows():
        text = f"Spent ${row['Amount']} on {row['Description']}"
        processed_text = preprocess_text(text)
        amount_cat = get_amount_category(row['Amount'])
        processed_text = f"spent {amount_cat} amount {processed_text}"
        
        prediction = model.predict([processed_text])[0]
        confidence = max(model.predict_proba([processed_text])[0])
        
        print(f"Original Text: {text}")
        print(f"Predicted: {prediction} (Confidence: {confidence:.2%})")
        print(f"Actual: {row['Category']}")
        print("-" * 30) 