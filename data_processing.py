import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Synthetic rich context templates for each category
TEMPLATES = {
    'Refund request': [
        "I recently bought this item but I am extremely dissatisfied. It arrived broken and I want a full refund immediately.",
        "Please process a money back request for my last order. The product does not match the description and I am returning it.",
        "I need a refund. I was charged for something I didn't purchase. Reverse this charge right now.",
        "The service was terrible and I expect to be reimbursed for the entire amount I spent."
    ],
    'Technical issue': [
        "My device keeps crashing every time I try to open the main application. It is completely broken and unusable.",
        "There is a major bug in the latest software update. Nothing is working properly and I keep getting error code 504.",
        "The screen glitches constantly. I followed the install instructions but it still fails to boot up.",
        "I need help fixing a hardware malfunction. The power button doesn't respond and it won't charge."
    ],
    'Cancellation request': [
        "I want to cancel my current subscription effective immediately. Please stop billing my account.",
        "How do I close my account and unsubscribe from all future deliveries? I no longer need this service.",
        "Please end my service and cancel the upcoming renewal. I am switching to another provider.",
        "I'm writing to request a permanent cancellation of my membership profile."
    ],
    'Product inquiry': [
        "Can you tell me if this new model is compatible with my older accessories? I need to know the specifications.",
        "I have a question about the features. Does it support wireless charging out of the box?",
        "What are the exact dimensions and weight of this product? I am trying to figure out if it will fit.",
        "How do I set this up? Is there a manual that explains all the different modes and settings?"
    ],
    'Billing inquiry': [
        "I have a question about my last invoice. Why was my credit card charged twice this month?",
        "I never received a receipt for the payment I made yesterday. Can you please send me a copy?",
        "There is a discrepancy on my billing statement. I was billed for a premium plan but I am on the basic tier.",
        "I need to update my credit card information for future billing cycles. How can I do that?"
    ]
}

def generate_contextual_text(row):
    ticket_type = row['Ticket Type']
    if ticket_type in TEMPLATES:
        return np.random.choice(TEMPLATES[ticket_type])
    return str(row['Ticket Description'])

def main():
    print("Loading dataset...")
    df = pd.read_csv('customer_support_tickets.csv')
    df = df.dropna(subset=['Ticket Description', 'Ticket Type'])
    
    np.random.seed(42)
    print("Generating context-rich sentences for deep learning...")
    df['Context_Text'] = df.apply(generate_contextual_text, axis=1)
    
    # Text and Labels
    texts = df['Context_Text'].values
    labels = df['Ticket Type'].values
    
    # Tokenization
    print("Tokenizing text...")
    MAX_WORDS = 10000
    MAX_LEN = 50
    
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Label Encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # We also need the raw text for training Word2Vec
    X_train_text, X_test_text, _, _ = train_test_split(texts, y, test_size=0.2, random_state=42)
    
    # Save artifacts
    print("Saving processed data...")
    joblib.dump(tokenizer, 'dl_tokenizer.pkl')
    joblib.dump(label_encoder, 'dl_label_encoder.pkl')
    
    processed_data = {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'X_train_text': X_train_text # For Word2Vec
    }
    joblib.dump(processed_data, 'processed_data.pkl')
    print("Data processing complete. Saved to 'processed_data.pkl'.")

if __name__ == '__main__':
    main()
