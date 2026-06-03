import numpy as np
import joblib
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    print("Loading processed data and tokenizer...")
    tokenizer = joblib.load('dl_tokenizer.pkl')
    data = joblib.load('processed_data.pkl')
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_train_text = data['X_train_text']
    
    vocab_size = min(10000, len(tokenizer.word_index) + 1)
    embedding_dim = 100
    
    print("Training Word2Vec model on the training text...")
    # Tokenize the raw text for Word2Vec (list of lists of words)
    sentences = [str(text).lower().split() for text in X_train_text]
    
    w2v_model = Word2Vec(sentences, vector_size=embedding_dim, window=5, min_count=1, workers=4)
    print("Word2Vec training complete.")
    
    print("Building Embedding Matrix...")
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    # Iterate through words in tokenizer and map to Word2Vec vector
    for word, i in tokenizer.word_index.items():
        if i >= vocab_size:
            continue
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
            
    print(f"Embedding matrix shape: {embedding_matrix.shape}")
    
    # Identify number of classes
    num_classes = len(np.unique(y_train))
    print(f"Number of target classes: {num_classes}")
    
    print("Building Deep Learning LSTM Model...")
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            trainable=True, # Allow fine-tuning of embeddings
            mask_zero=True
        ),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    print("Training the DL model...")
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # We use a validation split from the train set for early stopping
    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    
    print("Saving the Keras model...")
    model.save('dl_model.keras')
    print("Model saved to 'dl_model.keras'.")

if __name__ == '__main__':
    main()
