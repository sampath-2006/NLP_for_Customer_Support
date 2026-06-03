import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    print("Loading test data...")
    data = joblib.load('processed_data.pkl')
    X_test = data['X_test']
    y_test = data['y_test']
    
    print("Loading Label Encoder...")
    label_encoder = joblib.load('dl_label_encoder.pkl')
    
    print("Loading DL Model...")
    model = tf.keras.models.load_model('dl_model.keras')
    
    print("Generating predictions...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nDeep Learning Model Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    target_names = label_encoder.classes_
    print(classification_report(y_test, y_pred, target_names=target_names))

if __name__ == '__main__':
    main()
