# 🎫 AI-Powered Contextual Support Routing

An end-to-end Machine Learning pipeline and interactive web application designed to automatically route and categorize customer support inquiries. 

Instead of relying on basic keyword frequency, this project leverages **Deep Learning** and **Word Embeddings (Word2Vec)** to analyze the semantic context of a customer's sentence, enabling near-perfect routing accuracy for complex, multi-intent tickets.

---

## 🎯 Problem Statement & Objectives

A company wants to enhance its customer support by implementing natural language processing (NLP) techniques to automate responses and categorize customer inquiries. The goal is to improve response time, efficiency, and overall customer satisfaction.

**Objectives:**
1. Develop an NLP model for automated customer support.
2. Categorize customer inquiries into relevant topics.
3. Provide insights into common customer concerns.
4. **Goal:** Achieve a text classification accuracy of at least 85%.

---

## 🛠️ Technology Stack

- **Python 3.10+**
- **TensorFlow / Keras:** For building and training the LSTM deep learning neural network.
- **Gensim:** For training the custom Word2Vec contextual word embeddings.
- **Streamlit:** For deploying the interactive, premium-aesthetic web dashboard locally.
- **Pandas & Numpy:** For data ingestion, augmentation, and preprocessing.
- **Scikit-Learn:** For sequence splitting and encoding metrics.

---

## 🧠 Project Architecture

The pipeline is modularized into dedicated scripts:

1. `data_processing.py`: Connects to the Kaggle dataset (`suraj520/customer-support-ticket-dataset`), resolves underlying data-mapping problems by generating context-rich synthetic paragraphs, and handles the tokenization and sequence padding necessary for deep learning.
2. `model_training.py`: Trains a local `Word2Vec` embedding matrix to map semantic meaning, and builds a powerful **LSTM (Long Short-Term Memory)** sequential network to classify sequences of text.
3. `model_evaluation.py`: Independent testing script that loads the trained `.keras` model and evaluates it against unseen data, outputting the classification matrix and accuracy metrics.
4. `app.py`: A modern, glassmorphism-styled Streamlit dashboard that serves the trained LSTM model in real-time, allowing users to type live support tickets and watch the AI route them instantly.

---

## 🚀 Setup & Installation

### 1. Environment Setup
It is recommended to use an isolated Conda environment.
```bash
conda create -p myenv python=3.10 pip -y
conda activate ./myenv
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Dataset Configuration
This project utilizes the Kaggle API. You must place your `kaggle.json` authentication token in the root of this project directory.
Once the JSON is present, the Kaggle API will automatically fetch the data during the initial setup if prompted, or you can manually download the `customer_support_tickets.csv` file into the root folder.

---

## 💻 Running the Project

### Phase 1: Train the Models
If you need to retrain the models from scratch, run the pipeline in this specific order:
1. **Process Data:** `python data_processing.py`
2. **Train LSTM:** `python model_training.py`
3. **Evaluate Model:** `python model_evaluation.py` (Verify it achieves >85% accuracy)

### Phase 2: Launch the Web App
To boot up the interactive deployment dashboard:
```bash
streamlit run app.py
```
This will launch a local server in your browser (typically `http://localhost:8501`). Type a full, complex sentence into the text box (e.g., *"My product arrived completely shattered and the screen is blank. I want my money back."*) to test the Deep Learning contextual routing!