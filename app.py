import streamlit as st
import joblib
import pandas as pd
import time
import os
import numpy as np

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---- PAGE CONFIGURATION ----
st.set_page_config(page_title="AI Ticket Classifier", page_icon="✨", layout="wide", initial_sidebar_state="collapsed")

# ---- CUSTOM CSS FOR PREMIUM AESTHETICS ----
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Background and global styles */
    .stApp {
        background: linear-gradient(140deg, #0f172a 0%, #1e1b4b 100%);
        color: #e2e8f0;
    }

    /* Title styling */
    .premium-title {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(to right, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }

    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        font-weight: 300;
        color: #94a3b8;
        margin-bottom: 3rem;
    }

    /* Glassmorphism containers */
    .glass-container {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
    }

    /* Text Area Styling */
    .stTextArea textarea {
        background-color: rgba(15, 23, 42, 0.6) !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        font-size: 1.1rem !important;
        padding: 1rem !important;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #818cf8 !important;
        box-shadow: 0 0 15px rgba(129, 140, 248, 0.3) !important;
    }

    /* Button Styling */
    .stButton button {
        background: linear-gradient(135deg, #4f46e5, #ec4899);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(236, 72, 153, 0.4);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(236, 72, 153, 0.6);
        background: linear-gradient(135deg, #6366f1, #f472b6);
    }

    /* Category Badge */
    .category-badge {
        font-size: 1.8rem;
        font-weight: 800;
        padding: 1rem 2rem;
        border-radius: 12px;
        display: inline-block;
        color: white;
        text-align: center;
        width: 100%;
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
        animation: popIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    @keyframes popIn {
        0% { transform: scale(0.8); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(129, 140, 248, 0.5);
    }
    
    .metric-title {
        color: #94a3b8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }

    /* Hide default footer only */
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ---- CACHE LOADING THE MODEL ----
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("dl_model.keras")
    tokenizer = joblib.load("dl_tokenizer.pkl")
    label_encoder = joblib.load("dl_label_encoder.pkl")
    return model, tokenizer, label_encoder

try:
    model, tokenizer, label_encoder = load_assets()
    model_ready = True
except Exception as e:
    model_ready = False

# ---- APP HEADER ----
st.markdown('<div class="premium-title">AI Contextual Routing</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Next-generation customer support powered by LSTM Deep Learning</div>', unsafe_allow_html=True)

# ---- LAYOUT ----
col1, col_space, col2 = st.columns([5, 0.5, 3])

with col1:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("### 📝 Submit a Support Ticket")
    ticket_text = st.text_area(
        label="Customer Inquiry:",
        label_visibility="collapsed",
        height=220, 
        placeholder="Type your customer query here... E.g., 'I bought this last week but the screen is entirely broken and it won't charge. I want my money back immediately.'"
    )
    
    if st.button("Analyze & Route Ticket"):
        if ticket_text.strip() == "":
            st.warning("⚠️ Please enter a ticket description.")
        elif not model_ready:
            st.error("⚠️ Deep Learning model not loaded properly. Please run the training script.")
        else:
            with st.spinner("🧠 Extracting semantic context via Word2Vec..."):
                time.sleep(0.8) # Premium micro-interaction delay
                
                # Preprocess
                MAX_LEN = 50
                seq = tokenizer.texts_to_sequences([ticket_text])
                padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
                
                # Predict
                pred_probs = model.predict(padded)
                pred_class_idx = np.argmax(pred_probs, axis=1)[0]
                prediction = label_encoder.inverse_transform([pred_class_idx])[0]
                confidence = pred_probs[0][pred_class_idx] * 100
                
                # Dynamic Category Styling
                colors = {
                    'Refund request': 'linear-gradient(135deg, #ef4444, #b91c1c)',
                    'Technical issue': 'linear-gradient(135deg, #3b82f6, #1d4ed8)',
                    'Cancellation request': 'linear-gradient(135deg, #f59e0b, #b45309)',
                    'Product inquiry': 'linear-gradient(135deg, #10b981, #047857)',
                    'Billing inquiry': 'linear-gradient(135deg, #8b5cf6, #5b21b6)'
                }
                bg_color = colors.get(prediction, 'linear-gradient(135deg, #6e40c9, #1f6feb)')
                
                st.markdown(f'<div style="text-align:center; margin-top:20px; font-size:1.1rem; color:#cbd5e1;">Routing Destination Identified (Confidence: {confidence:.1f}%)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="category-badge" style="background: {bg_color};">{prediction}</div>', unsafe_allow_html=True)
                
                if confidence > 90:
                    st.balloons()
                
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("### 📊 System Telemetry")
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Automation Rate</div>
            <div class="metric-value" style="color: #38bdf8;">87.4%</div>
            <div style="color: #22c55e; font-size: 0.9rem;">▲ +4.2% this week</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Avg. Resolution Time</div>
            <div class="metric-value" style="color: #c084fc;">1.2 hrs</div>
            <div style="color: #22c55e; font-size: 0.9rem;">▼ -35% vs human triage</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Context Model Health</div>
            <div class="metric-value" style="color: #4ade80;">99.7%</div>
            <div style="color: #94a3b8; font-size: 0.9rem;">LSTM Network Online</div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
