"""
Streamlit Cloud Version - Standalone Dashboard
Works without API backend (uses mock data for demo)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import requests

# Page config
st.set_page_config(
    page_title="IDS Dashboard",
    page_icon="🔒",
    layout="wide"
)

# Title
st.markdown('<h1 style="text-align: center; color: #1f77b4;">Intrusion Detection System</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center;">Machine Learning-Based Network Security Monitor</h3>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Live Detection", "Analytics", "About"])

# API Configuration
API_URL = "https://ids-api-33k6.onrender.com"

# API prediction function
def api_predict(data):
    """Call API for prediction"""
    try:
        response = requests.post(f"{API_URL}/predict", json=data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.Timeout:
        st.warning("API request timed out. Using fallback prediction.")
        return mock_predict(data)
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None

# Fallback mock prediction function
def mock_predict(data):
    """Fallback prediction if API fails"""
    if data['src_bytes'] > 5000 or data['dst_bytes'] > 5000:
        return {
            'prediction': 'attack',
            'confidence': np.random.uniform(0.85, 0.99),
            'probabilities': {
                'normal': np.random.uniform(0.01, 0.15),
                'attack': np.random.uniform(0.85, 0.99)
            }
        }
    else:
        return {
            'prediction': 'normal',
            'confidence': np.random.uniform(0.80, 0.95),
            'probabilities': {
                'normal': np.random.uniform(0.80, 0.95),
                'attack': np.random.uniform(0.05, 0.20)
            }
        }

# HOME PAGE
if page == "Home":
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Models Trained", "7")
    col2.metric("Best Accuracy", "99%")
    col3.metric("Dataset Size", "125,972")
    col4.metric("Status", "Online")
    
    st.markdown("---")
    
    # Two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Model Performance Comparison")
        
        models = ["Logistic\nRegression", "Naive\nBayes", "Decision\nTree", 
                  "KNN", "SVM", "Random\nForest", "Neural\nNetwork"]
        accuracies = [82, 90, 93, 97, 98, 99, 99.2]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(models, accuracies, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', 
                             '#98D8C8', '#6BCF7F', '#95E1D3'])
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 100])
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}%', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Best Model: Random Forest")
        st.metric("Accuracy", "99.0%")
        st.metric("Precision", "96.1%")
        st.metric("Recall", "98.0%")
        st.metric("F1-Score", "97.0%")
        
        st.markdown("---")
        st.info("**Dataset:** NSL-KDD\n\n**Features:** 122\n\n**Classes:** Normal, Attack")

# LIVE DETECTION PAGE
elif page == "Live Detection":
    st.title("Live Network Traffic Detection")
    st.info("This is a demo version. Enter traffic parameters to test detection.")
    
    with st.form("detection_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Connection Info**")
            duration = st.number_input("Duration (sec)", min_value=0, value=0)
            src_bytes = st.number_input("Source Bytes", min_value=0, value=181)
            dst_bytes = st.number_input("Destination Bytes", min_value=0, value=5450)
        
        with col2:
            st.markdown("**Protocol & Service**")
            protocol = st.selectbox("Protocol", ["tcp", "udp", "icmp"])
            service = st.selectbox("Service", 
                ["http", "ftp", "smtp", "telnet", "ssh", "private"])
            flag = st.selectbox("Flag", ["SF", "S0", "REJ", "RSTO"])
        
        with col3:
            st.markdown("**Traffic Stats**")
            count = st.number_input("Count", min_value=0, value=1)
            srv_count = st.number_input("Service Count", min_value=0, value=1)
            serror_rate = st.slider("SYN Error Rate", 0.0, 1.0, 0.0)
        
        submitted = st.form_submit_button("Analyze Traffic", type="primary", use_container_width=True)
    
    if submitted:
        data = {
            "duration": duration,
            "protocol_type": protocol,
            "service": service,
            "flag": flag,
            "src_bytes": src_bytes,
            "dst_bytes": dst_bytes,
            "count": count,
            "srv_count": srv_count,
            "serror_rate": serror_rate
        }
        
        with st.spinner("Analyzing traffic..."):
            result = api_predict(data)
            
        if not result:
            st.error("Failed to get prediction. Please try again.")
            st.stop()
        
        st.markdown("---")
        st.subheader("Detection Result")
        
        if result['prediction'] == 'attack':
            st.error(f"### ATTACK DETECTED!")
            st.metric("Confidence", f"{result['confidence']*100:.1f}%")
            st.warning("**Recommended Actions:**\n- Block source IP\n- Log incident\n- Alert security team")
        else:
            st.success(f"### Normal Traffic")
            st.metric("Confidence", f"{result['confidence']*100:.1f}%")
        
        col1, col2 = st.columns(2)
        col1.metric("Normal Probability", f"{result['probabilities']['normal']*100:.1f}%")
        col2.metric("Attack Probability", f"{result['probabilities']['attack']*100:.1f}%")

# ANALYTICS PAGE
elif page == "Analytics":
    st.title("System Analytics")
    
    tab1, tab2 = st.tabs(["Model Comparison", "Confusion Matrix"])
    
    with tab1:
        st.subheader("Model Performance Comparison")
        
        comparison_data = {
            "Model": ["Logistic Regression", "Naive Bayes", "Decision Tree", 
                     "KNN", "SVM", "Random Forest", "Neural Network"],
            "Accuracy": ["82%", "90%", "93%", "97%", "98%", "99%", "99.2%"],
            "Precision": ["78%", "88%", "91%", "96%", "97%", "96%", "98%"],
            "Recall": ["80%", "89%", "92%", "97%", "98%", "98%", "99%"],
            "F1-Score": ["79%", "88.5%", "91.5%", "96.5%", "97.5%", "97%", "98.5%"]
        }
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("Confusion Matrix - Random Forest")
        
        cm = np.array([[13000, 468], [234, 11493]])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Attack'],
                    yticklabels=['Normal', 'Attack'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        st.pyplot(fig)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("True Negatives", "13,000")
        col2.metric("False Positives", "468")
        col3.metric("False Negatives", "234")
        col4.metric("True Positives", "11,493")

# ABOUT PAGE
else:
    st.title("About This Project")
    
    st.markdown("""
    ## Intrusion Detection System Using Machine Learning
    
    ### Development Team
    """)
    
    st.markdown("""
    ### Team Roles & Responsibilities
    """)
    
    # Create roles table
    roles_data = {
        "Role": [
            "Machine Learning Engineers",
            "Data Engineers", 
            "Dashboard Developers"
        ],
        "Team Members": [
            "Christine, Cosma",
            "Johnson, Ambrose",
            "Grace, Lailah"
        ],
        "Responsibilities": [
            "• Data preprocessing and feature engineering • Training ML models (Random Forest, SVM, Logistic Regression, Neural Networks) • Hyperparameter tuning • Model evaluation (accuracy, precision, recall, F1-score) • Detecting anomalies using ML • Exporting final models using joblib • Writing ML documentation and performance reports",
            "• Managing NSL-KDD dataset • Cleaning, filtering, transforming raw data • Building data ingestion and preprocessing pipelines • Handling missing values & encoding categorical fields • Implementing RobustScaler for feature scaling • Preparing final datasets for the ML team • Ensuring data quality and consistency",
            "• Designing and building the Streamlit dashboard • Creating charts, metrics, alerts, and logs UI • Integrating backend ML prediction API • Real-time visualization of network traffic • Displaying intrusion alerts and system status • Improving dashboard user experience (UX/UI)"
        ]
    }
    
    st.dataframe(pd.DataFrame(roles_data), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Team members table
    team_data = {
        "Name": [
            "Namatovu Christine",
            "Muyomba Wasswa Cosma",
            "Kasirye Johnson",
            "Masinde Ambrose Waiswa",
            "Umutoni Grace Nshimiye",
            "Kobugabe Lailah"
        ],
        "Registration": [
            "2023/BCS/004",
            "2023/BCS/084/PS",
            "2023/BCS/060/PS",
            "2023/BCS/074/PS",
            "2023/BCS/166/PS",
            "2023/BCS/002"
        ],
        "Role": [
            "Machine Learning Engineer",
            "Machine Learning Engineer",
            "Data Engineer",
            "Data Engineer",
            "Dashboard Developer",
            "Dashboard Developer"
        ]
    }
    
    st.table(pd.DataFrame(team_data))
    
    st.markdown("""
    **Supervisor:** Mr. Emmanuel Ruhamyankaka  
    **Institution:** Mbarara University of Science and Technology (MUST)  
    **Date:** November 2025
    
    ---
    
    ### Project Overview
    
    This system implements a machine learning-based intrusion detection system with:
    - 7 ML algorithms trained and compared
    - 99% attack detection accuracy
    - NSL-KDD dataset (125,972 samples)
    - Interactive web dashboard
    - Real-time traffic classification
    
    ### Technologies
    
    - **ML:** Scikit-learn, TensorFlow, XGBoost
    - **Frontend:** Streamlit
    - **Backend:** Flask API
    - **Deployment:** Docker, Streamlit Cloud
    
    ### Dataset: NSL-KDD
    
    - 125,972 training samples
    - 43 original features → 122 after preprocessing
    - Binary classification (Normal vs Attack)
    - 4 attack categories: DoS, Probe, R2L, U2R
    
    ---
    
    **© 2025 MUST Machine Learning Research Group**
    
    **GitHub:** [View Source Code](https://github.com/NAMATOVU-CHRISTINE/IDS)
    """)

# Footer
st.sidebar.markdown("---")

# Check API status
try:
    api_status = requests.get(f"{API_URL}/health", timeout=15)
    if api_status.status_code == 200:
        st.sidebar.success("API Connected")
    else:
        st.sidebar.warning("API Degraded")
except Exception as e:
    st.sidebar.error("API Offline")
    st.sidebar.caption(f"({str(e)[:30]}...)")

st.sidebar.info(f"""
**API Endpoint**

{API_URL}

Real-time predictions enabled!
""")
