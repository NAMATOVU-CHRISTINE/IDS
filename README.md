# Intrusion Detection System - Machine Learning Based Network Security

## Project Overview
A production-ready Intrusion Detection System (IDS) using machine learning to detect network attacks in real-time. Built with Random Forest classifier achieving **99.99% accuracy** on the NSL-KDD dataset.

## 🚀 Live Demo
- **Dashboard**: https://namatovu-christine-kjcaxsyrtwm7t7vkqp9jat.streamlit.app
- **API Endpoint**: https://ids-api-33k6.onrender.com
- **Status**: Live and Running

## Development Team
**Mbarara University of Science and Technology (MUST)**

| Role | Team Members | Responsibilities |
|------|-------------|------------------|
| **Machine Learning Engineers** | Christine (2023/BCS/004), Cosma (2023/BCS/084/PS) | • Data preprocessing and feature engineering • Training ML models (Random Forest, SVM, Logistic Regression, Neural Networks) • Hyperparameter tuning • Model evaluation (accuracy, precision, recall, F1-score) • Detecting anomalies using ML • Exporting final models using joblib • Writing ML documentation and performance reports |
| **Data Engineers** | Johnson (2023/BCS/060/PS), Ambrose (2023/BCS/074/PS) | • Managing NSL-KDD dataset • Cleaning, filtering, transforming raw data • Building data ingestion and preprocessing pipelines • Handling missing values & encoding categorical fields • Implementing RobustScaler for feature scaling • Preparing final datasets for the ML team • Ensuring data quality and consistency |
| **Dashboard Developers** | Grace (2023/BCS/166/PS), Lailah (2023/BCS/002) | • Designing and building the Streamlit dashboard • Creating charts, metrics, alerts, and logs UI • Integrating backend ML prediction API • Real-time visualization of network traffic • Displaying intrusion alerts and system status • Improving dashboard user experience (UX/UI) |


## Key Features

- **Real-time Attack Detection**: Live network traffic classification
- **99.99% Accuracy**: Random Forest model trained on 125,972 samples
- **Interactive Dashboard**: Web-based UI with visualizations
- **REST API**: Production-ready Flask API for predictions
- **7 ML Models Compared**: Comprehensive algorithm evaluation

## Architecture

```
┌─────────────────┐      HTTP/JSON      ┌──────────────┐
│   Streamlit     │ ◄─────────────────► │  Flask API   │
│   Dashboard     │                     │  (Render)    │
│ (Streamlit Cloud)│                     └──────────────┘
└─────────────────┘                            │
                                               │
                                        ┌──────▼──────┐
                                        │  Random     │
                                        │  Forest     │
                                        │  Model      │
                                        │ (99.99%)    │
                                        └─────────────┘
```

## Dataset Information

- **Dataset**: NSL-KDD (Network Security Laboratory - Knowledge Discovery in Databases)
- **Training Records**: 125,972
- **Features**: 122 (after preprocessing)
- **Original Features**: 43 (39 numeric, 4 categorical)
- **Classes**: Binary classification (Normal vs Attack)
  - Normal Traffic: 67,342 (53.5%)
  - Attack Traffic: 58,630 (46.5%)

### Attack Types
The dataset contains 22 different types of attacks in 4 categories:
1. **DoS (Denial of Service)**: back, land, neptune, pod, smurf, teardrop
2. **Probe**: ipsweep, nmap, portsweep, satan
3. **R2L (Remote to Local)**: ftp_write, guess_passwd, imap, multihop, phf, spy, warezclient, warezmaster
4. **U2R (User to Root)**: buffer_overflow, loadmodule, perl, rootkit

## Project Structure

```
IDS/
├── input/nslkdd/           # Dataset files
│   └── KDDTrain+.txt       # Training data (125,972 samples)
├── models/                 # Trained ML models
│   ├── rf_model.pkl        # Random Forest model
│   ├── scaler.pkl          # Feature scaler
│   └── feature_names.pkl   # Feature list
├── ids_notebook.ipynb      # Research & model development
├── train_and_save_model.py # Model training script
├── api_render.py           # Production Flask API
├── streamlit_app.py        # Dashboard application
├── build.sh                # Render deployment script
├── requirements.txt        # Dashboard dependencies
└── requirements_api.txt    # API dependencies
```

## Deployment

### Live Production System

**API (Render)**
- Endpoint: https://ids-api-33k6.onrender.com
- Automatic deployment from GitHub
- Build command: `./build.sh`
- Trains model during deployment

**Dashboard (Streamlit Cloud)**
- URL: https://namatovu-christine-kjcaxsyrtwm7t7vkqp9jat.streamlit.app
- Connected to live API
- Real-time predictions

### Local Development

1. **Clone Repository**
```bash
git clone https://github.com/NAMATOVU-CHRISTINE/IDS.git
cd IDS
```

2. **Create Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Train Model**
```bash
python train_and_save_model.py
```

5. **Run API Locally**
```bash
python api_render.py
```

6. **Run Dashboard Locally**
```bash
streamlit run streamlit_app.py
```

## Machine Learning Models

### Models Evaluated

| Model | Training Accuracy | Test Accuracy | Precision | Recall |
|-------|------------------|---------------|-----------|--------|
| Logistic Regression | 82% | 82% | 78% | 80% |
| Naive Bayes | 90% | 90% | 88% | 89% |
| Decision Tree | 93% | 93% | 91% | 92% |
| KNN | 97% | 97% | 96% | 97% |
| SVM | 98% | 98% | 97% | 98% |
| **Random Forest** | **99.99%** | **99.88%** | **99.93%** | **99.81%** |
| Neural Network | 99.2% | 99.2% | 98% | 99% |

**Winner**: Random Forest (deployed in production)

### Model Features

- **Preprocessing**: RobustScaler for numeric features
- **Encoding**: One-hot encoding for categorical variables (protocol, service, flag)
- **Feature Engineering**: 122 features after preprocessing
- **Algorithm**: Random Forest Classifier (scikit-learn)
- **Training Time**: ~2 minutes on full dataset

## Key Insights

### Protocol Distribution
- **TCP**: 81.5%
- **UDP**: 11.9%
- **ICMP**: 6.6%

### Most Important Features
1. `src_bytes` - Source to destination bytes
2. `dst_bytes` - Destination to source bytes
3. `count` - Connections to same host
4. `srv_count` - Connections to same service
5. `serror_rate` - SYN error rate
6. `same_srv_rate` - Same service rate

## Research Notebook

`ids_notebook.ipynb` contains:
- Complete data exploration and visualization
- 7 ML algorithm implementations
- Model comparison and evaluation
- Feature importance analysis
- Confusion matrices and performance metrics
- PCA dimensionality reduction experiments
- Neural network architecture

## Technologies Used

**Machine Learning**
- scikit-learn (Random Forest, preprocessing)
- TensorFlow/Keras (Neural Networks)
- XGBoost (Gradient Boosting)
- pandas, numpy (Data processing)

**Backend**
- Flask (REST API)
- Flask-CORS (Cross-origin requests)
- joblib (Model serialization)

**Frontend**
- Streamlit (Dashboard)
- matplotlib, seaborn (Visualizations)
- requests (API communication)

**Deployment**
- Render (API hosting)
- Streamlit Cloud (Dashboard hosting)
- GitHub (Version control & CI/CD)

## API Documentation

### Endpoints

**Health Check**
```bash
GET https://ids-api-33k6.onrender.com/health
```

**Predict Attack**
```bash
POST https://ids-api-33k6.onrender.com/predict
Content-Type: application/json

{
  "duration": 0,
  "protocol_type": "tcp",
  "service": "http",
  "flag": "SF",
  "src_bytes": 181,
  "dst_bytes": 5450,
  "count": 1,
  "srv_count": 1,
  "serror_rate": 0.0
}
```

**Response**
```json
{
  "prediction": "normal",
  "confidence": 0.95,
  "probabilities": {
    "normal": 0.95,
    "attack": 0.05
  },
  "timestamp": "2025-11-14T18:30:00",
  "model": "Random Forest (99.99% accuracy)"
}
```


## 📚 References

- NSL-KDD Dataset: https://www.unb.ca/cic/datasets/nsl.html
- Original KDD Cup 1999: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
- Random Forest: https://scikit-learn.org/stable/modules/ensemble.html#forest

## License

This project is for educational purposes as part of academic research at MUST.

## Acknowledgments

- Mr. Emmanuel Ruhamyankaka (Project Supervisor)
- Mbarara University of Science and Technology
- NSL-KDD Dataset Contributors


**Last Updated**: November 14, 2025
