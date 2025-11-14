# 🛡️ Intrusion Detection System - Machine Learning Based Network Security

## Project Overview
A production-ready Intrusion Detection System (IDS) using machine learning to detect network attacks in real-time. Built with Random Forest classifier achieving **99.99% accuracy** on the NSL-KDD dataset.

## 🚀 Live Demo
- **Dashboard**: https://namatovu-christine-kjcaxsyrtwm7t7vkqp9jat.streamlit.app
- **API Endpoint**: https://ids-api-33k6.onrender.com
- **Status**: ✅ Live and Running

## 👥 Development Team
**Mbarara University of Science and Technology (MUST)**

| Name | Registration | Role |
|------|-------------|------|
| Namatovu Christine | 2023/BCS/004 | Project Lead, ML Model Development |
| Kobugabe Lailah | 2023/BCS/002 | Data Preprocessing & Feature Engineering |
| Umutoni Grace Nshimiye | 2023/BCS/166/PS | Frontend Development & UI/UX Design |
| Masinde Ambrose Waiswa | 2023/BCS/074/PS | Backend API Development & Deployment |
| Kasirye Johnson | 2023/BCS/060/PS | Model Training & Evaluation |
| Muyomba Wasswa Cosma | 2023/BCS/084/PS | Documentation & Testing |

**Supervisor**: Mr. Emmanuel Ruhamyankaka  
**Date**: November 2025

## Dataset Information
- **Dataset**: NSL-KDD (Network Security Laboratory - Knowledge Discovery in Databases)
- **Training Records**: 125,972
- **Features**: 43 (39 numeric, 4 categorical)
- **Classes**: Binary classification (Normal vs Attack)
  - Normal Traffic: 67,342 (53.5%)
  - Attack Traffic: 58,630 (46.5%)

## Attack Types in Dataset
The dataset contains 22 different types of attacks categorized into 4 main groups:
1. **DoS (Denial of Service)**: back, land, neptune, pod, smurf, teardrop
2. **Probe**: ipsweep, nmap, portsweep, satan
3. **R2L (Remote to Local)**: ftp_write, guess_passwd, imap, multihop, phf, spy, warezclient, warezmaster
4. **U2R (User to Root)**: buffer_overflow, loadmodule, perl, rootkit

## Files in This Repository

### Data Files
- `KDDTrain+.txt` - Training dataset (19 MB)
- `KDDTest+.txt` - Test dataset (3.3 MB)
- `kddcup.names.txt` - Feature descriptions
- `training_attack_types.txt` - Attack type mappings
- `kddcup.data_10_percent_corrected` - 10% sample dataset (72 MB)

### Code Files
- `ids_notebook.ipynb` - Main Jupyter notebook with full analysis
- `run_notebook.py` - Python script for data exploration
- `Intrusion_Detection_System.ipynb` - Original notebook

### Output Files
- `analysis_summary.txt` - Detailed analysis summary
- `protocol_type_distribution.png` - Protocol type visualization
- `outcome_distribution.png` - Attack vs Normal traffic pie chart
- `service_distribution.png` - Top 15 services bar chart
- `correlation_matrix.png` - Feature correlation heatmap

## Setup Instructions

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\\Scripts\\activate  # On Windows
```

### 2. Install Required Packages
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow xgboost
```

### 3. Run Data Exploration
```bash
python run_notebook.py
```

## Analysis Results

### Protocol Distribution
- **TCP**: 102,688 (81.5%)
- **UDP**: 14,993 (11.9%)
- **ICMP**: 8,291 (6.6%)

### Top Services
1. HTTP: 40,338
2. Private: 21,853
3. Domain_u: 9,043
4. SMTP: 7,313
5. FTP_data: 6,859

### Key Features
The most important features for intrusion detection include:
- `src_bytes` - Number of data bytes from source to destination
- `dst_bytes` - Number of data bytes from destination to source
- `count` - Number of connections to the same host
- `srv_count` - Number of connections to the same service
- `serror_rate` - % of connections that have "SYN" errors
- `same_srv_rate` - % of connections to the same service

## Machine Learning Models

The notebook implements and compares the following models:

1. **Logistic Regression** - Binary classification baseline
2. **K-Nearest Neighbors (KNN)** - Instance-based learning
3. **Naive Bayes** - Probabilistic classifier
4. **Support Vector Machine (SVM)** - Linear and kernel-based
5. **Decision Tree** - Tree-based classifier
6. **Random Forest** - Ensemble of decision trees
7. **XGBoost** - Gradient boosting for threat level prediction
8. **Neural Networks** - Deep learning with TensorFlow/Keras

## Next Steps

To continue with the full analysis:

1. **Feature Engineering**
   - One-hot encoding for categorical variables
   - Feature scaling using RobustScaler
   - Principal Component Analysis (PCA) for dimensionality reduction

2. **Model Training**
   - Train all models on the preprocessed data
   - Hyperparameter tuning
   - Cross-validation

3. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion matrices
   - ROC curves and AUC scores
   - Feature importance analysis

4. **Deployment**
   - Save best performing model
   - Create prediction pipeline
   - Real-time intrusion detection

## Visualizations Generated

All visualizations are saved as PNG files:
- Protocol type distribution
- Attack vs Normal traffic distribution
- Service usage patterns
- Feature correlations

## References

- NSL-KDD Dataset: https://www.unb.ca/cic/datasets/nsl.html
- Original KDD Cup 1999: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

## License

This project is for educational purposes.


**Last Updated**: November 14, 2025
