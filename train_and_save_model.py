"""
Train Random Forest model on NSL-KDD dataset and save for production use
Based on ids_notebook.ipynb preprocessing and training
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
import joblib
import os

print("Loading dataset...")
columns = (['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot'
,'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations'
,'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate'
,'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count'
,'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate'
,'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','outcome','level'])

# Read data from correct path
data_train = pd.read_csv("input/nslkdd/KDDTrain+.txt")
data_train.columns = columns
print(f"Dataset loaded: {data_train.shape[0]} samples")

# Preprocessing function from notebook
def Scaling(df_num, cols):
    std_scaler = RobustScaler()
    std_scaler_temp = std_scaler.fit_transform(df_num)
    std_df = pd.DataFrame(std_scaler_temp, columns=cols)
    return std_df, std_scaler

cat_cols = ['is_host_login','protocol_type','service','flag','land', 'logged_in','is_guest_login', 'level', 'outcome']

def preprocess(dataframe):
    df_num = dataframe.drop(cat_cols, axis=1)
    num_cols = df_num.columns
    scaled_df, scaler = Scaling(df_num, num_cols)
    
    dataframe.drop(labels=num_cols, axis="columns", inplace=True)
    dataframe[num_cols] = scaled_df[num_cols]
    
    dataframe.loc[dataframe['outcome'] == "normal", "outcome"] = 0
    dataframe.loc[dataframe['outcome'] != 0, "outcome"] = 1
    
    dataframe = pd.get_dummies(dataframe, columns=['protocol_type', 'service', 'flag'])
    return dataframe, scaler

print("Preprocessing data...")
# Convert outcome to binary
data_train.loc[data_train['outcome'] == "normal", "outcome"] = 'normal'
data_train.loc[data_train['outcome'] != 'normal', "outcome"] = 'attack'

scaled_train, scaler = preprocess(data_train)

# Prepare features and target
x = scaled_train.drop(['outcome', 'level'], axis=1).values
y = scaled_train['outcome'].values.astype('int')
feature_names = list(scaled_train.drop(['outcome', 'level'], axis=1).columns)

print(f"Features: {x.shape[1]}, Samples: {x.shape[0]}")

# Train Random Forest (same as notebook)
print("Training Random Forest model...")
rf = RandomForestClassifier().fit(x, y)

# Calculate accuracy
train_accuracy = rf.score(x, y)
print(f"Training Accuracy: {train_accuracy*100:.2f}%")

# Save model artifacts
print("Saving model artifacts...")
os.makedirs('models', exist_ok=True)
joblib.dump(rf, 'models/rf_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(feature_names, 'models/feature_names.pkl')

print("✅ Model training complete!")
print(f"   - Model: models/rf_model.pkl")
print(f"   - Scaler: models/scaler.pkl")
print(f"   - Features: models/feature_names.pkl ({len(feature_names)} features)")
print(f"   - Training Accuracy: {train_accuracy*100:.2f}%")
