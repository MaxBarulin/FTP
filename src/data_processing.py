import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):
    data = pd.read_csv(filepath)
    print("название колонок в загруженном DataFrame:", data.columns.tolist())
    return data

def preprocess_data(data):
    scaler = MinMaxScaler()
    features = data[['D', 'd', 'H']]
    features_scaled = scaler.fit_transform(features)
    targets = data['norm_time'].values
    return features_scaled, targets, scaler