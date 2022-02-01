import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import torch

def preprocess():
    features = pd.read_csv('~/teams/DSC180A_FA21_A00/a13group1/songset_features_170k')
    features = features.drop(columns=['Unnamed: 0', 'type', 'id', 'uri', 'track_href', 'analysis_url'])

    std_scaler = StandardScaler()
    features['loudness'] = std_scaler.fit_transform(features[['loudness']])
    features['tempo'] = std_scaler.fit_transform(features[['tempo']])

    one_hot = OneHotEncoder()

    key_names = [f'key{i}' for i in range(12)]
    one_hot_key = pd.DataFrame(one_hot.fit_transform(features[['key']]).toarray())
    one_hot_key.columns = key_names

    ts_names = [f'ts{i}' for i in range(5)]
    one_hot_ts = pd.DataFrame(one_hot.fit_transform(features[['time_signature']]).toarray())
    one_hot_ts.columns = ts_names

    features = pd.concat([features, one_hot_key, one_hot_ts], axis=1)
    features = features.drop(columns=['key', 'time_signature'])

    features_arr = torch.tensor(np.array(features))
  
    
    return features_arr