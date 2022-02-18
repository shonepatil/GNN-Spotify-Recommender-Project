# @author: Benjamin Becze
# 01/13/2022
# Translates all saved JSON files from API request to CSV and exports.

import pandas as pd
import json
import tqdm

f = open('/home/s3patil/DSC 180/GraphSage/GNN-Spotify-Recommender-Project/data/a13group1/spotify_scrape/songset0.json')
data = json.load(f)
f.close
songset = pd.DataFrame(data['audio_features'])
print('Starting csv creation')

for i in tqdm.tqdm(range(1, 4619)):
    f = open(f'/home/s3patil/DSC 180/GraphSage/GNN-Spotify-Recommender-Project/data/a13group1/spotify_scrape/songset{i}.json')
    data = json.load(f)
    f.close

    data = pd.DataFrame(data['audio_features'])
    songset = pd.concat([songset, data], ignore_index=True)

songset_trim = songset.drop(columns=['type', 'id', 'uri', 'track_href', 'analysis_url'])

print(songset_trim.shape)
songset.to_csv('460k_songset_features.csv')
print('Csv created for songset')