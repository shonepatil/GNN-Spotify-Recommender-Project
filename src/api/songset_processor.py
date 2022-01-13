## @author: Benjamin Becze
## 01/13/2022
## Translates all saved JSON files from API request to CSV and exports.

# import pandas as pd
# import json

# f = open('C:/Users/Administrator/data/spotify_scrape/songset0.json')
# data = json.load(f)
# f.close
# songset = pd.DataFrame(data['audio_features'])
# for i in range(1, 1701):
#     f = open(f'C:/Users/Administrator/data/spotify_scrape/songset{i}.json')
#     data = json.load(f)
#     f.close

#     data = pd.DataFrame(data['audio_features'])
#     songset = pd.concat([songset, data], ignore_index=True)

# songset_trim = songset.drop(columns=['type', 'id', 'uri', 'track_href', 'analysis_url'])

# songset.to_csv('C:/Users/Administrator/data/spotify_features/songset_features.csv')