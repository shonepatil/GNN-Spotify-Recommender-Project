## @author: Benjamin Becze
## 01/13/2022
## Script to perform spotify API requests for 170,000 songs.
## Takes roughly 5 hours and saves to disk.
## DO NOT RUN

import pandas as pd
import numpy as np 
import json
import time
import spotifyAPI as spot

client_id = ''
client_secret = ''

spotify = spot.SpotifyAPI(client_id, client_secret)

def get_data(query, api, num):
    chunk = api.get_resource(query, 'audio-features', 'v1')
    json_object = json.dumps(chunk, indent = 4)

    with open(f'C:/Users/Administrator/data/spotify_scrape/songset{num}.json', 'w') as outfile:
        outfile.write(json_object)

ids_list = np.array(pd.read_csv('C:/Users/Administrator/data/170k_songs.csv')['track_uri'])
splitted_first_half = (np.array_split(ids_list[:170000], 1700))
splitted_second_half = list(ids_list[170000:])
bruh = list(splitted_first_half)
bruh.append(splitted_second_half)

def perform():
    tracker = 0
    for i in bruh:
        qry = ','.join(i)
        complete = False
        while complete != True:
            try:
                get_data(qry, spotify, tracker)
                complete = True
            except :
                print('http error')
                time.sleep(60)
                
        tracker += 1
        if tracker % 1000 == 0:
            print(tracker)
        time.sleep(10)

#perform()
