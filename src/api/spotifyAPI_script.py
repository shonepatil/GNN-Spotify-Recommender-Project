## @author: Benjamin Becze
## 01/13/2022
## Script to perform spotify API requests for 170,000 songs.
## Takes roughly 5 hours and saves to disk.
## DO NOT RUN

import pandas as pd
import numpy as np 
import json
import time
import api.spotifyAPI as spot
import math
import os

def round_down(x):
    return int(math.floor(x / 100.0)) * 100

def pull_audio_features(num_nodes):
    client_id = None
    client_secret = None

    spotify = spot.SpotifyAPI(client_id, client_secret)

    os.mkdir('./data/spotify_scrape')

    def get_data(query, api, num):
        chunk = api.get_resource(query, 'audio-features', 'v1')
        json_object = json.dumps(chunk, indent = 4)

        with open(f'./data/spotify_scrape/songset{num}.json', 'w') as outfile:
            outfile.write(json_object)

    # 460k songs has 461880 songs 106486690 edges
    ids_list = np.array(pd.read_csv('./data/songs.csv')['track_uri'])
    # splitted_first_half = (np.array_split(ids_list[:461800], 4618))
    # splitted_second_half = list(ids_list[461800:])
    splitted_first_half = (np.array_split(ids_list[:round_down(num_nodes)], num_nodes // 100))
    splitted_second_half = list(ids_list[round_down(num_nodes):])
    bruh = list(splitted_first_half)
    bruh.append(splitted_second_half)

    def perform():
        tracker = 0
        for i in range(0, len(bruh)):
            qry = ','.join(bruh[i])
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

    perform()
    print('Done pulling all song data from Spotify API!')
