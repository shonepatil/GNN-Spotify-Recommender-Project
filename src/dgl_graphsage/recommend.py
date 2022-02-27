import sys
import json
import time
from collections import defaultdict
import pandas as pd
import os
import numpy as np

# sys.path.insert(0, 'src/data')
# sys.path.insert(0, 'src/dgl_graphsage')

sys.path.insert(1, '..')
from api.spotifyAPI import SpotifyAPI
from train_updated import train

import dgl
import torch
import torch.nn.functional as F
from dgl import save_graphs
from dgl import load_graphs

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, './src/features')
import re
from torch.nn.functional import normalize

# uses original files to make the graph from scratch

# def scratch(feat_dir, normal=True):
#     print('Loading feature data...')
#     data = np.genfromtxt(feat_dir, delimiter=',', skip_header=True, dtype=str)
#     features = np.array(np.delete(data[:,2:], -3, 1), dtype=float)
#     if normal:
#         features = normalize(torch.Tensor(features), dim=0)
#     uris = data[:, 1]
#     uris = [re.sub('spotify:track:', '', uri) for uri in uris]
#     uri_map = {n: i for i,n in enumerate(uris)}
    
#     G = graph_from_scratch('./data/a13group1/data/', 'Spotify Playlist', 0, 0, 0)
#     print('graph created')
#     src, dest = [], [] 
#     adj_list = defaultdict(set)    
#     for e in G.edges:
#         u,v = uri_map[e[0]], uri_map[e[1]]
#         adj_list[u].add(v)
#         adj_list[v].add(u)
#         src.append(u)
#         dest.append(v)
        
#     src = torch.tensor(src)
#     dest = torch.tensor(dest)
        
    
#     print('adj list created')
    
#     dgl_G = dgl.graph((src, dest), num_nodes=len(G.nodes))
#     return features, adj_list, dgl_G, uri_map



# '''
# Creates graph from scratch using from_networkx
# results in double edges
# '''
# def scratch(feat_dir, normal=True):
#     print('Loading feature data...')
#     data = np.genfromtxt(feat_dir, delimiter=',', skip_header=True, dtype=str)
#     features = np.array(np.delete(data[:,2:], -3, 1), dtype=float)
#     if normal:
#         features = normalize(torch.Tensor(features), dim=0)
#     uris = data[:, 1]
#     uris = [re.sub('spotify:track:', '', uri) for uri in uris]
#     uri_map = {n: i for i,n in enumerate(uris)}
    
#     G = graph_from_scratch('./data/a13group1/data/', 'Spotify Playlist', 0, 0, 0)
#     print('graph created')
#     src, dest = [], [] 
#     adj_list = defaultdict(set)    
#     for e in G.edges:
#         u,v = uri_map[e[0]], uri_map[e[1]]
#         adj_list[u].add(v)
#         adj_list[v].add(u)
#         src.append(u)
#         dest.append(v)
#     print('adj list created')
    
#     #dgl_G = dgl.graph((src, dest), num_nodes=len(G.nodes))
#     dgl_G = dgl.from_networkx(G)
#     return features, adj_list, dgl_G, uri_map

'''
Loads the double edged graph
feat_dir = feature directory
'''
def load_double_edge(feat_dir, double_edge_dir, normal=True):
    print('Loading feature data...')
    data = np.genfromtxt(feat_dir, delimiter=',', skip_header=True, dtype=str)
    features = np.array(np.delete(data[:,2:], -3, 1), dtype=float)
    if normal:
        features = normalize(torch.Tensor(features), dim=0)
    uris = data[:, 1]
    uris = [re.sub('spotify:track:', '', uri) for uri in uris]
    uri_map = {n: i for i,n in enumerate(uris)}
    listed = list(uri_map)
    
    G  = load_graphs(double_edge_dir)[0][0]
    print('Loaded DGL Graph')
    sources = G.edges()[0] 
    destinations = G.edges()[1]
    
    src, dest = [], [] 
    adj_list = defaultdict(set)
    for e in range(len((G.edges()[0]))):
        u,v = sources[e].item(), destinations[e].item()
        adj_list[u].add(v)
        adj_list[v].add(u)
        src.append(u)
        dest.append(v)
        
    
    print('adj list created')
    return features, adj_list, G, uri_map

'''
Get a list of eligible slice files (first 10000 playlists)
thelist: list of directories
'''
def get_eligible(thelist):
    eligible = []
    for x in thelist:
        if x == 'playlists':continue
        nums = pd.Series(x.strip('mpd.slice.json').split('-')).astype(int)
        if nums[0] <= 9999:
            eligible.append(x)
    return eligible

'''
Gets a random playlist
'''
def get_random_playlist():
    data_path = (os.path.join(os.path.expanduser('~'), '/teams/DSC180A_FA21_A00/a13group1/data/'))
    file_samp = np.random.choice(get_eligible(pd.Series(os.listdir(os.path.join(os.path.expanduser('~'), '/teams/DSC180A_FA21_A00/a13group1/data/')))), replace=True)
    fname = os.path.join(data_path, file_samp)
    with open(fname) as f:
        data = json.load(f)
        item = np.random.choice(data['playlists'])
    print(fname)    
    return item

'''
Gets the track names of the original tracks in the playlist
'''
def get_playlist_info(item):
    print('Playlist ID:', item['pid'])
    print('Playlist Length:', len(item['tracks']))
    
    # Get track names---artist
    original_tracks = []
    for i in range(len(item['tracks'])):
        name = item['tracks'][i]['track_name']+'---'+item['tracks'][i]['artist_name']
        original_tracks.append(name)
        
    # Get track uris
    seeds = []
    for i in item['tracks']:
        uri = i['track_uri'].split(':')[-1]
        seeds.append(uri)
        
    return item, original_tracks, seeds

'''
Creates dictionary of highest scored recommendation (of songs not in playlist) for each song in playlist
seeds: list of track uris from user's playlist
dgl_G: DGL Graph
z: embeddings generated from model
pred: predictor from model
feat_data: matrix of feature data
'''
def recommend(seeds, dgl_G, z, pred, neigh, feat_data, uri_map):

    listed = list(uri_map) #parse through uri map for uri --> integer

    score_dict = defaultdict(dict)
    for s in seeds:
        s = uri_map[s]
        _, candidates = dgl_G.out_edges(s, form='uv')
        s_embed = z[s].unsqueeze(dim=0)
        edge_embeds = [torch.cat([s_embed, z[c.item()].unsqueeze(dim=0)],1) for c in candidates]
        #print('Node Value:', s, 'Possible Recs:', len(edge_embeds))
        edge_embeds = torch.cat(edge_embeds, 0)
        scores = pred.W2(F.relu(pred.W1(edge_embeds))).squeeze(1)
        val = list(zip(candidates.detach().numpy(), scores.detach().numpy()))
        val.sort(key=lambda x:x[1], reverse=True)
        
        # Make sure the song is not already in the playlist
        # score_dict[s] = val[0]
        inc = 0
        while True and inc < len(val):
            if listed[val[inc][0]] not in seeds:
                score_dict[s] = val[inc][0]
                break
            if inc == (len(val) - 1):
                # If no co-occurence, use 5-NN based on features -- COLD START
                # print('Cold Start, Using Feature Data Instead')
                closest = neigh.kneighbors(feat_data[[s]], 25, return_distance=False)[0]
                for i in closest:
                    if listed[i] not in seeds:
                        score_dict[s] = i
                        break
                break
                    
            else:
                inc += 1
                
    # Get uris            
    uri_recs = []
    for i in score_dict.keys():
        cur_uri = listed[score_dict[i]]
        uri_recs.append(cur_uri)
        
    return uri_recs

def get_data_spotify(query, api, num):
    chunk = api.get_resource(query, 'tracks', 'v1')
    return chunk

def get_rec_names(uri_recs, api, sleep_time):
    unique_recs, unique_recs_counts = np.unique(uri_recs, return_counts=True)
    rec_count = dict(zip(unique_recs, unique_recs_counts))

    splitted = []
    sub_splitted = []
    counter = 0
    all_counter = 0
    for i in unique_recs:
        if counter != 50:
            sub_splitted.append(i)
            counter += 1
        elif all_counter == len(unique_recs):
            sub_splitted.append(i)
            splitted.append(sub_splitted)
            break
        else:
            splitted.append(sub_splitted)
            sub_splitted = []
            counter = 0
            sub_splitted.append(i)

        all_counter += 1

    if len(splitted) == 0:
        splitted.append(sub_splitted)
    
    rec_track_names = []
    for part in splitted:
        part = str(part).replace("'", "").strip('[]').replace(' ', '')
        one = get_data_spotify(part, api, 1)
        
        for each in one['tracks']:
            
            trackname = each['name']
            firstartist = each['artists'][0]['name']

            the_rec = trackname+'---'+firstartist
            rec_track_names.append((the_rec, rec_count[each['id']]))


        time.sleep(2)
        
    return rec_track_names