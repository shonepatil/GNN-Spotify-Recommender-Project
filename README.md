# Graph Neural Networks for Song Recommendation on Spotify Playlists

This project tackles the task of creating meaningful and accurate song recommendations to Spotify Playlists by using Graph Neural Networks. The goal is to better capture the characteristics of songs by analyzing co-occurence of song pairs across thousands of playlists in the form of a graph.

To obtain the spotify playlist data, visit https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge and put the files in `data/playlists`. You will have to create these subfolders. Confirm data paths for future files in `config/data-params`. To add song features and create graph from scratch if you only have the playlist data, set `create_graph_from_scratch` to be `True` in data-params. For obtaining song features using the Spotify API, put secret key and secret id in `spotifyAPI_script.py` within `/src/api`. Also within data-params, we suggest setting `playlist-num` to be 1000 such that the Spotify API requesting only takes 15-20 minutes. If you set it to 10000 for a larger dataset, the code may take 4 hours.

To run the GraphSAGE based model on the Spotify Playlist data, use this command from the root folder: `python run.py data model`.

To customize the model parameters, edit `config/model-params`.
