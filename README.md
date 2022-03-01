# Graph Neural Networks for Song Recommendation on Spotify Playlists

This branch has the code to create and run Node2Vec with K-Nearest Neighbor. To obtain the spotify playlist data, visit https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge and put the files in `data/playlists`. You will have to create these subfolders. Confirm data paths for future files in `config/data-params`. Also set `node2vec_from_scratch` and `graph_from_scratch` to `True` if running for the first time.

To run the Node2Vec and KNN based model on the Spotify Playlist data, use this command from the root folder: `python run.py data model`.
