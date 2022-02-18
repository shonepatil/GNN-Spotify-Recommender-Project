#!/usr/bin/env python

import sys
import json

sys.path.insert(0, 'src/data')
sys.path.insert(0, 'src/dgl_graphsage')

from utils import load_features, load_graph
from train_updated import train


def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis', 'model'. 
    
    `main` runs the targets in order of data=>analysis=>model.
    '''

    if 'data' in targets:
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)

        # make the data target
        feat_data, uri_map = load_features(data_cfg['feat_dir'])
        dgl_G, weights = load_graph(data_cfg['gpickle_dir'], data_cfg['create_graph_from_scratch'], uri_map)

    if 'analysis' in targets:
        with open('config/analysis-params.json') as fh:
            analysis_cfg = json.load(fh)

        # make the analysis target
        compute_aggregates(**analysis_cfg)

    if 'model' in targets:
        with open('config/model-params.json') as fh:
            model_cfg = json.load(fh)
    
        # make the model target
        train(dgl_G, weights, feat_data, **model_cfg)

    if 'test' in targets:
        with open('config/test-data-params.json') as fh:
            data_cfg = json.load(fh)
        with open('config/test-model-params.json') as fh:
            model_cfg = json.load(fh)

        # load test data
        feat_data, uri_map = load_features(data_cfg['feat_dir'])
        dgl_G, weights = load_graph(data_cfg['gpickle_dir'], data_cfg['create_graph_from_scratch'], uri_map)

        # make the test target
        train(dgl_G, weights, feat_data, **model_cfg)

    return


if __name__ == '__main__':
    # run via:
    # python main.py data model
    targets = sys.argv[1:]
    main(targets)
