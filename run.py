#!/usr/bin/env python

import sys
import json

sys.path.insert(0, 'src/data')
sys.path.insert(0, 'src/dgl_graphsage')

from utils import load_data
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
        feat_data, adj_list, dgl_G = load_data(**data_cfg)

    if 'analysis' in targets:
        with open('config/analysis-params.json') as fh:
            analysis_cfg = json.load(fh)

        # make the analysis target
        compute_aggregates(**analysis_cfg)

    if 'model' in targets:
        with open('config/model-params.json') as fh:
            model_cfg = json.load(fh)
    
        # make the model target
        train(dgl_G, feat_data, adj_list, **model_cfg)

    if 'test' in targets:
        with open('config/test-data-params.json') as fh:
            data_cfg = json.load(fh)
        with open('config/test-model-params.json') as fh:
            model_cfg = json.load(fh)

        # load test data
        feat_data, adj_list, dgl_G = load_data(**data_cfg)

        # make the test target
        train(dgl_G, feat_data, adj_list, **model_cfg)

    return


if __name__ == '__main__':
    # run via:
    # python main.py data model
    targets = sys.argv[1:]
    main(targets)
