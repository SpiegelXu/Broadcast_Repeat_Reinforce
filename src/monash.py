"""Read Monash pickle (pairs, scaler), pairs = [(history, target), ...]。"""

import pickle

import numpy as np

from .experiment_config import project_path
from .data_generate import generate_cos_poly


def get_datasets(datasets_to_run, data_dir="datasets/monash", shuffle_seed=0):
    benchmarks = {}
    for dsname in datasets_to_run:
        with open(project_path(data_dir) / f"{dsname}.pkl", "rb") as f:
            pairs, _scaler = pickle.load(f)
        ind = np.random.RandomState(shuffle_seed).permutation(len(pairs))
        train = [np.asarray(pairs[i][0]) for i in ind]
        test = [np.asarray(pairs[i][1]) for i in ind]
        benchmarks[dsname] = (train, test)
    return benchmarks


def load_experiment_data(config):
    if config["source"] == "monash":
        return get_datasets(config["datasets_to_run"], config["data_dir"], config["shuffle_seed"])
    if config["source"] == "synthetic":
        _, _, _, y_hist, _, y_pred, _ = generate_cos_poly(**config["synthetic"])
        return {"cos_poly": ([y_hist], [y_pred])}
    raise ValueError(config["source"])
