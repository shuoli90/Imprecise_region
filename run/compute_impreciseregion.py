import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # <--- Add this line
import jax
import jax.numpy as jnp
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle

from misc import conformal_prediction
from misc import monte_carlo
from misc import p_value_combination
from misc import plausibility_regions
from misc import classification_metrics
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--dataset', type=str, default='toy')
    args = parser.parse_args()

    # Load data.
    with open('../data/toy/toy_data.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('../data/toy/toy_predictions0.pkl', 'rb') as f:
        predictions = pickle.load(f)
    
    trials = 1
    split = 0.5
    keys = jax.random.split(jax.random.PRNGKey(0), 2 * trials)
    for t in range(trials):
        permutation = jax.random.permutation(keys[2 * t], predictions.shape[0])
        val_examples = int(predictions.shape[0]*split)
        val_predictions = predictions[permutation[:val_examples]]
        test_predictions = predictions[permutation[val_examples:]]
        val_human_ground_truth = data['test_smooth_labels'][permutation[:val_examples]]
        test_human_ground_truth = data['test_smooth_labels'][permutation[val_examples:]]
        val_ground_truth = data['test_labels'][permutation[:val_examples]]
        test_ground_truth = data['test_labels'][permutation[val_examples:]]

        threshold = plausibility_regions.calibrate_plausibility_regions(
            conformity_scores=val_predictions,
            plausibilities=val_human_ground_truth,
            alpha=args.alpha,
        )

        test_plausibility_regions, plausibility_regions_includes = plausibility_regions.predict_plausibility_regions(
            conformity_scores=test_predictions,
            threshold=threshold,
        )

        mins_list = []
        maxs_list = []
        for test_idx in range(test_predictions.shape[0]):
            regions_include = plausibility_regions_includes[test_idx]
            regions = test_plausibility_regions[regions_include]
            mins = regions.min(axis=0)
            maxs = regions.max(axis=0)
            mins_list.append(mins)
            maxs_list.append(maxs)
        
        As = []
        for test_idx in range(test_predictions.shape[0]):
            plausibility_distribution = np.array([mins_list[test_idx], maxs_list[test_idx]])
            A = plausibility_regions.Imprecise_region(
                plausible_distribution=plausibility_distribution,
                delta=args.delta
            )
            As.append(A)