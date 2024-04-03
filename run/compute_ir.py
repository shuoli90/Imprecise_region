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
import json

from misc import conformal_prediction
from misc import monte_carlo
from misc import p_value_combination
from misc import plausibility_regions
from misc import classification_metrics
from misc import formats
from misc import selectors_utils
from misc import irn
import argparse
from collections import defaultdict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='toy')
    args = parser.parse_args()

    # Load data.
    if args.dataset == 'toy':
        with open('../data/toy/toy_data.pkl', 'rb') as f:
            data = pickle.load(f)
        with open('../data/toy/toy_predictions0.pkl', 'rb') as f:
            predictions = pickle.load(f)
    elif args.dataset == 'cifar10h':
        with open('../data/cifar10h/cnn_softmax.npy', 'rb') as f:
            predictions = jnp.load(f)
        with open('../data/cifar10h/uncertainty.npy', 'rb') as f:
            uncertainty = jnp.load(f)
        with open('../data/cifar10h/label.npy', 'rb') as f:
            labels = jnp.load(f)
        indices = labels <= 2
        predictions = predictions[indices, :3] / jnp.sum(predictions[indices, :3], axis=1, keepdims=True)
        uncertainty = uncertainty[indices, :3] / jnp.sum(uncertainty[indices, :3], axis=1, keepdims=True)
        labels = labels[indices]
        data = {'test_smooth_labels': uncertainty, 'test_labels': labels}
    elif args.dataset == 'derm':
        with open('../data/derm/dermatology_risks.txt', 'r') as f:
            risk_levels = f.readlines()
        risk_levels = np.array([int(r) for r in risk_levels])
        with open('../data/derm/dermatology_predictions0.txt', 'r') as f:
            predictions = f.readlines()
        predictions = np.array([[float(tmp) for tmp in p.split(' ')] for p in predictions])
        predictions_tmp = np.zeros((predictions.shape[0], 3))
        for r in [0, 1, 2]:
            predictions_tmp[:, r] = np.sum(predictions * (risk_levels == r), axis=1)
        predictions = predictions_tmp
        # convert to numpy array
        with open('../data/derm/dermatology_selectors.json', 'r') as f:
            selectors = json.load(f)
        padded_selectors = selectors_utils.pad_selectors(selectors, 10)
        rankings, groups = formats.convert_selectors_to_rankings(padded_selectors, 419)
        plausibilities = irn.aggregate_irn(rankings, groups)
        plausibilities_tmp = np.zeros((plausibilities.shape[0], 3))
        for r in [0, 1, 2]:
            plausibilities_tmp[:, r] = np.sum(plausibilities * (risk_levels == r), axis=1)
        plausibilities = plausibilities_tmp
        labels = np.argmax(plausibilities, axis=1)
        data = {'test_smooth_labels': plausibilities, 'test_labels': labels}

        # np.sum(plausibilities[i] * (risk_levels == r)) 

    n_classes = data['test_smooth_labels'].shape[1]
    trials = 20
    split = 0.5
    keys = jax.random.split(jax.random.PRNGKey(0), 2 * trials)
    results_list = []
    for alpha in [0.05, 0.1, 0.15]:
        for t in range(trials):
            alpha_imprecise = alpha / 2
            delta_imprecise = alpha / 2
            results = defaultdict(list)
            results['alpha'] = alpha
            results['trials'] = t
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
                alpha=alpha_imprecise,
            )
            print('Threshold:', threshold)

            test_plausibility_regions, plausibility_regions_includes = plausibility_regions.predict_plausibility_regions(
                conformity_scores=test_predictions,
                threshold=threshold,
            )

            coverage = plausibility_regions.check_plausibility_regions(
                conformity_scores=test_predictions,
                plausibilities=test_human_ground_truth,
                threshold=threshold,
            )
            print('Coverage:', np.mean(coverage))

            regions = []
            for test_idx in range(test_predictions.shape[0]):
                regions_include = plausibility_regions_includes[test_idx]
                region = test_plausibility_regions[regions_include]
                regions.append(region)
            
            As = []
            indices = []
            for test_idx in range(test_predictions.shape[0]):
                sets = [0] * n_classes
                try:
                    A = plausibility_regions.Imprecise_region(
                        plausible_distribution=regions[test_idx],
                        delta=delta_imprecise
                    )
                    for idx in A:
                        sets[idx] = 1
                except:
                    pass
                As.append(sets)
                indices.append(test_idx)
            confidence_sets = jnp.array(As).astype(np.int32)

            # results['inefficiencies_ir'].append(classification_metrics.size(
            #     confidence_sets))
            sizes = classification_metrics.size(confidence_sets)
            results['inefficiencies_ir_average'] = np.mean(sizes).item()
            print(results['inefficiencies_ir_average'])

            total = test_ground_truth.shape[0]
            test_ground_truth_valid = test_ground_truth[jnp.array(indices)]
            test_human_ground_truth_valid = test_human_ground_truth[jnp.array(indices)]

            test_one_hot_ground_truth = jax.nn.one_hot(
                test_ground_truth_valid, confidence_sets.shape[1])
            true_coverages_ir = classification_metrics.aggregated_coverage(
                confidence_sets, test_one_hot_ground_truth)
            # results['true_coverages_ir'].append(classification_metrics.aggregated_coverage(
            #     confidence_sets, test_one_hot_ground_truth))
            results['true_coverages_ir_average'] = (np.sum(true_coverages_ir) / total).item()
            print(results['true_coverages_ir_average'])
            test_top1_ground_truth = jax.nn.one_hot(
                jnp.argmax(test_human_ground_truth_valid, axis=1),
                test_human_ground_truth.shape[1])
            top1_coverages_ir = classification_metrics.aggregated_coverage(
                confidence_sets, test_top1_ground_truth)
            # results['top1_coverages_ir'].append(
            #     classification_metrics.aggregated_coverage(
            #         confidence_sets, test_top1_ground_truth))
            results['top1_coverages_ir_average'] = (np.sum(top1_coverages_ir) / total).item()
            print(results['top1_coverages_ir_average'])
            aggregated_coverages_ir = classification_metrics.aggregated_coverage(
                confidence_sets, test_human_ground_truth_valid)
            # results['aggregated_coverages_ir'].append(
            #     classification_metrics.aggregated_coverage(
            #         confidence_sets, test_human_ground_truth_valid))
            results['aggregated_coverage_ir_average'] = (np.sum(aggregated_coverages_ir) / total).item()
            print(results['aggregated_coverage_ir_average'])
            
            threshold = monte_carlo.calibrate_mc_threshold(
                keys[2 * t + 1], 
                val_predictions, 
                val_human_ground_truth, 
                num_samples=10, 
                alpha=alpha)
            confidence_sets = conformal_prediction.predict_threshold(
                test_predictions, threshold)

            # results['inefficiencies_mc'].append(classification_metrics.size(
            #     confidence_sets))
            inefficiencies_mc = classification_metrics.size(confidence_sets)
            results['inefficiencies_mc_average'] = (np.mean(inefficiencies_mc)).item()
            print(results['inefficiencies_mc_average'])
            test_one_hot_ground_truth = jax.nn.one_hot(
                test_ground_truth, confidence_sets.shape[1])
            # results['true_coverages_mc'].append(classification_metrics.aggregated_coverage(
            #     confidence_sets, test_one_hot_ground_truth))
            true_coverages_mc = classification_metrics.aggregated_coverage(
                confidence_sets, test_one_hot_ground_truth)
            results['true_coverages_mc_average'] = (np.mean(true_coverages_mc)).item()
            print(results['true_coverages_mc_average'])
            test_top1_ground_truth = jax.nn.one_hot(
                jnp.argmax(test_human_ground_truth, axis=1),
                test_human_ground_truth.shape[1])
            # results['top1_coverages_mc'].append(
            #     classification_metrics.aggregated_coverage(
            #         confidence_sets, test_top1_ground_truth))
            top1_coverages_mc = classification_metrics.aggregated_coverage(
                confidence_sets, test_top1_ground_truth)
            results['top1_coverages_mc_average'] = (np.mean(top1_coverages_mc)).item()
            print(results['top1_coverages_mc_average'])
            # results['aggregated_coverages_mc'].append(
            #     classification_metrics.aggregated_coverage(
            #         confidence_sets, test_human_ground_truth))
            aggregated_coverages_mc = classification_metrics.aggregated_coverage(
                confidence_sets, test_human_ground_truth)
            results['aggregated_coverage_mc_average'] = (np.mean(aggregated_coverages_mc)).item()
            print(results['aggregated_coverage_mc_average'])

            results_list.append(results)
    
    with open(f'../collected/{args.dataset}_results.json', 'w') as f:
        json.dump(results_list, f)