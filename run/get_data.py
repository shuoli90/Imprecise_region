import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
import sklearn.neural_network

from misc import formats
from misc import irn as aggregation
from misc import gaussian_toy_dataset as gtd

def get_data(config):
  """Generate data using the config."""
  # Defines a dataset of multiple overlapping Gaussians.
  generator = gtd.GaussianToyDataset(
      config['rng'], jnp.array(config['class_weights']),
      config['class_sigmas'], config['dimensionality'], config['sigma'])
  num_examples = config['train_examples'] + config['test_examples']
  # Sample points x from the overlapping Gaussian distributions.
  examples, ground_truths = generator.sample_points(num_examples)
  # Compute the true posterior distributions p(y|x).
  human_ground_truths = generator.evaluate_points(examples)
  # Sample annotator rankings for all points.
  rankings, groups = generator.sample_rankings(
      human_ground_truths,
      config['reader_sharpness'],
      config['expected_length'],
      config['grouping_threshold'])
  # Convert rankings and compute IRN aggregation.
  selectors = formats.convert_rankings_to_selectors(rankings, groups)
  irn = aggregation.aggregate_irn(rankings, groups)
  return {
      'config': config,
      'train_examples': examples[:config['train_examples']],
      'train_labels': ground_truths[:config['train_examples']],
      'train_smooth_labels': human_ground_truths[:config['train_examples']],
      'train_rankings': rankings[:config['train_examples']],
      'train_groups': groups[:config['train_examples']],
      'train_selectors': selectors[:config['train_examples']],
      'train_irn': irn[:config['train_examples']],
      'test_examples': examples[config['train_examples']:],
      'test_labels': ground_truths[config['train_examples']:],
      'test_smooth_labels': human_ground_truths[config['train_examples']:],
      'test_rankings': rankings[config['train_examples']:],
      'test_groups': groups[config['train_examples']:],
      'test_selectors': selectors[config['train_examples']:],
      'test_irn': irn[config['train_examples']:],
  }

config = {}
config['rng'] = gtd.PRNGSequence(5)
config['dimensionality'] = 2
config['sigma'] = 0.3
config['class_weights'] = [1]*3
config['class_sigmas'] = 0.1
config['train_examples'] = 1000
# Note that in the paper we used 20000 test examples.
config['test_examples'] = 1000
config['expected_length'] = 1.5
config['grouping_threshold'] = 0.05
# Number of readers and their sharpness.
config['reader_sharpness'] = jnp.array([500000, 100000, 50000, 1000000, 500000, 150000, 100000, 1000000, 100000, 90000])

data = get_data(config)
with open('toy_data.pkl', 'wb') as f:
  pickle.dump(data, f)
exit(0)

predictions = []
for seed in range(4):
  classifier = sklearn.neural_network.MLPClassifier(alpha=1, max_iter=(seed + 1) * 25, random_state=seed)
  classifier.fit(
      data['train_examples'],
      jax.nn.one_hot(data['train_labels'], 3))
  predictions_k = classifier.predict_log_proba(data['test_examples'])
  predictions_k = jax.nn.softmax(predictions_k)
  predictions.append(predictions_k)
predictions = jnp.array(predictions)

for seed in range(predictions.shape[0]):
    print(seed, jnp.mean(data['test_labels'] == jnp.argmax(predictions[seed], axis=1)))
    with open(f'toy_predictions{seed}.pkl', 'wb') as f:
        pickle.dump(predictions[seed], f)