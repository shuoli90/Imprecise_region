# Introduction
This is the repo for the conformal prediction with ambiguous data points project.

# Implementation
## cifar_label_conditional.py
- Compute thresholds for each label
- Construct ambiguous data points using cifar-10h (human-labeled label distribution; include class $k$ into the data point if its confidence is above 0.3.)
- Use class-conditioned thresholds to disambiguate ambiguous data points.
- Compute average set size
- To run, just *python cifar_label_conditional.py*

## cifar_label_subset.py
- For every set, construct calibration set *calibration_tmp* basing on the ambiguous label set
- Compute thresholds *tau_tmp* via extracting *1-alpha* quantile of *calibration_tmp*
- Use *tau_tmp* to disambiguous the ambiguous data points
- Compute average set size
- To run, just *python cifar_label_subset.py*

## Remaining questions:
- How should we compare the two result besides computing the average set size?