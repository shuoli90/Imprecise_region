# Introduction
This is the repo for the conformal prediction with ambiguous data points project.

# Implementation
## cifar.py
- Compute thresholds for each label
- Construct ambiguous data points using cifar-10h (human-labeled label distribution; include class $k$ into the data point if its confidence is above 0.3.)
- Use class-conditioned thresholds to disambiguate ambiguous data points.
- To run, just *python cifar.py*