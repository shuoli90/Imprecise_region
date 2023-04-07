from numpy import load
import matplotlib.pyplot as plt
import numpy as np

# load collected data
cnn_softmax = load('./data/cnn_softmax.npy')
uncertainty = load("./data/uncertainty.npy")
label = load("./data/label.npy")
alpha = 0.01
calibration_softmax = cnn_softmax[:5000,]
calibration_label = label[:5000,]
test_softmax = cnn_softmax[5000:,]
test_label = label[5000:,]
taus = []

# compute label-conditional thresholds
for k in range(10):
    print("Label %d" % k)
    indices = np.where(calibration_label == k)[0]
    ncm_k = 1 - calibration_softmax[indices, k]
    tau = np.quantile(ncm_k, 1 - alpha, method='higher')
    taus.append(tau)

# compute testing error
for k in range(10):
    print("Label %d" % k)
    indices = np.where(test_label == k)[0]
    ncm_k = 1 - test_softmax[indices, k]
    error = np.sum(ncm_k > taus[k]) / len(ncm_k)
    print("Error: %f" % error)

# construct ambiguity set
(image_idx, label_idx) = np.nonzero(uncertainty[5000:,] >= 0.1)
sets = []
for i in range(5000):
    indices = np.where(image_idx == i)[0]
    labels = label_idx[indices]
    sets.append(labels)

# disambiguate the ambiguity set
sets_disambiguate = []
for idx, set in enumerate(sets):
    set_tmp = []
    for label in set:
        if (1-test_softmax[idx, label]) <= taus[label]:
            set_tmp.append(label)
    sets_disambiguate.append(set_tmp)

print('average set size', np.mean([len(s) for s in sets_disambiguate]))