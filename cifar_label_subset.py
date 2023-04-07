from numpy import load
import matplotlib.pyplot as plt
import numpy as np

# load collected data
cnn_softmax = load('./data/cnn_softmax.npy')
uncertainty = load("./data/uncertainty.npy")
label = load("./data/label.npy")
alpha = 0.01
calibration_label = label[:5000,]
calibration_softmax = cnn_softmax[np.arange(5000), calibration_label]
test_softmax = cnn_softmax[5000:,]
test_label = label[5000:,]


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
    calibration_tmp = calibration_softmax[np.isin(calibration_label, set)]
    try:
        tau_tmp = np.quantile(calibration_tmp, 1 - alpha, method='higher')
    except IndexError as e:
        sets_disambiguate.append(set)
        continue
    for label in set:
        if (1-test_softmax[idx, label]) <= tau_tmp:
            set_tmp.append(label)
    sets_disambiguate.append(set_tmp)

print('average set size', np.mean([len(s) for s in sets_disambiguate]))