import utils
import numpy as np
import os


#experiment params
alpha = 0.95
lamdas = [0.0, 0.5, 0.75, 0.9, 0.95]
num_trials = 100
experiment_directory = './results/random_grid_no_terminal/'
experiment_name = 'grid7x7_cvar_regret_birl_l2.txt'

#result reader
from numpy import genfromtxt
policy_losses = genfromtxt(os.path.join(experiment_directory,experiment_name), delimiter=',')
print(policy_losses)
print("mean, map, cvar")
print("mean", np.mean(policy_losses, axis=0))
print("std", np.std(policy_losses, axis=0))
print("max", np.max(policy_losses, axis=0))
print("min", np.min(policy_losses, axis=0))

labels = ['Mean', 'MAP']
labels.extend([str(l) for l in lamdas])
print(labels)


#plot worst 10% histograms
sorted_plosses = np.sort(policy_losses, axis=0)

print('sample CVaRs', np.mean(sorted_plosses[95:,:], axis=0))

import matplotlib.pyplot as plt
# plt.plot(sorted_plosses, label = labels)
plt.hist(sorted_plosses[80:,:],label=labels, linewidth=2)
plt.legend()
plt.show()