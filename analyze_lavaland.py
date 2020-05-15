import utils
import numpy as np
import os

#used earlier probabilities of grass and lava and dirt and L2 ball norm on MCMC


experiment_directory = './results/lavaland/'
experiment_name = 'grid5x5_ploss_lavaocc.txt'

#result reader
from numpy import genfromtxt
policy_losses = genfromtxt(os.path.join(experiment_directory,experiment_name), delimiter=',')

mean_plosses = np.mean(policy_losses, axis=0)
print(mean_plosses)
import matplotlib.pyplot as plt

plt.figure()
plt.title("Policy Loss")
print()
x = np.arange(5)
plt.bar(x,mean_plosses[:5])
plt.xticks(x, ('MAP', 'Mean', 'Robust', 'Regret', 'Worst-Case'))
plt.figure()
plt.title("Lava Occupancy")
plt.bar(x, mean_plosses[5:])
plt.xticks(x, ('MAP', 'Mean', 'Robust', 'Regret', 'Worst-Case'))

labels = ['MAP', 'Mean', 'Robust', 'Regret', 'Worst-Case']
plt.figure()
for w in range(5):
    plt.plot(policy_losses[:,5+w], label=labels[w])
    #plt.plot(policy_losses[:,-1], label=['worst-case'])
plt.legend()
print(policy_losses[:,-2:])
plt.show()