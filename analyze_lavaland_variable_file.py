import utils
import numpy as np
import os




experiment_directory = './results/lavaland/'
experiment_name = 'grid5x5_ploss_lavaocc_v4_None.txt'    #this experiment used L2-norm constraint
# experiment_name = 'grid5x5_ploss_lavaocc_v2_lessgrass_fewer_inits_alpha99.txt'  #this experiment used L1-norm constraint

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
    print(w)
    plt.plot(policy_losses[:,5+w], label=labels[w])
    #plt.plot(policy_losses[:,-1], label=['worst-case'])
plt.legend()
print(np.mean(policy_losses, axis=0))
print(np.std(policy_losses, axis=0))
plt.show()