import utils
import numpy as np
import os


norms = ["None", "l2", "l1", 'inf'] #TODO: add inf norm

norm_labels = {"None": "None", "l2": "2-norm", "l1":"1-norm", "inf": "inf-norm"}
width = 0.5

experiment_directory = './results/lavaland/'
lava_occs = []

x = np.linspace(0,11,5)
for i,n in enumerate(norms):
    print(n)
    experiment_name = 'grid5x5_ploss_lavaocc_v4_' + n + '.txt'    #this experiment used L2-norm constraint
    # experiment_name = 'grid5x5_ploss_lavaocc_v2_lessgrass_fewer_inits_alpha99.txt'  #this experiment used L1-norm constraint

    #result reader
    from numpy import genfromtxt
    policy_losses = genfromtxt(os.path.join(experiment_directory,experiment_name), delimiter=',')

    mean_plosses = np.mean(policy_losses, axis=0)
    #print(mean_plosses)
    import matplotlib.pyplot as plt

    
    lava_occs = mean_plosses[5:]
    stdev = np.std(policy_losses, axis=0)
    lava_std = stdev[5:]
    print("mean", lava_occs)
    print("std", lava_std)

    # plt.figure()
    # plt.title("Policy Loss with " + n + " norm")
    # print()
    
    # plt.bar(x,mean_plosses[:5])
    # plt.xticks(x, ('MAP', 'Mean', 'Robust', 'Regret', 'Worst-Case'))

    
    # labels = ['MAP', 'Mean', 'Robust', 'Regret', 'Worst-Case']
    # plt.figure()
    # for w in range(5):
    #     print(w)
    #     plt.plot(policy_losses[:,5+w], label=labels[w])
    #     #plt.plot(policy_losses[:,-1], label=['worst-case'])
    # plt.legend()
    # print(np.mean(policy_losses, axis=0))
    #print(np.std(policy_losses, axis=0))

    
    plt.bar(x + width * i, lava_occs, width, label=norm_labels[n], alpha=0.75, yerr=0.25*lava_std)
plt.xticks(x + 1.5*width, ('MAP', 'Mean', 'Robust', 'Regret', 'Worst-Case'), fontsize=15)
#plt.xlabel("Policy Optimization Method", fontsize=18)
plt.ylabel("Average Lava Occupancy", fontsize=15)
plt.legend(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.tick_params(axis = "x", which = "both", bottom = False, top = False)
plt.savefig("./figs/lava_occupancy.png")



plt.show()