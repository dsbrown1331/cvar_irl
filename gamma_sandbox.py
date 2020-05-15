import numpy as np
# shape = 1/2
# scale = 20.0
# size = 1000
# samples = np.random.gamma(shape, scale, size)
# print(np.mean(samples), np.std(samples), np.max(samples))
# import matplotlib.pyplot as plt
# plt.hist(samples,100)
#plt.show()


from scipy.stats import gamma
import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1, 1)



x = np.linspace(0,120, 100)
linestyles=['-','--','-.',':']
num_states = 4
#shape = 0.5
# shapes = [1,1,1,3]
# scales = [20,30,50,50]
shapes = [.5,.5,.5,.5]
scales = [20, 40, 80,190]

print('means', np.array(shapes) * np.array(scales))

for i in range(num_states):
    plt.figure()
    size = 1000
    samples = np.random.gamma(shapes[i], scales[i], size)
    print(np.mean(samples), np.std(samples), np.max(samples))
    import matplotlib.pyplot as plt
    plt.hist(samples,100)




plt.figure()
for i in range(num_states):
    
    rv = gamma(a = shapes[i], scale=scales[i])
    plt.plot(x, rv.pdf(x), linestyles[i],lw=3, label='Do Nothing at state {}'.format(i))

plt.xticks(fontsize=16) 
plt.yticks(fontsize=16) 
plt.xlabel('Cost', fontsize=19)
plt.ylabel('Probability', fontsize=19)
plt.legend(loc='best', fontsize=16)
plt.tight_layout()
plt.show()