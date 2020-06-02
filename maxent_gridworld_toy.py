from maxent import *

#Question 2

# Build domain, features, and demos
trans_mat = build_trans_mat_gridworld()
num_features = 2
state_features = np.concatenate((np.ones((26,1)), np.zeros((26,1))), axis=1)#np.eye(26,25)  # Terminal state has no features, forcing zero reward
state_features[25,:] = np.zeros(2)
state_features[8,:] = np.array([0,1])
print(state_features)
demos = [[0,1,2,3,4,9,14,19,24,25],[0,5,10,15,20,21,22,23,24,25],[0,5,6,11,12,17,18,23,24,25],[0,1,6,7,12,13,18,19,24,25]]
seed_weights = np.zeros(num_features)

# Parameters
n_epochs = 100
horizon = 10
learning_rate = 0.5
n_states = np.shape(trans_mat)[0]
start_dist = np.zeros(n_states)
start_dist[0] = 1 #start at state 0 

# Main algorithm call
r_weights, grads = maxEntIRL(trans_mat, state_features, demos, seed_weights, n_epochs, horizon, learning_rate, start_dist)

# Construct reward function from weights and state features
reward_fxn = []
for s_i in range(25):
    reward_fxn.append( np.dot(r_weights, state_features[s_i]) )
reward_fxn = np.reshape(reward_fxn, (5,5))

# Plot reward function
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = np.arange(0, 5, 1)
Y = np.arange(0, 5, 1)
X, Y = np.meshgrid(X, Y)
surf = ax.plot_surface(X, Y, reward_fxn, rstride=1, cstride=1, cmap=cm.coolwarm,
		linewidth=0, antialiased=False)
plt.show()
