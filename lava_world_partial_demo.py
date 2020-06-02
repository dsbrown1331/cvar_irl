import mdp
import mdp_worlds
import utils
import numpy as np
import random 
import bayesian_irl
import plot_gridworld as pg


#used this to plot the diagrams in the paper currently.

init_seed = 1#4 + 10
np.random.seed(init_seed)
random.seed(init_seed)

slip_prob = 0.3
demo_horizon = 10
num_demos = 1

###BIRL
beta = 5.0
step_stdev = 0.2
burn = 500
skip = 5
num_samples = 2000
mcmc_norm = "inf"
likelihood = "birl"

mdp_env = mdp_worlds.lava_ambiguous_corridor()
opt_sa = mdp.solve_mdp_lp(mdp_env)


print("Cliff world")
print("Optimal Policy")
utils.print_policy_from_occupancies(opt_sa, mdp_env)
print("reward")
utils.print_as_grid(mdp_env.r_s, mdp_env)
print("features")
utils.display_onehot_state_features(mdp_env)

init_demo_state = 1#mdp_env.num_cols * (mdp_env.num_rows - 1)
traj_demonstrations = []
demo_set = set()
for d in range(num_demos):
    # np.random.seed(init_seed + d)
    # random.seed(init_seed + d)
    s = init_demo_state #mdp_env.init_states[0] # only one initial state
    demo = utils.rollout_from_usa(s, demo_horizon, opt_sa, mdp_env)
    print("demo", d, demo)
    traj_demonstrations.append(demo)
    for s_a in demo:
        demo_set.add(s_a)
demonstrations = list(demo_set)
print("demonstration")
print(demonstrations)

state_feature_list = [tuple(fs) for fs in mdp_env.state_features]
pg.get_policy_string_from_trajectory(traj_demonstrations[0], state_feature_list, mdp_env, filename="./figs/lava_corridor/demo_lava_world.png")


#Now let's run Bayesian IRL on this demo in this mdp with a placeholder feature to see what happens.


birl = bayesian_irl.BayesianIRL(mdp_env, beta, step_stdev, debug=False, mcmc_norm=mcmc_norm, likelihood=likelihood)

# r_hyps = [#[0,-1],
#         [-1,0],
#         #[-0.1232111, -0.8767889],
#         #[-0.25691869, -0.74308131],
#         [-0.05609254, -0.94390746],
#         [-0.57066167,  0.42933833],
#         [-0.55, 0.45],
#         [-.1,-0.9]
#         ]

# for i,r in enumerate(r_hyps):
#     print()
#     print("TRIAL ", i)
#     #generate random rewards
#     #r = r / np.linalg.norm(r)
#     print("weights", r)
#     r_sa = mdp_env.transform_to_R_sa(r)
#     print('mean reward')
#     r_s = np.dot(mdp_env.state_features, r)
#     utils.print_as_grid(r_s, mdp_env)
    
#     occ_freqs, q_vals = birl.solve_optimal_policy(r)
#     print("Optimal Policy under weights")
#     utils.print_policy_from_occupancies(occ_freqs, mdp_env)
#     print("policy probs")
#     utils.print_stochastic_policy_action_probs(occ_freqs, mdp_env)
    
#     print(q_vals.shape)
#     print(q_vals)
#     utils.print_q_vals_pretty(q_vals, mdp_env)
#     #q_vals2 = mdp.get_q_values(occupancy_frequencies, mdp_env)
#     ll = birl.log_likelihood(r, q_vals, demonstrations)
#     print("log likelihood ", ll)

# input()
map_w, map_u, r_chain, u_chain = birl.sample_posterior(demonstrations, num_samples, True)
print(r_chain)
import matplotlib.pyplot as plt
plt.figure()
for w in range(len(r_chain[0])):
    plt.plot(r_chain[:,w],label="feature {}".format(w))
plt.legend()
plt.show()


r_chain_burned = r_chain[burn::skip]

print("MAP")
print("map_weights", map_w)
map_r = np.dot(mdp_env.state_features, map_w)
print("map reward")
utils.print_as_grid(map_r, mdp_env)
print("Map policy")
utils.print_policy_from_occupancies(map_u, mdp_env)

print("MEAN")
mean_w = np.mean(r_chain_burned, axis=0)
print("mean_weights", mean_w)
mean_r = np.dot(mdp_env.state_features, mean_w)
mean_r_sa = mdp_env.transform_to_R_sa(mean_w)
mean_u_sa = mdp.solve_mdp_lp(mdp_env, reward_sa=mean_r_sa) #use optional argument to replace standard rewards with sample
print('mean reward')
utils.print_as_grid(mean_r, mdp_env)
print("mean policy")
utils.print_policy_from_occupancies(mean_u_sa, mdp_env)

#Now let's see what CVaR optimization does.
lambdas = [0., 0.1, 0.3, 0.5, 0.8, 1.0]
alpha = 0.95
for lamda in lambdas:
    print()
    print("ALPHA", alpha, "LAMBDA", lamda)

    debug = False
    

    n = r_chain_burned.shape[0]
    posterior_probs = np.ones(n) / n  #uniform dist since samples from MCMC

    
    print("------ Robust Solution ---------")
    u_expert = np.zeros(mdp_env.num_actions * mdp_env.num_states)
    robust_opt_usa, cvar_value, exp_ret = mdp.solve_max_cvar_policy(mdp_env, u_expert, r_chain_burned.transpose(), posterior_probs, alpha, debug, lamda)
    #utils.print_stochastic_policy_action_probs(cvar_opt_usa, mdp_env_A)
    print("Policy for lambda={} and alpha={}".format(lamda, alpha))
    utils.print_policy_from_occupancies(robust_opt_usa, mdp_env)

    pi = utils.get_policy_string_from_occupancies(robust_opt_usa, mdp_env)
    state_feature_list = [tuple(fs) for fs in mdp_env.state_features]
    pg.plot_optimal_policy_lists(pi, state_feature_list, mdp_env.num_rows, mdp_env.num_cols, "./figs/lava_corridor/robust_alpha" + str(alpha) + "lambda" + str(lamda) + ".png")
    

    print("------ Regret Solution ---------")
    traj_demonstrations = [demonstrations]
    u_expert = utils.u_sa_from_demos(traj_demonstrations, mdp_env)
    
    regret_opt_usa, cvar_value, exp_ret = mdp.solve_max_cvar_policy(mdp_env, u_expert, r_chain_burned.transpose(), posterior_probs, alpha, debug, lamda)
    #utils.print_stochastic_policy_action_probs(cvar_opt_usa, mdp_env_A)
    print("Policy for lambda={} and alpha={}".format(lamda, alpha))
    utils.print_policy_from_occupancies(regret_opt_usa, mdp_env)

    pi = utils.get_policy_string_from_occupancies(regret_opt_usa, mdp_env)
    state_feature_list = [tuple(fs) for fs in mdp_env.state_features]
    pg.plot_optimal_policy_lists(pi, state_feature_list, mdp_env.num_rows, mdp_env.num_cols, "./figs/lava_corridor/regret_alpha" + str(alpha) + "lambda" + str(lamda) + ".png")
    
# print("-------- IRD Solution -------")
# u_expert = utils.u_sa_from_demos(traj_demonstrations, mdp_env)#mdp_env.state_features[init_demo_state]#np.zeros(mdp_env.get_reward_dimensionality())#
# ird_w = utils.get_worst_case_feature_weights_binary_ird(r_chain_burned, u_expert, mdp_env)
# ird_r = np.dot(mdp_env.state_features, ird_w)
# ird_r_sa = mdp_env.transform_to_R_sa(ird_w)
# ird_u_sa = mdp.solve_mdp_lp(mdp_env, reward_sa=ird_r_sa) #use optional argument to replace standard rewards with sample
# print('ird reward')
# utils.print_as_grid(ird_r, mdp_env)
# print("ird policy")
# utils.print_policy_from_occupancies(ird_u_sa, mdp_env)
