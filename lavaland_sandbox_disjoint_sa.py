import mdp
import mdp_worlds
import utils
import numpy as np
import bayesian_irl
import random

# mdp_env_A = mdp_worlds.lava_ambiguous_ird_fig2a()

# u_sa_A = mdp.solve_mdp_lp(mdp_env_A)

# print("mdp A")
# print("Policy")
# utils.print_policy_from_occupancies(u_sa_A, mdp_env_A)
# print("reward")
# utils.print_as_grid(mdp_env_A.r_s, mdp_env_A)


# mdp_env_B = mdp_worlds.lava_ambiguous_ird_fig2b()

# u_sa_B = mdp.solve_mdp_lp(mdp_env_B)

# print("mdp B")
# print("Policy")
# utils.print_policy_from_occupancies(u_sa_B, mdp_env_B)
# print("reward")
# utils.print_as_grid(mdp_env_B.r_s, mdp_env_B)



seed = 12131
np.random.seed(seed)
random.seed(seed)


#let's try out BIRL on a simpler version and see what happens

#first let's give a demo in the A version that doesn't have lava

mdp_env_A = mdp_worlds.lava_ird_simplified_a()
# mdp_env_A = mdp_worlds.lava_ambiguous_ird_fig2a()
u_sa_A = mdp.solve_mdp_lp(mdp_env_A)

print("mdp A")
print("Policy")
utils.print_policy_from_occupancies(u_sa_A, mdp_env_A)
print("reward")
utils.print_as_grid(mdp_env_A.r_s, mdp_env_A)
print("features")
utils.display_onehot_state_features(mdp_env_A)


#generate demo for Dylan's NeurIPS world
# demonstrations = utils.rollout_from_usa(51, 15, u_sa_A, mdp_env_A)
#generate demo for my simplified world

demonstrations = utils.rollout_from_usa(10, 100, u_sa_A, mdp_env_A)
print("demonstration")
print(demonstrations)

#Now let's run Bayesian IRL on this demo in this mdp with a placeholder feature to see what happens.

beta = 100.0
step_stdev = 0.1
burn = 100
skip = 5
num_samples = 2000
sample_norm = None
birl = bayesian_irl.BayesianIRL(mdp_env_A, beta, step_stdev, debug=False, mcmc_norm=sample_norm)


map_w, map_u, r_chain, u_chain = birl.sample_posterior(demonstrations, num_samples, True)

import matplotlib.pyplot as plt
for w in range(len(r_chain[0])):
    plt.plot(r_chain[:,w],label="feature {}".format(w))
plt.legend()
plt.show()

print("MAP")
print("map_weights", map_w)
map_r = np.dot(mdp_env_A.state_features, map_w)
print("map reward")
utils.print_as_grid(map_r, mdp_env_A)
print("Map policy")
utils.print_policy_from_occupancies(map_u, mdp_env_A)

print("MEAN")
mean_w = np.mean(r_chain[burn::skip], axis=0)
print("mean_weights", mean_w)
mean_r = np.dot(mdp_env_A.state_features, mean_w)
mean_r_sa = mdp_env_A.transform_to_R_sa(mean_w)
mean_u_sa = mdp.solve_mdp_lp(mdp_env_A, reward_sa=mean_r_sa) #use optional argument to replace standard rewards with sample
print('mean reward')
utils.print_as_grid(mean_r, mdp_env_A)
print("mean policy")
utils.print_policy_from_occupancies(mean_u_sa, mdp_env_A)


#Now let's explore transfering the reward to a new domain.
#---print("mdp B")
mdp_env_B = mdp_worlds.lava_ird_simplified_b()
# mdp_env_B = mdp_worlds.lava_ambiguous_ird_fig2b()

u_sa_B = mdp.solve_mdp_lp(mdp_env_B)

print("===========================")
print("mdp B")
print("===========================")
print("Opt Policy under true reward")
utils.print_policy_from_occupancies(u_sa_B, mdp_env_B)
print("reward")
utils.print_as_grid(mdp_env_B.r_s, mdp_env_B)
print("features")
utils.display_onehot_state_features(mdp_env_B)

print("MAP")
print("map_weights", map_w)
map_r = np.dot(mdp_env_B.state_features, map_w)
print("map reward")
utils.print_as_grid(map_r, mdp_env_B)
#compute new policy for mdp_B for map rewards
map_r_sa = mdp_env_B.transform_to_R_sa(map_w)
map_u_sa = mdp.solve_mdp_lp(mdp_env_B, reward_sa=map_r_sa) #use optional argument to replace standard rewards with sample
print("Map policy")
utils.print_policy_from_occupancies(map_u_sa, mdp_env_B)

print("MEAN")
mean_w = np.mean(r_chain[burn::skip], axis=0)
print("mean_weights", mean_w)
mean_r = np.dot(mdp_env_B.state_features, mean_w)
mean_r_sa = mdp_env_B.transform_to_R_sa(mean_w)
mean_u_sa = mdp.solve_mdp_lp(mdp_env_B, reward_sa=mean_r_sa) #use optional argument to replace standard rewards with sample
print('mean reward')
utils.print_as_grid(mean_r, mdp_env_B)
print("mean policy")
utils.print_policy_from_occupancies(mean_u_sa, mdp_env_B)


#Now let's see what CVaR optimization does.
alpha = 0.99
debug = False
lamda = 0.0
r_chain_burned = r_chain[burn::skip]
n = r_chain_burned.shape[0]
posterior_probs = np.ones(n) / n  #uniform dist since samples from MCMC

print("MDP A")    
print("features")
utils.display_onehot_state_features(mdp_env_A)

print("------ Robust Solution ---------")
u_expert = np.zeros(mdp_env_B.num_actions * mdp_env_B.num_states)
cvar_opt_usa, cvar_value, exp_ret = mdp.solve_max_cvar_policy(mdp_env_A, u_expert, r_chain_burned.transpose(), posterior_probs, alpha, debug, lamda)
#utils.print_stochastic_policy_action_probs(cvar_opt_usa, mdp_env_A)
print("Policy for lambda={} and alpha={}".format(lamda, alpha))
utils.print_policy_from_occupancies(cvar_opt_usa, mdp_env_A)
print("solving for CVaR reward")
cvar_reward, q = mdp.solve_minCVaR_reward(mdp_env_A, u_expert, r_chain_burned.transpose(), posterior_probs, alpha)
# print("cvar reward weights", cvar_reward)
print("cvar reward weights", np.dot(q, r_chain_burned))


print("------ Regret Solution ---------")
traj_demonstrations = [demonstrations]
u_expert = utils.u_sa_from_demos(traj_demonstrations, mdp_env_A)
print('expert u_sa', u_expert)

cvar_opt_usa, cvar_value, exp_ret = mdp.solve_max_cvar_policy(mdp_env_A, u_expert, r_chain_burned.transpose(), posterior_probs, alpha, debug, lamda)
#utils.print_stochastic_policy_action_probs(cvar_opt_usa, mdp_env_A)
print("Policy for lambda={} and alpha={}".format(lamda, alpha))
utils.print_policy_from_occupancies(cvar_opt_usa, mdp_env_A)
print("solving for CVaR reward")
cvar_reward, q = mdp.solve_minCVaR_reward(mdp_env_A, u_expert, r_chain_burned.transpose(), posterior_probs, alpha)
# print("cvar reward weights", cvar_reward)
print("cvar reward weights", np.dot(q, r_chain_burned))

print("\n===============")
print("MDP B")  
print("===============\n")
print("features")
utils.display_onehot_state_features(mdp_env_B)


print("------ Robust Solution ---------")
u_expert = np.zeros(mdp_env_B.num_actions * mdp_env_B.num_states)
cvar_opt_usa, cvar_value, exp_ret = mdp.solve_max_cvar_policy(mdp_env_B, u_expert, r_chain_burned.transpose(), posterior_probs, alpha, debug, lamda)
#utils.print_stochastic_policy_action_probs(cvar_opt_usa, mdp_env_B)
print("Policy for lambda={} and alpha={}".format(lamda, alpha))
utils.print_policy_from_occupancies(cvar_opt_usa, mdp_env_B)

print("solving for CVaR reward")
cvar_reward, q = mdp.solve_minCVaR_reward(mdp_env_B, u_expert, r_chain_burned.transpose(), posterior_probs, alpha)
# print("cvar reward weights", cvar_reward)
print("cvar reward weights", np.dot(q, r_chain_burned))


print("------ Regret Solution ---------")
#NOTE that I'm using mdp_env_A since we don't have feature counts for env B
u_expert = utils.u_sa_from_demos(traj_demonstrations, mdp_env_A)
cvar_opt_usa, cvar_value, exp_ret = mdp.solve_max_cvar_policy(mdp_env_B, u_expert, r_chain_burned.transpose(), posterior_probs, alpha, debug, lamda)
print('expert u_sa', u_expert)
#utils.print_stochastic_policy_action_probs(cvar_opt_usa, mdp_env_B)
print("Policy for lambda={} and alpha={}".format(lamda, alpha))
utils.print_policy_from_occupancies(cvar_opt_usa, mdp_env_B)

print("solving for CVaR reward")
cvar_reward, q = mdp.solve_minCVaR_reward(mdp_env_B, u_expert, r_chain_burned.transpose(), posterior_probs, alpha)
#print("cvar reward weights", cvar_reward)
print("cvar reward weights", np.dot(q, r_chain_burned))


print("-------- IRD Solution -------")
u_expert = utils.u_sa_from_demos(traj_demonstrations, mdp_env_A)
ird_w = utils.get_worst_case_feature_weights_binary_ird(r_chain_burned, u_expert, mdp_env_B)
ird_r = np.dot(mdp_env_B.state_features, ird_w)
ird_r_sa = mdp_env_B.transform_to_R_sa(ird_w)
ird_u_sa = mdp.solve_mdp_lp(mdp_env_B, reward_sa=ird_r_sa) #use optional argument to replace standard rewards with sample
print('ird reward')
utils.print_as_grid(ird_r, mdp_env_B)
print("ird policy")
utils.print_policy_from_occupancies(ird_u_sa, mdp_env_B)


print("-------- IRD Zero Baseline Solution -------")
u_expert = np.zeros(mdp_env_B.num_actions * mdp_env_B.num_states)
ird_w = np.min(r_chain_burned, axis=0)#utils.get_worst_case_feature_weights_binary_ird(r_chain_burned, u_expert, mdp_env_B)
ird_r = np.dot(mdp_env_B.state_features, ird_w)
ird_r_sa = mdp_env_B.transform_to_R_sa(ird_w)
ird_u_sa = mdp.solve_mdp_lp(mdp_env_B, reward_sa=ird_r_sa) #use optional argument to replace standard rewards with sample
print('ird reward')
utils.print_as_grid(ird_r, mdp_env_B)
print("ird policy")
utils.print_policy_from_occupancies(ird_u_sa, mdp_env_B)
