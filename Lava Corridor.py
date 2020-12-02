import mdp
import mdp_worlds
import utils
import numpy as np
import random 
import bayesian_irl
import plot_gridworld as pg

init_seed = 1#4 + 10
np.random.seed(init_seed)
random.seed(init_seed)

slip_prob = 0.3
demo_horizon = 10
num_demos = 1

###BIRL
beta = 10.0
step_stdev = 0.2
burn = 50
skip = 5
num_samples = 200
mcmc_norm = "l2"
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
pg.get_policy_string_from_trajectory(traj_demonstrations[0], state_feature_list, mdp_env)


    
# In[4]:

#LPAL solution
u_expert = utils.u_sa_from_demos(traj_demonstrations, mdp_env)
lpal_usa = mdp.solve_lpal_policy(mdp_env, u_expert)
#utils.print_stochastic_policy_action_probs(cvar_opt_usa, mdp_env_A)
print("lpal policy")
utils.print_policy_from_occupancies(lpal_usa, mdp_env)
utils.print_stochastic_policy_action_probs(lpal_usa, mdp_env)
pi_dict = utils.get_stoch_policy_string_dictionary_from_occupancies(lpal_usa, mdp_env)
state_feature_list = [tuple(fs) for fs in mdp_env.state_features]
pg.plot_optimal_policy_stochastic(pi_dict, state_feature_list, mdp_env.num_rows, mdp_env.num_cols)


# In[4]:


import maxent
#just keep states in traj_demos
maxent_demos = []
for d in traj_demonstrations:
    #add only states to demos
    demo = []
    for s,a in d:
        demo.append(s)
    maxent_demos.append(demo)
maxent_usa, r_weights, maxent_pi = maxent.calc_max_ent_u_sa(mdp_env, maxent_demos)
print("max ent policy")
utils.print_policy_from_occupancies(maxent_usa, mdp_env)
utils.print_stochastic_policy_action_probs(maxent_usa, mdp_env)
pi_dict = utils.get_stoch_policy_string_dictionary_from_occupancies(maxent_usa, mdp_env)
state_feature_list = [tuple(fs) for fs in mdp_env.state_features]
pg.plot_optimal_policy_stochastic(pi_dict, state_feature_list, mdp_env.num_rows, mdp_env.num_cols)
