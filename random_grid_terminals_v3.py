from mdp_worlds import random_gridworld_corner_terminal
import mdp
import utils
import bayesian_irl
import numpy as np
import copy
import random

#going to try even more ambiguous demonstrations


#experiment params
num_trials = 100
debug = True
init_seed = 1331
experiment_directory = './results/random_grid_terminal/'
experiment_name = 'grid6x6_cvar_birl_v3.txt'
import os
if not os.path.exists(experiment_directory):
    os.makedirs(experiment_directory)

#result writer
f = open(os.path.join(experiment_directory, experiment_name), 'w')


#MDP params

num_rows = 8
num_cols = 8
num_features = 6

#demo params
demo_state = num_cols * (num_rows - 1) #start bottom left state
horizon =30

#CVaR parameters
alpha = 0.95
lamdas = [0.0, 0.5, 0.75, 0.9, 0.95]


#B-IRL params
beta = 100.0
step_stdev = 0.05
num_samples = 3000
burn = 100
skip = 5



for i in range(num_trials):
    print("="*10)
    print("iteration", i)
    print("="*10)

    seed = init_seed + i * 13
    np.random.seed(seed)
    random.seed(seed)

    mdp_env = random_gridworld_corner_terminal(num_rows, num_cols, num_features)
    opt_u_sa = mdp.solve_mdp_lp(mdp_env, debug=debug)
    true_r_sa = mdp_env.r_sa
    true_w = mdp_env.feature_weights
    
    

    #generate demontration from bottom left corner
    demonstrations = utils.rollout_from_usa(demo_state, horizon, opt_u_sa, mdp_env)
    print("demonstration")
    print(demonstrations)


    ###Run Bayesian IRL to get posterior

    birl = bayesian_irl.BayesianIRL(mdp_env, beta, step_stdev, debug=False)
    map_w, map_u_sa, w_chain, u_chain = birl.sample_posterior(demonstrations, num_samples, False)


    if debug:
        print("-------")
        print("true weights", true_w)
        print("features")
        utils.display_onehot_state_features(mdp_env)
        print("optimal policy")
        utils.print_policy_from_occupancies(opt_u_sa, mdp_env)
        print("optimal values")
        v = mdp.get_state_values(opt_u_sa, mdp_env)
        utils.print_as_grid(v, mdp_env)


    if debug: 
        print("map_weights", map_w)
        map_r = np.dot(mdp_env.state_features, map_w)
        print("MAP reward")
        utils.print_as_grid(map_r, mdp_env)
        print("Map policy")
        utils.print_policy_from_occupancies(map_u_sa, mdp_env)

    w_chain_burned = w_chain[burn::skip]

    ###compute mean reward policy

    mean_w = np.mean(w_chain_burned, axis=0)
    #reuse mdp env details but solve with mean reward
    mean_u_sa = mdp.solve_mdp_lp(mdp_env, reward_sa=mdp_env.transform_to_R_sa(mean_w) ,debug=True)
    
    if debug:
        print("mean w", mean_w)
        print("Mean policy from posterior")
        utils.print_policy_from_occupancies(mean_u_sa, mdp_env) 
        mean_r = np.dot(mdp_env.state_features, mean_w)
        print("Mean rewards")
        utils.print_as_grid(mean_r, mdp_env)
    

    ###Compute policy loss wrt true reward
    mean_ploss = utils.policy_loss(mean_u_sa, mdp_env, opt_u_sa)
    map_ploss = utils.policy_loss(map_u_sa, mdp_env, opt_u_sa)
    
    print("mean = {}, map = {}".format(mean_ploss, map_ploss))

    ###run CVaR IRL to get policy

    #running just the robust version for now
    u_expert = np.zeros(mdp_env.num_actions * mdp_env.num_states)
    
    n = w_chain_burned.shape[0]
    posterior_probs = np.ones(n) / n  #uniform dist since samples from MCMC
    cvar_losses = []
    for lamda in lamdas:
        cvar_u_sa, cvar, exp_ret = mdp.solve_max_cvar_policy(mdp_env, u_expert, w_chain_burned.transpose(), posterior_probs, alpha, False, lamda=lamda)
        if debug: 
            print("CVaR policy")
            utils.print_policy_from_occupancies(cvar_u_sa, mdp_env)


        cvar_ploss = utils.policy_loss(cvar_u_sa, mdp_env, opt_u_sa)
        cvar_losses.append(cvar_ploss)

    cvar_ploss_str = ""
    for loss in cvar_losses:
        cvar_ploss_str += ", {}".format(loss)


    print("cvar = {}".format(cvar_ploss_str))
    
    ### write to file
    f.write("{}, {}{}\n".format(mean_ploss, map_ploss, cvar_ploss_str))

f.close()

    #TODO: examine results

    #TODO: compare to LPAL and IRD