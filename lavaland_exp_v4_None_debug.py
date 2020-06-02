import mdp
import mdp_worlds
import utils
import numpy as np
import bayesian_irl
import random


#Okay, so it seems that using fewer demos is better for our method than for IRD so that's good.

init_seed = 1331

num_trials = 1

#CVaR optimization params
alpha = 0.95
debug = False
lamda = 0.0

#Bayesian IRL params
beta = 100.0
step_stdev = 0.15
burn = 200
skip = 5
num_samples = 1000

plot_chain = True

demo_states = [0]

demo_horizon = 100

birl_norm = None


experiment_directory = './results/lavaland/'
experiment_name = 'grid5x5_ploss_lavaocc_v4_None.txt'
import os
if not os.path.exists(experiment_directory):
    os.makedirs(experiment_directory)

#result writer
f = open(os.path.join(experiment_directory, experiment_name), 'w')




for t in range(num_trials):
    print("##############")
    print("Trial ", t)
    print("##############")
    print()
    seed = init_seed + 13 * t
    np.random.seed(seed)
    random.seed(seed)


    #train mdp
    mdp_env_A = mdp_worlds.lavaland_smaller(contains_lava=False)
    #test mdp ((probably) has lava)
    mdp_env_B = mdp_worlds.lavaland_smaller(contains_lava=True)

    u_sa_A = mdp.solve_mdp_lp(mdp_env_A)

    print("===========================")
    print("Training MDP with No Lava")
    print("===========================")

    print("Optimal Policy")
    utils.print_policy_from_occupancies(u_sa_A, mdp_env_A)
    print("reward")
    utils.print_as_grid(mdp_env_A.r_s, mdp_env_A)
    print("features")
    utils.display_onehot_state_features(mdp_env_A)

    #generate demonstration from top left corner
    traj_demonstrations = []
    demo_set = set()
    for s in demo_states:#range(mdp_env_A.get_num_states()):
        if mdp_env_A.init_dist[s] > 0:
            demo = utils.rollout_from_usa(s, demo_horizon, u_sa_A, mdp_env_A)
            traj_demonstrations.append(demo)
            for s_a in demo:
                demo_set.add(s_a)
    demonstrations = list(demo_set)
    print("demonstration")
    print(demonstrations)

    #CVaR stuff needs expected feature counts from a list of trajectories
    #traj_demonstrations = [demonstrations]

    #Now let's run Bayesian IRL on this demo in this mdp with a placeholder feature to see what happens.
    birl = bayesian_irl.BayesianIRL(mdp_env_A, beta, step_stdev, debug=False, mcmc_norm=birl_norm)

    map_w, _, r_chain, _ = birl.sample_posterior(demonstrations, num_samples, False)

    

    r_chain_burned = r_chain[burn::skip]
    n = r_chain_burned.shape[0]
    posterior_probs = np.ones(n) / n  #uniform dist since samples from MCMC

    if plot_chain:
        import matplotlib.pyplot as plt
        for w in range(len(r_chain[0])):
            plt.plot(r_chain[:,w],label="feature {}".format(w))
        plt.legend()
        plt.show()

    

    print("===========================")
    print("Testing MDP with Lava")
    print("===========================")
    u_sa_B = mdp.solve_mdp_lp(mdp_env_B)
    print("Opt Policy under true reward")
    utils.print_policy_from_occupancies(u_sa_B, mdp_env_B)
    print("reward")
    utils.print_as_grid(mdp_env_B.r_s, mdp_env_B)
    print("features")
    utils.display_onehot_state_features(mdp_env_B)

    print("MAP on testing env")
    print("map_weights", map_w)
    map_r = np.dot(mdp_env_B.state_features, map_w)
    print("map reward")
    utils.print_as_grid(map_r, mdp_env_B)
    #compute new policy for mdp_B for map rewards
    map_r_sa = mdp_env_B.transform_to_R_sa(map_w)
    print("map r_sa")
    print(map_r_sa)
    map_u_sa = mdp.solve_mdp_lp(mdp_env_B, reward_sa=map_r_sa) #use optional argument to replace standard rewards with sample
    print("Map policy map_u_sa")
    utils.print_policy_from_occupancies(map_u_sa, mdp_env_B)
    
    print("MEAN policy on test env")
    mean_w = np.mean(r_chain_burned, axis=0)
    print("mean_weights", mean_w)
    mean_r = np.dot(mdp_env_B.state_features, mean_w)
    mean_r_sa = mdp_env_B.transform_to_R_sa(mean_w)
    mean_u_sa = mdp.solve_mdp_lp(mdp_env_B, reward_sa=mean_r_sa) #use optional argument to replace standard rewards with sample
    print('mean reward')
    utils.print_as_grid(mean_r, mdp_env_B)
    print("mean policy")
    utils.print_policy_from_occupancies(mean_u_sa, mdp_env_B)

    print("features")
    utils.display_onehot_state_features(mdp_env_B)

    print("opt)")
    print(u_sa_B)
    print("map")
    print(map_u_sa)
    print(u_sa_B - map_u_sa)
    print("mean)")
    print(mean_u_sa)
    print(u_sa_B - mean_u_sa)
    print("r_sa")
    print(mdp_env_B.r_sa)
    

    map_ploss = np.dot(mdp_env_B.r_sa, u_sa_B - map_u_sa)
    mean_ploss = np.dot(mdp_env_B.r_sa, u_sa_B - mean_u_sa)
    

    print("MAP policy loss", map_ploss)
    print("mean policy loss", mean_ploss)
    
    lava_states = []
    for s,f in enumerate(mdp_env_B.state_features):
        if (f == (0,0,0,1)).all(): #hard coded lava feature
            lava_states.append(s)

    print("lava states", lava_states)

    print("initial dist")
    print(mdp_env_B.init_dist)

    print("map_u")
    print(np.sum(map_u_sa))
    utils.print_policy_occupancies_pretty(map_u_sa, mdp_env_B)
    utils.print_stochastic_policy_action_probs(map_u_sa, mdp_env_B)

    print("mean_u")
    print(np.sum(mean_u_sa))
    utils.print_policy_occupancies_pretty(mean_u_sa, mdp_env_B)
    utils.print_stochastic_policy_action_probs(mean_u_sa, mdp_env_B)

    num_states = mdp_env_B.get_num_states()
    map_lava = 0
    for s in lava_states:
        map_lava += np.sum(map_u_sa[s::num_states])

    print("map lava", map_lava)


    num_states = mdp_env_B.get_num_states()
    mean_lava = 0
    for s in lava_states:
        mean_lava += np.sum(mean_u_sa[s::num_states])

    print("mean lava", mean_lava)
    #collect the lava portions of the state-action occupancies
    #stack the weight vectors
    stacked_weights = []
    for a in range(mdp_env_B.num_actions):
        stacked_weights.append(np.array(mdp_env_B.state_features))
    stacked_weights = np.concatenate(stacked_weights)

    map_lava = np.dot(map_u_sa, stacked_weights)
    mean_lava = np.dot(mean_u_sa, stacked_weights)
    
    print("MAP lava occupancy", map_lava)
    print("Mean lava occupancy", mean_lava)
    