import mdp
import mdp_worlds
import utils
import numpy as np
import bayesian_irl
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mcmc_norm', type=str, default='None',
                    help='None, "inf", "l2", or "l1"')

args = parser.parse_args()


#Okay, so it seems that using fewer demos is better for our method than for IRD so that's good.

init_seed = 1212

num_trials = 30

#CVaR optimization params
alpha = 0.95
debug = False
lamda = 0.0

#Bayesian IRL params
beta = 100.0
step_stdev = 0.15
burn = 200
skip = 5
num_samples = 2000

plot_chain = False

demo_states = [0]

demo_horizon = 30

if args.mcmc_norm == 'None':
        birl_norm = None
        experiment_name = 'grid5x5_ploss_lavaocc_v4_None.txt'
else:
    birl_norm = args.mcmc_norm
    experiment_name = 'grid5x5_ploss_lavaocc_v4_' + args.mcmc_norm + '.txt'


experiment_directory = './results/lavaland/'

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
    mdp_env_A = mdp_worlds.lavaland_small(contains_lava=False)
    #test mdp ((probably) has lava)
    mdp_env_B = mdp_worlds.lavaland_small(contains_lava=True)

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

    map_w, map_u_A, r_chain, _ = birl.sample_posterior(demonstrations, num_samples, False)


    r_chain_burned = r_chain[burn::skip]
    n = r_chain_burned.shape[0]
    posterior_probs = np.ones(n) / n  #uniform dist since samples from MCMC

    if plot_chain:
        import matplotlib.pyplot as plt
        for w in range(len(r_chain[0])):
            plt.plot(r_chain[:,w],label="feature {}".format(w))
        plt.legend()
        plt.show()

    print("MAP Policy on Train MDP")
    print("map_weights", map_w)
    map_r = np.dot(mdp_env_A.state_features, map_w)
    print("map reward")
    utils.print_as_grid(map_r, mdp_env_A)
    print("Map policy")
    utils.print_policy_from_occupancies(map_u_A, mdp_env_A)

    print("MEAN policy on Train MDP")
    mean_w = np.mean(r_chain[burn::skip], axis=0)
    print("mean_weights", mean_w)
    mean_r = np.dot(mdp_env_A.state_features, mean_w)
    mean_r_sa = mdp_env_A.transform_to_R_sa(mean_w)
    mean_u_A = mdp.solve_mdp_lp(mdp_env_A, reward_sa=mean_r_sa) #use optional argument to replace standard rewards with sample
    print('mean reward')
    utils.print_as_grid(mean_r, mdp_env_A)
    print("mean policy")
    utils.print_policy_from_occupancies(mean_u_A, mdp_env_A)

    print("Optimal Policy")
    utils.print_policy_from_occupancies(u_sa_A, mdp_env_A)
    

    print("MAP policy loss", np.dot(mdp_env_A.r_sa, u_sa_A - map_u_A))
    print("Mean policy loss", np.dot(mdp_env_A.r_sa, u_sa_A - mean_u_A))



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
    map_u_sa = mdp.solve_mdp_lp(mdp_env_B, reward_sa=map_r_sa) #use optional argument to replace standard rewards with sample
    print("Map policy")
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


    print("------ Robust Solution ---------")
    u_expert = np.zeros(mdp_env_B.num_actions * mdp_env_B.num_states)
    cvar_robust_usa, cvar_value, exp_ret = mdp.solve_max_cvar_policy(mdp_env_B, u_expert, r_chain_burned.transpose(), posterior_probs, alpha, debug, lamda)
    #utils.print_stochastic_policy_action_probs(cvar_opt_usa, mdp_env_B)
    print("Policy for lambda={} and alpha={}".format(lamda, alpha))
    utils.print_policy_from_occupancies(cvar_robust_usa, mdp_env_B)

    # print("solving for CVaR reward")
    # cvar_reward, q = mdp.solve_minCVaR_reward(mdp_env_B, u_expert, r_chain_burned.transpose(), posterior_probs, alpha)
    # # print("cvar reward weights", cvar_reward)
    # print("cvar reward weights", np.dot(q, r_chain_burned))


    print("------ Regret Solution ---------")
    #NOTE that I'm using mdp_env_A since we don't have feature counts for env B
    u_expert = utils.u_sa_from_demos(traj_demonstrations, mdp_env_A)
    cvar_regret_usa, cvar_value, exp_ret = mdp.solve_max_cvar_policy(mdp_env_B, u_expert, r_chain_burned.transpose(), posterior_probs, alpha, debug, lamda)
    print('expert u_sa', u_expert)
    #utils.print_stochastic_policy_action_probs(cvar_opt_usa, mdp_env_B)
    print("Policy for lambda={} and alpha={}".format(lamda, alpha))
    utils.print_policy_from_occupancies(cvar_regret_usa, mdp_env_B)

    # print("solving for CVaR reward")
    # cvar_reward, q = mdp.solve_minCVaR_reward(mdp_env_B, u_expert, r_chain_burned.transpose(), posterior_probs, alpha)
    # #print("cvar reward weights", cvar_reward)
    # print("cvar reward weights", np.dot(q, r_chain_burned))

    print("-------- IRD Solution -------")
    ird_r = utils.get_worst_case_state_rewards_ird(r_chain_burned, u_expert, mdp_env_B)
    ird_r_sa = mdp_env_B.transform_to_R_sa(ird_r)
    ird_u_sa = mdp.solve_mdp_lp(mdp_env_B, reward_sa=ird_r_sa) #use optional argument to replace standard rewards with sample
    print('ird reward')
    utils.print_as_grid(ird_r, mdp_env_B)
    print("ird policy")
    utils.print_policy_from_occupancies(ird_u_sa, mdp_env_B)


    map_ploss = np.dot(mdp_env_B.r_sa, u_sa_B - map_u_sa)
    mean_ploss = np.dot(mdp_env_B.r_sa, u_sa_B - mean_u_sa)
    robust_ploss = np.dot(mdp_env_B.r_sa, u_sa_B - cvar_robust_usa)
    regret_ploss = np.dot(mdp_env_B.r_sa, u_sa_B - cvar_regret_usa)
    ird_ploss = np.dot(mdp_env_B.r_sa, u_sa_B - ird_u_sa)


    print("MAP policy loss", map_ploss)
    print("mean policy loss", mean_ploss)
    print("robust policy loss", robust_ploss)
    print("regret policy loss", regret_ploss)
    print("ird policy loss", ird_ploss)

    # lava_states = []
    # for s,f in enumerate(mdp_env_B.state_features):
    #     if (f == (0,0,0,1)).all(): #hard coded lava feature
    #         lava_states.append(s)

    # print("lava states", lava_states)
    #collect the lava portions of the state-action occupancies
    #stack the weight vectors
    stacked_weights = []
    for a in range(mdp_env_B.num_actions):
        stacked_weights.append(np.array(mdp_env_B.state_features))
    stacked_weights = np.concatenate(stacked_weights)

    map_lava = np.dot(map_u_sa, stacked_weights)[3]
    mean_lava = np.dot(mean_u_sa, stacked_weights)[3]
    robust_lava = np.dot(cvar_robust_usa, stacked_weights)[3]
    regret_lava = np.dot(cvar_regret_usa, stacked_weights)[3]
    ird_lava = np.dot(ird_u_sa, stacked_weights)[3]

    print("MAP lava occupancy", map_lava)
    print("Mean lava occupancy", mean_lava)
    print("Robust lava occupancy", robust_lava)
    print("Regret lava occupancy", regret_lava)
    print("IRD lava occupancy", ird_lava)
    utils.write_line([map_ploss, mean_ploss, robust_ploss, regret_ploss, ird_ploss, map_lava, mean_lava, robust_lava, regret_lava, ird_lava], f)
f.close()