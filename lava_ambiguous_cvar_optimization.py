import bayesian_irl
import mdp_worlds
import utils
import mdp
import numpy as np
import scipy
import random



if __name__=="__main__":
    seed = 1234
    np.random.seed(seed)
    scipy.random.seed(seed)
    random.seed(seed)
    #mdp_env = mdp_worlds.two_state_chain()
    #demonstrations = [(1,0), (0,0)]

    # mdp_env = mdp_worlds.machine_teaching_toy_featurized()
    # demonstrations = [(2,3),(5,0),(4,0),(3,2)]


    mdp_env = mdp_worlds.lava_ambiguous_aaai18()
    u_sa = mdp.solve_mdp_lp(mdp_env)
    #generate demo from state 5 to terminal
    demonstrations = utils.rollout_from_usa(5, 10, u_sa, mdp_env)
    print(demonstrations)

    traj_demonstrations = [demonstrations]


    beta = 100.0
    step_stdev = 0.2
    birl = bayesian_irl.BayesianIRL(mdp_env, beta, step_stdev, debug=False)

    num_samples = 1000
    burn = 50
    skip = 2
    map_w, map_u, r_chain, u_chain = birl.sample_posterior(demonstrations, num_samples)
    print("map_weights", map_w)
    map_r = np.dot(mdp_env.state_features, map_w)
    utils.print_as_grid(map_r, mdp_env)
    #print("Map policy")
    #utils.print_policy_from_occupancies(map_u, mdp_env)

    # print("chain")
    # for r in r_chain:
    #     print(r)

    worst_index = np.argmin(r_chain[:,1])
    print(r_chain[worst_index])
    print(np.sum(r_chain[:,1] < -0.82), "out of ", len(r_chain))

    r_chain_burned = r_chain[burn::skip]
    # print("chain after burn and skip")
    # for r in r_chain_burned:
    #     print(r)
    #input()
    worst_index = np.argmin(r_chain_burned[:,1])
    print(r_chain_burned[worst_index])
    print(np.sum(r_chain_burned[:, 1]< -0.82), "out of", len(r_chain_burned))
    #input()

    print("MAP policy")
    utils.print_policy_from_occupancies(map_u, mdp_env)
    
    
    #let's actually try using the optimal policy to get the feature counts and see if the regret method works?
    u_expert = u_sa
    alpha = 0.95
    n = r_chain_burned.shape[0]
    posterior_probs = np.ones(n) / n  #uniform dist since samples from MCMC
    cvar_opt_usa_regret, cvar, exp_ret = mdp.solve_max_cvar_policy(mdp_env, u_expert, r_chain_burned.transpose(), posterior_probs, alpha, False)
    print("{}-CVaR policy regret optimal u_E".format(alpha))
    utils.print_policy_from_occupancies(cvar_opt_usa_regret, mdp_env)

    #let's actually try the robust formulation but with respect to the empirical feature counts in the demo
    u_expert = utils.u_sa_from_demos(traj_demonstrations, mdp_env)
    alpha = 0.95
    n = r_chain_burned.shape[0]
    posterior_probs = np.ones(n) / n  #uniform dist since samples from MCMC
    cvar_opt_usa_regret, cvar, exp_ret = mdp.solve_max_cvar_policy(mdp_env, u_expert, r_chain_burned.transpose(), posterior_probs, alpha, False)
    print("{}-CVaR policy regret approx u_E".format(alpha))
    utils.print_policy_from_occupancies(cvar_opt_usa_regret, mdp_env)



    #run CVaR optimization, maybe just the robust version for now
    u_expert = np.zeros(mdp_env.num_actions * mdp_env.num_states)
    cvar_opt_usa_robust, cvar, exp_ret = mdp.solve_max_cvar_policy(mdp_env, u_expert, r_chain_burned.transpose(), posterior_probs, alpha, False)
    print("{}-CVaR policy robust".format(alpha))
    utils.print_policy_from_occupancies(cvar_opt_usa_robust, mdp_env)



    # #debug, what if I hand craft the posterior?
    # R_toy = np.array([[-.18, -.82],
    #                   [-.1, -.9],
    #                   [0.01, -0.99], 
    #                   [-.1, -.9],
    #                   [0.01, -0.99], 
    #                   [-.1, -.9],
    #                   [0.01, -0.99], 
    #                   [-.1, -.9],
    #                   [0.01, -0.99], 
    #                   [-.1, -.9],
    #                   [0.01, -0.99], 
    #                   [-.1, -.9],
    #                   [0.01, -0.99], 
    #                   [-.1, -.9],
    #                   [0.01, -0.99], 
    #                   [-.1, -.9],
    #                   [0.01, -0.99], 
    #                   [-.1, -.9],
    #                   [0.01, -0.99], 
    #                   [-.1, -.9],
    #                   [0.01, -0.99], 
    #                   [-0.21686709, -0.78313291]]).transpose()
    # # R_toy = np.array([[-0.01983656, -0.98016344],
    # #     [-0.18720474, -0.81279526],
    # #     [-0.08654517, -0.91345483],
    # #     [-0.07859799, -0.92140201],
    # #     [-0.01983656, -0.98016344]]).transpose()
    # k,n = R_toy.shape
    # posterior_probs_toy = np.ones(n)/n
    # alpha = 0.5
    # cvar_opt_usa = mdp.solve_max_cvar_policy(mdp_env, u_expert, R_toy, posterior_probs_toy, alpha, True)
    # print("CVaR policy")
    # utils.print_policy_from_occupancies(cvar_opt_usa, mdp_env)