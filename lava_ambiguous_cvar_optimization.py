import bayesian_irl
import mdp_worlds
import utils
import mdp
import numpy as np



if __name__=="__main__":
    #mdp_env = mdp_worlds.two_state_chain()
    #demonstrations = [(1,0), (0,0)]

    # mdp_env = mdp_worlds.machine_teaching_toy_featurized()
    # demonstrations = [(2,3),(5,0),(4,0),(3,2)]


    mdp_env = mdp_worlds.lava_ambiguous_aaai18()
    u_sa = mdp.solve_mdp_lp(mdp_env)
    #generate demo from state 5 to terminal
    demonstrations = utils.rollout_from_usa(5, 10, u_sa, mdp_env)
    print(demonstrations)


    beta = 100.0
    step_stdev = 0.01
    birl = bayesian_irl.BayesianIRL(mdp_env, beta, step_stdev, debug=False)

    num_samples = 2000
    map_w, map_u, r_chain, u_chain = birl.sample_posterior(demonstrations, num_samples)
    print("map_weights", map_w)
    map_r = np.dot(mdp_env.state_features, map_w)
    utils.print_as_grid(map_r, mdp_env)
    print("Map policy")
    utils.print_policy_from_occupancies(map_u, mdp_env)

    # print("chain")
    # for r in r_chain:
    #     print(r)

    worst_index = np.argmin(r_chain[:,1])
    print(r_chain[worst_index])
    print(np.sum(r_chain[:,1] < -0.82))


    #run CVaR optimization, maybe just the robust version for now
    u_expert = np.zeros(mdp_env.num_actions * mdp_env.num_states)
    alpha = 0.99
    cvar_opt_usa = mdp.solve_max_cvar_policy(mdp_env, u_expert, r_chain.transpose(), np.ones(num_samples) / num_samples, alpha)
    utils.print_policy_from_occupancies(cvar_opt_usa, mdp_env)
    utils.print_policy_from_occupancies(cvar_opt_usa, mdp_env)
    