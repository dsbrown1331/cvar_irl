import bayesian_irl
import mdp_worlds
import utils
import mdp
import numpy as np
import scipy
import random


def generate_reward_sample(num_states):
    #costs for no-op are -N(0,10-4) except last state that is -N(100,800)
    r_noop = np.concatenate((-np.random.randn(num_states-1) * 10e-4, -100 + -800*np.random.randn(1)))
    
    #costs for repair are -N(130,1) for all but last state where it is -N(130,20)
    r_repair = np.concatenate((-130 + -1 * np.random.randn(num_states-1), -130 + -20*np.random.randn(1))) 
    
    return np.concatenate((r_noop, r_repair))

def generate_reward_mean(num_states):
    #costs for no-op are -N(0,10-4) except last state that is -N(100,800)
    r_noop = np.concatenate((np.zeros(num_states-1), -100*np.ones(1)))
    
    #costs for repair are -N(130,1) for all but last state where it is -N(130,20)
    r_repair = np.concatenate((-130*np.ones(num_states-1), -130*np.ones(1))) 
    
    return np.concatenate((r_noop, r_repair))

def generate_posterior_samples(num_states, num_samples):
    print("samples")
    all_samples = []
    for i in range(num_samples):
        r_sample = generate_reward_sample(num_states)
        all_samples.append(r_sample)
        # r_string = ""
        # for r in r_sample:
        #     r_string += "{:.1f}\t".format(r)
        # print(r_string)

    print("mean of posterior from samples")
    print(np.mean(all_samples, axis=0))
    print(generate_reward_mean(3))

    posterior = np.array(all_samples)

    return posterior.transpose()  #each column is a reward sample


if __name__=="__main__":
    # seed = 1234
    # np.random.seed(seed)
    # scipy.random.seed(seed)
    # random.seed(seed)
    num_states = 50
    num_samples = 2000
    #r_noop = np.array([0,0,-100])
    #r_repair = np.array([-50,-50,-50])
    gamma = 0.8
    alpha = 0.9
    lamda = 0

    r_sa = generate_reward_mean(num_states)
    init_distribution = np.ones(num_states)/num_states  #uniform distribution
    mdp_env = mdp.MachineReplacementMDP(num_states, r_sa, gamma, init_distribution)
    print(mdp_env.Ps)

    u_sa = mdp.solve_mdp_lp(mdp_env, debug=True)
    print("MAP/Mean policy from posterior")
    utils.print_policy_from_occupancies(u_sa, mdp_env) 
    print("rewards")
    print(mdp_env.r_sa)

    posterior = generate_posterior_samples(num_states, num_samples)
    print(posterior)
    print(posterior.shape)


    #run CVaR optimization, maybe just the robust version for now
    u_expert = np.zeros(mdp_env.num_actions * mdp_env.num_states)
    

    posterior_probs = np.ones(num_samples) / num_samples  #uniform dist since samples from MCMC
    cvar_opt_usa, cvar, exp_ret = mdp.solve_max_cvar_policy(mdp_env, u_expert, posterior, posterior_probs, alpha, False, lamda)
    
    print("MAP/Mean policy from posterior")
    utils.print_policy_from_occupancies(u_sa, mdp_env) 
    print("rewards")
    print(mdp_env.r_sa)
    
    print("CVaR policy")
    utils.print_policy_from_occupancies(cvar_opt_usa, mdp_env)
    utils.print_stochastic_policy_action_probs(cvar_opt_usa, mdp_env)

    