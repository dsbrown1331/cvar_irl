import numpy as np

def get_optimal_policy_from_usa(u_sa, mdp_env):
    num_states, num_actions = mdp_env.num_states, mdp_env.num_actions
    opt_stoch_pi = np.zeros((num_states, num_actions))
    for s in range(num_states):
        #compute the total occupancy for that state across all actions
        s_tot_occupancy = np.sum(u_sa[s::num_states])
        for a in range(num_actions):
            opt_stoch_pi[s][a] = u_sa[s+a*num_states] / max(s_tot_occupancy, 0.000001)
    return opt_stoch_pi
