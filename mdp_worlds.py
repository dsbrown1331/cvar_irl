import mdp
import numpy as np
import utils

def machine_teaching_toy():
    num_rows = 2
    num_cols = 3
    num_states = num_rows * num_cols
    r_s = np.array( [-1, -10, -1,
                     -1, -1, -1] )
    gamma = 0.9
    init_dist = 1/(num_states) * np.ones(num_states)
    terminals = [0]
    mdp_env = mdp.BasicGridMDP(num_rows, num_cols, r_s, gamma, init_dist, terminals, True)
    return mdp_env

def machine_teaching_toy_featurized():
    num_rows = 2
    num_cols = 3
    num_states = num_rows * num_cols
    white = (1,0)
    gray = (0,1)
    state_features = np.array([white, gray,  white,
                               white, white, white])
    weights = np.array([-1, -10])
    gamma = 0.9
    init_dist = 1/(num_states) * np.ones(num_states)
    terminals = [0]
    mdp_env = mdp.FeaturizedGridMDP(num_rows, num_cols, state_features, weights, gamma, init_dist, terminals, True)
    return mdp_env


def lava_ambiguous_aaai18():
    num_rows = 5
    num_cols = 5
    num_states = num_rows * num_cols
    white = (1,0)
    red = (0,1)
    state_features = np.array([white, white, white, red, white,
                               white, red, white, red, white,
                               white, red, white, red, white,
                               white, red, white, red, white,
                               white, red, white, white, white ])
    weights = np.array([-.18, -.82])
    gamma = 0.95
    init_dist = np.zeros(num_states)
    init_states = [5,4]
    for si in init_states:
        init_dist[si] = 1.0 / len(init_states)
    term_states = [12]
    init_dist = 1/(num_states) * np.ones(num_states)
    mdp_env = mdp.FeaturizedGridMDP(num_rows, num_cols, state_features, weights, gamma, init_dist, term_states)
    return mdp_env



def two_state_chain():
    num_states = 2
    gamma = 0.9
    p0 = np.array([0.5,0.5])
    r_sa = np.array([1,-1, +1, -1])
    chain = mdp.ChainMDP(num_states, r_sa, gamma, p0)
    return chain


if __name__=="__main__":
    #mdp_env = machine_teaching_toy_featurized()
    mdp_env = lava_ambiguous_aaai18()
    u_sa = mdp.solve_mdp_lp(mdp_env, debug=True)
    print("optimal policy")
    utils.print_policy_from_occupancies(u_sa, mdp_env)
    print("optimal policy")
    v = mdp.get_state_values(u_sa, mdp_env)
    utils.print_as_grid(v, mdp_env)
