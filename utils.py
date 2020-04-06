import numpy as np
import random

def get_optimal_policy_from_usa(u_sa, mdp_env):
    num_states, num_actions = mdp_env.num_states, mdp_env.num_actions
    opt_stoch_pi = np.zeros((num_states, num_actions))
    for s in range(num_states):
        #compute the total occupancy for that state across all actions
        s_tot_occupancy = np.sum(u_sa[s::num_states])
        for a in range(num_actions):
            opt_stoch_pi[s][a] = u_sa[s+a*num_states] / max(s_tot_occupancy, 0.000001)
    return opt_stoch_pi


def print_table_row(vals):
    row_str = ""
    for i in range(len(vals) - 1):
        row_str += "{:0.2} & ".format(vals[i])
    row_str += "{:0.2} \\\\".format(vals[-1])
    return row_str


def logsumexp(x):
    max_x = np.max(x)
    sum_exp = 0.0
    for xi in x:
        sum_exp += np.exp(xi - max_x)
    return max(x) + np.log(sum_exp)

def print_policy_from_occupancies(proposal_occupancies, mdp_env):
    policy = get_optimal_policy_from_usa(proposal_occupancies, mdp_env)
    cnt = 0
    for r in range(mdp_env.num_rows):
        row_str = ""
        for c in range(mdp_env.num_cols):
            if cnt not in mdp_env.terminals:
                row_str += mdp_env.get_readable_actions(np.argmax(policy[cnt])) + "\t"
            else:
                row_str += ".\t"  #denote terminal with .
            cnt += 1
        print(row_str)

def print_as_grid(x, mdp_env):
    #print into a num_rows by num_cols grid
    cnt = 0
    for r in range(mdp_env.num_rows):
        row_str = ""
        for c in range(mdp_env.num_cols):
            row_str += "{:.2f}\t".format(x[cnt])
            cnt += 1
        print(row_str)


def rollout_from_usa(start, horizon, u_sa, mdp_env):
    #generate a demonstration starting at start of length horizon
    demonstration = []
    #first get the stochastic optimal policy
    policy = get_optimal_policy_from_usa(u_sa,mdp_env)

    #rollout for H steps or until a terminal is reached
    curr_state = start
    #print('start',curr_state)
    steps = 0
    while curr_state not in mdp_env.terminals and steps < horizon:
        #print('actions',policy[curr_state])
        #select an action choice according to policy action probs
        a = np.random.choice(range(mdp_env.num_actions), p = policy[curr_state])
        #print(a)
        demonstration.append((curr_state, a))
        #sample transition
        action_transition_probs = mdp_env.Ps[a][curr_state]
        s_next = np.random.choice(range(mdp_env.num_states), p = action_transition_probs)
        curr_state = s_next
        steps += 1
        #print('next state', curr_state)
    if curr_state in mdp_env.terminals:
        #append the terminal state
        demonstration.append((curr_state, None))  #no more actions available
    return demonstration
