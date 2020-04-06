# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:39:34 2020

@author: dsbrown
"""


import numpy as np
import utils
from scipy.optimize import linprog
from interface import implements, Interface
import sys

#acts as abstract class
class MDP(Interface):
    def get_num_actions(self):
        pass

    def get_reward_dimensionality(self):
        pass

    def set_reward_fn(self, new_reward):
        pass

    def get_transition_prob(self, s1,a,s2):
        pass

    def get_num_states(self):
        pass

    def get_readable_actions(self, action_num):
        pass

    def get_state_action_rewards(self):
        pass

    def transform_to_R_sa(self, reward_weights):
        #mainly used for BIRL to take hypothesis reward and transform it
        #take in representation of reward weights and return vectorized version of R_sa
        #R_sa = [R(s0,a0), .., R(sn,a0), ...R(s0,am),..., R(sn,am)]
        pass

    

        


class ChainMDP(implements(MDP)):
    #basic MDP class that has two actions (left, right), no terminal states and is a chain mdp with deterministic transitions
    def __init__(self, num_states, r_sa, gamma, init_dist):
        self.num_actions = 2
        self.num_rows = 1
        self.num_cols = num_states
        self.num_states =  num_states
        self.gamma = gamma
        self.init_dist = init_dist
       
        self.r_sa = r_sa


        self.P_left = self.get_transitions(policy="left")
        #print("P_left\n",self.P_left)
        self.P_right = self.get_transitions(policy="right")
        #print("P_right\n",self.P_right)
        self.Ps = [self.P_left, self.P_right]


    def get_num_actions(self):
        return self.num_actions

    def transform_to_R_sa(self, reward_weights):
        #Don't do anything, reward_weights should be r_sa 
        assert(len(reward_weights) == len(self.r_sa))
        return reward_weights

    def get_readable_actions(self, action_num):
        if action_num == 0:
            return "<"
        elif action_num == 1:
            return ">"
        else:
            print("error, only two possible actions")
            sys.exit()

    def get_num_states(self):
        return self.num_states

    def get_reward_dimensionality(self):
        return len(self.r_sa)

    def set_reward_fn(self, new_reward):
        self.r_sa = new_reward

    def get_state_action_rewards(self):
        return self.r_sa

    def get_transition_prob(self, s1,a,s2):
        return self.Ps[a][s1][s2]

    def get_transitions(self, policy):
        P_pi = np.zeros((self.num_states, self.num_states))
        if policy == "left":  #action 0
            #always transition one to left unless already at left border
            cnt = 0
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    if c > 0:
                        P_pi[cnt, cnt - 1] = 1.0
                    else:
                        P_pi[cnt,cnt] = 1.0
                    #increment state count
                    cnt += 1
        elif policy == "right":  #action 1
            #always transition one to right unless already at right border
            cnt = 0
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    if c < self.num_cols - 1:
                        #transition to next state to right
                        P_pi[cnt, cnt + 1] = 1.0
                    else:
                        #self transition
                        P_pi[cnt,cnt] = 1.0
                    #increment state count
                    cnt += 1
        return P_pi
        

class BasicGridMDP(implements(MDP)):
    #basic MDP class that has four actions, no terminal states and is a grid with deterministic transitions
    def __init__(self, num_rows, num_cols, r_s, gamma, init_dist, terminals = [], debug=False):
        self.num_actions = 4
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_states =  num_rows * num_cols
        self.gamma = gamma
        self.init_dist = init_dist
        self.terminals = terminals
        self.debug = debug
        self.r_s = r_s
        self.r_sa = self.transform_to_R_sa(self.r_s)
        #print("transformed R(s,a)", self.r_sa)

        self.P_left = self.get_transitions(policy="left")
        if self.debug: print("P_left\n",self.P_left)
        self.P_right = self.get_transitions(policy="right")
        if self.debug: print("P_right\n",self.P_right)
        self.P_up = self.get_transitions(policy="up")
        if self.debug: print("_up\n",self.P_up)
        self.P_down = self.get_transitions(policy="down")
        if self.debug: print("P_down\n",self.P_down)
        self.Ps = [self.P_left, self.P_right, self.P_up, self.P_down] #actions:0,1,2,3

    def get_num_actions(self):
        return self.num_actions

    def get_num_states(self):
        return self.num_states

   
    def get_readable_actions(self, action_num):
        if action_num == 0:
            return "<"
        elif action_num == 1:
            return ">"
        elif action_num == 2:
            return "^"
        elif action_num == 3:
            return "v"
        else:
            print("error, only four possible actions")
            sys.exit()


    def get_transition_prob(self, s1,a,s2):
        return self.Ps[a][s1][s2]

    #Note that I'm using r_s as the reward dim not r_sa!
    def get_reward_dimensionality(self):
        return len(self.r_s)


    def get_state_action_rewards(self):
        return self.r_sa

    #assume new reward is of the form r_s
    def set_reward_fn(self, new_reward):
        self.r_s = new_reward
        #also update r_sa
        self.r_sa = self.transform_to_R_sa(self.r_s)



    #transform R(s) into R(s,a) for use in LP
    def transform_to_R_sa(self, reward_weights):
        #assume that reward_weights is r_s
        #tile to get r_sa from r

        '''input: numpy array R_s, output R_sa'''
        #print(len(R_s))
        #print(self.num_states)
        #just repeat values since R_sa = [R(s1,a1), R(s2,a1),...,R(sn,a1), R(s1,a2), R(s2,a2),..., R(sn,am)]
        assert(len(reward_weights) == self.num_states)
        return np.tile(reward_weights, self.num_actions)

    def get_transitions(self, policy):
        P_pi = np.zeros((self.num_states, self.num_states))
        if policy == "left":  #action 0 
            #always transition one to left unless already at left border
            cnt = 0
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    if cnt not in self.terminals: #no transitions out of terminal
                        if c > 0:
                            P_pi[cnt, cnt - 1] = 1.0
                        else:
                            P_pi[cnt,cnt] = 1.0
                    #increment state count
                    cnt += 1
        elif policy == "right":  #action 1
            #always transition one to right unless already at right border
            cnt = 0
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    if cnt not in self.terminals: #no transitions out of terminal
                        if c < self.num_cols - 1:
                            #transition to next state to right
                            P_pi[cnt, cnt + 1] = 1.0
                        else:
                            #self transition
                            P_pi[cnt,cnt] = 1.0
                    #increment state count
                    cnt += 1
        elif policy == "up": #action 2
            #always transition one to left unless already at left border
            cnt = 0
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    if cnt not in self.terminals: #no transitions out of terminal
                        if r > 0:
                            P_pi[cnt, cnt - self.num_cols] = 1.0
                        else:
                            P_pi[cnt,cnt] = 1.0
                    #increment state count
                    cnt += 1
        elif policy == "down":  #action 3
            #always transition one to left unless already at left border
            cnt = 0
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    if cnt not in self.terminals: #no transitions out of terminal
                        if r < self.num_rows - 1:
                            P_pi[cnt, cnt + self.num_cols] = 1.0
                        else:
                            P_pi[cnt,cnt] = 1.0
                    #increment state count
                    cnt += 1
        return P_pi


class FeaturizedGridMDP(BasicGridMDP):


    def __init__(self,num_rows, num_cols, state_feature_matrix, feature_weights, gamma, init_dist, terminals = [], debug=False):
        self.num_actions = 4
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_states =  num_rows * num_cols
        self.gamma = gamma
        self.init_dist = init_dist
        self.terminals = terminals
        self.debug = debug
        
        self.P_left = self.get_transitions(policy="left")
        if self.debug: print("P_left\n",self.P_left)
        self.P_right = self.get_transitions(policy="right")
        if self.debug: print("P_right\n",self.P_right)
        self.P_up = self.get_transitions(policy="up")
        if self.debug: print("_up\n",self.P_up)
        self.P_down = self.get_transitions(policy="down")
        if self.debug: print("P_down\n",self.P_down)
        self.Ps = [self.P_left, self.P_right, self.P_up, self.P_down] #actions:0,1,2,3

        #figure out reward function
        self.state_features = state_feature_matrix
        self.feature_weights = feature_weights
        r_s = np.dot(self.state_features, self.feature_weights)
        #print("r_s", r_s)
        self.r_s = r_s
        self.r_sa = self.transform_to_R_sa(self.feature_weights)
        #print("transformed R(s,a)", self.r_sa)


    def get_reward_dimensionality(self):
        return len(self.feature_weights)


    def set_reward_fn(self, new_reward):
        #input is the new_reward weights
        assert(len(new_reward) == len(self.feature_weights))
        #update feature weights
        self.feature_weights = new_reward.copy()
        #update r_s
        self.r_s = np.dot(self.state_features, new_reward)
        #update r_sa
        self.r_sa = np.tile(self.r_s, self.num_actions)


    def transform_to_R_sa(self, reward_weights):
        #assumes that inputs are the reward feature weights
        #returns the vectorized R_sa 
        
        #first get R_s
        R_s = np.dot(self.state_features, reward_weights)
        return np.tile(R_s, self.num_actions)
        



def get_state_values(occupancy_frequencies, mdp_env):
    num_states, gamma = mdp_env.num_states, mdp_env.gamma
    r_sa = mdp_env.get_state_action_rewards()
    #get optimal stochastic policy
    stochastic_policy = utils.get_optimal_policy_from_usa(occupancy_frequencies, mdp_env)
    
    reward_policy = get_policy_rewards(stochastic_policy, r_sa)
    transitions_policy = get_policy_transitions(stochastic_policy, mdp_env)
    A = np.eye(num_states) - gamma * transitions_policy 
    b = reward_policy
    #solve for value function
    state_values = np.linalg.solve(A, b)

    return state_values
    

def get_q_values(occupancy_frequencies, mdp_env):
    num_actions, gamma = mdp_env.num_actions, mdp_env.gamma
    r_sa = mdp_env.get_state_action_rewards()
    #get state values
    state_values = get_state_values(occupancy_frequencies, mdp_env)
    #get state-action values
    Ps = tuple(mdp_env.Ps[i] for i in range(num_actions))
    P_column = np.concatenate(Ps, axis=0)
    #print(P_column)
    q_values = r_sa + gamma * np.dot(P_column, state_values)
    return q_values
    

def solve_mdp_lp(mdp_env, reward_sa=None, debug=False):
    '''method that uses Linear programming to solve MDP
        if reward_sa is not None, then it uses reward_sa in place of mdp_env.r_sa
    
    '''

    I_s = np.eye(mdp_env.num_states)
    gamma = mdp_env.gamma

    if mdp_env.num_actions == 4:
        A_eq = np.concatenate((I_s - gamma * mdp_env.P_left.transpose(),
                        I_s - gamma * mdp_env.P_right.transpose(),
                        I_s - gamma * mdp_env.P_up.transpose(),
                        I_s - gamma * mdp_env.P_down.transpose()),axis =1)
    else:
        A_eq = np.concatenate((I_s - gamma * mdp_env.P_left.transpose(),
                        I_s - gamma * mdp_env.P_right.transpose()),axis =1)
    b_eq = mdp_env.init_dist
    if reward_sa is not None:
        c = -1.0 * reward_sa  #we want to maximize r_sa^T c so make it negative since scipy minimizes by default
    else:
        c = -1.0 * mdp_env.r_sa  #we want to maximize r_sa^T c so make it negative since scipy minimizes by default

    sol = linprog(c, A_eq=A_eq, b_eq = b_eq)
    #minimize:
    #c @ x
    #such that:
    #A_ub @ x <= b_ub
    #A_eq @ x == b_eq
    #all variables are non-negative by default
    #print(sol)

    if debug: print("expeced value MDP LP", -sol['fun'])  #need to negate the value to get the maximum
    #print("state_action occupancies", sol['x'])
    u_sa = sol['x'] 

    #print("expected value dot product", np.dot(u_sa, mdp_env.r_sa))
    #calculate the optimal policy
    return u_sa

def solve_max_cvar_policy(mdp_env, u_expert, R, p_R, alpha):
    '''input mdp_env: the mdp
        u_expert: the state-action occupancies of the expert
        R: a matrix with each column a reward hypothesis
        p_R: a posterior probability mass function over the reward hypotheses
        alpha: the risk sensitivity, higher is more conservative. We look at the (1-alpha)*100% average worst-case

        returns the occupancy frequencies of the policy optimal wrt cvar
    '''
    num_states, num_actions, gamma = mdp_env.num_states, mdp_env.num_actions, mdp_env.gamma
    _,n = R.shape  #k is dimension of reward function and n is the number of samples in the posterior
    #get number of state-action occupancies
    k = mdp_env.num_states * mdp_env.num_actions
    posterior_probs = p_R
    #new objective is 
    #max \sigma - 1/(1-\alpha) * p^T z for vector of auxiliary variables z.

    #so the decision variables are (in the following order) all the u(s,a) and sigma, and all the z's.

    #we want to maximize so take the negative of this vector and minimize via scipy 
    c_cvar = -1. * np.concatenate((np.zeros(num_states * num_actions), #for the u(s,a)'s not in objective any more
                        np.ones(1),                 #for sigma
                        -1.0/(1.0 - alpha) * posterior_probs))  #for the auxiliary variables z

    #constraints: for each of the auxiliary variables we have a constraint >=0 and >= the stuff inside the ReLU

    #create constraint for each auxiliary variable should have |R| + 1 (for sigma) + n (for samples) columns 
    # and n rows (one for each z variable)
    auxiliary_constraints = np.zeros((n, k + 1 + n))
    for i in range(n):
        z_part = np.zeros(n)
        z_part[i] = -1.0 #make the part for the auxiliary variable >= the part in the relu
        z_row = np.concatenate((-R[:,i],  #-R_i(s,a)'s
                                np.ones(1),    #sigma
                                z_part))
        auxiliary_constraints[i,:] = z_row

    #add the upper bounds for these constraints:
    auxiliary_b = -1. * np.dot(R.transpose(), u_expert)

    #add the non-negativitity constraints for the vars u(s,a) and z(R). 
    #mu's greater than or equal to zero
    auxiliary_u_geq0 = -np.eye(k, M=k+1+n)  #negative since constraint needs to be Ax<=b
    auxiliary_bu_geq0 = np.zeros(k)

    auxiliary_z_geq0 = np.concatenate((np.zeros((n, k+1)), -np.eye(n)), axis=1)
    auxiliary_bz_geq0 = np.zeros(n)

    #don't forget the normal MDP constraints over the mu(s,a) terms
    I_s = np.eye(num_states)
    if mdp_env.num_actions == 4:
        A_eq = np.concatenate(( I_s - gamma * mdp_env.P_left.transpose(),
                                I_s - gamma * mdp_env.P_right.transpose(),
                                I_s - gamma * mdp_env.P_up.transpose(),
                                I_s - gamma * mdp_env.P_down.transpose()),axis =1)
    else:
        A_eq = np.concatenate((I_s - gamma * mdp_env.P_left.transpose(),
                               I_s - gamma * mdp_env.P_right.transpose()),axis =1)
    b_eq = mdp_env.init_dist
    A_eq_plus = np.concatenate((A_eq, np.zeros((mdp_env.num_states,1+n))), axis=1)  #add zeros for sigma and the auxiliary z's

    A_cvar = np.concatenate((auxiliary_constraints,
                            auxiliary_u_geq0,
                            auxiliary_z_geq0), axis=0)
    b_cvar = np.concatenate((auxiliary_b, auxiliary_bu_geq0, auxiliary_bz_geq0))

    #solve the LP
    sol = linprog(c_cvar, A_eq=A_eq_plus, b_eq = b_eq, A_ub=A_cvar, b_ub = b_cvar, bounds=(None, None)) #TODO:might be good to explicitly make the bounds here rather than via constraints...
    print("solution to optimizing CVaR")
    print(sol)
    cvar = -sol['fun'] #take negative since we minimized negative CVaR but really wanted to maximize positive CVaR
    #the solution of the LP corresponds to the CVaR
    var_sigma = sol['x'][k] #get sigma (this is VaR (at least close))
    cvar_opt_usa = sol['x'][:k]
    print("CVaR = ", cvar)
    print("policy u(s,a) = ", cvar_opt_usa)
    cvar_opt_stoch_pi = utils.get_optimal_policy_from_usa(cvar_opt_usa, mdp_env)
    print("CVaR opt stochastic policy")
    print(cvar_opt_stoch_pi)

    policy_losses = np.dot(R.transpose(), cvar_opt_usa - u_expert)
    print("policy losses:", policy_losses)
    print("expert returns:", np.dot(R.transpose(), u_expert))
    print("my returns:", np.dot(R.transpose(), cvar_opt_usa))

    return cvar_opt_usa


def solve_minCVaR_reward(mdp_env, u_expert, R, p_R, alpha):
    '''
    Solves the dual problem
      input:
        mdp_env: the mdp
        u_expert: the state-action occupancies of the expert
        R: a matrix with each column a reward hypothesis
        p_R: a posterior probability mass function over the reward hypotheses
        alpha: the risk sensitivity, higher is more conservative. We look at the (1-alpha)*100% average worst-case

       output:
        The adversarial reward and the q weights on the reward posterior. Optimizing for this reward should yield the CVaR optimal policy

    '''
    num_states, num_actions, gamma = mdp_env.num_states, mdp_env.num_actions, mdp_env.gamma
    p0 = mdp_env.init_dist
    k,n = R.shape  #k is dimension of reward function and n is the number of samples in the posterior
    posterior_probs = p_R
    #objective is min p_0^Tv - u_E^T R q

    #the decision variables are (in the following order) q (an element for each reward in the posterior) and v(s) for all s
    #coefficients on objective
    c_q = np.concatenate((np.dot(-R.transpose(), u_expert), p0))  #for the auxiliary variables z

    #constraints: 

    #sum of q's should equal 1
    A_eq = np.concatenate((np.ones((1,n)), np.zeros((1,num_states))), axis = 1)
    b_eq = np.ones(1)
    
    #leq constraints

    #first do the q <= 1/(1-alpha) p
    A_q_leq_p = np.concatenate((np.eye(n), np.zeros((n, num_states))), axis=1)
    b_q_leq_p = 1.0/(1 - alpha) * p_R

    #next do the value iteration equations
    I_s = np.eye(num_states)
    if mdp_env.num_actions == 4:
        trans_dyn = np.concatenate(( I_s - gamma * mdp_env.P_left,
                                I_s - gamma * mdp_env.P_right,
                                I_s - gamma * mdp_env.P_up,
                                I_s - gamma * mdp_env.P_down), axis=0)
    else:
        trans_dyn = np.concatenate((I_s - gamma * mdp_env.P_left,
                               I_s - gamma * mdp_env.P_right), axis=0)
    
    A_vi = np.concatenate((R, -trans_dyn), axis=1)
    b_vi = np.zeros(num_states * num_actions)

    #last add constraint that all q >= 0
    A_q_geq_0 = np.concatenate((-np.eye(n), np.zeros((n, num_states))), axis=1)
    b_q_geq_0 = np.zeros(n)

    #stick them all together
    A_leq = np.concatenate((A_q_leq_p,
                            A_vi,
                            A_q_geq_0), axis=0)
    b_geq = np.concatenate((b_q_leq_p, b_vi, b_q_geq_0))

    #solve the LP
    sol = linprog(c_q, A_eq=A_eq, b_eq=b_eq, A_ub=A_leq, b_ub=b_geq, bounds=(None, None)) #TODO:might be good to explicitly make the bounds here rather than via constraints...
    print("solution to optimizing for CVaR reward")
    print(sol)
    cvar = sol['fun'] #I think the objective value should be the same?
    #the solution of the LP corresponds to the CVaR
    q = sol['x'][:n] #get sigma (this is VaR (at least close))
    values = sol['x'][n:]
    print("CVaR = ", cvar)
    print("policy v(s) under Rq = ", values)
    print("expected value", np.dot(mdp_env.init_dist, values))
    
    print("q weights:", q)
    cvar_reward_fn = np.dot(R,q)
    print("CVaR reward Rq =", cvar_reward_fn)

    return cvar_reward_fn, q


def get_policy_rewards(stoch_pi, rewards_sa):
    num_states, num_actions = stoch_pi.shape
    policy_rewards = np.zeros(num_states)
    for s, a_probs in enumerate(stoch_pi):
        expected_reward = 0.0
        for a, prob in enumerate(a_probs):
            index = s + num_states * a
            expected_reward += prob * rewards_sa[index]
        policy_rewards[s] = expected_reward
    return policy_rewards


#TODO: might be able to vectorize this and speed it up!
def get_policy_transitions(stoch_pi, mdp_env):
    num_states, num_actions = mdp_env.num_states, mdp_env.num_actions
    P_pi = np.zeros((num_states, num_states))
    #calculate expectations
    for s1 in range(num_states):
        for s2 in range(num_states):
            cum_prob = 0.0
            for a in range(num_actions):
                cum_prob += stoch_pi[s1,a] * mdp_env.get_transition_prob(s1,a,s2)
            P_pi[s1,s2] =  cum_prob
    return P_pi


def two_by_two_mdp():
    num_rows = 2
    num_cols = 2
    r_s = np.array( [-1, -10, -1, 1] )
    gamma = 0.5
    init_dist = np.array([0.5,0.5,0,0])
    mdp = BasicGridMDP(num_rows, num_cols, r_s, gamma, init_dist)
    u_sa_opt = solve_mdp_lp(mdp)
    print(u_sa_opt)
    print("u[state,actions]=")
    print(np.reshape(u_sa_opt, (num_rows*num_cols, 4)).transpose())
    pi_opt = utils.get_optimal_policy_from_usa(u_sa_opt, mdp)
    print("optimal policy")
    print("pi(a|s) for left, right, up, down")
    print(pi_opt)


def two_state_chain():
        
    num_states = 2
    num_actions = 2
    gamma = 0.5
    p0 = np.array([0.5,0.5])
    r_sa = np.array([1,-1, +1, -1])
    chain = ChainMDP(num_states, r_sa, gamma, p0)
    #u_sa_opt = solve_mdp_lp(chain)  #np.zeros(np.shape(r_sa))
    u_sa_opt = np.zeros(np.shape(r_sa))
    print(u_sa_opt)
    print("u[state,actions]=")
    print(np.reshape(u_sa_opt, (num_states, num_actions)).transpose())
    pi_opt = utils.get_optimal_policy_from_usa(u_sa_opt, chain)
    print("optimal policy")
    print("pi(a|s) for left, right")
    print(pi_opt)
    print('optimizing CVAR policy')
    R1 = np.array([0., +1., 0., 0.])
    R2 = np.array([0.5, 0, 0.5, 0])
    R3 = np.array([0.5, 0.5, 0., 0.])
    R = np.vstack((R1,R2,R3)).transpose()  #each reward hypothesis is a column so stack as rows then transpose
    print("reward posterior")
    print(R)
    k,n = R.shape  #k is dimension of reward function and n is the number of samples in the posterior
    posterior_probs = np.array([1/3, 1/3, 1/3])
    #we need to add auxiliary variables to get the ReLU in the objective.

    alpha = 0.5 #TODO:change back  #(1-alpha)% average worst-case. 

    cvar_opt_usa = solve_max_cvar_policy(chain, u_sa_opt, R, posterior_probs, alpha)

    u_expert = u_sa_opt
    policy_losses = np.dot(R.transpose(), cvar_opt_usa - u_expert)




    #TODO: check below to see if it is correct.
    print("let's check the solution")
    #first we need the u^Tri terms for each reward in the posterior. Turns out I already computed them above.

    c_ploss = policy_losses  #multiply by negative since we min in the lp solver

    _,num_rew_samples = R.shape

    #add constraint q <= 1/(1-alpha) * p 
    A = np.eye(num_rew_samples)
    b = 1 / (1-alpha) * posterior_probs 
    #and add constraint that sum q = 1
    A_eq = np.ones((1,num_rew_samples))
    b_eq = np.ones(1)

    #print(A)
    #print(b)
    #solve the LP
    sol = linprog(c_ploss, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq) #use default bounds since they are set to (0,infty)
    print("solving robust form to get cvar worst-case reward")
    print(sol)
    q_star = sol['x']
    print("Cvar robust", sol['fun'])
    R_cvar = np.dot(R, q_star)
    print("CVaR reward is", R_cvar)
    cvar2 = np.dot(cvar_opt_usa, R_cvar) - np.dot(u_expert, R_cvar)
    print("alternative CVaR calculation yields", cvar2)
    cvar3 = np.dot(q_star, np.dot(R.transpose(), cvar_opt_usa - u_expert))
    print("another CVaR calc", cvar3)

    print("solving for optimal policy for CVaR reward")
    new_mdp = ChainMDP(num_states, R_cvar, gamma, p0)
    new_pi = solve_mdp_lp(new_mdp)
    print("new opt policy")
    print(new_pi)
    print(utils.get_optimal_policy_from_usa(new_pi, new_mdp))

    print(utils.print_table_row(np.concatenate((R_cvar, q_star))))

    cvar_r_dual, q_dual = solve_minCVaR_reward(chain, u_expert, R, posterior_probs, alpha)

    #okay so let's check and see what happens if we optimize cvar_r_dual
    dual_reward_mdp = ChainMDP(num_states, cvar_r_dual, gamma, p0)
    dual_opt_pi = solve_mdp_lp(new_mdp)
    print("dual opt policy")
    print(dual_opt_pi)
    print(utils.get_optimal_policy_from_usa(dual_opt_pi, dual_reward_mdp))
    #interesting, this doesn't give the same policy either...I guess this makes sense since we're taking a game and then 
    #we're taking away one player so the other player (in this case the policy optimizer) should get higher value since it can play
    # a best response. Kind of like how the GAIL reward function isn't good by itself but only works when optimized with the agent.

    #The CVaR's match so that's good.

    #does this reward match the reward found by taking 



if __name__ == "__main__":
    two_state_chain()