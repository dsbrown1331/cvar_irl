# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:39:34 2020

@author: dsbrown
"""


import numpy as np
import utils
from scipy.optimize import linprog

class ChainMDP:
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


    def get_transitions(self, policy):
        P_pi = np.zeros((self.num_states, self.num_states))
        if policy == "left":
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
        elif policy == "right":
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
        

class BasicMDP:
    #basic MDP class that has four actions, no terminal states and is a grid with deterministic transitions
    def __init__(self, num_rows, num_cols, r_s, gamma, init_dist):
        self.num_actions = 4
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_states =  num_rows * num_cols
        self.gamma = gamma
        self.init_dist = init_dist
        self.r_s = r_s
        self.r_sa = self.transform_to_R_sa(self.r_s, self.num_actions)
        print("transformed R(s,a)", self.r_sa)

        self.P_left = self.get_transitions(policy="left")
        #print("P_left\n",self.P_left)
        self.P_right = self.get_transitions(policy="right")
        #print("P_right\n",self.P_right)
        self.P_up = self.get_transitions(policy="up")
        #print("_up\n",self.P_up)
        self.P_down = self.get_transitions(policy="down")
        #print("P_down\n",self.P_down)

    #transform R(s) into R(s,a) for use in LP
    def transform_to_R_sa(self, R_s, num_actions):
        '''input: numpy array R_s, output R_sa'''
        print(len(R_s))
        print(self.num_states)
        #just repeat values since R_sa = [R(s1,a1), R(s2,a1),...,R(sn,a1), R(s1,a2), R(s2,a2),..., R(sn,am)]
        assert(len(R_s) == self.num_states)
        return np.tile(R_s, num_actions)

    def get_transitions(self, policy):
        P_pi = np.zeros((self.num_states, self.num_states))
        if policy == "left":
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
        elif policy == "right":
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
        elif policy == "up":
            #always transition one to left unless already at left border
            cnt = 0
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    if r > 0:
                        P_pi[cnt, cnt - self.num_cols] = 1.0
                    else:
                        P_pi[cnt,cnt] = 1.0
                    #increment state count
                    cnt += 1
        elif policy == "down":
            #always transition one to left unless already at left border
            cnt = 0
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    if r < self.num_rows - 1:
                        P_pi[cnt, cnt + self.num_cols] = 1.0
                    else:
                        P_pi[cnt,cnt] = 1.0
                    #increment state count
                    cnt += 1
        return P_pi


def solve_mdp_lp(mdp_env):
    '''method that uses Linear programming to solve MDP'''
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
    c = -1.0 * mdp_env.r_sa  #we want to maximize r_sa^T c so make it negative since scipy minimizes by default

    sol = linprog(c, A_eq=A_eq, b_eq = b_eq)
    #minimize:
    #c @ x
    #such that:
    #A_ub @ x <= b_ub
    #A_eq @ x == b_eq
    #all variables are non-negative by default
    #print(sol)

    print("expeced value MDP LP", -sol['fun'])  #need to negate the value to get the maximum
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
    '''
    num_states, num_actions, gamma = mdp_env.num_states, mdp_env.num_actions, mdp_env.gamma
    k,n = R.shape  #k is dimension of reward function and n is the number of samples in the posterior
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
    auxiliary_b = np.dot(R.transpose(), u_expert)

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
    print("returns:", np.dot(R.transpose(), cvar_opt_usa))
    return cvar_opt_usa


def two_by_two_mdp():
    num_rows = 2
    num_cols = 2
    r_s = np.array( [-1, -10, -1, 1] )
    gamma = 0.5
    init_dist = np.array([0.5,0.5,0,0])
    mdp = BasicMDP(num_rows, num_cols, r_s, gamma, init_dist)
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
    u_sa_opt = solve_mdp_lp(chain)
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
    print(R)
    k,n = R.shape  #k is dimension of reward function and n is the number of samples in the posterior
    posterior_probs = np.array([1/3, 1/3, 1/3])
    #we need to add auxiliary variables to get the ReLU in the objective.

    alpha = 0.5   #(1-alpha% average worst-case. #TODO: check this and make sure it makes sense...

    cvar_opt_usa = solve_max_cvar_policy(chain, u_sa_opt, R, posterior_probs, alpha)

    u_expert = u_sa_opt
    policy_losses = np.dot(R.transpose(), cvar_opt_usa - u_expert)




    #TODO: check below to see if it is correct.
    print("let's check the solution")
    #first we need the u^Tri terms for each reward in the posterior. Turns out I already computed them above.

    c_ploss = -1. * policy_losses  #multiply by negative since we min in the lp solver

    _,num_rew_samples = R.shape

    #add constraint q <= 1/(1-alpha) * p 
    A = np.eye(num_rew_samples)
    b = 1 / (1-alpha) * posterior_probs 
    #and add constraint that sum q = 1
    A_eq = np.ones((1,num_rew_samples))
    b_eq = np.ones(1)

    print(A)
    print(b)
    #solve the LP
    sol = linprog(c_ploss, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq) #use default bounds since they are set to (0,infty)
    print("solving robust form to get cvar worst-case reward")
    print(sol)
    q_star = sol['x']

    R_cvar = np.dot(R, q_star)
    print("CVaR reward is", R_cvar)
    cvar2 = np.dot(cvar_opt_usa, R_cvar) - np.dot(u_expert, R_cvar)
    print("alternative CVaR calculation yields", cvar2)

if __name__ == "__main__":
    two_state_chain()