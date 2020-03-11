# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:39:34 2020

@author: dsbrown
"""


import numpy as np


#baby MDP

#two states, s1, s2 with self transitions and 
#transitoins between 


#let's assume rewards are only a function
#of state for now
num_states = 2
num_actions = 2
gamma = 0.5
p0 = np.array([0.5,0.5])
r = np.array([1,-1, +1, -1])
#policy always goes or stays in first state
P_pi = np.array([[1.0, 0.0],
                 [1.0, 0.0]])
r_pi = np.array([+1, -1])

#state occupancy frequency for policy
u_pi = np.dot(np.linalg.inv(np.eye(num_states) - gamma*P_pi.transpose()),  p0)
print("occupancy frequency for a policy")
print(u_pi)

#state value function of policy
v_pi = np.dot(np.linalg.inv(np.eye(num_states) - gamma * P_pi), r_pi)
print("state value function of policy")
print(v_pi)

#return for policy
rho_pi = np.dot(v_pi, p0)
print('policy expected return', rho_pi)
u_pi_T_r = np.dot(u_pi, r_pi)
print("policy expected return", u_pi_T_r)

#let's use linear programming to solve for optimal policy of the MDP
#we actually need r to be over states and actions, I think
r_sa = np.array([+1., -1., +1., -1.]) #s1-left, s2-left, s1-right, s2-right
P_left = np.array([[1., 0.],
                   [1., 0.]])
P_right = np.array([[0., 1.],
                    [0., 1.]])
I_s = np.eye(num_states)
#use same p0 as before
from scipy.optimize import linprog
A_eq = np.concatenate((I_s - gamma * P_left.transpose(),
                       I_s - gamma * P_right.transpose()), axis =1)
b_eq = p0
c = -1.0 * r_sa  #we want to maximize r_sa^T c so make it negative since scipy minimizes by default

sol = linprog(c, A_eq=A_eq, b_eq = b_eq)
#minimize:
#c @ x
#such that:
#A_ub @ x <= b_ub
#A_eq @ x == b_eq
#all variables are non-negative by default
print(sol)

print("expeced value LP", -sol['fun'])  #need to negate the value to get the maximum
print("state_action occupancies", sol['x'])
u_sa = sol['x'] 

print("expected value dot product", np.dot(u_sa, r_sa))
#calculate the optimal policy
#lp_opt_stochastic_pi = np.array([[u_sa[0]/(u_sa[0] + u_sa[2]), u_sa[2]/(u_sa[0] + u_sa[2])],  #action probs (left, right) for state 1
#                                 [u_sa[1]/(u_sa[1] + u_sa[3]), u_sa[3]/(u_sa[1] + u_sa[3])]]) #action probs (left, right) for state 2
lp_opt_stoch_pi = np.zeros((num_states, num_actions))
for s in range(num_states):
    #compute the total occupancy for that state across all actions
    s_tot_occupancy = np.sum(u_sa[s::num_actions])
    for a in range(num_actions):
        lp_opt_stoch_pi[s][a] = u_sa[s+a*num_actions] / s_tot_occupancy
print("lp opt stochastic policy")
print(lp_opt_stoch_pi)
lp_opt_det_pi = np.argmax(lp_opt_stoch_pi, axis=1)
print("optimal det policy")
print(lp_opt_det_pi) #0-left, 1-right


#okay so that works, cool! Let's try and optimize the CVaR!

#we need expert trajectories/mu's. Let's assume for now that we have mu_expert. TODO: talk to Marek about what to do if we don't...
u_expert = u_sa #assume expert is optimal for now.
#u_expert = np.zeros(u_sa.shape)  #Or just set to zero to be robust to distribution.


#let's assume we have three reward hypotheses
R1 = np.array([0., +1., 0., 0.])
R2 = np.array([0.5, 0, 0.5, 0])
R3 = np.array([0.5, 0.5, 0., 0.])
R = np.vstack((R1,R2,R3)).transpose()  #each reward hypothesis is a column so stack as rows then transpose
print(R)
k,n = R.shape  #k is dimension of reward function and n is the number of samples in the posterior
posterior_probs = np.array([1/3, 1/3, 1/3])
#we need to add auxiliary variables to get the ReLU in the objective.

alpha = 0.95   #(1-alpha% average worst-case. #TODO: check this and make sure it makes sense...

#new objective is 
#max \sigma - 1/(1-\alpha) * p^T z for vector of auxiliary variables z.

#so the variables are all the u(s,a) and sigma, and all the z's.

#we want to maximize so take the negative of this vector and minimize via scipy 
c_cvar = -1. * np.concatenate((np.zeros(len(r_sa)), #for the u(s,a)'s not in objective any more
                    np.ones(1),                 #for sigma
                    -1.0/(1.0 - alpha) * np.ones(n) * posterior_probs))  #for the auxiliary variables z

#constraints: for each of the auxiliary variables we have a constraint >=0 and >= the stuff inside the ReLU

#create constraint for each auxiliary variable should have |R| + 1 (for sigma) + n (for samples) columns 
# and n rows (one for relu part)
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

#add the non-negatvitity constraints for the vars u(s,a) and z(R). 
#mu's greater than or equal to zero
auxiliary_u_geq0 = -np.eye(k, M=k+1+n)  #negative since constraint needs to be Ax<=b
auxiliary_bu_geq0 = np.zeros(k)

auxiliary_z_geq0 = np.concatenate((np.zeros((n, k+1)), -np.eye(n)), axis=1)
auxiliary_bz_geq0 = np.zeros(n)

#don't forget the normal MDP constraints over the mu(s,a) terms
A_eq = np.concatenate((I_s - gamma * P_left.transpose(),
                       I_s - gamma * P_right.transpose()), axis =1)
A_eq_plus = np.concatenate((A_eq, np.zeros((num_states,1+n))), axis=1)  #add zeros for sigma and the auxiliary z's
b_eq = p0

A_cvar = np.concatenate((auxiliary_constraints,
                        auxiliary_u_geq0,
                        auxiliary_z_geq0), axis=0)
b_cvar = np.concatenate((auxiliary_b, auxiliary_bu_geq0, auxiliary_bz_geq0))

#solve the LP
sol = linprog(c_cvar, A_eq=A_eq_plus, b_eq = b_eq, A_ub=A_cvar, b_ub = b_cvar, bounds=(None, None)) #TODO:might be good to explicitly make the bounds here rather than via constraints...
print("solution to optimizing CVaR")
print(sol)
#TODO: figure out what the solution of the LP corresponds to...is the solution CVaR??
cvar_sigma = -1 * sol['x'][k] #get sigma (should be negative since we changed the max to min, rght?)
cvar_opt_usa = sol['x'][:k]
print("cvar = ", cvar_sigma)
print("policy u(s,a) = ", cvar_opt_usa)
cvar_opt_stoch_pi = np.zeros((num_states, num_actions))
for s in range(num_states):
    #compute the total occupancy for that state across all actions
    s_tot_occupancy = np.sum(cvar_opt_usa[s::num_actions])
    for a in range(num_actions):
        cvar_opt_stoch_pi[s][a] = cvar_opt_usa[s+a*num_actions] / s_tot_occupancy
print("lp opt stochastic policy")
print(cvar_opt_stoch_pi)

policy_losses = np.dot(R.transpose(), cvar_opt_usa - u_expert)
print("returns:", policy_losses)


print("let's check the solution")
#first we need the u^Tri terms for each reward in the posterior. Turns out I already computed them above.

c_ploss = -1. * policy_losses  #multiply by negative since we min in the lp solver

_,num_rew_samples = R.shape

#add constraint q <= 1/(1-alpha) * p 
A = np.eye(num_rew_samples)
b = 1 / (1-alpha) * posterior_probs 
#and add constraint that sum q = 1
A = np.concatenate((A,np.ones((1,num_rew_samples))), axis = 0)
b = np.append(b, 1)

print(A)
print(b)
#solve the LP
sol = linprog(c_ploss, A_ub=A, b_ub=b) #use default bounds since they are set to (0,infty)
print("solving robust form to get cvar worst-case reward")
print(sol)
q_star = sol['x']

R_cvar = np.dot(R, q_star)
print("CVaR reward is", R_cvar)
cvar2 = np.dot(cvar_opt_usa, R_cvar) - np.dot(u_expert, R_cvar)
print("alternative CVaR calculation yields", cvar2)
