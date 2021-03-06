import mdp
import numpy as np
import utils


def windy_cliff_world_small(slip_prob):
    #trying to replicate the ambiguous lava domain from Dylan's IRD paper figure 2 (a)
    #four features, 10x10 grid

    #trying with less grass and fewer starting states
    #setting all starting states to be dirt

    num_rows = 3
    num_cols = 3
    num_states = num_rows * num_cols
    #four types of terrain
    num_features = 2
    d = (1,0)
    c = (0,1)
    #create one-hot features
    f_vecs = np.eye(num_features)
    features = [tuple(f) for f in f_vecs]


    state_features = np.array([#d,d,d,
                               d,d,d,
                               d,d,d,
                               d,c,d])
    weights = np.array([-1,0])
    
    gamma = 0.99
    init_dist = np.zeros(num_states)
    init_state = num_cols * (num_rows - 1)
    # init_states = [init_state]
    # for si in init_states:
    #     state_features[si] = d
    #     init_dist[si] = 1.0 / len(init_states)
    term_states = [num_states - 1] #no terminal
    init_dist = 1/(num_states) * np.ones(num_states)
    print("init states", init_dist)
    print("term states", term_states)
    
    mdp_env = mdp.FeaturizedGridMDP(num_rows, num_cols, state_features, weights, gamma, init_dist, term_states)

    #set transition dynamics
    mdp_env.Ps = mdp.get_windy_down_const_prob_transitions(mdp_env, slip_prob)

    #set transitions for cliff to go to back to init state
    for s in range(num_states):
        if (state_features[s] == c).all():
            for P in mdp_env.Ps:
                P[s,:] = np.zeros(num_states)
                P[s,init_state] = 1.


    # for i,P in enumerate(mdp_env.Ps):
    #     print("action dir " + mdp_env.get_readable_actions(i))
    #     print(P)

    return mdp_env


def windy_cliff_world_large(slip_prob):
    #trying to replicate the ambiguous lava domain from Dylan's IRD paper figure 2 (a)
    #four features, 10x10 grid

    #trying with less grass and fewer starting states
    #setting all starting states to be dirt

    num_rows = 6
    num_cols = 7
    num_states = num_rows * num_cols
    #four types of terrain
    num_features = 3
    d = (1,0,0)
    c = (0,1,0)
    t = (0,0,1)
    #create one-hot features
    f_vecs = np.eye(num_features)
    features = [tuple(f) for f in f_vecs]


    state_features = np.array([d,d,d,d,d,d,d,
                               d,d,d,d,d,d,d,
                               d,d,d,d,d,d,d,
                               d,d,d,d,d,d,d,
                               d,d,d,d,d,d,d,
                               d,c,c,c,c,t,d])
    weights = np.array([-1,-100,+1])
    
    gamma = 0.99
    init_dist = np.zeros(num_states)
    init_state = num_cols * (num_rows - 1)
    # init_states = [init_state]
    # for si in init_states:
    #     state_features[si] = d
    #     init_dist[si] = 1.0 / len(init_states)
    #term_states = [num_states - 1] #no terminal
    term_states = []
    init_dist = 1/(num_states) * np.ones(num_states)
    mdp_env = mdp.FeaturizedGridMDP(num_rows, num_cols, state_features, weights, gamma, init_dist, term_states)

    #set transition dynamics
    mdp_env.Ps = mdp.get_windy_down_const_prob_transitions(mdp_env, slip_prob)

    #set transitions for cliff to go to back to init state
    for s in range(num_states):
        if (state_features[s] == c).all():
            for P in mdp_env.Ps:
                P[s,:] = np.zeros(num_states)
                P[s,init_state] = 1.


    # for i,P in enumerate(mdp_env.Ps):
    #     print("action dir " + mdp_env.get_readable_actions(i))
    #     print(P)

    return mdp_env


def windy_cliff_world(slip_prob):
    #trying to replicate the ambiguous lava domain from Dylan's IRD paper figure 2 (a)
    #four features, 10x10 grid

    #trying with less grass and fewer starting states
    #setting all starting states to be dirt

    num_rows = 5
    num_cols = 5
    num_states = num_rows * num_cols
    #four types of terrain
    num_features = 3
    d = (1,0,0)
    c = (0,1,0)
    t = (0,0,1)
    #create one-hot features
    f_vecs = np.eye(num_features)
    features = [tuple(f) for f in f_vecs]


    state_features = np.array([d,d,d,d,d,
                               d,d,d,d,d,
                               d,d,d,d,d,
                               d,d,d,d,d,
                               d,c,c,c,t])
    weights = np.array([-1,-100,+1])
    
    gamma = 0.99
    init_dist = np.zeros(num_states)
    init_state = num_cols * (num_rows - 1)
    # init_states = [init_state]
    # for si in init_states:
    #     state_features[si] = d
    #     init_dist[si] = 1.0 / len(init_states)
    #term_states = [num_states - 1] #no terminal
    term_states = []
    init_dist = 1/(num_states) * np.ones(num_states)
    mdp_env = mdp.FeaturizedGridMDP(num_rows, num_cols, state_features, weights, gamma, init_dist, term_states)

    #set transition dynamics
    mdp_env.Ps = mdp.get_windy_down_const_prob_transitions(mdp_env, slip_prob)

    #set transitions for cliff to go to back to init state
    for s in range(num_states):
        if (state_features[s] == c).all():
            for P in mdp_env.Ps:
                P[s,:] = np.zeros(num_states)
                P[s,init_state] = 1.


    # for i,P in enumerate(mdp_env.Ps):
    #     print("action dir " + mdp_env.get_readable_actions(i))
    #     print(P)

    return mdp_env



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
    mdp_env = mdp.FeaturizedGridMDP(num_rows, num_cols, state_features, weights, gamma, init_dist, terminals, False)
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
    weights = np.array([-0.18, -0.82])#np.array([-0.26750391, -0.96355677])#np.array([-.18, -.82])
    weights = weights / np.linalg.norm(weights)
    print(weights)
    gamma = 0.95
    init_dist = np.zeros(num_states)
    init_states = [5,4]
    for si in init_states:
        init_dist[si] = 1.0 / len(init_states)
    term_states = [12]
    init_dist = 1/(num_states) * np.ones(num_states)
    mdp_env = mdp.FeaturizedGridMDP(num_rows, num_cols, state_features, weights, gamma, init_dist, term_states)
    return mdp_env


def lava_ambiguous_corridor():
    num_rows = 3
    num_cols = 5
    num_states = num_rows * num_cols
    white = (1,0)
    red = (0,1)
    state_features = np.array([white, white, white, white, white,
                               white, red, red, red, red,
                               white, white, white, white, white])
    weights = np.array([-0.1, -0.9])#np.array([-0.26750391, -0.96355677])#np.array([-.18, -.82])
    weights = weights / np.linalg.norm(weights)
    print(weights)
    gamma = 0.99
    init_dist = np.zeros(num_states)
    # init_states = [5,4]
    # for si in init_states:
    #     init_dist[si] = 1.0 / len(init_states)
    term_states = [14]
    init_dist = 1/(num_states) * np.ones(num_states)
    mdp_env = mdp.FeaturizedGridMDP(num_rows, num_cols, state_features, weights, gamma, init_dist, term_states)
    return mdp_env

def lava_ambiguous_corridor3():
    num_rows = 5
    num_cols = 5
    num_states = num_rows * num_cols
    white = (1,0)
    red = (0,1)
    state_features = np.array([white, white, white, white, white,
                               white, red, red, red, red,
                               white, red, red, red, red,
                               white, red, red, red, red,
                               white, white, white, white, white])
    weights = np.array([-0.1, -0.9])#np.array([-0.26750391, -0.96355677])#np.array([-.18, -.82])
    weights = weights / np.linalg.norm(weights)
    print(weights)
    gamma = 0.99
    init_dist = np.zeros(num_states)
    # init_states = [5,4]
    # for si in init_states:
    #     init_dist[si] = 1.0 / len(init_states)
    term_states = [24]
    init_dist = 1/(num_states) * np.ones(num_states)
    mdp_env = mdp.FeaturizedGridMDP(num_rows, num_cols, state_features, weights, gamma, init_dist, term_states)
    return mdp_env



def lava_ambiguous_corridor2():
    num_rows = 3
    num_cols = 5
    num_states = num_rows * num_cols
    white = (1,0,0)
    red = (0,1,0)
    green = (0,0,1)
    state_features = np.array([white, white, white, white, white,
                               white, red, red, red, red,
                               white, white, white, white, green])
    weights = np.array([-1, -9, +1])#np.array([-0.26750391, -0.96355677])#np.array([-.18, -.82])
    #weights = weights / np.linalg.norm(weights)
    print(weights)
    gamma = 0.99
    init_dist = np.zeros(num_states)
    # init_states = [5,4]
    # for si in init_states:
    #     init_dist[si] = 1.0 / len(init_states)
    term_states = []
    init_dist = 1/(num_states) * np.ones(num_states)
    mdp_env = mdp.FeaturizedGridMDP(num_rows, num_cols, state_features, weights, gamma, init_dist, term_states)
    return mdp_env



def lava_ambiguous_ird_fig2a():
    #trying to replicate the ambiguous lava domain from Dylan's IRD paper figure 2 (a)
    #four features, 10x10 grid

    num_rows = 10
    num_cols = 10
    num_states = num_rows * num_cols
    d = (1,0,0,0)
    g = (0,1,0,0)
    t = (0,0,1,0)
    l = (0,0,0,1)
    state_features = np.array([d,d,g,g,g,g,g,g,d,d,
                               d,d,g,g,g,g,g,g,d,d,
                               d,d,g,g,g,g,g,g,d,d,
                               d,d,g,g,g,g,g,g,d,d,
                               d,d,d,g,g,g,g,d,d,d,
                               d,d,d,g,g,g,g,d,t,d,
                               d,d,d,g,g,g,g,d,d,d,
                               d,d,d,d,g,g,d,d,d,d,
                               d,d,d,d,g,g,d,d,d,d,
                               d,d,d,d,g,g,d,d,d,d])
    weights = np.array([-1,-5,+1,-100])
    #weights = weights / np.linalg.norm(weights)
    print(weights)
    gamma = 0.95
    init_dist = np.zeros(num_states)
    init_states = [51]
    for si in init_states:
        init_dist[si] = 1.0 / len(init_states)
    term_states = [58]
    init_dist = 1/(num_states) * np.ones(num_states)
    mdp_env = mdp.FeaturizedGridMDP(num_rows, num_cols, state_features, weights, gamma, init_dist, term_states)
    return mdp_env


def lava_ambiguous_ird_fig2b():
    #trying to replicate the ambiguous lava domain from Dylan's IRD paper figure 2 (a)
    #four features, 10x10 grid

    num_rows = 10
    num_cols = 10
    num_states = num_rows * num_cols
    d = (1,0,0,0)
    g = (0,1,0,0)
    t = (0,0,1,0)
    l = (0,0,0,1)
    state_features = np.array([d,d,g,g,g,g,g,g,d,d,
                               d,d,g,g,g,g,g,g,d,d,
                               d,d,g,g,g,g,g,g,d,d,
                               d,d,g,g,g,g,g,g,d,d,
                               d,d,d,l,l,l,l,d,d,d,
                               d,d,d,l,l,l,l,d,t,d,
                               d,d,d,l,l,l,l,d,d,d,
                               d,d,d,d,g,g,d,d,d,d,
                               d,d,d,d,g,g,d,d,d,d,
                               d,d,d,d,g,g,d,d,d,d])
    weights = np.array([-1,-5,+1,-100])
    #weights = weights / np.linalg.norm(weights)
    print(weights)
    gamma = 0.95
    init_dist = np.zeros(num_states)
    init_states = [51]
    for si in init_states:
        init_dist[si] = 1.0 / len(init_states)
    term_states = [58]
    init_dist = 1/(num_states) * np.ones(num_states)
    mdp_env = mdp.FeaturizedGridMDP(num_rows, num_cols, state_features, weights, gamma, init_dist, term_states)
    return mdp_env


def lava_ird_simplified_a():
    #trying to replicate the ambiguous lava domain from Dylan's IRD paper figure 2 (a)
    #four features, 10x10 grid

    num_rows = 5
    num_cols = 5
    num_states = num_rows * num_cols
    d = (1,0,0,0)
    g = (0,1,0,0)
    t = (0,0,1,0)
    l = (0,0,0,1)
    state_features = np.array([d,g,g,g,d,
                               d,g,g,g,d,
                               d,g,g,g,t,
                               d,g,g,g,d,
                               d,d,g,d,d])
    weights = np.array([-1,-5,+1,-100])
    #weights = weights / np.linalg.norm(weights)
    print(weights)
    gamma = 0.95
    init_dist = np.zeros(num_states)
    # init_states = [10]
    # for si in init_states:
    #     init_dist[si] = 1.0 / len(init_states)
    term_states = []#[14]
    init_dist = 1/(num_states) * np.ones(num_states)
    mdp_env = mdp.FeaturizedGridMDP(num_rows, num_cols, state_features, weights, gamma, init_dist, term_states)
    return mdp_env


def lava_ird_corner_a():
    #trying to replicate the ambiguous lava domain from Dylan's IRD paper figure 2 (a)
    #four features, 10x10 grid

    num_rows = 2
    num_cols = 2
    num_states = num_rows * num_cols
    d = (1,0,0,0)
    g = (0,1,0,0)
    t = (0,0,1,0)
    l = (0,0,0,1)
    state_features = np.array([d,d,
                               g,t])
    weights = np.array([-1,-5,+1,-100])
    #weights = weights / np.linalg.norm(weights)
    print(weights)
    gamma = 0.95
    init_dist = np.zeros(num_states)
    init_states = [0]
    for si in init_states:
        init_dist[si] = 1.0 / len(init_states)
    term_states = []#[14]
    # init_dist = 1/(num_states) * np.ones(num_states)
    mdp_env = mdp.FeaturizedGridMDP(num_rows, num_cols, state_features, weights, gamma, init_dist, term_states)
    return mdp_env

def lava_ird_corner_b():
    #trying to replicate the ambiguous lava domain from Dylan's IRD paper figure 2 (a)
    #four features, 10x10 grid

    num_rows = 2
    num_cols = 2
    num_states = num_rows * num_cols
    d = (1,0,0,0)
    g = (0,1,0,0)
    t = (0,0,1,0)
    l = (0,0,0,1)
    state_features = np.array([d,l,
                               g,t])
    weights = np.array([-1,-5,+1,-100])
    #weights = weights / np.linalg.norm(weights)
    print(weights)
    gamma = 0.95
    init_dist = np.zeros(num_states)
    init_states = [0]
    for si in init_states:
        init_dist[si] = 1.0 / len(init_states)
    term_states = []#[14]
    # init_dist = 1/(num_states) * np.ones(num_states)
    mdp_env = mdp.FeaturizedGridMDP(num_rows, num_cols, state_features, weights, gamma, init_dist, term_states)
    return mdp_env



def lava_ird_simplified_b():
    #trying to replicate the ambiguous lava domain from Dylan's IRD paper figure 2 (a)
    #four features, 10x10 grid

    num_rows = 5
    num_cols = 5
    num_states = num_rows * num_cols
    d = (1,0,0,0)
    g = (0,1,0,0)
    t = (0,0,1,0)
    l = (0,0,0,1)
    state_features = np.array([d,g,g,g,d,
                               d,g,g,g,d,
                               d,l,l,l,t,
                               d,l,l,l,d,
                               d,d,g,d,d])
    weights = np.array([-1,-5,+1,-100])
    #weights = weights / np.linalg.norm(weights)
    print(weights)
    gamma = 0.95
    init_dist = np.zeros(num_states)
    # init_states = [10]
    # for si in init_states:
    #     init_dist[si] = 1.0 / len(init_states)
    term_states = []#[14]
    init_dist = 1/(num_states) * np.ones(num_states)
    mdp_env = mdp.FeaturizedGridMDP(num_rows, num_cols, state_features, weights, gamma, init_dist, term_states)
    return mdp_env


def lavaland_small(contains_lava=False):
    #trying to replicate the ambiguous lava domain from Dylan's IRD paper figure 2 (a)
    #four features, 10x10 grid

    num_rows = 6
    num_cols = 6
    num_states = num_rows * num_cols
    #four types of terrain
    num_features = 4
    d = (1,0,0,0)
    g = (0,1,0,0)
    t = (0,0,1,0)
    l = (0,0,0,1)
    #create one-hot features
    f_vecs = np.eye(num_features)
    features = [tuple(f) for f in f_vecs]

    state_features = []
    for i in range(num_states):
        #select all but last state randomly from all but last feature, last feature is target feature
        if not contains_lava:
            #add lava randomly but at lower percentage
            p = [0.8,0.2,0.0,0.0]  #add dirt and grass in equal proportions
        else:
            p = [0.6,0.2,0,0.2]  #add dirt and grass and lava in equal proportions
        r_idx = np.random.choice(4,p=p) #sample from four features with probabilities p
        # r_idx = np.random.randint(num_features - 1)
        state_features.append(features[r_idx])
    #make a random state the target
    state_features[np.random.randint(num_states)] = t

    state_features = np.array(state_features)
    weights = np.array([-1,-5,+1,-100])
    #weights = weights / np.linalg.norm(weights)
    print(weights)
    gamma = 0.95
    # init_dist = np.zeros(num_states)
    # init_states = [0]
    # for si in init_states:
    #     init_dist[si] = 1.0 / len(init_states)
    term_states = [] #no terminal
    init_dist = 1/(num_states) * np.ones(num_states)
    mdp_env = mdp.FeaturizedGridMDP(num_rows, num_cols, state_features, weights, gamma, init_dist, term_states)
    return mdp_env


def negative_sideeffects_small(contains_lava=False):
    #inspired by ambiguous lava domain from Dylan's IRD paper figure 2 (a) and also by Jessie Huang and Doinna's paper 
    #has two features that are terrain types, a goal, and a "lava" feature that is unobserved in training but observed at testing

    num_rows = 6
    num_cols = 6
    num_states = num_rows * num_cols
    #four types of terrain
    num_features = 4
    d = (1,0,0,0)
    g = (0,1,0,0)
    t = (0,0,1,0)
    l = (0,0,0,1)
    #create one-hot features
    f_vecs = np.eye(num_features)
    features = [tuple(f) for f in f_vecs]

    state_features = []
    for i in range(num_states):
        #select all but last state randomly from all but last feature, last feature is target feature
        if not contains_lava:
            #add lava randomly but at lower percentage
            p = [0.5,0.5,0.0,0.0]  #add dirt and grass in equal proportions
        else:
            p = [0.4,0.4,0,0.2]  #add dirt and grass equally but add in some lava now
        r_idx = np.random.choice(4,p=p) #sample from four features with probabilities p
        # r_idx = np.random.randint(num_features - 1)
        state_features.append(features[r_idx])
    #make a random state the target
    goal_state = np.random.randint(num_states)
    state_features[goal_state] = t

    state_features = np.array(state_features)
    weights = np.array([-np.random.rand(),-np.random.rand(),+5,-5])
    weights = weights / np.linalg.norm(weights)
    print(weights)
    gamma = 0.95
    # init_dist = np.zeros(num_states)
    # init_states = [0]
    # for si in init_states:
    #     init_dist[si] = 1.0 / len(init_states)
    term_states = [goal_state] #no terminal
    init_dist = 1/(num_states) * np.ones(num_states)
    mdp_env = mdp.FeaturizedGridMDP(num_rows, num_cols, state_features, weights, gamma, init_dist, term_states)
    return mdp_env


def negative_sideeffects_goal(num_rows, num_cols, num_features, unseen_feature=False):
    #no terminal random rewards and features

    num_states = num_rows * num_cols

    if unseen_feature:
        assert(num_features >=3)

    #create one-hot features
    f_vecs = np.eye(num_features)
    features = [tuple(f) for f in f_vecs]

    state_features = []
    for i in range(num_states):
        #select all but last two states randomly (last state is goal, second to last state is possibly unseen)
        if unseen_feature:
            r_idx = np.random.randint(num_features-1)
        else:
            r_idx = np.random.randint(num_features-2)
        state_features.append(features[r_idx])
    
    #select goal
    goal_state = np.random.randint(num_states)
    state_features[goal_state] = features[-1]

    
    state_features = np.array(state_features)


    #sample from L2 ball
    weights = -np.random.rand(num_features)
    #set goal as positive
    weights[-1] = +2
    #set unseen as negative
    weights[-2] = -2
    weights = weights / np.linalg.norm(weights)
    
    print("weights", weights)
    gamma = 0.99
    #let's look at all starting states for now
    init_dist = np.ones(num_states) / num_states
    # init_states = [10]
    # for si in init_states:
    #     init_dist[si] = 1.0 / len(init_states)

    #no terminal
    term_states = [goal_state]
    
    mdp_env = mdp.FeaturizedGridMDP(num_rows, num_cols, state_features, weights, gamma, init_dist, term_states)
    return mdp_env


def lavaland_smaller(contains_lava=False):
    #trying to replicate the ambiguous lava domain from Dylan's IRD paper figure 2 (a)
    #four features, 10x10 grid

    num_rows = 2
    num_cols = 3
    num_states = num_rows * num_cols
    #four types of terrain
    num_features = 4
    d = (1,0,0,0)
    g = (0,1,0,0)
    t = (0,0,1,0)
    l = (0,0,0,1)
    #create one-hot features

    if not contains_lava:
        state_features = [d,d,t,
                        d,g,d]
    
    else:
        state_features = [l,d,t,
                        d,g,d]
    

    state_features = np.array(state_features)
    weights = np.array([-1,-5,+1,-100])
    #weights = weights / np.linalg.norm(weights)
    print(weights)
    gamma = 0.95
    init_dist = np.zeros(num_states)
    init_states = [0]
    for si in init_states:
        init_dist[si] = 1.0 / len(init_states)
    term_states = [] #no terminal
    # init_dist = 1/(num_states) * np.ones(num_states)
    mdp_env = mdp.FeaturizedGridMDP(num_rows, num_cols, state_features, weights, gamma, init_dist, term_states)
    return mdp_env



def lavaland_small_lessgrass(contains_lava=False):
    #trying to replicate the ambiguous lava domain from Dylan's IRD paper figure 2 (a)
    #four features, 10x10 grid

    #trying with less grass and fewer starting states
    #setting all starting states to be dirt

    num_rows = 5
    num_cols = 5
    num_states = num_rows * num_cols
    #four types of terrain
    num_features = 4
    d = (1,0,0,0)
    g = (0,1,0,0)
    t = (0,0,1,0)
    l = (0,0,0,1)
    #create one-hot features
    f_vecs = np.eye(num_features)
    features = [tuple(f) for f in f_vecs]

    state_features = []
    for i in range(num_states):
        #select all but last state randomly from all but last feature, last feature is target feature
        if not contains_lava:
            #add lava randomly but at lower percentage
            p = [0.9,0.1,0.0,0.0]  #add dirt and grass in equal proportions
        else:
            p = [0.65,0.2,0,0.15]  #add dirt and grass and lava in equal proportions
        r_idx = np.random.choice(4,p=p) #sample from four features with probabilities p
        # r_idx = np.random.randint(num_features - 1)
        state_features.append(features[r_idx])
    #make the middle state a target state
    state_features[12] = t

    state_features = np.array(state_features)
    weights = np.array([-1,-5,+1,-100])
    #weights = weights / np.linalg.norm(weights)
    print(weights)
    gamma = 0.95
    init_dist = np.zeros(num_states)
    init_states = [0,4,20,24]
    for si in init_states:
        state_features[si] = d
        init_dist[si] = 1.0 / len(init_states)
    term_states = [] #no terminal
    # init_dist = 1/(num_states) * np.ones(num_states)
    mdp_env = mdp.FeaturizedGridMDP(num_rows, num_cols, state_features, weights, gamma, init_dist, term_states)
    return mdp_env

def lavaland_small_somegrass(contains_lava=False):
    #trying to replicate the ambiguous lava domain from Dylan's IRD paper figure 2 (a)
    #four features, 10x10 grid

    #trying with less grass and fewer starting states
    #setting all starting states to be dirt

    num_rows = 5
    num_cols = 5
    num_states = num_rows * num_cols
    #four types of terrain
    num_features = 4
    d = (1,0,0,0)
    g = (0,1,0,0)
    t = (0,0,1,0)
    l = (0,0,0,1)
    #create one-hot features
    f_vecs = np.eye(num_features)
    features = [tuple(f) for f in f_vecs]

    state_features = []
    for i in range(num_states):
        #select all but last state randomly from all but last feature, last feature is target feature
        if not contains_lava:
            #add lava randomly but at lower percentage
            p = [0.85,0.15,0.0,0.0]  #add dirt and grass in equal proportions
        else:
            p = [0.7,0.15,0,0.15]  #add dirt and grass and lava in equal proportions
        r_idx = np.random.choice(4,p=p) #sample from four features with probabilities p
        # r_idx = np.random.randint(num_features - 1)
        state_features.append(features[r_idx])
    #make the middle state a target state
    state_features[14] = t

    state_features = np.array(state_features)
    weights = np.array([-1,-5,+1,-100])
    #weights = weights / np.linalg.norm(weights)
    print(weights)
    gamma = 0.95
    # init_dist = np.zeros(num_states)
    # init_states = [0,4,20,24]
    # for si in init_states:
    #     state_features[si] = d
    #     init_dist[si] = 1.0 / len(init_states)
    term_states = [] #no terminal
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



def random_gridworld_corner_terminal(num_rows, num_cols, num_features):
    #random grid world with terminal in bottom right corner

    num_states = num_rows * num_cols

    #create one-hot features
    f_vecs = np.eye(num_features)
    features = [tuple(f) for f in f_vecs]

    state_features = []
    for i in range(num_states-1):
        #select all but last state randomly from all but last feature, last feature is target feature
        r_idx = np.random.randint(num_features - 1)
        state_features.append(features[r_idx])
    #make the last state a target state and terminal
    state_features.append(features[-1])

    state_features = np.array(state_features)

    weights = utils.sample_l2_ball(num_features)
    #make the weights so all are non-positive except for the target which is non-negative
    for i,w in enumerate(weights):
        if i < num_features - 1:
            weights[i] = -np.abs(w)
        else:
            weights[i] = np.abs(w)
    
    print("weights", weights)
    gamma = 0.99
    #let's look at all starting states for now
    init_dist = np.ones(num_states) / num_states
    # init_states = [10]
    # for si in init_states:
    #     init_dist[si] = 1.0 / len(init_states)

    #make the last state the terminal state
    term_states = [num_states - 1]
    
    mdp_env = mdp.FeaturizedGridMDP(num_rows, num_cols, state_features, weights, gamma, init_dist, term_states)
    return mdp_env


def random_gridworld(num_rows, num_cols, num_features):
    #no terminal random rewards and features

    num_states = num_rows * num_cols

    #create one-hot features
    f_vecs = np.eye(num_features)
    features = [tuple(f) for f in f_vecs]

    state_features = []
    for i in range(num_states):
        #select all but last state randomly from all but last feature, last feature is target feature
        r_idx = np.random.randint(num_features)
        state_features.append(features[r_idx])
    
    state_features = np.array(state_features)

    weights = utils.sample_l2_ball(num_features)
    
    print("weights", weights)
    gamma = 0.99
    #let's look at all starting states for now
    init_dist = np.ones(num_states) / num_states
    # init_states = [10]
    # for si in init_states:
    #     init_dist[si] = 1.0 / len(init_states)

    #no terminal
    term_states = []
    
    mdp_env = mdp.FeaturizedGridMDP(num_rows, num_cols, state_features, weights, gamma, init_dist, term_states)
    return mdp_env



if __name__=="__main__":
    #mdp_env = machine_teaching_toy_featurized()
    # mdp_env = lava_ambiguous_aaai18()
    mdp_env = random_gridworld_corner_terminal(6,6,5)
    print("features")
    utils.display_onehot_state_features(mdp_env)
    u_sa = mdp.solve_mdp_lp(mdp_env, debug=True)
    print("optimal policy")
    utils.print_policy_from_occupancies(u_sa, mdp_env)
    print("optimal values")
    v = mdp.get_state_values(u_sa, mdp_env)
    utils.print_as_grid(v, mdp_env)

    
