import mdp
import mdp_worlds
import utils
import numpy as np
import bayesian_irl

# mdp_env_A = mdp_worlds.lava_ambiguous_ird_fig2a()

# u_sa_A = mdp.solve_mdp_lp(mdp_env_A)

# print("mdp A")
# print("Policy")
# utils.print_policy_from_occupancies(u_sa_A, mdp_env_A)
# print("reward")
# utils.print_as_grid(mdp_env_A.r_s, mdp_env_A)


# mdp_env_B = mdp_worlds.lava_ambiguous_ird_fig2b()

# u_sa_B = mdp.solve_mdp_lp(mdp_env_B)

# print("mdp B")
# print("Policy")
# utils.print_policy_from_occupancies(u_sa_B, mdp_env_B)
# print("reward")
# utils.print_as_grid(mdp_env_B.r_s, mdp_env_B)


#let's try out BIRL on a simpler version and see what happens

#first let's give a demo in the A version that doesn't have lava

mdp_env_B = mdp_worlds.lava_ird_simplified_b()
map_w = np.array([-0.30380369, -0.9159926,   0.10477373,  0.24017357])

print("MAP")
print("map_weights", map_w)
map_r = np.dot(mdp_env_B.state_features, map_w)
print("map reward")
utils.print_as_grid(map_r, mdp_env_B)
#compute new policy for mdp_B for map rewards
map_r_sa = mdp_env_B.transform_to_R_sa(map_w)
map_u_sa = mdp.solve_mdp_lp(mdp_env_B, reward_sa=map_r_sa) #use optional argument to replace standard rewards with sample
print("Map policy")
utils.print_policy_from_occupancies(map_u_sa, mdp_env_B)

