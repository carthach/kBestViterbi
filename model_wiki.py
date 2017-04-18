import numpy as np

obs_map = {"normal" : 0, "cold" : 1, "dizzy" : 2}
state_map = {"healthy" : 0, "fever" : 1}

pi = np.array([0.6, 0.4])

a = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])

b = np.array([
    [0.5, 0.4, 0.1],
    [0.1, 0.3, 0.6]
])

# obs = [obs_map["normal"], obs_map["cold"]]
obs = [obs_map["normal"], obs_map["cold"], obs_map["dizzy"]]