import numpy as np

a = np.array([[0.6, 0.2, 0.2], [0.5, 0.3, 0.2], [0.4, 0.1, 0.5]])
pi = np.array([0.5, 0.2, 0.3])
b = np.array([[0.7, 0.1, 0.2], [0.1, 0.6, 0.3], [0.3, 0.3, 0.4]])

states = UP, DOWN, UNCHANGED, = 0, 1, 2
obs = [UP, UP, DOWN]
obs = [UP, UP, DOWN, UNCHANGED, UNCHANGED, DOWN, UP, UP]