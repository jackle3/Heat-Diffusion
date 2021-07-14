from fmm import FMMDistance
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

np.set_printoptions(linewidth=200, formatter={'all': lambda y: "{:7.3f}".format(y)})

# We are going to use a one-source FMM
grid_size = 50
initial_conditions = [(int(.5 * grid_size), int(.5 * grid_size))]

fmm = FMMDistance(initial_conditions, grid_size)
max_iteration, distance_mat = fmm.calculate_distance()

def gaussian_height(sigma, x, mu=0):
    first = 1.0/(np.sqrt(2 * np.pi * sigma**2))
    second_pow_top = -1.0 * (x-mu)**2
    second_pow_bot = 2.0 * sigma ** 2
    second = np.exp(second_pow_top / second_pow_bot)
    return first * second

# print(gaussian_height(sigma=1, x=1))

res_mat = []
for layer in tqdm(range(0, len(distance_mat), 4), desc='Creating gaussian mat.'):
    curr = distance_mat[layer].copy()
    curr[curr > 1e9] = -1
    sigma = np.amax(curr)

    temp = np.empty(shape=curr.shape)

    for i in range(curr.shape[0]):
        for j in range(curr.shape[1]):
            temp[i, j] = gaussian_height(sigma=sigma, x=curr[i,j])

    res_mat.append(temp)

    
