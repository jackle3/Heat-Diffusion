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


fig, ax = plt.subplots(facecolor='white')
cax = ax.pcolormesh(res_mat[0], cmap=plt.cm.jet, vmin=0, vmax=np.amax(res_mat[len(res_mat)-1]))
fig.colorbar(cax)

def animate(i):
    ax.set_title(f"FMM Animation: {i}/{len(res_mat)}")
    cax.set_array(res_mat[i].flatten())

anim = animation.FuncAnimation(fig, animate, repeat_delay=2000,
                                interval=100, frames=tqdm(range(int(len(res_mat))), desc="Creating Animation"))
# anim.save(f'{}_source_fmm.gif')
plt.show()

print("Done")
