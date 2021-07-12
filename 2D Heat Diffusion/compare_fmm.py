from fmm import FMMDistance
import numpy as np
import skfmm

np.set_printoptions(linewidth=200, formatter={'float': lambda x: "{0:0.3f}".format(x)})

#grid_size = 10
for grid_size in [10]:
    print(f"Grid size: {grid_size}")

    # one source
    initial_conditions = [(int(.5 * grid_size), int(.5 * grid_size))]

    # two source
    # initial_conditions = [(int(.25 * grid_size), int(.5 * grid_size)), (int(.75 * grid_size), int(.5 * grid_size))]

    # ring source
    # spots = [(.5, .25), (.25, .5), (.5, .75), (.75, .5), (.32, .32), (.68, .68), (.68, .32), (.32, .68)]
    # initial_conditions = [(int(grid_size * spots[i][0]), int(grid_size*spots[i][1])) for i in range(len(spots))]

    dm = FMMDistance(initial_conditions, grid_size)
    dm.calculate_distance()
    res_mp = np.array([[dm.mat[i][j].get_value() for i in range(grid_size)]
                                          for j in range(grid_size)])
    # print(res_mp)

    phi = np.ones((grid_size, grid_size))
    for (i, j) in initial_conditions:
        if i < grid_size and j < grid_size:
            phi[i, j] = 0

    res_sk = skfmm.distance(phi)
    # print(res_sk)
    print("Mean difference (my_fmm - skfmm): %.5f" % np.mean(res_mp-res_sk))
    print("Total difference (my_fmm - skfmm): %.5f" % np.sum(res_mp-res_sk))
    print()

    euclid = np.zeros((grid_size, grid_size))
    for (i, j) in initial_conditions:
        if i < grid_size and j < grid_size:
            euclid[i, j] = 0
    for i in range(grid_size):
        for j in range(grid_size):
            # dist = np.linalg.norm([i, j] - initial_conditions[0])
            dist = np.linalg.norm(np.array((i, j)) - np.array(initial_conditions[0]))
            euclid[i, j] = dist

    print("Mean difference (my_fmm - euclid): %.5f" % np.mean(res_mp - euclid))
    print("Total difference (my_fmm - euclid): %.5f" % np.sum(res_mp - euclid))
    print()

    print(res_mp)

    print(res_sk)

    print(euclid)


