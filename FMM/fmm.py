"""
Author: Jack Le
"""

"""
Information is gathered from the follow sources:
1. https://www2.imm.dtu.dk/pubdb/edoc/imm841.pdf

>> Background details
1. Voxels (points) that form the initial condition are considered frozen (fr)
2. Vowels that have computed distances but are not frozen are considered narrow band (nb)
3. For each iteration of the central loop, the nb voxel with lowest distance value
is frozen, and the distances are computed for its neighbors
4. Frozen voxels are used to compute the values of other voxels abut are never computed again
5. Thus, the algorithm is like a "front" of narrow band voxels that propagate from the initial 
condition, freezing voxels as it moves along

>> Notes
1. An important data structure to use is the binary heap. The key for the binary heap is distance, 
and each element int he binary heap is a point to a voxel. The element with the lowest key is always 
on top in a binary heap, making it easy to find the narrow band voxel with the smallest distance value
2. Binary heap sounds like a priority queue - look more into it

>> Outline of the algorithm
1. Before the loop, tag the initial conditions (points) as frozen.
2. For each frozen voxel, we will visit all the neighbors, and at each of these neighbors, the distance 
will be computed using only information from frozen voxels.
3. These neighboring voxels are now tagged as a narrow band voxel and inserted into the binary heap
- Inside the loop
4. In each iteration, extract from the top of the heap the narrow band voxel with the smallest distance
    - This is the narrow band voxel that is closet to the curve, or initial condition
5. Tag the smallest distance narrow band voxel as frozen (we consider its distance value to be computed)
6. For each neighbor of the narrow band voxel that is not frozen, we compute the distance, tag it as narrow 
band, and insert it into the heap.
    - This means that if the neighbor is already a narrow band, we recompute the value and change its position 
    in the heap to reflect the new value
7. Loop back and extract the nw smallest distance narrow band voxel

I is a list that contains the initial voxels, or initial conditions
H is an empty binary heap
Initialization() {
    for each voxel v in I {
        Freeze v;
        for each neighbor vn of v {
            compute distance d at vn;
            if vn is not in the narrow band {
                tag vn as narrow band;
                insert (d, vn) into the heap H;
            }
            else {
                decrease the key of vn in H to d;
                    - in other words, change the existing (d_old, vn) to (d, vn)
            }
        }
    }
}

Loop() {
    while heap H is not empty {
        Extract smallest distance narrow band v from the top of H;
        Freeze v;
        for each neighbor vn of v {
            compute distance d at vn;
            if vn is not in the narrow band {
                tag vn as narrow band;
                insert (d, vn) into the heap H;
            }
            else {
                decrease the key of vn in H to d;
                    - in other words, change the existing (d_old, vn) to (d, vn)
            }
        }
    }
}

>> Computing Distances
1. The FMM works by solving the Eikonal equation, defined as || ∇T(x) || * F(x) = 1, where T is the arrival 
time of the front at point x, and F is the speed of the front at point x, where F >= 0. Because the travel
time can only expand, the arrival time at T is single valued.
2. To solve the Eikonal, we must find the distance value for the narrow band voxel so that the estimated length
of the gradient ||∇T|| is equal to 1/F. In other words, find distance for ||∇T|| = 1/F.
3. There is a proposed formula for the squared length of the gradient.
||∇T||^2 = max(Va - Vb, Va - Vc, 0)^2 + max(Va - Vd, Va - Ve, 0)^2
where Va is the unknown distance value, and Vb, Vc, Vd, Ve are the distance value of neighbouring voxels.
4. Plugging the equation from step 3 into step 2, we get the following
1/F^2 = max(Va - Vb, Va - Vc, 0)^2 + max(Va - Vd, Va - Ve, 0)^2
5. To solve this, we can look at each element (ie. look at just max(Va - Vb, Va - Vc, 0)^2 first). We will 
want to use the smaller of the value Vb and Vc.
6. In addition, we can only use frozen voxels. If neither Vb nor Vc are frozen, this term drops out of the
equation completely.
7. For example, if Vb < Vc, and Ve < Vd, and Vb and Ve are frozen, the equation is
(Va - Vb)^2 + (Va - Ve)^2 = F^-2
8. This is a quadratic equation, where we solve for Va. The larger solution of the two solutions is the one 
that we want.
"""

import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import sys


class Voxel:
    """
    A vertex/voxel on the matrix/grid

    :Attributes:
    - Status: "FAR" - unvisited voxel, "NAB" - narrow band/considered voxel, "FRZ" - frozen voxel
    - Value: distance at the vertex, computed by the T function
    """

    def __init__(self, status, value):
        self.status = status
        self.value = value

    def set_status_far(self):
        self.status = "FAR"

    def set_status_nb(self):
        self.status = "NAB"

    def set_status_frozen(self):
        self.status = "FRZ"

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value

    def is_far(self):
        return self.status == "FAR"

    def is_nb(self):
        return self.status == "NAB"

    def is_frozen(self):
        return self.status == "FRZ"

    def __str__(self):
        return '(%.1f %s)' % (self.value, self.status)

    def __repr__(self):
        return '(%.1f %s)' % (self.value, self.status)


class FMMDistance:
    """
    Creates the distance map from an initial condition using the FMM,
    computing distances via the Eikonal equation
    """

    def __init__(self, initial_conditions, grid_size):
        """
        Initializes the distance map object
        :param initial_conditions: the initial vertices (i, j) where T = 0
        :param grid_size: the size of the grid
        """
        self.mat = [[Voxel("FAR", np.inf) for j in range(grid_size)] for i in range(grid_size)]
        self.grid_size = grid_size
        self.F = np.ones((grid_size, grid_size))
        self.list_nb = []

        for (i, j) in initial_conditions:
            if i < grid_size and j < grid_size:
                self.mat[i][j].set_value(0)
                self.mat[i][j].set_status_frozen()
                for vn in self.non_frozen_neighbors(i, j):
                    self.set_distance(vn)

    def non_frozen_neighbors(self, i, j):
        """
        Returns list of non frozen neighboring voxels
        :param i: ith coordinate
        :param j: jth coordinate
        :return result: list of non frozen neighboring voxels
        """
        result = []
        dir_i = [1, 0, -1, 0]
        dir_j = [0, 1, 0, -1]
        for k in range(4):
            ti = i + dir_i[k]
            tj = j + dir_j[k]
            if (ti >= 0) and (ti < self.grid_size) and (tj >= 0) and (tj < self.grid_size):
                if not self.mat[ti][tj].is_frozen():
                    result.append((ti, tj))
        return result

    def set_distance(self, v):
        """
        Calculate/recalculate the distance T at the voxel v(i, j)
        :param v: the narrow band/far voxel to be updated
        """
        i, j = v
        # vd1 = min(Vb, Vc)
        # vd2 = min(Vd, Ve)
        if i == 0:
            vd1 = self.mat[i + 1][j].get_value()
        elif i == self.grid_size - 1:
            vd1 = self.mat[i - 1][j].get_value()
        else:
            vd1 = min(self.mat[i + 1][j].get_value(), self.mat[i - 1][j].get_value())

        if j == 0:
            vd2 = self.mat[i][j + 1].get_value()
        elif j == self.grid_size - 1:
            vd2 = self.mat[i][j - 1].get_value()
        else:
            vd2 = min(self.mat[i][j + 1].get_value(), self.mat[i][j - 1].get_value())

        if abs(vd1 - vd2) <= 1.0 / self.F[i, j]:
            t = .5 * (vd1 + vd2 + np.sqrt((vd1 + vd2) ** 2 - 2 * (vd1 ** 2 + vd2 ** 2 - 1.0 / (self.F[i, j] ** 2))))
        else:
            t = min(vd1, vd2) + self.F[i, j]

        t_old = self.mat[i][j].get_value()
        if t < t_old:
            self.mat[i][j].set_value(t)
            if self.mat[i][j].is_far():
                heapq.heappush(self.list_nb, (t, v))
                self.mat[i][j].set_status_nb()
            else:
                idx = self.list_nb.index((t_old, v))
                self.list_nb[idx] = (t, v)
                heapq.heapify(self.list_nb)

    def iterate(self):
        """
        Runs one iteration of the loop.
        """
        u = heapq.heappop(self.list_nb)

        u_i, u_j = u[1]

        self.mat[u_i][u_j].set_status_frozen()
        for vn in self.non_frozen_neighbors(u_i, u_j):
            self.set_distance(vn)

    def calculate_distance(self):
        """
        runs iterate until all distances are calculated
        """
        iteration = 0
        dist_map = []

        def generator():
            while self.list_nb != []:
                yield

        for _ in tqdm(generator(), desc="Calculating Distance", file=sys.stdout):
            self.iterate()
            iteration += 1
            dist_map.append(np.array([[self.mat[i][j].get_value() for i in range(self.grid_size)]
                                      for j in range(self.grid_size)]))

        print(f"Calculations finished with {iteration} iterations.")
        return iteration, dist_map


if __name__ == "__main__":
    np.set_printoptions(linewidth=200, formatter={'float': lambda x: "{0:0.3f}".format(x)})

    def create_anim(source_type, init, sz):
        plt.clf()
        fig, ax = plt.subplots(facecolor='white')

        dm = FMMDistance(init, sz)
        max_iter, fmm_mp = dm.calculate_distance()

        cax = ax.pcolormesh(fmm_mp[0], cmap=plt.cm.jet.reversed(), vmin=0, vmax=np.amax(fmm_mp[len(fmm_mp)-1]))
        fig.colorbar(cax)

        def animate(i):
            ax.set_title(f"FMM Animation: {i}/{max_iter}")
            cax.set_array(fmm_mp[i].flatten())

        anim = animation.FuncAnimation(fig, animate, repeat_delay=2000,
                                       interval=100, frames=tqdm(range(int(max_iter)), desc="Creating Animation"))
        anim.save(f'{source_type}_source_fmm.gif')
        plt.show()

        print("Done")

    grid_size = 20
    initial_conditions = [(int(.5*grid_size), int(.5*grid_size))]
    create_anim("one_slow", init=initial_conditions, sz=grid_size)

    # grid_size = 100
    # initial_conditions = [(int(.5*grid_size), int(.5*grid_size))]
    # create_anim("one", init=initial_conditions, sz=grid_size)
    #
    # initial_conditions = [(int(.25*grid_size), int(.5*grid_size)), (int(.75*grid_size), int(.5*grid_size))]
    # create_anim("two", init=initial_conditions, sz=grid_size)
    #
    # spots = [(.5, .25), (.25, .5), (.5, .75), (.75, .5), (.32, .32), (.68, .68), (.68, .32), (.32, .68)]
    # initial_conditions = [(int(grid_size * spots[i][0]), int(grid_size*spots[i][1])) for i in range(len(spots))]
    # create_anim("ring", init=initial_conditions, sz=grid_size)

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), facecolor='white')

    # # One source
    # grid_size = 100
    # initial_conditions = [(int(.5*grid_size), int(.5*grid_size))]
    # dm = FMMDistance(initial_conditions, grid_size)
    # dm.calculate_distance()
    # print(np.array(dm.mat), '\n')
    # plt_mat = np.array([[dm.mat[i][j].get_value() for i in range(grid_size)] for j in range(grid_size)])
    # print(np.array(plt_mat))
    # ax1.pcolormesh(plt_mat, cmap=plt.cm.jet.reversed())
    # ax1.set_title("My FMM: one-source")
    #
    # # Two source
    # grid_size = 100
    # initial_conditions = [(int(.25*grid_size), int(.5*grid_size)), (int(.75*grid_size), int(.5*grid_size))]
    # dm = FMMDistance(initial_conditions, grid_size)
    # dm.calculate_distance()
    # print(np.array(dm.mat), '\n')
    # plt_mat = np.array([[dm.mat[i][j].get_value() for i in range(grid_size)] for j in range(grid_size)])
    # print(np.array(plt_mat))
    # ax2.pcolormesh(plt_mat, cmap=plt.cm.jet.reversed())
    # ax2.set_title("My FMM: two-source")
    #
    # # Ring
    # grid_size = 100
    # spots = [(.5, .25), (.25, .5), (.5, .75), (.75, .5), (.32, .32), (.68, .68), (.68, .32), (.32, .68)]
    # initial_conditions = [(int(grid_size * spots[i][0]), int(grid_size*spots[i][1])) for i in range(len(spots))]
    # print(initial_conditions)
    # dm = FMMDistance(initial_conditions, grid_size)
    # dm.calculate_distance()
    # print(np.array(dm.mat), '\n')
    # plt_mat = np.array([[dm.mat[i][j].get_value() for i in range(grid_size)] for j in range(grid_size)])
    # print(np.array(plt_mat))
    # ax3.pcolormesh(plt_mat, cmap=plt.cm.jet.reversed())
    # ax3.set_title("My FMM: ring of sources")
    # plt.savefig("fmm.png")

    # plt.show()
