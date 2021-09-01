from mesh_reader import MeshReader
from vtk_fmm import FMMDistance
import numpy as np

file_path = "/home/jack/Code/GitHub/Heat-Diffusion/VTK-Mesh-Render/data/"
# file_name = input("Give filename: ")
file_name = 'patch7.vtk'

reader = MeshReader(file_path=file_path, file_name=file_name)
reader.setup()

fmm = FMMDistance(vertex_adj=reader.vertex_adj, initial_point=0)
hist = fmm.calculate_distance()

res_fmm = hist[-1]
print(res_fmm)

res_euclid = {}
for key in reader.vertex_adj:
    res_euclid[key] = np.linalg.norm(reader.vertex_adj[key].coord - reader.vertex_adj[0].coord)
print(res_euclid)

print("| %-8s | %-15s | %-20s | %-15s" % ('Index', 'FMM Distance', 'Euclidean Distance', 'Difference'))
diff_tot = []
for key in res_euclid:
    diff = abs(res_fmm[key] - res_euclid[key])
    print("| %-8s | %-15.10s | %-20.10s | %-15.10s |" % (key, res_fmm[key], res_euclid[key], diff))
    diff_tot.append(diff)
print("-"*70)
print("| %49s | %-15.10s |" % ('Mean Difference', np.mean(diff_tot)))
print("| %49s | %-15.10s |" % ('Total Difference', np.sum(diff_tot)))

