import vtkmodules.all as vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
"""
Doc pages to use:
- https://vtk.org/doc/nightly/html/classvtkUnstructuredGrid.html
- https://vtk.org/doc/nightly/html/classvtkUnstructuredGridReader.html
"""

class VertexNode:

    def __init__(self, coord, index):
        """
        Initializes the node
        :param coord: a 3D vector (1x3 matrix) containing coordinates of point
        :param index: the index of the point from the vtk file
        """
        self.coord = coord
        self.index = index
        self.children_vertices = []  # will store the connected nodes in adjacency list
        self.cell_index = []  # will store the index of all cells that contains/are adjacent to this point

    def __str__(self):
        return 'index: {} --> point_coords: [ {} ] --> next_vertex: {} --> reachable_cells: {}'.format(
            str(self.index),
            " | ".join(map(str, self.coord)),
            self.children_vertices,
            str(self.cell_index))

    def __repr__(self):
        return self.__str__()


class MeshReader:
    """
    Reader class for the reading an unstructured grid mesh
    """

    def __init__(self, file_path, file_name):
        """
        Initializes the class and stores the data paths, then loads the grid from the data path
        :param file_path: the path to the data files
        :param file_name: the name of the data file that needs to be read
        """
        file_path = file_path
        file_name = file_name
        self.data_path = file_path + file_name  # stores the path to the data file that needs to be read
        self.vtk_reader = None  # will store the vtk reader
        self.points = None  # will store the points that are read from the grid in the data path
        self.cells = None  # will store the cells that are read fro the grid in the data path
        self.np_points = None  # will store the points data in np format
        self.np_cell_list = None  # will store the cell list data in np format
        self.vertex_adj = None  # will store the vertices adjacency list

    def load_unstructured_grid(self):
        """
        Loads the unstructured grid, saving the reader, points, and cells
        :return: None
        """
        vtk_reader = vtk.vtkUnstructuredGridReader()
        vtk_reader.SetFileName(self.data_path)
        vtk_reader.Update()
        self.vtk_reader = vtk_reader
        self.points = self.vtk_reader.GetOutput().GetPoints()
        self.cells = self.vtk_reader.GetOutput().GetCells()

    def get_point_components(self, id):
        """
        https://vtk.org/doc/nightly/html/classvtkPoints.html#a9c44bfa0cedf2ef3bae40a4514d00ae2
        Copy point components into user provided array v[3] for specified id.
        :param id: index, or id, or a certain point
        :return point: coordinates of point at point id
        """
        point = [0, 0, 0]
        self.points.GetPoint(id, point)
        return point

    def convert_cell_list_to_np(self):
        """
        https://vtk.org/doc/nightly/html/classvtkCellArray.html
        creates cell list based on np instead of vtk
        :return cell_list: index-separated list of cells, with each index being a
                            certain cell containing index of points
        """
        cells = self.cells
        np_cells = vtk_to_numpy(cells.GetData())
        np_offsets = vtk_to_numpy(cells.GetOffsetsArray())  # originally returned as vtkDataArray, converted to np
        diffs = np.diff(np_offsets)
        prev = 0
        cell_list = []
        for diff in diffs:
            curr = prev + 1
            cell = np_cells[curr:curr + diff]
            prev = curr + diff
            cell_list.append(cell.tolist())
        self.np_cell_list = cell_list

    def convert_points_data_to_np(self):
        """
        Converts point data to numpy. Original point data is in vtkDataArray and isn't as usable.
        :return np_points: N x 3 matrix of points, with each row being coordinates in the three dimensions
        """
        points = self.points
        np_points = vtk_to_numpy(points.GetData())
        self.np_points = np_points

    def generate_vertices_adjacency_list(self):
        # {idx: Node storing coordinates, index, and adjacent vertices}
        points_map = {idx: VertexNode(data, idx) for idx, data in enumerate(self.np_points)}
        # Generates the direct adjacent cells
        for cell in self.np_cell_list:
            n = len(cell) - 1
            for i in range(n):
                points_map[cell[i]].children_vertices.append(cell[i + 1])
                points_map[cell[i + 1]].children_vertices.append(cell[i])
            points_map[cell[0]].children_vertices.append(cell[n])
            points_map[cell[n]].children_vertices.append(cell[0])
            # Diagonals
            points_map[cell[0]].children_vertices.append(cell[2])
            points_map[cell[2]].children_vertices.append(cell[0])
            # points_map[cell[1]].children_vertices.append(cell[1])
            # points_map[cell[3]].children_vertices.append(cell[3])
        for p_key in points_map:
            points_map[p_key].children_vertices = sorted(list(set(points_map[p_key].children_vertices)))
        # Generates the list of indices of adjacent cells
        for pt_idx in points_map:
            for cell_idx, cell in enumerate(self.np_cell_list):
                if pt_idx in cell:
                    points_map[pt_idx].cell_index.append(cell_idx)
        self.vertex_adj = points_map

    def setup(self):
        self.load_unstructured_grid()  # loads the unstructured grid into the instance variables
        self.convert_cell_list_to_np()  # creates the np cell list
        self.convert_points_data_to_np()  # creates the np points list
        self.generate_vertices_adjacency_list()  # generate the vertices adjacency list

# Test code
if __name__ == "__main__":
    # file = input("Give filename: ")
    # file = 'holes1.vtk'
    file = 'InternalNodes_2.vtk'
    reader = MeshReader('/home/jack/Code/GitHub/Heat-Diffusion/VTK-Mesh-Render/data/', file)

    reader.setup()
    for key in reader.vertex_adj:
        print(reader.vertex_adj[key])
