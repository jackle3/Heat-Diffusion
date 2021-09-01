import vtkmodules.all as vtk
from tqdm import tqdm

from mesh_reader import MeshReader
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from vtk_fmm import FMMDistance
import time

def create_line(point1, point2, color):
    """
    Creates a line to connect vertices
    :param point1: first vertex
    :param point2: second vertex
    :param color: color of line
    :return actor: line actor
    """

    '''
    vtkLineSource is a source object that creates a polyline defined by two endpoints or a collection of 
    connected line segments. To define the line by end points, use SetPoint1 and SetPoint2 methods. To define a 
    broken line comprising of multiple line segments, use SetPoints to provide the corner points that for the line.
    '''
    line_src = vtk.vtkLineSource()
    line_src.SetPoint1(point1)
    line_src.SetPoint2(point2)

    ''' Create a mapper for line '''
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(line_src.GetOutputPort())
    mapper.ScalarVisibilityOff()

    ''' Create an actor and set the appearance for the mapper '''
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetLineWidth(2.0)

    return actor


def create_sphere(point, color, radius=1):
    """
    Create a sphere to mark the picked vertex
    :param point: the point data of the picked vertex
    :param radius: the radius of the sphere
    :param color: the color to render the sphere in
    :return actor: sphere actor
    """

    ''' Create sphere source '''
    sphere_src = vtk.vtkSphereSource()
    sphere_src.SetCenter(point)
    sphere_src.SetRadius(radius)

    sphere_src.SetPhiResolution(50)
    sphere_src.SetThetaResolution(50)

    ''' Create a mapper for sphere '''
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(sphere_src.GetOutputPort())
    mapper.ScalarVisibilityOff()

    ''' Create an actor and set the appearance for the mapper '''
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(0.7)

    return actor


class MouseVertexInteractor(vtk.vtkInteractorStyleTrackballCamera):
    """
    vtkInteractorStyleTrackballCamera allows the user to interactively manipulate (rotate, pan, etc.) the
    camera, the viewpoint of the scene. In trackball interaction, the magnitude of the mouse motion is
    proportional to the camera motion associated with a particular mouse binding. For example, small left-button
    motions cause small changes in the rotation of the camera around its focal point. For a 3-button mouse,
    the left button is for rotation, the right button for zooming, the middle button for panning, ctrl + left
    button for spinning, and shift + right button for environment rotation. (With fewer mouse buttons, ctrl +
    shift + left button is for zooming, and shift + left button is for panning.)
    """

    def __init__(self, data, reader):
        """
        Intializes the interactor
        :param data: the data from the vtk_reader
        :param reader: the dataset reader
        """
        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
        self.reader = reader
        self.data = data
        self.point_locator = vtk.vtkPointLocator()
        self.point_locator.SetDataSet(data)
        self.picked_actors = []  # will store the actors for sphere and adjacent
        self.current_picked = None  # will store the current picked
        self.render_window = None  # will store the render window
        self.fmm = None  # will store the FMM

    def set_render_window(self, render_window):
        self.render_window = render_window

    def left_button_press_event(self, obj, event):
        """
        Event listener for mouse press, creates sphere on closest vertex
        :return closest_point: index of the closest point to mouse position
        """
        click_pos = self.GetInteractor().GetEventPosition()

        '''
        vtkPropPicker is used to pick an actor/prop given a selection point (in display coordinates) and a 
        renderer. This class uses graphics hardware/rendering system to pick rapidly (as compared to using ray 
        casting as does vtkCellPicker and vtkPointPicker). This class determines the actor/prop and pick position in 
        world coordinates; point and cell ids are not determined.
        '''
        point_picker = vtk.vtkPropPicker()
        point_picker.Pick(click_pos[0], click_pos[1], 0, self.GetDefaultRenderer())
        position = point_picker.GetPickPosition()
        closest_point = self.point_locator.FindClosestPoint(position)
        if np.sum(position) != 0 and closest_point != self.current_picked:
            point = self.data.GetPoint(closest_point)

            closest_children = self.reader.vertex_adj[closest_point].children_vertices

            children_dists = []
            for children in closest_children:
                choose = self.data.GetPoint(children)
                children_dists.append(np.min(np.linalg.norm(np.array(point) - np.array(choose))))
            dist = np.average(children_dists)
            print('-'*200)
            print("Picked point index %d\n%s" % (closest_point, self.reader.vertex_adj[closest_point]))

            if hasattr(self, "picked_actors"):
                for actor in self.picked_actors:
                    self.GetDefaultRenderer().RemoveActor(actor)
                self.GetDefaultRenderer().ResetCamera()
                self.render_window.Render()
                self.picked_actors = []

            self.create_visuals_on_renderer(point_idx=closest_point, point=point, dist=dist, color=[1, 0, 0])
            print("Done")
            self.current_picked = closest_point
        self.OnLeftButtonDown()

        return

    def create_visuals_on_renderer(self, point_idx, point, dist, color):
        sphere = create_sphere(point=point, color=color, radius=dist/10)
        self.picked_actors.append(sphere)
        self.GetDefaultRenderer().AddActor(sphere)

        self.fmm = FMMDistance(self.reader.vertex_adj, point_idx)
        hist = self.fmm.calculate_distance()

        step = 1
        if len(hist) > 70:
            step = int(np.ceil(len(hist)/70))
        print(len(hist), step)

        ''' Gaussian FMM to simulate energy preservation '''
        def gaussian_height(sigma, x, mu=0):
            first = 1.0 / (np.sqrt(2 * np.pi * sigma ** 2))
            second_pow_top = -1.0 * (x - mu) ** 2
            second_pow_bot = 2.0 * sigma ** 2
            second = np.exp(second_pow_top / second_pow_bot)
            return first * second

        res = []
        for iteration, layer in enumerate(hist):
            temp = {}
            curr = {key: -1 if layer[key] > 1e9 else layer[key] for key in layer}
            sigma = max(curr.values())
            # print(sigma)

            for key in curr:
                if curr[key] != -1:
                    temp[key] = gaussian_height(sigma=sigma*1.2, x=curr[key]*2) * 100000
                else:
                    temp[key] = gaussian_height(sigma=sigma*1.2, x=hist[-1][key]*2) * 100000

            # print(iteration, temp)
            for key in temp:
                if temp[key] > 1e9:
                    temp[key] = 0
            res.append(temp)

        max_dist = 0
        for key in res[-1]:
            max_dist = max(max_dist, res[-1][key])
        max_dist += 40
        print('max_dist', max_dist)

        curr = hist[-1]
        sigma = max(curr.values())
        for iteration in range(80):
            sigma *= 1.015
            temp = {}
            for key in curr:
                temp[key] = gaussian_height(sigma=sigma * 1.2, x=curr[key] * 2) * 100000

            for key in temp:
                if temp[key] > 1e9:
                    temp[key] = 0
            res.append(temp)

        for i in tqdm(range(0, len(res), step)):
            distances = res[i]
            surface = self.create_surface(distances=distances, max_dist=max_dist)
            self.picked_actors.append(surface)
            self.GetDefaultRenderer().AddActor(surface)
            self.render_window.Render()
            # time.sleep(0.5)

    def create_surface(self, distances, max_dist):
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetNumberOfTuples(len(self.reader.np_points))
        colors.SetName("point_colors")

        for i in range(len(self.reader.np_points)):
            if i in distances:
                dist = distances[i]
            else:
                dist = self.fmm.mat[i].get_value()
            jet = cm.get_cmap('jet')
            norm = Normalize(vmin=0, vmax=max_dist)
            res = jet(norm(dist))
            colors.SetTuple(i, [255*res[0], 255*res[1], 255*res[2]])

        self.reader.vtk_reader.GetOutput().GetPointData().SetScalars(colors)

        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(self.reader.vtk_reader.GetOutputPort())
        mapper.ScalarVisibilityOn()
        mapper.SelectColorArray('point_colors')
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SetColorModeToDirectScalars()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        return actor


def read_and_display():
    file_path = "/home/jack/Code/GitHub/Heat-Diffusion/VTK-Mesh-Render/data/"
    file_name = input("Give filename: ")
    # file_name = 'holes1.vtk'

    ''' Read in the unstructured grid and create adjacency lists '''
    reader = MeshReader(file_path=file_path, file_name=file_name)
    reader.setup()

    ''' Create a mapper '''
    mapper = vtk.vtkDataSetMapper()
    # print(">>>>>> ", reader.vtk_reader.GetOutputPort())
    mapper.SetInputConnection(reader.vtk_reader.GetOutputPort())
    mapper.ScalarVisibilityOff()

    ''' Create an actor and set the appearance for the mapper (edges) '''
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetRepresentationToWireframe()
    actor.GetProperty().SetLineWidth(2.0)

    ''' Create a render to set camera, lighting '''
    render = vtk.vtkRenderer()
    render.AddActor(actor)

    ''' Add actors for the diagonals '''
    for key in reader.vertex_adj:
        origin = reader.np_points[key]
        for children in reader.vertex_adj[key].children_vertices:
            point = reader.np_points[children]
            # print(key, origin, children, point)
            line = create_line(origin, point, [1, 1, 1])
            render.AddActor(line)

    ''' Create a render window to show the result '''
    window = vtk.vtkRenderWindow()
    window.AddRenderer(render)
    window.SetSize(900, 900)

    ''' Create a key press interactor and set default renderer '''
    key_press = MouseVertexInteractor(reader.vtk_reader.GetOutput(), reader)
    key_press.SetDefaultRenderer(render)
    key_press.set_render_window(window)

    ''' Add user interaction to the render window '''
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.SetInteractorStyle(key_press)

    ''' Add camera coordinates '''
    axes = vtk.vtkAxesActor()
    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOutlineColor(0.9300, 0.5700, 0.1300)
    widget.SetOrientationMarker(axes)
    widget.SetInteractor(interactor)
    widget.SetViewport(0.0, 0.0, 0.2, 0.2)
    widget.SetEnabled(1)
    widget.InteractiveOn()

    ''' Launch the window and interactor '''
    render.ResetCamera()
    window.SetWindowName(f"FMM Heat Diffusion - {file_name}")
    interactor.Initialize()
    interactor.Start()


if __name__ == "__main__":
    read_and_display()
