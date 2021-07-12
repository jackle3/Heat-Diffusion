import vtkmodules.all as vtk
from mesh_reader import MeshReader
import numpy as np


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
        self.current_picked = None  # will store the current picked point

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
            print("Picked point index %d\n%s" % (closest_point, self.reader.vertex_adj[closest_point]))

            reachable_cells = self.reader.vertex_adj[closest_point].cell_index
            cell_list = []
            for cell in reachable_cells:
                cell_list.append(self.reader.quad_adj[cell].points)

            self.create_source_visual_on_renderer(point, dist, [1, 0, 0], cell_list)
            for children in closest_children:
                point_data = self.data.GetPoint(children)
                self.create_child_visual_on_renderer(point, point_data, dist, [0, 0, 1])

            self.current_picked = closest_point
        self.OnLeftButtonDown()

        return

    def create_source_visual_on_renderer(self, point, dist, color, cell_list):
        """
        create the visual for the source (point clicked), as well as the polygon for neighbouring quads
        :param point: source point
        :param dist: average distance between this point and all its neighbors
        :param color: color to set the point
        :param cell_list: cell_list to use to construct the neighbouring quads
        """
        if hasattr(self, "picked_actors"):
            for actor in self.picked_actors:
                self.GetDefaultRenderer().RemoveActor(actor)
            self.picked_actors = []
        sphere = self.create_sphere(point=point, color=color, radius=dist/10)
        self.picked_actors.append(sphere)
        self.GetDefaultRenderer().AddActor(sphere)

        quads = self.create_polygon(cell_list, [0.5, 0, 0.5])
        self.picked_actors.append(quads)
        self.GetDefaultRenderer().AddActor(quads)

    def create_child_visual_on_renderer(self, origin, point, dist, color):
        """
        create the visual for the source (point clicked), as well as the polygon for neighbouring quads
        :param origin: origin/source point
        :param point: source point
        :param dist: average distance between this point and all its neighbors
        :param color: color to set the point
        """
        sphere = self.create_sphere(point=point, color=color, radius=dist/10)
        self.picked_actors.append(sphere)
        self.GetDefaultRenderer().AddActor(sphere)

        line = self.create_line(origin, point, color)
        self.picked_actors.append(line)
        self.GetDefaultRenderer().AddActor(line)

    def create_polygon_array(self, cell_list):
        """
        creates an array of polygons to plot the neighboring quads
        :param cell_list: list of cells, where each index of a list of points within that cell
        :return polygons: vtkCellArray of polygons/cells
        """
        polygons = vtk.vtkCellArray()
        for cell in cell_list:
            polygon = vtk.vtkPolygon()
            num_points = len(cell)
            polygon.GetPointIds().SetNumberOfIds(num_points)  # Specify the number of ids for this object to hold.
            for _i, point in enumerate(cell):
                polygon.GetPointIds().SetId(_i, point)
            polygons.InsertNextCell(polygon)
        return polygons

    def create_polygon(self, cell_list, color):
        """
        Create a polygon to map the neighbouring quads
        :param cell_list: list of cells, where each index of a list of points within that cell
        :param color: color of quads
        :return actor: polygon actor
        """
        '''
        Create the polygons of the neighbouring quads to map out
        '''
        polygons = self.create_polygon_array(cell_list)
        polygon_src = vtk.vtkPolyData()
        polygon_src.SetPoints(self.reader.points)  # load in all the points into the src
        polygon_src.SetPolys(polygons)  # define the cell array of the src with just the wanted polygons

        ''' Create a mapper for polygon '''
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polygon_src)

        ''' Create an actor and set the appearance for the mapper '''
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(0.3)

        return actor


    def create_line(self, point1, point2, color):
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
        actor.GetProperty().SetLineWidth(4)

        return actor

    def create_sphere(self, point, color, radius=1):
        """
        Create a sphere to mark the picked vertex
        :param point: the point data of the picked vertex
        :param radius: the radius of the sphere
        :param color: the color to render the sphere in
        :return actor: sphere actor
        """

        '''
        vtkSphereSource creates a sphere (represented by polygons) of specified radius centered at the origin. The 
        resolution (polygonal discretization) in both the latitude (phi) and longitude (theta) directions can be 
        specified. It also is possible to create partial spheres by specifying maximum phi and theta angles. By 
        default, the surface tessellation of the sphere uses triangles; however you can set LatLongTessellation to 
        produce a tessellation using quadrilaterals.
        '''
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


def read_and_display():
    file_path = "/home/jack/Code/GitHub/Heat-Diffusion/VTK-Mesh-Render/data/"
    file_name = input("Give filename: ")
    # file_name = 'InternalNodes_2.vtk'

    ''' Read in the unstructured grid and create adjacency lists '''
    reader = MeshReader(file_path=file_path, file_name=file_name)
    reader.setup()

    '''
    vtkDataSetMapper is a mapper to map data sets (i.e., vtkDataSet and all derived classes) to graphics
    primitives. The mapping procedure is as follows: all 0D, 1D, and 2D cells are converted into points,
    lines, and polygons/triangle strips and then mapped to the graphics system. The 2D faces of 3D cells
    are mapped only if they are used by only one cell, i.e., on the boundary of the data set.
    '''
    mapper = vtk.vtkDataSetMapper()
    print(">>>>>> ", reader.vtk_reader.GetOutputPort())
    mapper.SetInputConnection(reader.vtk_reader.GetOutputPort())
    mapper.ScalarVisibilityOff()

    ''' Create an actor and set the appearance for the mapper '''
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetRepresentationToWireframe()
    actor.GetProperty().SetLineWidth(2.0)

    ''' Create a render to set camera, lighting '''
    render = vtk.vtkRenderer()
    render.AddActor(actor)

    ''' Create a render window to show the result '''
    window = vtk.vtkRenderWindow()
    window.AddRenderer(render)
    window.SetSize(900, 900)

    ''' Create a key press interactor and set default renderer '''
    key_press = MouseVertexInteractor(reader.vtk_reader.GetOutput(), reader)
    key_press.SetDefaultRenderer(render)

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
    window.SetWindowName(f"Vertex Adjacency Picker - {file_name}")
    interactor.Initialize()
    interactor.Start()


if __name__ == "__main__":
    read_and_display()
