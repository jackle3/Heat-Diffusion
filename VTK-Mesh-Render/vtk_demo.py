# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 11:43:18 2020

@author: Guoning Chen
"""

import vtk


'''
    The main function.
'''
def main():
        
    print("Please input the full file name: ")
    input_file_name = input()
    
    ''' Step 1: Read a vtk data file '''
    vtk_reader = vtk.vtkDataSetReader()
    vtk_reader.SetFileName(input_file_name)
    
    '''Step 2: Get geometry using a filter '''
    vtk_geometry = vtk.vtkExtractEdges()
    #vtk_geometry.SetInputData(vtk_reader.GetPolyDataOutput())
    vtk_geometry.SetInputConnection(vtk_reader.GetOutputPort())
    vtk_geometry.Update()
    
    '''Step 3: use a mapper to get the geometry primitives '''
    vtk_poly_mapper = vtk.vtkPolyDataMapper()
    vtk_poly_mapper.SetInputConnection(vtk_geometry.GetOutputPort())
    vtk_poly_mapper.ScalarVisibilityOff()#Turn this on when showing scalar field
     
    '''Step 4: create an actor and set the appearance for the mapper'''
    vtk_actor = vtk.vtkActor()
    vtk_actor.SetMapper(vtk_poly_mapper)
    vtk_actor.GetProperty().SetColor(1, 1, 0)
    
    '''Step 5: create a render to set camera, lighting '''
    render = vtk.vtkRenderer()
    render.AddActor(vtk_actor)
    
    '''Step 6: set the render window to show the result '''
    window = vtk.vtkRenderWindow()
    window.AddRenderer(render)
    window.SetSize(600, 600)
       
    '''Step 7: add user interaction to the render window'''
    window_interactor = vtk.vtkRenderWindowInteractor()
    window_interactor.SetRenderWindow(window)   
    window_interactor.Initialize()
    
    ''' Lauch the window '''
    window.Render()
    window.SetWindowName('COSC 6344 Visualization')
    window_interactor.Start()
    
if __name__ == '__main__':
    main()
