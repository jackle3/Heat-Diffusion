# Heat Diffusion
This project contains the code for a summer project with Professor [Guoning Chen](http://www2.cs.uh.edu/~chengu) and Lei Si, where we try to simulate the heat diffusion process for a non-uniform unstructed mesh.

For this, I created an accurate visual approximation of the heat diffusion process by combining the [Fast Marching Method](https://en.wikipedia.org/wiki/Fast_marching_method), used to show the actual diffusion, and a Gaussian curve approximation, used to show the energy preservation process of the grid.

The grid and simulation itself was run and displayed using [VTK](https://vtk.org/). To get this to work, I had to create a custom data structure that would read in a grid displayed on VTK and store relevant information. This information would then be passed on and used by the Fast Marching Method for its calculations.

To see my entire progress and path through this project, as well as my thoughts and reflections on certain aspects of the project, you can check out my Notion journal, linked here: https://www.notion.so/jakl3/Jack-Le-Heat-Diffusion-b3dd84077f764ff1b22de8911493227a

