## CartesianGrids.jl

_Tools for working with spatial fields discretized on or immersed in Cartesian grids_

| Documentation | Build Status |
|:---:|:---:|
| [![docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaIBPM.github.io/CartesianGrids.jl/latest) | [![Build Status](https://travis-ci.com/JuliaIBPM/CartesianGrids.jl.svg?branch=master)](https://travis-ci.com/JuliaIBPM/CartesianGrids.jl) [![Build status](https://ci.appveyor.com/api/projects/status/6tokpjqb4x8999g0?svg=true)](https://ci.appveyor.com/project/JuliaIBPM/cartesiangrids-jl) [![codecov](https://codecov.io/gh/JuliaIBPM/CartesianGrids.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaIBPM/CartesianGrids.jl) |

## About the package

The purpose of this package is to enable easy setup of and operations on spatial fields on 2-d staggered Cartesian grids. Tools are provided for
*  Constructing field data that lie in the different finite-dimensional spaces of the grid: the centers, corners, and edges of the cells
*  Performing discretely-mimetic differential calculus operations on these data, such as div, grad, curl, etc. These operations are carried out in a manner that transforms them between the spaces.
*  Solving Poisson's equation on infinite grids using the lattice Green's function
*  Immersing data on co-dimension one and two entities (points, curves) into the grid
*  Performing operations on these co-dimension data.


Documentation can be found at https://JuliaIBPM.github.io/CartesianGrids.jl/latest.

**CartesianGrids.jl** is registered in the general Julia registry. To install, enter the package manager by typing
```julia
] add CartesianGrids
```

Then, in any version, type
```julia
julia> using CartesianGrids
```
For examples, consult the documentation or see the example Jupyter notebooks in the Examples folder.
