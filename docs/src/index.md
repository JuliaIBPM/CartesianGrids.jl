# CartesianGrids.jl

*a framework for working with spatial fields discretized on or immersed in Cartesian grids*

The purpose of this package is to enable easy setup of and operations on spatial fields on 2-d uniform staggered Cartesian grids. Tools are provided for
*  Constructing field data that lie in the different finite-dimensional spaces of the grid: the centers, corners, and edges of the cells
*  Performing discretely-mimetic differential calculus operations on these data, such as div, grad, curl, etc. These operations are carried out in a manner that transforms them between the spaces.
*  Solving Poisson's equation on infinite grids using the lattice Green's function
*  Immersing data on co-dimension one and two entities (points, curves) into the grid
*  Performing operations on these co-dimension data.

Many of the core aspects of the fluid-body interaction are based on the immersed boundary projection method, developed by Taira and Colonius [^1].

## Installation

This package works on Julia `1.0` and higher and is registered in the general Julia registry. To install, type
```julia
]add CartesianGrids
```

Then type
```julia
julia> using CartesianGrids
```

The plots in this documentation are generated using [Plots.jl](http://docs.juliaplots.org/latest/).
You might want to install that, too, to follow the examples.

## References

[^1]: Taira, K. and Colonius, T. (2007) "The immersed boundary method: a projection approach," *J. Comput. Phys.*, 225, 2118--2137.
