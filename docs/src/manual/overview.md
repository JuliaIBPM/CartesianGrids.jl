# Overview of staggered grids

```@meta
DocTestSetup = quote
  using CartesianGrids
  using Random
  Random.seed!(1)
end
```


```@setup create
using CartesianGrids
using Plots
```

In `CartesianGrids`, field data, such as velocity, vorticity and pressure, are stored on
a staggered uniform grid. Such a grid is divided into *cells*, with *edges* (which,
on a two-dimensional grid, are the same as *faces*) and *nodes* (cell centers).
Nodes hold scalar-valued data. Edges, on the other hand, hold the components of
vector-valued data that are normal to the respective edges; one component lies
on the vertical edges, while the other is on the horizontal edges.

Furthermore, there are two different cell types: *primal* and *dual*. On
the physical grid, these cell types are offset with respect to each other by half
a cell spacing in each direction. In other words, the four corners of the primal
(resp. dual) cell are the nodes of four dual (resp. primary) cells.

Thus, on a two-dimensional staggered grid, there are four distinct vector spaces,
associated with where the data are held on the grid:
- dual nodes,
- dual edges,
- primal nodes, and
- primal edges.
In `CartesianGrids`, these are each distinct data types. Furthermore, the relationships between these types are
defined by an underlying grid shared by all. By convention, this grid is defined by
the number of dual cells `NX` and `NY` in each direction; we will often refer to it
as the *dual grid*. For example, `Nodes{Dual,NX,NY}` is the type for dual node data
on this grid; `Edges{Primal,NX,NY}` is the type for edge data on the primal cells
within this same `NX` by `NY` dual grid. Note that, even though this latter type is
parameterized by `NX` and `NY`, these values do *not* correspond to the number of primal
edges in each direction on this dual grid. These values always correspond to the
number of dual cells on the grid, for any data type. This makes it clear the
grid is shared by all data.

## Setting up field data

Let's see an example of creating a blank set of dual node data and filling it with
something:

```@repl create
w = Nodes(Dual,(5,4))
w .= reshape(1:20,5,4)
```

Other data types on the same grid can be set up in similar fashion. To ensure
that they have a size that is consistent with the dual node data `w`, we can use
this in place of the size:
```@repl create
q = Edges(Primal,w);
q.u[2,3] = 1;
q
```

## Index

```@index
Pages = ["overview.md"]
```
