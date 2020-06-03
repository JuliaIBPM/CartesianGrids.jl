# Setting up field data

```@meta
DocTestSetup = quote
  using CartesianGrids
end
```

```@setup create
using CartesianGrids
using Plots
pyplot()
```

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
Pages = ["fielddata.md"]
```
