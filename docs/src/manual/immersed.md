# Immersed data and their operations

```@meta
DocTestSetup = quote
  using CartesianGrids
  using Random
  Random.seed!(1)
end
```

```math
\def\ddt#1{\frac{\mathrm{d}#1}{\mathrm{d}t}}

\renewcommand{\vec}{\boldsymbol}
\newcommand{\uvec}[1]{\vec{\hat{#1}}}
\newcommand{\utangent}{\uvec{\tau}}
\newcommand{\unormal}{\uvec{n}}

\renewcommand{\d}{\,\mathrm{d}}
```


```@setup create
using CartesianGrids
using Plots
```

## The grid in physical space

Thus far, we have not had to consider the relationship between the grid's index space
and some physical space.
All of the operations thus far have acted on the entries in the discrete fields, based only on their
relative indices, and not on their physical coordinates. In this section, we will
discuss the relationship between the grid's index space and physical space, and then
in the next section we'll discuss how we can transfer data between these spaces.

Generically, we can write the relationship between the physical coordinates $x$ and $y$, and
the indices $i$ and $j$ of any grid point as

$$x(i) = (i - \Delta i - i_0)\Delta x, \quad y(j) = (j - \Delta j - j_0)\Delta x$$

The scaling between these spaces is controlled by $\Delta x$, which represents the
uniform size of each grid cell; note that grid cells are presumed to be square in `CartesianGrids`.
The indices $I_0 = (i_0,j_0)$ represent the location of the origin *in the index space for primal nodes*.
Why primal nodes? Since the underlying grid is composed of dual cells, then primal nodes
sit at the corners of the domain, so it is the most convenient for anchoring the grid to a specific
point. But, since some field data of the same index are shifted by half a cell in one or both
directions, then $\Delta i$ and $\Delta j$ are included for such purposes; these are either $0$ or $1/2$,
depending on the field type. For example, for a primal node, $\Delta i = 0$, so that $x(i_0) = 0$;
for a dual node, $\Delta i = 1/2$, so that $x(i_0) = -\Delta x/2$.

In particular, for our four different data types and their components

- Primal nodes: $\Delta i = 0$, $\Delta j = 0$
- Dual nodes: $\Delta i = 1/2$, $\Delta j = 1/2$
- Primal edges u: $\Delta i = 1/2$, $\Delta j = 0$
- Primal edges v: $\Delta i = 0$, $\Delta j = 1/2$
- Dual edges u: $\Delta i = 0$, $\Delta j = 1/2$
- Dual edges v: $\Delta i = 1/2$, $\Delta j = 0$

## Regularization and interpolation

Based on this relationship between the physical space and the index space, we
can now construct a means of transferring data between a point $(x,y)$ in the
physical space and the grid points in its immediate vicinity. We say that such a
point is *immersed* in the grid. The process of transferring from the point to the
 grid is called *regularization*, since we
are effectively smearing this data over some extended neighborhood; the
opposite operation, transferring grid field data to an arbitrary point, is
 *interpolation*. In `CartesianGrids`, both operations are carried out with the *discrete
 delta function* (DDF), which is a discrete analog of the Dirac delta function. The
 DDF generally has compact support, so that
 it only interacts with a small number of grid points in the vicinity of a
 given physical location. Since each of the different field types reside at
 slightly different locations, the range of indices invoked in this interaction
 will be different for each field type.

Regularization can actually take different forms. It can be a simple point-wise
interpolation, the discrete analog of simply multiplying by the Dirac delta function:

$$f_i \delta(\mathbf{x} - \mathbf{x}_i)$$

to immerse a value $f_i$ based at point $\mathbf{x}_i = (x_i,y_i)$.

Alternatively, regularization can be carried out over a curve $\mathbf{X}(s)$, the analog of

$$\int f(s) \delta(\mathbf{x} - \mathbf{X}(s))\mathrm{d}s$$

or it can be performed volumetrically, corresponding to

$$\int f(\mathbf{y}) \delta(\mathbf{x} - \mathbf{y})\mathrm{d}\mathbf{y}$$

In this case, the function $f$ is distributed over some region of space. In each of
these cases, the discrete version is simply a sum over data at a finite number of
discrete points, and the type of regularization is specified by providing an
optional argument specifying the arclength, area or volume associated
with each discrete point. These arguments are used to weight the sum.

 Let's see the regularization and interpolation in action. We will set up a ring
 of 100 points on a circle of radius $1/4$ centered at $(1/2,1/2)$. This curve-
 type regularization will be weighted by the arclength, $ds$, associated with each
 of the 100 points.
 On these points, we will
 set vector-valued data in which the $x$ component is uniformly equal to 1.0,
 while the $y$ component is set equal to the vertical position relative to the
 circle center. We will regularize these vector data to a primal
 edge field on the grid in which these points are immersed.

```@setup regularize
using CartesianGrids
using Plots
pyplot()
```

```@repl regularize
n = 100;
θ = range(0,stop=2π,length=n+1);
x = 0.5 .+ 0.25*cos.(θ[1:n]);
y = 0.5 .+ 0.25*sin.(θ[1:n]);
ds = 2π/n*0.25;
X = VectorData(x,y);
```

The variable `X` now holds the coordinates of the immersed points. Now we will set
up the vector-valued data on these points

```@repl regularize
f = VectorData(X);
fill!(f.u,1.0);
f.v .= X.v.-0.5;
```

Note that we have ensured that `f` has the correct dimensions by supplying the
coordinate data `X`. This first step also initializes the data to zeros.

Now, let's set up the grid. The physical domain will be of size $1.0 \times 1.0$,
and we will use $100$ dual grid cells in each direction. Allowing a single layer of ghost
cells surrounding the domain, we use $102$ cells, and set the cell size to 0.01.
Also, we will set the $(x,y)$ origin to coincide with the lower left corner of
the domain.

```@repl regularize
nx = 102; ny = 102;
q = Edges(Primal,(nx,ny));
Lx = 1.0;
dx = Lx/(nx-2)
```

Now we set up the regularization operator. To set it up, it needs to know
the coordinate data of the set of immersed points, the grid cell size, and the
weight to apply to each immersed point. Since this is a regularization of a curve,
this weight is the differential arc length `ds` associated with each point.
(This last argument is supplied as a scalar, since it is uniform.)

```@repl regularize
H = Regularize(X,dx,weights=ds)
```

We have omitted some optional arguments. For example, it chooses a default DDF
kernel (the 3rd-order Yang kernel `Yang3`); this can be changed with the `ddftype` argument. Also,
the lower left corner, where we've set the origin, is the location of the $(1,1)$
primal node; this is the default choice for `I0` (the tuple $I_0$ of coordinates in index
space discussed in the previous section).

Now we can apply the regularization operator. We supply the target field `q` as the
first argument and the source data `f` as the second argument.

```@repl regularize
H(q,f);
plot(q)
savefig("regq.svg"); nothing # hide
```
![](regq.svg)

We could also regularize this to a field of dual edges.

```@repl regularize
p = Edges(Dual,(nx,ny));
H(p,f);
plot(p)
savefig("regp.svg"); nothing # hide
```
![](regp.svg)

Scalar-valued data on the immersed points can only be regularized to nodal fields;
the syntax is similar, and the regularization operator does not need to be
reconstructed:

```@repl regularize
g = ScalarData(X);
fill!(g,1.0);
w = Nodes(Dual,(nx,ny));
H(w,g);
plot(w)
savefig("regw.svg"); nothing # hide
```
![](regw.svg)

For a given regularization operator, $H$, there is a companion interpolation operator,
$E$. In `CartesianGrids`, this interpolation is also carried out
with the same constructed operator, but with the arguments reversed: the grid field
data are the source and the immersed points are the target. Note that interpolation
is always a volumetric operation, so the weights assigned during the construction
of the operator are not used in interpolation. Let's interpolate our regularized field back onto the immersed points.

```@repl regularize
f2 = VectorData(X);
H(f2,q);
plot(f2.u,lab="u")
plot!(f2.v,lab="v")
savefig("interpf.svg"); nothing # hide
```
![](interpf.svg)

Note that interpolation is *not* the inverse of regularization; we don't recover the original data
when we regularize and then interpolate. However, there is generally a way to scale the quantities on the immersed points and on the grid so that $H = E^T$. If we want to force these
operations to be transposes of each other, we can supply the `issymmetric=true` flag. This flag will override any supplied weights. But here, we will exclude it so that it defaults to the asymmetric form.

```@repl regularize
H = Regularize(X,dx)
```


If we expect to carry out the regularization and interpolation a lot, then it
is often sensible to construct matrix versions of these operators. This
construction is sometimes a bit slow, but the resulting operators perform their
operations much faster than the matrix-free operators described above. To
generate these matrix operators, we have to supply the data types of the
source and target of the operation. For example, for regularization from
scalar field data to dual node data,
```@repl regularize
g = ScalarData(X);
w = Nodes(Dual,(nx,ny));
Hmat = RegularizationMatrix(H,g,w);
fill!(g,1.0);
w .= Hmat*g;
```
In general, the interpolation matrix is separately constructed, and the source and target
are reversed:
```@repl regularize
Emat = InterpolationMatrix(H,w,g);
g .= Emat*w;
```

Alternatively, if the regularization and interpolation are symmetric, then we
can get them both when we call for the regularization matrix:
```@repl regularize
H = Regularize(X,dx,issymmetric=true)
Hmat, Emat = RegularizationMatrix(H,g,w);
```
It might seem a bit funny to store them separately if they are just transposes
of each other, but it is essential for the method dispatch that they are
given separate types.

## Other operations with point-type data

```@setup vector
using CartesianGrids
```

We have seen point-type data structures, `ScalarData` and `VectorData`; there is also a tensor type of data, `TensorData`, which holds the four components of a 2x2 tensor. One can regularize and interpolate with this tensor data, as well; its companion grid data structure is the `EdgeGradient` type, which is a wrapper for four `Nodes` structures: two `Dual`, and two `Primal`, where the four tensor components are naturally held on the grid.

There are also some extensions of standard operations to the `VectorData` type. For example, we can add a tuple of two numbers to vector data, and these numbers get added to each entry in the set of points, component-wise. For example,
```@repl vector
Y = VectorData(4)
Y + (1,2)
```

Subtraction also works, and the operations are commutable.

Another useful operation is a cross product, which can be carried out between a single scalar (treated as though it was the component of an out-of-plane vector) and `VectorData`:

```@repl vector
using LinearAlgebra
X = VectorData(4)
fill!(X.u,1)
pointwise_cross(2.0,X)
```
