struct Regularize{N,F}

  "x values of points, normalized to grid index space"
  x :: Vector{Real}

  "y values of points, normalized to grid index space"
  y :: Vector{Real}

  "1/dV factor"
  overdv :: Float64

  "weights for each point (e.g. arclengths), divided by dV"
  wgt :: Vector{Float64}

  "Discrete Delta function"
  ddf :: AbstractDDF

  "Radius of the DDF"
  ddf_radius :: Float64

  "Symmetry flag"
  _issymmetric :: Bool

end


"""
    Regularize(x,y,dx,[ddftype=Yang3],[graddir=0],[I0=(1,1)], [weights=1.0], [filter=false],
                       [issymmetric=false])

Constructor to set up an operator for regularizing and interpolating data from/to
points immersed in the grid to/from fields on the grid itself. The supplied
`x` and `y` represent physical coordinates of the immersed points, and `dx`
denotes a uniform physical cell size of the grid. The separate arguments `x` and
`y` can be replaced by a single argument `X` of type `VectorData` holding the
coordinates.

The operations of regularization and interpolation are carried out with a discrete
delta function (ddf), which defaults to the type `Yang3`. Others are also possible,
such as `Roma`, `Goza` or `M3`. The optional argument `graddir`, if set to 1 or 2, will
generate an interpolation operator that evaluates the negative of the
respective component of the gradient of a grid field at the immersed points. The
default value of this argument is 0, which simply interpolates. Note that the
regularization form of this gradient type is also possible.

The optional tuple
`I0` represents the indices of the primary node that coincides with `(x,y) = (0,0)`.
This defaults to `(1,1)`, which leaves one layer of ghost (dual) cells and sets
the physical origin in the lower left corner of the grid of interior dual cells.

Another optional parameter, `weights`, sets the weight of each point in the
regularization. This would generally be set with, say, the differential arc
length for regularization of data on a curve. It can be a vector (of the same length
as x and y) or a scalar if uniform. It defaults to 1.0.

The optional Boolean parameter `filter` can be set to `true` if it is desired to
apply filtering (see Goza et al, J Comput Phys 2016) to the grid data before
interpolating. This is generally only used in the context of preconditioning
the solution for forces on the immersed points.

If the optional Boolean parameter `issymmetric` is set to `true`, then the
regularization and interpolation are constructed to be transposes of each other.
Note that this option overrides any supplied weights. The default of this
parameter is `false`.

The resulting operator can be used in either direction, regularization and
interpolation, with the first argument representing the *target* (the entity
to regularize/interpolate to), and the second argument
the *source* (the entity to regularize/interpolate from). The regularization
does not use the filtering option.

# Example

In the example below, we set up a 12 x 12 grid. Using the default value for `I0`
and setting `dx = 0.1`, the physical dimensions of the non-ghost part of the grid
are 1.0 x 1.0. Three points are set up in the interior, and a vector field is assigned
to them, with the x component of each of them set to 1.0. These data are regularized
to a field of primal edges on the grid, using the Roma DDF kernel.

```jldoctest
julia> x = [0.25,0.75,0.25]; y = [0.75,0.25,0.25];

julia> X = VectorData(x,y);

julia> q = Edges(Primal,(12,12));

julia> dx = 0.1;

julia> H = Regularize(x,y,dx;ddftype=Roma)
Regularization/interpolation operator with non-filtered interpolation
  DDF type CartesianGrids.Roma
  3 points in grid with cell area 0.01

julia> f = VectorData(X);

julia> fill!(f.u,1.0);

julia> H(q,f)
Edges{Primal,12,12,Float64} data
u (in grid orientation)
11×12 Array{Float64,2}:
 0.0  0.0  0.0       0.0     0.0      …  0.0       0.0     0.0      0.0  0.0
 0.0  0.0  0.0       0.0     0.0         0.0       0.0     0.0      0.0  0.0
 0.0  0.0  8.33333  33.3333  8.33333     0.0       0.0     0.0      0.0  0.0
 0.0  0.0  8.33333  33.3333  8.33333     0.0       0.0     0.0      0.0  0.0
 0.0  0.0  0.0       0.0     0.0         0.0       0.0     0.0      0.0  0.0
 0.0  0.0  0.0       0.0     0.0      …  0.0       0.0     0.0      0.0  0.0
 0.0  0.0  0.0       0.0     0.0         0.0       0.0     0.0      0.0  0.0
 0.0  0.0  8.33333  33.3333  8.33333     8.33333  33.3333  8.33333  0.0  0.0
 0.0  0.0  8.33333  33.3333  8.33333     8.33333  33.3333  8.33333  0.0  0.0
 0.0  0.0  0.0       0.0     0.0         0.0       0.0     0.0      0.0  0.0
 0.0  0.0  0.0       0.0     0.0      …  0.0       0.0     0.0      0.0  0.0
v (in grid orientation)
12×11 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
```
"""
function Regularize(x::AbstractVector{D},y::AbstractVector{D},dx::T;
                    ddftype::DataType=Yang3,graddir::Int=0,
                    I0::Tuple{Int,Int}=(1,1),
                    weights::Union{T,Vector{T}}=1.0,
                    filter::Bool = false,
                    issymmetric::Bool = false) where {T<:Real,D<:Real}

  _issymmetric = (filter ? false : issymmetric)

  n = length(x)
  @assert length(y)==n
  if !_issymmetric
    if typeof(weights) == T
      wtvec = similar(x,T)
      fill!(wtvec,weights/(dx*dx))
    else
      @assert length(weights)==n
      wtvec = deepcopy(weights)./(dx*dx)
    end
  else
    # if the regularization and interpolation are symmetric, then the
    # weights are automatically set to be the cell area in order to cancel it
    # in the denominator of the regularization operator.
    wtvec = similar(FD.value.(x))
    fill!(wtvec,1.0)
  end

  baseddf = DDF(ddftype=ddftype,dx=1.0)
  if graddir == 0
    ddf = baseddf
  else
    ddf = GradDDF(graddir,ddftype=ddftype,dx=1.0)
  end

  Regularize{length(x),filter}(x./dx.+I0[1],y./dx.+I0[2],1.0/(dx*dx),
                      wtvec,ddf,_get_regularization_radius(baseddf),_issymmetric)
end

Regularize(x::T,y::T,dx::Real;b...) where {T<:Real} = Regularize([x],[y],dx;b...)

Regularize(x::VectorData,dx::Real;b...) = Regularize(x.u.data,x.v.data,dx;b...)

function Base.show(io::IO, H::Regularize{N,F}) where {N,F}
    filter = F ? "filtered" : "non-filtered"
    ddftype,_ = typeof(H.ddf).parameters
    op = H._issymmetric ? "Symmetric regularization/interpolation" : "Regularization/interpolation"
    println(io, "$op operator with $filter interpolation")
    println(io, "  DDF type $ddftype")
    println(io, "  $N points in grid with cell area $(sprint(show,1.0/H.overdv;context=:compact => true))")
end

function _get_regularization_radius(ddf::DDF)
  v = 1.0
  r = 0.0
  dr = 0.01
  while (v = abs(ddf(r))) > eps()
    r += dr
  end
  return r
end

_delta_block(radius,shift) = -radius+shift, radius+shift
_index_range(x,xmin,xmax,n,dn) = max(1,ceil(Int,FD.value(x)+xmin)), min(n-dn,floor(Int,FD.value(x)+xmax))
_distance_list(x,imin,imax,shift) = float(imin)-shift-x:float(imax)-shift-x


"""
    RegularizationMatrix(H::Regularize,f::PointData,u::CellData) -> Hmat

Construct and store a matrix representation of regularization associated with `H`
for data of type `f` to data of type `u`. The resulting matrix `Hmat` can then be
used to apply on point data of type `f` to regularize it to grid data of type `u`,
using `mul!(u,Hmat,f)`. It can also be used as just `Hmat*f`.

If `H` is a symmetric regularization and interpolation operator, then this
actually returns a tuple `Hmat, Emat`, where `Emat` is the interpolation matrix.
"""
struct RegularizationMatrix{TU,TF,T} <: AbstractMatrix{T}
  M :: SparseMatrixCSC{T,Int64}
end
#struct RegularizationMatrix{TU,TF} <: AbstractMatrix{Real}
#  M :: SparseMatrixCSC{Real,Int64}
#end


"""
    InterpolationMatrix(H::Regularize,u::CellData,f::PointData) -> Emat

Construct and store a matrix representation of interpolation associated with `H`
for data of type `u` to data of type `f`. The resulting matrix `Emat` can then be
used to apply on grid data of type `u` to interpolate it to point data of type `f`,
using `mul!(f,Emat,u)`. It can also be used as just `Emat*u`.
"""
struct InterpolationMatrix{TU,TF,T} <: AbstractMatrix{T}
  M :: SparseMatrixCSC{T,Int64}
end
#struct InterpolationMatrix{TU,TF} <: AbstractMatrix{Real}
#  M :: SparseMatrixCSC{Real,Int64}
#end

@wraparray RegularizationMatrix M 2
@wraparray InterpolationMatrix M 2


# ======  Regularization and interpolation operators of scalar types ======== #

pointtype = :ScalarData
for (gridtype,ctype,dnx,dny,shiftx,shifty) in @generate_scalarlist(SCALARLIST)

# Regularization
  @eval function (H::Regularize{N,F})(target::$gridtype{$ctype,NX,NY,T,DDT},source::$pointtype{N,S,DT}) where {N,F,NX,NY,S,T,DT,DDT}
        radius = H.ddf_radius
        fill!(target,0.0)
        xmin, xmax = _delta_block(radius,$shiftx)
        ymin, ymax = _delta_block(radius,$shifty)
        nzinds = findall(!iszero,source)
        @inbounds for pt in nzinds
            fact = source[pt]*H.wgt[pt]
            minx, maxx = _index_range(H.x[pt],xmin,xmax,NX,$dnx)
            miny, maxy = _index_range(H.y[pt],ymin,ymax,NY,$dny)
            prangex = _distance_list(H.x[pt],minx,maxx,$shiftx)
            prangey = _distance_list(H.y[pt],miny,maxy,$shifty)
            target[minx:maxx,miny:maxy] .+= fact.*H.ddf(prangex,prangey)
        end
        target
  end


# Interpolation
  @eval function (H::Regularize{N,false})(target::$pointtype{N,S,DT},source::$gridtype{$ctype,NX,NY,T,DDT}) where {N,NX,NY,S,T,DT,DDT}
        radius = H.ddf_radius
        fill!(target,0.0)
        xmin, xmax = _delta_block(radius,$shiftx)
        ymin, ymax = _delta_block(radius,$shifty)
        @inbounds for pt in eachindex(target)
            minx, maxx = _index_range(H.x[pt],xmin,xmax,NX,$dnx)
            miny, maxy = _index_range(H.y[pt],ymin,ymax,NY,$dny)
            prangex = _distance_list(H.x[pt],minx,maxx,$shiftx)
            prangey = _distance_list(H.y[pt],miny,maxy,$shifty)
            target[pt] += sum(source[minx:maxx,miny:maxy].*H.ddf(prangex,prangey))
        end
        target
  end

# Interpolation with filtering
  @eval function (H::Regularize{N,true})(target::$pointtype{N,S,DT},source::$gridtype{$ctype,NX,NY,T,DDT}) where {N,NX,NY,S,T,DT,DDT}
        tmp = typeof(source)()
        radius = H.ddf_radius
        fill!(target,0.0)
        xmin, xmax = _delta_block(radius,$shiftx)
        ymin, ymax = _delta_block(radius,$shifty)
        @inbounds for pt in eachindex(target)
            minx, maxx = _index_range(H.x[pt],xmin,xmax,NX,$dnx)
            miny, maxy = _index_range(H.y[pt],ymin,ymax,NY,$dny)
            prangex = _distance_list(H.x[pt],minx,maxx,$shiftx)
            prangey = _distance_list(H.y[pt],miny,maxy,$shifty)
            tmp[minx:maxx,miny:maxy] .+= H.wgt[pt].*H.ddf(prangex,prangey)
        end
        nzinds = findall(x -> abs(x) > eps(),tmp)
        tmp[nzinds] .= inv.(tmp[nzinds])

        @inbounds for pt in eachindex(target)
            minx, maxx = _index_range(H.x[pt],xmin,xmax,NX,$dnx)
            miny, maxy = _index_range(H.y[pt],ymin,ymax,NY,$dny)
            prangex = _distance_list(H.x[pt],minx,maxx,$shiftx)
            prangey = _distance_list(H.y[pt],miny,maxy,$shifty)
            target[pt] += sum(tmp[minx:maxx,miny:maxy].*
                              source[minx:maxx,miny:maxy].*
                              H.ddf(prangex,prangey))
            #end
        end
        target
  end

  # Construct regularization matrix
  @eval function RegularizationMatrix(H::Regularize{N,F},
    f::$pointtype{N,S,DT},
    u::$gridtype{$ctype,NX,NY,T,DDT}) where {N,F,NX,NY,S,T,DT,DDT}

    linI = LinearIndices(u)
    rad = H.ddf_radius
    xmin, xmax = _delta_block(rad,$shiftx)
    ymin, ymax = _delta_block(rad,$shifty)

    rows = Int64[]
    cols = Int64[]
    vals = eltype(u)[]

    for pt = 1:N
      minx, maxx = _index_range(H.x[pt],xmin,xmax,NX,$dnx)
      miny, maxy = _index_range(H.y[pt],ymin,ymax,NY,$dny)
      prangex = _distance_list(H.x[pt],minx,maxx,$shiftx)
      prangey = _distance_list(H.y[pt],miny,maxy,$shifty)

      I1 = vec(linI[minx:maxx,miny:maxy])
      append!(rows,I1)
      append!(cols,fill(pt,length(I1)))
      append!(vals,H.wgt[pt].*vec(H.ddf(prangex,prangey)))
    end
    Hmat = sparse(rows,cols,vals,length(u),length(f))
    if H._issymmetric
      # In symmetric case, these matrices are identical. (Interpolation is stored
      # as its transpose.)
      return RegularizationMatrix{typeof(u),typeof(f),eltype(Hmat)}(Hmat),InterpolationMatrix{typeof(u),typeof(f),eltype(Hmat)}(Hmat)
    else
      return RegularizationMatrix{typeof(u),typeof(f),eltype(Hmat)}(Hmat)
    end
  end

  # Construct interpolation matrix
  @eval function InterpolationMatrix(H::Regularize{N,false},
    u::$gridtype{$ctype,NX,NY,T,DDT},
    f::$pointtype{N,S,DT}) where {N,NX,NY,S,T,DT,DDT}

    linI = LinearIndices(u)
    rad = H.ddf_radius
    xmin, xmax = _delta_block(rad,$shiftx)
    ymin, ymax = _delta_block(rad,$shifty)

    rows = Int64[]
    cols = Int64[]
    vals = eltype(u)[]

    for pt = 1:N
      minx, maxx = _index_range(H.x[pt],xmin,xmax,NX,$dnx)
      miny, maxy = _index_range(H.y[pt],ymin,ymax,NY,$dny)
      prangex = _distance_list(H.x[pt],minx,maxx,$shiftx)
      prangey = _distance_list(H.y[pt],miny,maxy,$shifty)

      I1 = vec(linI[minx:maxx,miny:maxy])
      append!(rows,I1)
      append!(cols,fill(pt,length(I1)))
      append!(vals,vec(H.ddf(prangex,prangey)))
    end
    Emat = sparse(rows,cols,vals,length(u),length(f))
    InterpolationMatrix{typeof(u),typeof(f),eltype(Emat)}(Emat)
  end

  # Construct interpolation matrix with filtering. Because we construct
  # the matrix by supplying unit vectors from the points, it is easier
  # to carry out the filtering here rather than call the filtered operator
  # we defined above
  @eval function InterpolationMatrix(H::Regularize{N,true},
    u::$gridtype{$ctype,NX,NY,T,DDT},
    f::$pointtype{N,S,DT}) where {N,NX,NY,S,T,DT,DDT}

    linI = LinearIndices(u)
    rad = H.ddf_radius
    xmin, xmax = _delta_block(rad,$shiftx)
    ymin, ymax = _delta_block(rad,$shifty)

    g = similar(f)
    v = similar(u)
    fill!(g,1.0)
    wt = sparsevec(H(v,g))
    wt.nzval .= inv.(wt.nzval)

    rows = Int64[]
    cols = Int64[]
    vals = eltype(u)[]
    for pt = 1:N
      minx, maxx = _index_range(H.x[pt],xmin,xmax,NX,$dnx)
      miny, maxy = _index_range(H.y[pt],ymin,ymax,NY,$dny)
      prangex = _distance_list(H.x[pt],minx,maxx,$shiftx)
      prangey = _distance_list(H.y[pt],miny,maxy,$shifty)

      I1 = vec(linI[minx:maxx,miny:maxy])
      append!(rows,I1)
      append!(cols,fill(pt,length(I1)))
      append!(vals,wt[I1].*vec(H.ddf(prangex,prangey)))
    end
    Emat = sparse(rows,cols,vals,length(u),length(f))

    InterpolationMatrix{typeof(u),typeof(f),eltype(Emat)}(Emat)
  end

end


# ===== Regularization and interpolation operators of vector data to edges ===== #
pointtype = :VectorData
for (gridtype,ctype,dunx,duny,dvnx,dvny,shiftux,shiftuy,shiftvx,shiftvy) in @generate_collectionlist(VECTORLIST)

# Regularization
  @eval function (H::Regularize{N,F})(target::$gridtype{$(ctype...),NX,NY,T,DDT},source::$pointtype{N,S,DT}) where {N,F,NX,NY,S,T,DT,DDT}
        H(target.u,source.u)
        H(target.v,source.v)
        target
  end

# Interpolation
  @eval function (H::Regularize{N,F})(target::$pointtype{N,S,DT},source::$gridtype{$(ctype...),NX,NY,T,DDT}) where {N,F,NX,NY,S,T,DT,DDT}
        H(target.u,source.u)
        H(target.v,source.v)
        target
  end

  # Construct regularization matrix
  @eval function RegularizationMatrix(H::Regularize{N,F},src::$pointtype{N,S,DT},target::$gridtype{$(ctype...),NX,NY,T,DDT}) where {N,F,NX,NY,S,T,DT,DDT}

    lenu = length(target.u)
    lenv = length(target.v)
    Hmat = spzeros(lenu+lenv,2N)

    if H._issymmetric
      # In symmetric case, these matrices are identical. (Interpolation is stored
      # as its transpose.)
      Hmat[1:lenu,          1:N]    = RegularizationMatrix(H,src.u,target.u)[1].M
      Hmat[lenu+1:lenu+lenv,N+1:2N] = RegularizationMatrix(H,src.v,target.v)[1].M
      return RegularizationMatrix{typeof(target),typeof(src),eltype(Hmat)}(Hmat),InterpolationMatrix{typeof(target),typeof(src),eltype(Hmat)}(Hmat)
    else
      Hmat[1:lenu,          1:N]    = RegularizationMatrix(H,src.u,target.u).M
      Hmat[lenu+1:lenu+lenv,N+1:2N] = RegularizationMatrix(H,src.v,target.v).M
      return RegularizationMatrix{typeof(target),typeof(src),eltype(Hmat)}(Hmat)
    end
  end

  # Construct interpolation matrix
  @eval function InterpolationMatrix(H::Regularize{N,F},src::$gridtype{$(ctype...),NX,NY,T,DDT},target::$pointtype{N,S,DT}) where {N,F,NX,NY,S,T,DT,DDT}

    # note that we store interpolation matrices in the same shape as regularization matrices
    lenu = length(src.u)
    lenv = length(src.v)
    Emat = spzeros(lenu+lenv,2N)
    Emat[1:lenu,          1:N]    = InterpolationMatrix(H,src.u,target.u).M
    Emat[lenu+1:lenu+lenv,N+1:2N] = InterpolationMatrix(H,src.v,target.v).M
    InterpolationMatrix{typeof(src),typeof(target),eltype(Emat)}(Emat)
  end

end


# ======  Regularization and interpolation operators of tensor data to edge gradients ==== #
# Here, u describes both diagonal components and v the off diagonal
# We do not use the scalar-wise operations here because there is double use of
# the same ddf evaluation, so this saves us quite a bit of time.

pointtype = :TensorData
for (gridtype,ctype,dunx,duny,dvnx,dvny,shiftux,shiftuy,shiftvx,shiftvy) in @generate_collectionlist(TENSORLIST)

# Regularization
  @eval function (H::Regularize{N,F})(target::$gridtype{$(ctype...),NX,NY,T,DDT},source::$pointtype{N,S,DT}) where {N,F,NX,NY,S,T,DT,DDT}
    radius = H.ddf_radius
    fill!(target.dudx,0.0)
    fill!(target.dvdy,0.0)
    xmin, xmax = _delta_block(radius,$shiftux)
    ymin, ymax = _delta_block(radius,$shiftuy)
    @inbounds for pt in 1:N
      minx, maxx = _index_range(H.x[pt],xmin,xmax,NX,$dunx)
      miny, maxy = _index_range(H.y[pt],ymin,ymax,NY,$duny)
      prangex = _distance_list(H.x[pt],minx,maxx,$shiftux)
      prangey = _distance_list(H.y[pt],miny,maxy,$shiftuy)
      tmp = H.wgt[pt].*H.ddf(prangex,prangey)
      target.dudx[minx:maxx,miny:maxy] .+= source.dudx[pt].*tmp
      target.dvdy[minx:maxx,miny:maxy] .+= source.dvdy[pt].*tmp
    end
    fill!(target.dudy,0.0)
    fill!(target.dvdx,0.0)
    xmin, xmax = _delta_block(radius,$shiftvx)
    ymin, ymax = _delta_block(radius,$shiftvy)
    @inbounds for pt in 1:N
      minx, maxx = _index_range(H.x[pt],xmin,xmax,NX,$dvnx)
      miny, maxy = _index_range(H.y[pt],ymin,ymax,NY,$dvny)
      prangex = _distance_list(H.x[pt],minx,maxx,$shiftvx)
      prangey = _distance_list(H.y[pt],miny,maxy,$shiftvy)
      tmp = H.wgt[pt].*H.ddf(prangex,prangey)
      target.dudy[minx:maxx,miny:maxy] .+= source.dudy[pt].*tmp
      target.dvdx[minx:maxx,miny:maxy] .+= source.dvdx[pt].*tmp
    end
    target
  end

  # @eval function (H::Regularize{N,F})(target::$ctype,source::$ftype) where {N,F,NX,NY,S,T}
  #   H(target.dudx,source.dudx)
  #   H(target.dudy,source.dudy)
  #   H(target.dvdx,source.dvdx)
  #   H(target.dvdy,source.dvdy)
  #   target
  # end

# Interpolation
  @eval function (H::Regularize{N,false})(target::$pointtype{N,S,DT},source::$gridtype{$(ctype...),NX,NY,T,DDT}) where {N,NX,NY,S,T,DT,DDT}
    radius = H.ddf_radius
    fill!(target.dudx,0.0)
    fill!(target.dvdy,0.0)
    xmin, xmax = _delta_block(radius,$shiftux)
    ymin, ymax = _delta_block(radius,$shiftuy)
    @inbounds for pt in 1:N
      minx, maxx = _index_range(H.x[pt],xmin,xmax,NX,$dunx)
      miny, maxy = _index_range(H.y[pt],ymin,ymax,NY,$duny)
      prangex = _distance_list(H.x[pt],minx,maxx,$shiftux)
      prangey = _distance_list(H.y[pt],miny,maxy,$shiftuy)
      tmp = H.ddf(prangex,prangey)
      target.dudx[pt] += sum(source.dudx[minx:maxx,miny:maxy].*tmp)
      target.dvdy[pt] += sum(source.dvdy[minx:maxx,miny:maxy].*tmp)
    end
    fill!(target.dudy,0.0)
    fill!(target.dvdx,0.0)
    xmin, xmax = _delta_block(radius,$shiftvx)
    ymin, ymax = _delta_block(radius,$shiftvy)
    @inbounds for pt in 1:N
      minx, maxx = _index_range(H.x[pt],xmin,xmax,NX,$dvnx)
      miny, maxy = _index_range(H.y[pt],ymin,ymax,NY,$dvny)
      prangex = _distance_list(H.x[pt],minx,maxx,$shiftvx)
      prangey = _distance_list(H.y[pt],miny,maxy,$shiftvy)
      tmp = H.ddf(prangex,prangey)
      target.dudy[pt] += sum(source.dudy[minx:maxx,miny:maxy].*tmp)
      target.dvdx[pt] += sum(source.dvdx[minx:maxx,miny:maxy].*tmp)
    end
    target
  end

  # @eval function (H::Regularize{N,F})(target::$ftype,source::$ctype) where {N,F,NX,NY,S,T}
  #   H(target.dudx,source.dudx)
  #   H(target.dudy,source.dudy)
  #   H(target.dvdx,source.dvdx)
  #   H(target.dvdy,source.dvdy)
  #   target
  # end

# Interpolation with filtering -- need to speed up
  @eval function (H::Regularize{N,true})(target::$pointtype{N,S,DT},source::$gridtype{$(ctype...),NX,NY,T,DDT}) where {N,NX,NY,S,T,DT,DDT}
    tmp = typeof(source)()
    radius = H.ddf_radius
    fill!(target.dudx,0.0)
    fill!(target.dvdy,0.0)
    xmin, xmax = _delta_block(radius,$shiftux)
    ymin, ymax = _delta_block(radius,$shiftuy)
    @inbounds for pt in 1:N
        minx, maxx = _index_range(H.x[pt],xmin,xmax,NX,$dunx)
        miny, maxy = _index_range(H.y[pt],ymin,ymax,NY,$duny)
        prangex = _distance_list(H.x[pt],minx,maxx,$shiftux)
        prangey = _distance_list(H.y[pt],miny,maxy,$shiftuy)
        tmp.dudx[minx:maxx,miny:maxy] .+= H.wgt[pt].*H.ddf(prangex,prangey)
    end
    nzinds = findall(x -> abs(x) > eps(),tmp.dudx)
    tmp.dudx[nzinds] .= inv.(tmp.dudx[nzinds])
    tmp.dvdy .= tmp.dudx

    @inbounds for pt in 1:N
        minx, maxx = _index_range(H.x[pt],xmin,xmax,NX,$dunx)
        miny, maxy = _index_range(H.y[pt],ymin,ymax,NY,$duny)
        prangex = _distance_list(H.x[pt],minx,maxx,$shiftux)
        prangey = _distance_list(H.y[pt],miny,maxy,$shiftuy)
        d = H.ddf(prangex,prangey)
        target.dudx[pt] += sum(tmp.dudx[minx:maxx,miny:maxy].*
                               source.dudx[minx:maxx,miny:maxy].*d)
        target.dvdy[pt] += sum(tmp.dvdy[minx:maxx,miny:maxy].*
                               source.dvdy[minx:maxx,miny:maxy].*d)
    end

    fill!(target.dudy,0.0)
    fill!(target.dvdx,0.0)
    xmin, xmax = _delta_block(radius,$shiftvx)
    ymin, ymax = _delta_block(radius,$shiftvy)
    @inbounds for pt in 1:N
        minx, maxx = _index_range(H.x[pt],xmin,xmax,NX,$dvnx)
        miny, maxy = _index_range(H.y[pt],ymin,ymax,NY,$dvny)
        prangex = _distance_list(H.x[pt],minx,maxx,$shiftvx)
        prangey = _distance_list(H.y[pt],miny,maxy,$shiftvy)
        tmp.dudy[minx:maxx,miny:maxy] .+= H.wgt[pt].*H.ddf(prangex,prangey)
    end
    nzinds = findall(x -> abs(x) > eps(),tmp.dudy)
    tmp.dudy[nzinds] .= inv.(tmp.dudy[nzinds])
    tmp.dvdx .= tmp.dudy

    @inbounds for pt in 1:N
        minx, maxx = _index_range(H.x[pt],xmin,xmax,NX,$dvnx)
        miny, maxy = _index_range(H.y[pt],ymin,ymax,NY,$dvny)
        prangex = _distance_list(H.x[pt],minx,maxx,$shiftvx)
        prangey = _distance_list(H.y[pt],miny,maxy,$shiftvy)
        d = H.ddf(prangex,prangey)
        target.dudy[pt] += sum(tmp.dudy[minx:maxx,miny:maxy].*
                               source.dudy[minx:maxx,miny:maxy].*d)
        target.dvdx[pt] += sum(tmp.dvdx[minx:maxx,miny:maxy].*
                               source.dvdx[minx:maxx,miny:maxy].*d)
    end
    target
  end

  # Construct regularization matrix
  @eval function RegularizationMatrix(H::Regularize{N,F},src::$pointtype{N,S,DT},target::$gridtype{$(ctype...),NX,NY,T,DDT}) where {N,F,NX,NY,S,T,DT,DDT}

    # note that we only need to compute two distinct matrices, since there are
    # only two types of cell data in this tensor

    lenu = length(target.dudx)
    lenv = length(target.dudy)
    Hmat = spzeros(2lenu+2lenv,4N)

    if H._issymmetric
      # In symmetric case, these matrices are identical. (Interpolation is stored
      # as its transpose.)
      Hdudx, _ = RegularizationMatrix(H,src.dudx,target.dudx)
      Hdudy, _ = RegularizationMatrix(H,src.dudy,target.dudy)
      Hmat[1:lenu,          1:N]    = Hdudx.M
      Hmat[lenu+1:lenu+lenv,N+1:2N] = Hdudy.M
      Hmat[lenu+lenv+1:lenu+2lenv,2N+1:3N] = Hdudy.M
      Hmat[lenu+2lenv+1:2lenu+2lenv,3N+1:4N] = Hdudx.M
      return RegularizationMatrix{typeof(target),typeof(src),eltype(Hmat)}(Hmat),InterpolationMatrix{typeof(target),typeof(src),eltype(Hmat)}(Hmat)
    else
      Hdudx = RegularizationMatrix(H,src.dudx,target.dudx)
      Hdudy = RegularizationMatrix(H,src.dudy,target.dudy)
      Hmat[1:lenu,          1:N]    = Hdudx.M
      Hmat[lenu+1:lenu+lenv,N+1:2N] = Hdudy.M
      Hmat[lenu+lenv+1:lenu+2lenv,2N+1:3N] = Hdudy.M
      Hmat[lenu+2lenv+1:2lenu+2lenv,3N+1:4N] = Hdudx.M
      return RegularizationMatrix{typeof(target),typeof(src),eltype(Hmat)}(Hmat)
    end
  end

  # Construct interpolation matrix
  @eval function InterpolationMatrix(H::Regularize{N,F},src::$gridtype{$(ctype...),NX,NY,T,DDT},target::$pointtype{N,S,DT}) where {N,F,NX,NY,S,T,DT,DDT}

    # note that we store interpolation matrices in the same shape as regularization matrices
    lenu = length(src.dudx)
    lenv = length(src.dudy)
    Emat = spzeros(2lenu+2lenv,4N)
    Edudx = InterpolationMatrix(H,src.dudx,target.dudx)
    Edudy = InterpolationMatrix(H,src.dudy,target.dudy)

    Emat[1:lenu,          1:N]    = Edudx.M
    Emat[lenu+1:lenu+lenv,N+1:2N] = Edudy.M
    Emat[lenu+lenv+1:lenu+2lenv,2N+1:3N] = Edudy.M
    Emat[lenu+2lenv+1:2lenu+2lenv,3N+1:4N] = Edudx.M

    InterpolationMatrix{typeof(src),typeof(target),eltype(Emat)}(Emat)
  end

end

###### Matrix multiplication extended to these Regularization/Interpolation matrices ########

# Regularize
# We need to restrict the action of the matrices to the point and grid types
# they are built for, but allow the underyling data type (DT parameter)
# to be different, since one might be, e.g., a SubArray and the other an Array.
for f in [:Nodes,:XEdges,:YEdges]
    @eval mul!(u::S1,Hmat::RegularizationMatrix{S2,F},f::PointData) where {F,S1<:$f{C,NX,NY,T},S2<:$f{C,NX,NY,T}} where {C,NX,NY,T} = _mul!(u,Hmat,f)
    @eval mul!(f,Emat::InterpolationMatrix{S2,F},u::S1) where {F,S1<:$f{C,NX,NY,T},S2<:$f{C,NX,NY,T}} where {C,NX,NY,T} = _mul!(f,Emat,u)
    @eval (*)(Emat::InterpolationMatrix{S2,F},u::S1) where {F,S1<:$f{C,NX,NY,T},S2<:$f{C,NX,NY,T}} where {C,NX,NY,T} = mul!(F(),Emat,u)
    # This is meant to generate a MethodError for non-matching point types:
    @eval (*)(Emat::InterpolationMatrix{S2,F},u::GridData) where {F,S2<:$f{C,NX,NY,T}} where {C,NX,NY,T} = mul!(F(),Emat,u)
end
# Handle all collected grid data in a stricter fashion, since underlying data
# is always stored as Vector type
mul!(u::G,Hmat::RegularizationMatrix{G,F},f::PointData) where {G <: CollectedGridData,F} = _mul!(u,Hmat,f)
mul!(f,Emat::InterpolationMatrix{G,F},u::G) where {G <: CollectedGridData,F} = _mul!(f,Emat,u)
(*)(Emat::InterpolationMatrix{G,F},u::H) where {F,G <: CollectedGridData, H <: GridData} = mul!(F(),Emat,u)
# This is meant to generate a MethodError for non-matching point types:
#(*)(Emat::InterpolationMatrix{G,F},u) where {F,G <: CollectedGridData} = mul!(F(),Emat,u)


# Now dispatch on the point data type
for f in [:ScalarData,:VectorData,:TensorData]
  @eval _mul!(u,Hmat::RegularizationMatrix{G,S1},f::S2) where {G,S1<:$f{N,T},S2<:$f{N,T}} where {N,T} = _unsafe_mul!(u,Hmat,f)
  @eval _mul!(f::S2,Emat::InterpolationMatrix{G,S1},u) where {G,S1<:$f{N,T},S2<:$f{N,T}} where {N,T} = _unsafe_mul!(f,Emat,u)
  @eval (*)(Hmat::RegularizationMatrix{G,S1},f::S2) where {G,S1<:$f{N,T},S2<:$f{N,T}} where {N,T} = mul!(G(),Hmat,f)
  # This is meant to generate a MethodError for non-matching point types:
  for g in [:ScalarData,:VectorData,:TensorData]
    g == f && continue
    @eval (*)(Hmat::RegularizationMatrix{G,S1},f::S2) where {G,S1<:$f{N,T},S2<:$g{N,T}} where {N,T} = mul!(G(),Hmat,f)
  end
end


# Regularization without checking types
function _unsafe_mul!(u,Hmat::RegularizationMatrix,f)
  fill!(u,0.0)
  nzv = Hmat.M.nzval
  rv = Hmat.M.rowval
  @inbounds for col = 1:Hmat.M.n
    fj = f[col]
    for j = Hmat.M.colptr[col]:(Hmat.M.colptr[col + 1] - 1)
        u[rv[j]] += nzv[j]*fj
    end
  end
  u
end

# Interpolation without checking types
function _unsafe_mul!(f,Emat::InterpolationMatrix,u)
  fill!(f,0.0)
  nzv = Emat.M.nzval
  rv = Emat.M.rowval
  @inbounds for col = 1:Emat.M.n
      tmp = zero(eltype(f))
      for j = Emat.M.colptr[col]:(Emat.M.colptr[col + 1] - 1)
          tmp += transpose(nzv[j])*u[rv[j]]
      end
      f[col] += tmp
  end
  f
end

# Interpolation of regularization, used for developing the filtering matrix
function mul!(C::Array{T},Emat::InterpolationMatrix{G,F},
  Hmat::RegularizationMatrix{G,F}) where {G<:GridData,F<:PointData,T<:Real}
fill!(C,0.0)
Enzv = Emat.M.nzval
Erv = Emat.M.rowval
@inbounds for row = 1:Emat.M.n, col = 1:Hmat.M.n
tmp = zero(eltype(C))
for j = Emat.M.colptr[row]:(Emat.M.colptr[row + 1] - 1)
tmp += transpose(Enzv[j])*Hmat[Erv[j],col]
end
C[row,col] += tmp
end
return C
end


(*)(Emat::InterpolationMatrix,Hmat::RegularizationMatrix) =
        mul!(Array{eltype(Emat),2}(undef,Emat.M.n,Hmat.M.n),Emat,Hmat)

# This fixes a method ambiguity
(*)(Emat::InterpolationMatrix{G,F},Hmat::RegularizationMatrix) where {G <: Union{VectorGridData,TensorGridData},F}=
        mul!(Array{eltype(Emat),2}(undef,Emat.M.n,Hmat.M.n),Emat,Hmat)



function Base.summary(io::IO, H::RegularizationMatrix{TU,TF}) where {TU,TF}
    print(io, "Regularization matrix acting on type $TF and returning type $TU")
end

function Base.summary(io::IO, H::InterpolationMatrix{TU,TF}) where {TU,TF}
    print(io, "Interpolation matrix acting on type $TU and returning type $TF")
end
