# Laplacian

"""
    laplacian!(v,w)

Evaluate the discrete Laplacian of `w` and return it as `v`. The data `w` can be
of type dual/primary nodes or edge components or edges; `v` must be of the same type.

# Example

```jldoctest
julia> w = Nodes(Dual,(8,6));

julia> v = deepcopy(w);

julia> w[4,3] = 1.0;

julia> laplacian!(v,w)
Nodes{Dual,8,6,Float64} data
Printing in grid orientation (lower left is (1,1))
6×8 Array{Float64,2}:
 0.0  0.0  0.0   0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0   0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0   1.0  0.0  0.0  0.0  0.0
 0.0  0.0  1.0  -4.0  1.0  0.0  0.0  0.0
 0.0  0.0  0.0   1.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0   0.0  0.0  0.0  0.0  0.0
```
"""
function laplacian!(out::Nodes{Dual,NX, NY}, w::Nodes{Dual,NX, NY}) where {NX, NY}
    view(out,2:NX-1,2:NY-1) .= view(w,2:NX-1,1:NY-2) .+ view(w,1:NX-2,2:NY-1) .+
                               view(w,3:NX,2:NY-1) .+ view(w,2:NX-1,3:NY) .-
                               4.0.*view(w,2:NX-1,2:NY-1)
    #@inbounds for y in 2:NY-1, x in 2:NX-1
    #    out[x,y] = w[x,y-1] + w[x-1,y] - 4w[x,y] + w[x+1,y] + w[x,y+1]
    #end
    out
end

function laplacian!(out::Nodes{Primal,NX, NY}, w::Nodes{Primal,NX, NY}) where {NX, NY}
    view(out,2:NX-2,2:NY-2) .= view(w,2:NX-2,1:NY-3) .+ view(w,1:NX-3,2:NY-2) .+
                               view(w,3:NX-1,2:NY-2) .+ view(w,2:NX-2,3:NY-1) .-
                               4.0.*view(w,2:NX-2,2:NY-2)
    #@inbounds for y in 2:NY-2, x in 2:NX-2
    #    out[x,y] = w[x,y-1] + w[x-1,y] - 4w[x,y] + w[x+1,y] + w[x,y+1]
    #end
    out
end

function laplacian!(out::XEdges{Dual,NX, NY}, w::XEdges{Dual,NX, NY}) where {NX, NY}
    view(out,2:NX-2,2:NY-1) .= view(w,2:NX-2,1:NY-2) .+ view(w,1:NX-3,2:NY-1) .+
                               view(w,3:NX-1,2:NY-1) .+ view(w,2:NX-2,3:NY) .-
                               4.0.*view(w,2:NX-2,2:NY-1)
    out
end

function laplacian!(out::YEdges{Dual,NX, NY}, w::YEdges{Dual,NX, NY}) where {NX, NY}
    view(out,2:NX-1,2:NY-2) .= view(w,2:NX-1,1:NY-3) .+ view(w,1:NX-2,2:NY-2) .+
                               view(w,3:NX,2:NY-2)   .+ view(w,2:NX-1,3:NY-1) .-
                               4.0.*view(w,2:NX-1,2:NY-2)
    out
end


function laplacian!(out::XEdges{Primal,NX, NY}, w::XEdges{Primal,NX, NY}) where {NX, NY}
   view(out,2:NX-1,2:NY-2) .= view(w,2:NX-1,1:NY-3) .+ view(w,1:NX-2,2:NY-2) .+
                              view(w,3:NX,2:NY-2) .+ view(w,2:NX-1,3:NY-1) .-
                              4.0.*view(w,2:NX-1,2:NY-2)
    out
end

function laplacian!(out::YEdges{Primal,NX, NY}, w::YEdges{Primal,NX, NY}) where {NX, NY}
   view(out,2:NX-2,2:NY-1) .= view(w,2:NX-2,1:NY-2) .+ view(w,1:NX-3,2:NY-1) .+
                                view(w,3:NX-1,2:NY-1) .+ view(w,2:NX-2,3:NY) .-
                               4.0.*view(w,2:NX-2,2:NY-1)
    out
end

function laplacian!(out::Edges{C,NX, NY}, w::Edges{C,NX, NY}) where {C,NX, NY}
  laplacian!(out.u,w.u)
  laplacian!(out.v,w.v)
  out
end

function laplacian!(out::EdgeGradient{C,D,NX, NY}, w::EdgeGradient{C,D,NX, NY}) where {C,D,NX, NY}
  laplacian!(out.dudx,w.dudx)
  laplacian!(out.dvdx,w.dvdx)
  laplacian!(out.dudy,w.dudy)
  laplacian!(out.dvdy,w.dvdy)
  out
end



#=
function laplacian!(out::Edges{Dual,NX, NY}, w::Edges{Dual,NX, NY}) where {NX, NY}
  view(out.u,2:NX-2,2:NY-1) .= view(w.u,2:NX-2,1:NY-2) .+ view(w.u,1:NX-3,2:NY-1) .+
                          view(w.u,3:NX-1,2:NY-1) .+ view(w.u,2:NX-2,3:NY) .-
                          4.0.*view(w.u,2:NX-2,2:NY-1)
  view(out.v,2:NX-1,2:NY-2) .= view(w.v,2:NX-1,1:NY-3) .+ view(w.v,1:NX-2,2:NY-2) .+
                               view(w.v,3:NX,2:NY-2) .+ view(w.v,2:NX-1,3:NY-1) .-
                               4.0.*view(w.v,2:NX-1,2:NY-2)
  #@inbounds for y in 2:NY-1, x in 2:NX-2
  #    out.u[x,y] = w.u[x,y-1] + w.u[x-1,y] - 4w.u[x,y] + w.u[x+1,y] + w.u[x,y+1]
  #end
  #@inbounds for y in 2:NY-2, x in 2:NX-1
  #    out.v[x,y] = w.v[x,y-1] + w.v[x-1,y] - 4w.v[x,y] + w.v[x+1,y] + w.v[x,y+1]
  #end
  out
end
=#

#=
function laplacian!(out::Edges{Primal,NX, NY}, w::Edges{Primal,NX, NY}) where {NX, NY}
  view(out.u,2:NX-1,2:NY-2) .= view(w.u,2:NX-1,1:NY-3) .+ view(w.u,1:NX-2,2:NY-2) .+
                               view(w.u,3:NX,2:NY-2) .+ view(w.u,2:NX-1,3:NY-1) .-
                              4.0.*view(w.u,2:NX-1,2:NY-2)
  view(out.v,2:NX-2,2:NY-1) .= view(w.v,2:NX-2,1:NY-2) .+ view(w.v,1:NX-3,2:NY-1) .+
                                view(w.v,3:NX-1,2:NY-1) .+ view(w.v,2:NX-2,3:NY) .-
                               4.0.*view(w.v,2:NX-2,2:NY-1)
  #@inbounds for y in 2:NY-2, x in 2:NX-1
  #    out.u[x,y] = w.u[x,y-1] + w.u[x-1,y] - 4w.u[x,y] + w.u[x+1,y] + w.u[x,y+1]
  #end
  #@inbounds for y in 2:NY-1, x in 2:NX-2
  #    out.v[x,y] = w.v[x,y-1] + w.v[x-1,y] - 4w.v[x,y] + w.v[x+1,y] + w.v[x,y+1]
  #end
  out
end
=#

"""
    laplacian(w)

Evaluate the discrete Laplacian of `w`. The data `w` can be of type dual/primary
nodes or edges. The returned result is of the same type as `w`.

# Example

```jldoctest
julia> q = Edges(Primal,(8,6));

julia> q.u[2,2] = 1.0;

julia> laplacian(q)
Edges{Primal,8,6,Float64} data
u (in grid orientation)
5×8 Array{Float64,2}:
 0.0   0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0   0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0   1.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  -4.0  1.0  0.0  0.0  0.0  0.0  0.0
 0.0   0.0  0.0  0.0  0.0  0.0  0.0  0.0
v (in grid orientation)
6×7 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
```
"""
function laplacian(w::Nodes{C,NX,NY}) where {C<:CellType,NX,NY}
  laplacian!(Nodes(C,w), w)
end

function laplacian(w::XEdges{C,NX,NY}) where {C<:CellType,NX,NY}
  laplacian!(XEdges(C,w), w)
end

function laplacian(w::YEdges{C,NX,NY}) where {C<:CellType,NX,NY}
  laplacian!(YEdges(C,w), w)
end

function laplacian(w::Edges{C,NX,NY}) where {C<:CellType,NX,NY}
  laplacian!(Edges(C,w), w)
end

function laplacian(w::EdgeGradient{C,D,NX,NY}) where {C<:CellType,D<:CellType,NX,NY}
  laplacian!(EdgeGradient(C,w), w)
end

"""
    laplacian_symm!(v,w)

Evaluate the symmetric 5-point discrete Laplacian of `w` and return it as `v`. The data `w` can be
of type dual nodes only for now. This symmetric Laplacian also evaluates the
partial Laplacians (using only available stencil data) on the ghost nodes.
"""
function laplacian_symm!(out::Nodes{Dual,NX, NY}, w::Nodes{Dual,NX, NY}) where {NX, NY}
    @inbounds for y in 2:NY-1, x in 2:NX-1
        out[x,y] = w[x,y-1] + w[x-1,y] - 4w[x,y] + w[x+1,y] + w[x,y+1]
    end
    @inbounds for y in 2:NY-1
        out[1,y]  = w[1,y-1]            - 4w[1,y] + w[2,y] + w[1,y+1]
        out[NX,y] = w[NX,y-1] + w[NX-1,y]- 4w[NX,y]        + w[NX,y+1]
    end
    @inbounds for x in 2:NX-1
        out[x,1]  = w[x-1,1] + w[x+1,1] - 4w[x,1] + w[x,2]
        out[x,NY] = w[x-1,NY]+ w[x+1,NY]- 4w[x,NY] + w[x,NY-1]
    end
    out[1,1] = -4w[1,1] + w[1,2] + w[2,1]
    out[NX,1] = -4w[NX,1] + w[NX-1,1] + w[NX,2]
    out[1,NY] = -4w[1,NY] + w[1,NY-1] + w[2,NY]
    out[NX,NY] = -4w[NX,NY] + w[NX,NY-1] + w[NX-1,NY]
    out
end


"""
    plan_laplacian(dims::Tuple,[with_inverse=false],[fftw_flags=FFTW.ESTIMATE],
                          [factor=1.0],[dx=1.0],[dtype=Float64])

Constructor to set up an operator for evaluating the discrete Laplacian on
dual or primal nodal data of dimension `dims`. If the optional keyword
`with_inverse` is set to `true`, then it also sets up the inverse Laplacian
(the lattice Green's function, LGF). These can then be applied, respectively, with
`*` and `\\` operations on data of the appropriate size. The optional
parameter `factor` is a scalar used to multiply the result of the operator and
divide the inverse. The optional parameter
`dx` is used in adjusting the uniform value of the LGF to match the behavior
of the continuous analog at large distances; this is set to 1.0 by default. The
type of data on which to act is floating point by default, but can also be ComplexF64.
This is specified with the optional parameter `dtype`

Instead of the first argument, one can also supply `w::Nodes` to specify the
size of the domain.

# Example

```jldoctest
julia> w = Nodes(Dual,(5,5));

julia> w[3,3] = 1.0;

julia> L = plan_laplacian(5,5;with_inverse=true)
Discrete Laplacian (and inverse) on a (nx = 5, ny = 5) grid acting on Float64 data with spacing 1.0

julia> s = L\\w
Nodes{Dual,5,5,Float64} data
Printing in grid orientation (lower left is (1,1))
5×5 Array{Float64,2}:
 0.16707    0.129276     0.106037     0.129276    0.16707
 0.129276   0.0609665   -0.00734343   0.0609665   0.129276
 0.106037  -0.00734343  -0.257343    -0.00734343  0.106037
 0.129276   0.0609665   -0.00734343   0.0609665   0.129276
 0.16707    0.129276     0.106037     0.129276    0.16707

julia> L*s ≈ w
true
```
"""
function plan_laplacian end

"""
    plan_laplacian!(dims::Tuple,[with_inverse=false],[fftw_flags=FFTW.ESTIMATE],
                          [factor=1.0][,nthreads=length(Sys.cpu_info())])

Same as [`plan_laplacian`](@ref), but operates in-place on data. The number of threads `threads` defaults to the number of
logical CPU cores on the system.
"""
function plan_laplacian! end

struct Laplacian{NX, NY, T, R,inplace}
    factor::T
    dx::Float64
    conv::Union{CircularConvolution{NX, NY, T},Nothing}
end

Base.eltype(::Laplacian{NX,NY,T}) where {NX,NY,T} = T


for (lf,inplace) in ((:plan_laplacian,false),
                     (:plan_laplacian!,true))
    @eval function $lf(dims::Tuple{Int,Int};
                   with_inverse = false, fftw_flags = FFTW.ESTIMATE, factor::Real = 1.0, dx = 1.0, dtype = Float64, nthreads = MAX_NTHREADS)
        NX, NY = dims
        if !with_inverse
            return Laplacian{NX, NY, dtype, false, $inplace}(convert(dtype,factor),convert(Float64,dx),nothing)
        end

        G = view(LGF_TABLE, 1:NX, 1:NY)
        Laplacian{NX, NY, dtype, true, $inplace}(convert(dtype,factor),convert(Float64,dx),CircularConvolution(G, fftw_flags,dtype=dtype,nthreads=nthreads))
    end

    @eval function $lf(nx::Int, ny::Int;
        with_inverse = false, fftw_flags = FFTW.ESTIMATE, factor = 1.0, dx = 1.0, dtype = Float64, nthreads = MAX_NTHREADS)
        $lf((nx, ny), with_inverse = with_inverse, fftw_flags = fftw_flags, factor = factor, dx = dx, dtype = dtype, nthreads = nthreads)
    end

    # Base the size on the dual grid associated with any grid data, since this
    # is what the efficient grid size in PhysicalGrid has been established with
    @eval function $lf(::GridData{NX,NY};
        with_inverse = false, fftw_flags = FFTW.ESTIMATE, factor = 1.0, dx = 1.0, dtype = Float64, nthreads = MAX_NTHREADS) where {NX,NY}
        $lf((NX,NY), with_inverse = with_inverse, fftw_flags = fftw_flags, factor = factor, dx = dx, dtype = dtype, nthreads = nthreads)
    end
end



function Base.show(io::IO, L::Laplacian{NX, NY, T, R, inplace}) where {NX, NY, T, R, inplace}
    nodedims = "(nx = $NX, ny = $NY)"
    inverse = R ? " (and inverse)" : ""
    isinplace = inplace ? " in-place" : ""
    print(io, "Discrete$isinplace Laplacian$inverse on a $nodedims grid acting on $T data with
               factor $(L.factor) and spacing $(L.dx)")
end

mul!(out::T, L::Laplacian, s::T) where T<:GridData = (laplacian!(out, s); out .*= L.factor)

*(L::Laplacian{MX,MY,T,R,false}, s::GridData) where {MX,MY,T,R} =
      L.factor*laplacian(s)

# function (*)(L::Laplacian{MX,MY,T,R,true}, s::GridData) where {MX,MY,T,R}
#     mul!(s,L,deepcopy(s))
# end

#==== * accepting ForwardDiff.Dual numbers ====#
function (*)(L::Laplacian{MX,MY,T,R,true}, s::GridData) where {MX,MY,T,R}
  valmat = FD.value.(s.data)
  outval = deepcopy(valmat)
  mul!(outval,L,s)
  out.data .= outval

  parmat = FD.partials.(s.data)
  if !(any(isempty, parmat))
    idx = findfirst(x -> x != 0, s.data)
    tag = get_tag(s.data[idx])
    # matrix including partials of FD.Dual numbers
    parval = similar(valmat)
    npar = length(parmat[1,1])
    outpar = Vector{typeof(parval)}(undef,npar)

    for k in 1:npar
      fill!(parval, 0)
      parval .= FD.partials.(s.data,k)
      outpar[k] = deepcopy(parval)
      mul!(outpar[k],L,parval)
    end

    out.data .= [FD.Dual{tag}(outval[i,j], [outpar[k][i,j] for k in 1:npar]...) for i in 1:size(out.data, 1), j in 1:size(out.data, 2)]
  end
  out
end

#=
function ldiv!(out::Nodes{C,NX, NY,T},
                   L::Laplacian{MX, MY, T, true, inplace},
                   s::Nodes{C, NX, NY,T}) where {C <: CellType, NX, NY, MX, MY, T, inplace}

    mul!(out.data, L.conv, s.data)
    inv_factor = 1.0/L.factor

    # Adjust the behavior at large distance to match continuous kernel
    out.data .-= (sum(s.data)/2π)*(GAMMA+log(8)/2-log(L.dx))
    out.data .*= inv_factor
    out
end
=#
for (datatype) in (:Nodes, :XEdges, :YEdges)
  @eval function ldiv!(out::$datatype{C,NX, NY,T},
                   L::Laplacian{MX, MY, T, true, inplace},
                   s::$datatype{C, NX, NY,T}) where {C <: CellType, NX, NY, MX, MY, T<:Union{Float64,ComplexF64}, inplace}

    mul!(out.data, L.conv, s.data)
    inv_factor = 1.0/L.factor

    # Adjust the behavior at large distance to match continuous kernel
    out.data .-= (sum(s.data)/2π)*(GAMMA+log(8)/2-log(L.dx))
    out.data .*= inv_factor
    out
  end

  #==== ldiv! accepting ForwardDiff.Dual numbers ====#
  @eval function ldiv!(out::$datatype{C,NX,NY,T},
                    L::Laplacian{MX, MY, TL, true, inplace},
                    s::$datatype{C,NX,NY,T}) where {C<:CellType, NX, NY, MX, MY, T<:Real, TL, inplace}

    # matrix including values of FD.Dual numbers
    valmat = FD.value.(s.data)
    outval = deepcopy(valmat)
    mul!(outval,L.conv,valmat)
    out.data .= outval

    parmat = FD.partials.(s.data)
    if !(any(isempty, parmat))
      idx = findfirst(x -> x != 0, s.data)
      tag = get_tag(s.data[idx])
      # matrix including partials of FD.Dual numbers
      parval = similar(valmat)
      npar = length(parmat[1,1])
      outpar = Vector{typeof(parval)}(undef,npar)

      for k in 1:npar
        fill!(parval, 0)
        parval .= FD.partials.(s.data,k)
        mul!(outpar[k],L.conv,parval)
      end

      out.data .= [FD.Dual{tag}(outval[i,j], [outpar[k][i,j] for k in 1:npar]...) for i in 1:size(out.data, 1), j in 1:size(out.data, 2)]
    end
    inv_factor = 1.0/L.factor
    # Adjust the behavior at large distance to match continuous kernel
    out.data .-= (sum(s.data)/2π)*(GAMMA+log(8)/2-log(L.dx))
    out.data .*= inv_factor
    out
  end

  #@eval \(L::Laplacian{MX,MY,T,R,false},s::$datatype{C,NX,NY}) where {MX,MY,T,R,C <: CellType,NX,NY} =
  #  ldiv!($datatype(C,s), L, s)

end

function get_tag(::FD.Dual{T}) where T
  return T
end

function ldiv!(out::Edges{C,NX,NY},L::Laplacian,s::Edges{C,NX,NY}) where {C,NX,NY}
  ldiv!(out.u,L,s.u)
  ldiv!(out.v,L,s.v)
  out
end

function ldiv!(out::EdgeGradient{C,D,NX,NY},L::Laplacian,s::EdgeGradient{C,D,NX,NY}) where {C,D,NX,NY}
  ldiv!(out.dudx,L,s.dudx)
  ldiv!(out.dvdx,L,s.dvdx)
  ldiv!(out.dudy,L,s.dudy)
  ldiv!(out.dvdy,L,s.dvdy)
  out
end

\(L::Laplacian{MX,MY,T,R,false},s::G) where {MX,MY,T,R,G<:GridData} =
  ldiv!(G(), L, s)

\(L::Laplacian{MX,MY,T,R,true},s::GridData{NX,NY}) where {MX,MY,T,R,NX,NY} =
    ldiv!(s, L, deepcopy(s))


#=
\(L::Laplacian{MX,MY,T,R,false},s::Nodes{C,NX,NY}) where {MX,MY,T,R,C <: CellType,NX,NY} =
  ldiv!(Nodes(C,s), L, s)

\(L::Laplacian{MX,MY,T,R,true},s::Nodes{C,NX,NY}) where {MX,MY,T,R,C <: CellType,NX,NY} =
  ldiv!(s, L, deepcopy(s))
=#
