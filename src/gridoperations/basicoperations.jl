import Base: -, +, *, /, \, ∘, zero, conj, real, imag, abs, abs2
import LinearAlgebra: transpose!, transpose

export tensorproduct!

# Identity operator

struct Identity end

(*)(::Identity,s::GridData) = s








### On scalar grid data ####

# Set it to negative of itself
function (-)(p_in::GridData)
  p = deepcopy(p_in)
  p.data .= -p.data
  return p
end

function (-)(p1::T,p2::T) where {T <: GridData}
   return T(p1.data .- p2.data)
 end

function (+)(p1::T,p2::T) where {T <: GridData}
  return T(p1.data .+ p2.data)
end

# Multiply and divide by a constant
function (*)(p::T,c::Number) where {T<:GridData}
  return T(c*p.data)
end

(*)(c::Number,p::T) where {T<:GridData} = *(p,c)

function (/)(p::T,c::Number) where {T<:GridData}
  return T(p.data ./ c)
end

"""
    product!(out::GridData,p::GridData,q::GridData)

Compute the Hadamard (i.e. element by element) product of grid data
`p` and `q` (of the same type) and return the result in `out`.

# Example

```jldoctest
julia> q = Edges(Dual,(8,6));

julia> out = p = deepcopy(q);

julia> q.u[3,2] = 0.3;

julia> p.u[3,2] = 0.2;

julia> product!(out,p,q)
Edges{Dual,8,6,Float64} data
u (in grid orientation)
6×7 Array{Float64,2}:
 0.0  0.0  0.0   0.0  0.0  0.0  0.0
 0.0  0.0  0.0   0.0  0.0  0.0  0.0
 0.0  0.0  0.0   0.0  0.0  0.0  0.0
 0.0  0.0  0.0   0.0  0.0  0.0  0.0
 0.0  0.0  0.06  0.0  0.0  0.0  0.0
 0.0  0.0  0.0   0.0  0.0  0.0  0.0
v (in grid orientation)
5×8 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
```
"""
@inline product!(out::T, p::T, q::T) where {T <: GridData} = (out .= p.*q)

for f in (:Nodes, :XEdges, :YEdges, :Edges)
  @eval @inline product!(out::$f{C,NX,NY},p::$f{C,NX,NY},q::$f{C,NX,NY}) where {C<:CellType,NX,NY} = (out .= p.*q)
end

@inline product!(out::EdgeGradient{C,D,NX,NY},p::EdgeGradient{C,D,NX,NY},q::EdgeGradient{C,D,NX,NY}) where {C<:CellType,D<:CellType,NX,NY} = (out .= p.*q)


"""
    product(p::Edges/Nodes,q::Edges/Nodes) --> Edges/Nodes

Compute the Hadamard product of edge or nodal (primal or dual) data `p` and `q` and return
the result. This operation can also be carried out with the `∘` operator:

# Example

```jldoctest
julia> q = Edges(Dual,(8,6));

julia> p = deepcopy(q);

julia> q.u[3,2] = 0.3;

julia> p.u[3,2] = 0.2;

julia> p∘q
Edges{Dual,8,6,Float64} data
u (in grid orientation)
6×7 Array{Float64,2}:
 0.0  0.0  0.0   0.0  0.0  0.0  0.0
 0.0  0.0  0.0   0.0  0.0  0.0  0.0
 0.0  0.0  0.0   0.0  0.0  0.0  0.0
 0.0  0.0  0.0   0.0  0.0  0.0  0.0
 0.0  0.0  0.06  0.0  0.0  0.0  0.0
 0.0  0.0  0.0   0.0  0.0  0.0  0.0
v (in grid orientation)
5×8 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
```
"""
@inline product(p::T, q::T) where {T <: GridData} = product!(T(), p, q)

@inline function (∘)(p::GridData{T,NX,NY}, q::GridData) where {T,NX,NY}
  product!(typeof(p)(), p, q)
end

"""
    tensorproduct!(q::EdgeGradient,u::Edges,v::Edges,ut::EdgeGradient,vt::EdgeGradient)

In-place tensor product of `u` and `v`, with the result returned in `q`. The
`ut` and `vt` are supplied as temporary storage.
"""
function tensorproduct!(q::EdgeGradient{C,D,NX,NY},u::Edges{C,NX,NY},v::Edges{C,NX,NY},
                        ut::EdgeGradient{C,D,NX,NY},vt::EdgeGradient{C,D,NX,NY}) where {C<:CellType,D<:CellType,NX,NY}
    grid_interpolate!(ut,u)
    grid_interpolate!(vt,v)
    q.dudx .= ut.dudx ∘ vt.dudx
    q.dudy .= ut.dvdx ∘ vt.dudy
    q.dvdx .= ut.dudy ∘ vt.dvdx
    q.dvdy .= ut.dvdy ∘ vt.dvdy
    return q
end

tensorproduct!(q::EdgeGradient{C,D,NX,NY},u::Edges{C,NX,NY},v::Edges{C,NX,NY}) where {C<:CellType,D<:CellType,NX,NY} =
        tensorproduct!(q,u,v,zero(q),zero(q))

(*)(u::Edges{C,NX,NY},v::Edges{C,NX,NY}) where {C<:CellType,NX,NY} = tensorproduct!(EdgeGradient(C,u),u,v)


"""
    transpose!(pt::EdgeGradient,p::EdgeGradient)

In-place element-by-element transpose of `EdgeGradient` `p`.
"""
function transpose!(qt::EdgeGradient{C,D,NX,NY},q::EdgeGradient{C,D,NX,NY}) where {C<:CellType,D<:CellType,NX,NY}
    qt.dudx .= q.dudx
    qt.dudy .= q.dvdx
    qt.dvdx .= q.dudy
    qt.dvdy .= q.dvdy
    return qt
end

"""
    transpose(p::EdgeGradient) -> EdgeGradient

Element-by-element transpose of `EdgeGradient` `p`
"""
transpose(q::EdgeGradient{C,D,NX,NY}) where {C<:CellType,D<:CellType,NX,NY} =
    transpose!(zero(q),q)


zero(::Type{T}) where {T <: GridData} = T()

#### ON COMPLEX GRID DATA

for f in (:conj,)
    @eval function $f(A::GridData{NX,NY,T}) where {NX,NY,T <: ComplexF64}
        Acopy = deepcopy(A)
        Acopy.data .= broadcast($f,Acopy.data)
        return Acopy
    end
end

for f in (:real, :imag, :abs, :abs2)
  @eval function $f(A::GridData{NX,NY,T}) where {NX,NY,T <: ComplexF64}
      Acopy = similar(A,element_type=Float64)
      Acopy.data .= broadcast($f,A.data)
      return Acopy
  end
end


# This enables fused in-place broadcasting of all grid data
Base.BroadcastStyle(::Type{<:GridData}) = Broadcast.ArrayStyle{GridData}()


function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{GridData}},::Type{T}) where {T}
    similar(unpack(bc),element_type=T)
end

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{GridData}})
    similar(unpack(bc))
end


#=
@inline unpack(bc::Broadcast.Broadcasted, i) = Broadcast.Broadcasted(bc.f, unpack_args(i, bc.args))
unpack(x,::Any) = x
unpack(x::GridData, ::Nothing) = x

@inline unpack_args(i, args::Tuple) = (unpack(args[1], i), unpack_args(i, Base.tail(args))...)
unpack_args(i, args::Tuple{Any}) = (unpack(args[1], i),)
unpack_args(::Any, args::Tuple{}) = ()
=#

function Base.copyto!(dest::T,bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{GridData}}) where {T <: GridData}
    copyto!(dest.data,unpack_data(bc, nothing))
    dest
end
