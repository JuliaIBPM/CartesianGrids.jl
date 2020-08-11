import Base: -, +, *, /, \, ∘, zero, conj, real, imag, abs, abs2


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


zero(::Type{T}) where {T <: GridData} = T()

#### ON COMPLEX GRID DATA

for f in (:conj,)
    @eval function $f(A::GridData{NX,NY,T}) where {NX,NY,T <: ComplexF64}
        Acopy = deepcopy(A)
        Acopy .= broadcast($f,Acopy)
        return Acopy
    end
end

for f in (:real, :imag, :abs, :abs2)
  @eval function $f(A::GridData{NX,NY,T}) where {NX,NY,T <: ComplexF64}
      Acopy = similar(A,element_type=Float64)
      Acopy .= broadcast($f,A)
      return Acopy
  end
end
