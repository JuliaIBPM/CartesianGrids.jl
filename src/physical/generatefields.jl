### Routines for constructing smooth functions that generate field data ###

import Base: *, -, +, show

export AbstractSpatialField, Gaussian, radius, center, strength,
       EmptySpatialField,
       SpatialGaussian, GeneratedField, datatype,
       PulseField

abstract type AbstractSpatialField end


## Empty spatial field

"""
    EmptySpatialField()

Create a blank spatial field. This is primarily useful for initializing a
sum of spatial fields.

# Example
```jldoctest
julia> g = EmptySpatialField()
EmptySpatialField()

julia> g(2,3)
0.0
```
"""
struct EmptySpatialField <: AbstractSpatialField end
(g::EmptySpatialField)(x,y) = Float64(0)

## Gaussian

"""
    Gaussian(σ,x0,A)

Construct a 1-d Gaussian function centered at `x0` with standard deviation `σ`
and amplitude `A`. The resulting function can be evaluated at any real-valued
number.

# Example

```jldoctest
julia> g = Gaussian(0.2,0,1)
Gaussian(0.2, 0, 1, 2.8209479177387813)

julia> g(0.2)
1.0377687435514866
```
"""
struct Gaussian
  σ :: Real
  x0 :: Real
  A :: Real
  fact :: Float64
end

Gaussian(σ,x0,A) = Gaussian(σ,x0,A,A/sqrt(π*σ^2))

radius(g::Gaussian) = g.σ
center(g::Gaussian) = g.x0
strength(g::Gaussian) = g.A

@inline gaussian(r;tol=6.0) = abs(r) < tol ? exp(-r^2) : 0.0

(g::Gaussian)(x) = g.fact*gaussian((x-center(g))/radius(g))

## Spatial Gaussian field ##

struct SpatialGaussian <: AbstractSpatialField
  gx :: Gaussian
  gy :: Gaussian
  A :: Real
end

SpatialGaussian(σx::Real,σy::Real,x0::Real,y0::Real,A::Real) =
          SpatialGaussian(Gaussian(σx,x0,A),Gaussian(σy,y0,1),A)

SpatialGaussian(σ::Real,x0::Real,y0::Real,A::Real) = SpatialGaussian(σ,σ,x0,y0,A)

(g::SpatialGaussian)(x,y) = g.gx(x)*g.gy(y)

## Scaling spatial fields

struct ScaledField{N <: Real, P <: AbstractSpatialField} <: AbstractSpatialField
    s::N
    p::P
end
function show(io::IO, p::ScaledField)
    print(io, "$(p.s) × ($(p.p))")
end
s::Number * p::AbstractSpatialField = ScaledField(s, p)
-(p::AbstractSpatialField) = ScaledField(-1, p)
(p::ScaledField)(x,y) = p.s*p.p(x,y)

struct AddedFields{T <: Tuple} <: AbstractSpatialField
    ps::T
end
function show(io::IO, Σp::AddedFields)
    println(io, "AddedFields:")
    for p in Σp.ps
        println(io, "  $p")
    end
end

## Adding spatial fields together

"""
    p₁::AbstractSpatialField + p₂::AbstractSpatialField

Add the fields so that `(p₁ + p₂)(x,y) = p₁(x,y) + p₂(x,y)`.

"""
+(p::AbstractSpatialField, Σp::AddedFields) = AddedFields((p, Σp.ps...))
+(Σp::AddedFields, p::AbstractSpatialField) = AddedFields((Σp.ps..., p))
function +(Σp₁::AddedFields, Σp₂::AddedFields)
    AddedFields((Σp₁..., Σp₂...))
end

-(p₁::AbstractSpatialField, p₂::AbstractSpatialField) = p₁ + (-p₂)

+(p::AbstractSpatialField...) = AddedFields(p)

function (Σp::AddedFields)(x,y)
    f = 0.0
    for p in Σp.ps
        f += p(x,y)
    end
    f
end

## For generating an actual instance of a spatial field
"""
    GeneratedField(d::ScalarGridData,field::AbstractSpatialField,grid::PhysicalGrid)

Create an instance of a spatial field function `field` on scalar grid data `d`,
based on a grid `grid`. After creating the instance `g = GeneratedField(d,field,grid)`,
then the resulting grid data can be accessed by typing `g()`.
"""
struct GeneratedField{T <: ScalarGridData}
    fielddata :: T
    field :: AbstractSpatialField
    xg :: AbstractRange
    yg :: AbstractRange
end

function GeneratedField(d::ScalarGridData,field::AbstractSpatialField,g::PhysicalGrid)
    xg, yg = coordinates(d,g)
    tmp = typeof(d)(field.(xg*ones(1,length(yg)),ones(length(xg))*yg'))
    GeneratedField{typeof(tmp)}(tmp,field,xg,yg)
end

(f::GeneratedField)() = f.fielddata
datatype(f::GeneratedField{T}) where {T} = T

## For generating a transient form of a spatial field

"""
    PulseField(g::GeneratedField,t0,σ)

Create a transient form of a generated spatial field, useful for
introducing a short-lived forcing field onto the grid. The supplied field `g`
is modulated by a Gaussian pulse centered at time `t0` with width `σ`. This pulse's
maximum is unity, so that the amplitude of the overall field is set by `g`. The
resulting object can be evaluated with a single argument (time) and returns a
`ScalarGridData` type object of the same type contained in `g`.
"""
struct PulseField
  gfield :: GeneratedField
  timemod :: Gaussian
end

PulseField(gfield::GeneratedField,t0::Real,σt::Real) = PulseField(gfield,Gaussian(σt,t0,sqrt(π*σt^2)))

(f::PulseField)(t) = f.timemod(t)*f.gfield()


datatype(f::PulseField) = datatype(f.gfield)


## Allow all spatial fields to accept a vector argument

#for T in InteractiveUtils.subtypes(AbstractSpatialField)
#    @eval (f::$T)(x::Vector{R}) where {R <: Float64} = f(x...)
#end
