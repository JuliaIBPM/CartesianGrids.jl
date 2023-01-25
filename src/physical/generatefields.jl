### Routines for constructing smooth functions that generate field data ###

import Base: *, -, +, show

abstract type AbstractSpatialField end
abstract type AbstractGeneratedField end

#include("motionprofiles.jl")
using SpaceTimeFields
import SpaceTimeFields: Abstract1DProfile, radius, strength


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
(g::EmptySpatialField)(a...) = Float64(0)

## Basic spatial field ##
"""
    SpatialField(f::Function)

Creates a lazy instance of a spatial field from a function `f`, which
must have the signature `f(x,y)`.
"""
struct SpatialField{FT<:Function} <: AbstractSpatialField
  f :: FT
end
(field::SpatialField)(x,y) = field.f(x,y)
(field::SpatialField)(x,y,t) = field.f(x,y) # ignore the time argument

"""
    SpatialTemporalField(f::Function)

Creates a lazy instance of a spatial-temporal field from a function `f`, which
must have the signature `f(x,y,t)`.
"""
struct SpatialTemporalField{FT<:Function} <: AbstractSpatialField
  f :: FT
end
(field::SpatialTemporalField)(x,y,t) = field.f(x,y,t)


## Spatial Gaussian field ##

"""
    SpatialGaussian(σx,σy,x0,y0,A[,derivdir=0])

Set up a spatial field in the form of a Gaussian centered at `x0,y0` with
radii `σx` and `σy` in the respective directions and amplitude `A`. If the
optional parameter `deriv` is set to 1 or 2, then it returns the first
derivative of a Gaussian in that direction (`x` or `y`, respectively).

`SpatialGaussian(σx,σy,x0,y0,A,u,v[,derivdir=0])` generates a Gaussian
that convects at velocity `(u,v)`. It can be evaluated with an additional
argument for time.
"""
struct SpatialGaussian{CT,GX,GY} <: AbstractSpatialField
  gx :: GX
  gy :: GY
  A :: Float64
  u :: Float64
  v :: Float64
  SpatialGaussian(gx::Abstract1DProfile,gy::Abstract1DProfile,A,u,v) = new{true,typeof(gx),typeof(gy)}(gx,gy,A,u,v)
  SpatialGaussian(gx::Abstract1DProfile,gy::Abstract1DProfile,A) = new{false,typeof(gx),typeof(gy)}(gx,gy,A,0.0,0.0)
end

SpatialGaussian(σx::Real,σy::Real,x0::Real,y0::Real,A::Real;deriv::Int=0) =
                _spatialdgaussian(σx,σy,x0,y0,A,Val(deriv))
SpatialGaussian(σx::Real,σy::Real,x0::Real,y0::Real,A::Real,u::Real,v::Real;deriv::Int=0) =
                _spatialdgaussian(σx,σy,x0,y0,A,u,v,Val(deriv))


_spatialdgaussian(σx,σy,x0,y0,A,::Val{0}) = SpatialGaussian(Gaussian(σx,A) >> x0,
                                                            Gaussian(σy,1) >> y0,A)
_spatialdgaussian(σx,σy,x0,y0,A,::Val{1}) = SpatialGaussian(DGaussian(σx,A) >> x0,
                                                            Gaussian(σy,1) >> y0,A)
_spatialdgaussian(σx,σy,x0,y0,A,::Val{2}) = SpatialGaussian(Gaussian(σx,A) >> x0,
                                                            DGaussian(σy,1) >> y0,A)

_spatialdgaussian(σx,σy,x0,y0,A,u,v,::Val{0}) = SpatialGaussian(Gaussian(σx,A) >> x0,
                                                                Gaussian(σy,1) >> y0,A,u,v)
_spatialdgaussian(σx,σy,x0,y0,A,u,v,::Val{1}) = SpatialGaussian(DGaussian(σx,A) >> x0,
                                                                Gaussian(σy,1) >> y0,A,u,v)
_spatialdgaussian(σx,σy,x0,y0,A,u,v,::Val{2}) = SpatialGaussian(Gaussian(σx,A) >> x0,
                                                                DGaussian(σy,1) >> y0,A,u,v)


SpatialGaussian(σ,x0,y0,A;deriv::Int=0) = SpatialGaussian(σ,σ,x0,y0,A,deriv=deriv)


(g::SpatialGaussian{GX,GY})(x,y) where {GX,GY} = g.gx(x)*g.gy(y)
# ignore the time argument if it is called with this...
(g::SpatialGaussian{false,GX,GY})(x,y,t) where {GX,GY} = g(x,y)
(g::SpatialGaussian{true,GX,GY})(x,y,t) where {GX,GY} = g.gx(x-g.u*t)*g.gy(y-g.v*t)


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
(p::ScaledField)(x,y,t) = p.s*p.p(x,y,t)


## Adding spatial fields together.

struct AddedFields{T <: Tuple} <: AbstractSpatialField
    ps::T
end
function show(io::IO, Σp::AddedFields)
    println(io, "AddedFields:")
    for p in Σp.ps
        println(io, "  $p")
    end
end

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

# Evaluate at x,y , assuming that t = 0 for any time-varying member
function (Σp::AddedFields)(x,y)
    f = 0.0
    for p in Σp.ps
        f += p(x,y,0.0)
    end
    f
end

# Evaluate at x,y,t, ignoring t for any constant member
function (Σp::AddedFields)(x,y,t)
    f = 0.0
    for p in Σp.ps
        f += p(x,y,t)
    end
    f
end

## For generating an actual instance of a spatial field
"""
    GeneratedField(d::GridData,field::AbstractSpatialField...,grid::PhysicalGrid)

Create an instance of a spatial field function `field` on scalar grid data `d`,
based on a grid `grid`. After creating the instance `g = GeneratedField(d,field,grid)`,
then the resulting grid data can be accessed by typing `g()`. For vector grid data,
a separate `field` must be supplied for each component.

If the fields are time dependent, then you can also evaluate `g(t)` at the
desired time. The time argument is ignored if the fields are static.
"""
struct GeneratedField{T <: GridData}
    fielddata :: T
    fieldfcns :: Vector{AbstractSpatialField}
    grid :: PhysicalGrid
end

function GeneratedField(d::ScalarGridData,field::AbstractSpatialField,g::PhysicalGrid)
    #xg, yg = coordinates(d,g)
    #tmp = _generatedfield(xg,yg,field,d,0.0)
    tmp = similar(d)
    _generatedfield!(tmp,field,g,0.0)
    GeneratedField{typeof(tmp)}(tmp,AbstractSpatialField[field],g)
end

function GeneratedField(d::CollectedGridData,fields::Vector{AbstractSpatialField},g::PhysicalGrid)
  @assert length(fields) == _numberofcomponents(typeof(d))
  tmp = similar(d)
  _generatedfield!(tmp,fields,g,0.0)
  GeneratedField{typeof(tmp)}(tmp,fields,g)
end

function _generatedfield!(d::ScalarGridData,field::AbstractSpatialField,g::PhysicalGrid,t::Real)
    xg, yg = coordinates(d,g)
    d .= _generatedfield(xg,yg,field,d,t)
    return d
end

_generatedfield!(d::ScalarGridData,fields::Vector{AbstractSpatialField},g::PhysicalGrid,t::Real) =
    _generatedfield!(d,fields[1],g,t)


function _generatedfield!(d::CollectedGridData,fields::Vector{AbstractSpatialField},g::PhysicalGrid,t::Real)
  cnt = 0
  for (i,fname) in enumerate(propertynames(d))
    if typeof(getfield(d,fname)) <: GridData
        cnt += 1
        x, y = coordinates(getfield(d,fname),g)
        getfield(d,fname) .= _generatedfield(x,y,fields[cnt],getfield(d,fname),t)
    end
  end
  return d
end

GeneratedField(d::VectorGridData,
      fieldu::AbstractSpatialField,fieldv::AbstractSpatialField,
      g::PhysicalGrid) = GeneratedField(d,AbstractSpatialField[fieldu,fieldv],g)


_generatedfield(xg,yg,field::AbstractSpatialField,d::ScalarGridData,t) =
          typeof(d)(_evaluatefield(field,xg,yg,t))

_evaluatefield(field,xg,yg,t) = field.(xg*ones(1,length(yg)),ones(length(xg))*yg',t)

datatype(f::GeneratedField{T}) where {T} = T
grid(f::GeneratedField{T}) where {T} = f.grid

(f::GeneratedField{T})() where {T} = f.fielddata

function (f::GeneratedField)(t::Real)
    _generatedfield!(f.fielddata,f.fieldfcns,f.grid,t)
end



## For generating a transient form of a spatial field
"""
    ModulatedField(g::GeneratedField,modfcn::Abstract1DProfile)

Create a time-modulated form of a generated spatial field, useful for
introducing a forcing field onto the grid. The supplied field `g`
is modulated by a function `modfcn` with a specified profle shape. The
resulting object can be evaluated with a single argument (time) and returns a
`GridData` type object of the same type contained in `g`.
"""
struct ModulatedField
  gfield :: GeneratedField
  modfcn :: Abstract1DProfile
end

(f::ModulatedField)(t) = f.modfcn(t)*f.gfield()

datatype(f::ModulatedField) = datatype(f.gfield)


"""
    PulseField(g::GeneratedField,t0,σ)

Create a transient form of a generated spatial field, useful for
introducing a short-lived forcing field onto the grid. The supplied field `g`
is modulated by a Gaussian pulse centered at time `t0` with width `σ`. This pulse's
maximum is unity, so that the amplitude of the overall field is set by `g`. The
resulting object can be evaluated with a single argument (time) and returns a
`ScalarGridData` type object of the same type contained in `g`.
"""
#=
struct PulseField
  gfield :: GeneratedField
  modfcn :: Abstract1DProfile
end
=#

PulseField(gfield::GeneratedField{T},t0::Real,σt::Real) where {T <: GridData} =
            ModulatedField(gfield,Gaussian(σt,sqrt(π*σt^2)) >> t0)


#(f::PulseField)(t) = f.modfcn(t)*f.gfield()

#datatype(f::PulseField) = datatype(f.gfield)


## Allow all spatial fields to accept a vector argument

#for T in InteractiveUtils.subtypes(AbstractSpatialField)
#    @eval (f::$T)(x::Vector{R}) where {R <: Float64} = f(x...)
#end
