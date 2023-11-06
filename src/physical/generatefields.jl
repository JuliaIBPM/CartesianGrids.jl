### Routines for generating field data with specified functions ###


abstract type AbstractGeneratedField end

#include("motionprofiles.jl")
#import SpaceTimeFields: Abstract1DProfile, AbstractSpatialField


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
