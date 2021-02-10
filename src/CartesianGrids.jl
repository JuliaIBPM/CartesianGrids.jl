module CartesianGrids

import Base: @propagate_inbounds, show, summary, fill!

#using Compat
using FFTW
using SpecialFunctions
using Statistics

using LinearAlgebra
using SparseArrays
using Interpolations

import LinearAlgebra: mul!, ldiv!, cross, ×, dot, ⋅

import Base: parentindices
const GAMMA = MathConstants.γ

export Primal, Dual, ScalarGridData, VectorGridData, TensorGridData, GridData,
       CollectedGridData,
       PointData, ScalarData, VectorData, TensorData,
       celltype, griddatatype, indexshift,
       diff!,grid_interpolate!,
       curl, curl!, Curl, divergence, divergence!, Divergence,
       grad, grad!, Grad,
       laplacian, laplacian!, laplacian_symm!, plan_laplacian, plan_laplacian!,
       helmholtz, helmholtz!, plan_helmholtz, plan_helmholtz!,
       plan_intfact,plan_intfact!,Identity,
       product, product!, ∘,
       magsq!,magsq,mag!,mag,
       directional_derivative!, directional_derivative_conserve!, curl_cross!,
       convective_derivative!, convective_derivative_rot!,
       DDF, GradDDF,
       Regularize, RegularizationMatrix, InterpolationMatrix,
       CircularConvolution,
       AbstractSpatialField, Gaussian, DGaussian, radius, center, strength,
       EmptySpatialField,
       SpatialGaussian,GeneratedField, datatype, grid, PulseField, ModulatedField

abstract type CellType end
abstract type Primal <: CellType end
abstract type Dual <: CellType end

abstract type GridData{NX,NY,T} <: AbstractMatrix{T} end
abstract type ScalarGridData{NX,NY,T} <: GridData{NX,NY,T} end
abstract type VectorGridData{NX,NY,T} <: GridData{NX,NY,T} end
abstract type TensorGridData{NX,NY,T} <: GridData{NX,NY,T} end
CollectedGridData = Union{VectorGridData,TensorGridData}

# List of scalar grid types. Each pair of numbers specifies
# the number of grid points in each direction for this data type, relative
# to the reference grid. The two pairs of numbers correspond to Primal
# and Dual versions of this grid data type.
const SCALARLIST = [ :Nodes, (-1,-1), (0,0)],
                   [ :XEdges, (0,-1), (-1,0)],
                   [ :YEdges, (-1,0), (0,-1)]

# List of collection grid types. This specifies the list of different types
# to create regularization and coordinates routines for.
const VECTORLIST = [:Edges, [:Primal]],
                   [:Edges, [:Dual]],
                   [:NodePair, [:Primal,:Dual]],
                   [:NodePair, [:Dual,:Primal]]

const TENSORLIST = [:EdgeGradient, [:Dual,:Primal]],
                   [:EdgeGradient, [:Primal,:Dual]]


function othertype end

macro othertype(celltype, k)
    esc(quote
        othertype(::$celltype) = $k
        othertype(::Type{$celltype}) = $k
    end)
end

@othertype Primal Dual
@othertype Dual Primal
@othertype CellType CellType

#@wraparray ScalarGridData data 2

include("fields/fieldmacros.jl")
include("fields/scalargrid.jl")

# Generate the scalar grid field types and associated functions
for (wrapper,primaldn,dualdn) in SCALARLIST
    @eval @scalarfield $wrapper $primaldn $dualdn
end

include("fields/collections.jl")

include("gridoperations/basicoperations.jl")
include("gridoperations/innerproducts.jl")
include("gridoperations/convolution.jl")
include("gridoperations/lgf.jl")
include("gridoperations/lgf-helmholtz.jl")
include("gridoperations/laplacian.jl")
include("gridoperations/helmholtz.jl")
include("gridoperations/intfact.jl")
include("gridoperations/diffcalculus.jl")
include("gridoperations/differencing1d.jl")
include("gridoperations/interpolation1d.jl")
include("gridoperations/shift.jl")
include("gridoperations/nlcalculus.jl")

include("points/points.jl")
include("points/ddf.jl")
include("points/regularization.jl")

include("physical/physicalgrid.jl")
include("physical/interpolation.jl")
include("physical/generatefields.jl")


#== Plot Recipes ==#

include("plot_recipes.jl")

end
