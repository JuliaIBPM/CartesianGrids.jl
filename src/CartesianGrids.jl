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

export Primal, Dual, ScalarGridData, VectorGridData, GridData,
       Points, ScalarData, VectorData, TensorData,
       celltype, griddatatype, indexshift,
       diff!,grid_interpolate!,
       curl, curl!, Curl, divergence, divergence!, Divergence,
       grad, grad!, Grad,
       laplacian, laplacian!, laplacian_symm!, plan_laplacian, plan_laplacian!,
       helmholtz, helmholtz!, plan_helmholtz, plan_helmholtz!,
       plan_intfact,plan_intfact!,Identity,
       product, product!, ∘,
       directional_derivative!, directional_derivative_conserve!, curl_cross!,
       convective_derivative!, convective_derivative_rot!,
       DDF, GradDDF,
       Regularize, RegularizationMatrix, InterpolationMatrix,
       CircularConvolution

abstract type CellType end
abstract type Primal <: CellType end
abstract type Dual <: CellType end

abstract type GridData{NX,NY,T} <: AbstractMatrix{T} end

abstract type ScalarGridData{NX,NY,T} <: GridData{NX,NY,T} end

abstract type VectorGridData{NX,NY,T} <: GridData{NX,NY,T} end

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

include("fieldmacros.jl")
include("scalargrid.jl")

# Generate the scalar grid field types and associated functions
for (wrapper,primaldn,dualdn) in SCALARLIST
    @eval @scalarfield $wrapper $primaldn $dualdn
end

include("collections.jl")

include("basicoperations.jl")
include("points.jl")

#CollectedData = Union{EdgeGradient{R,S,NX,NY,T},NodePair{R,S,NX,NY,T}} where {R,S,NX,NY,T}

include("physicalgrid.jl")
include("operators.jl")


end
