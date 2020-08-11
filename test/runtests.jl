
using CartesianGrids
using Test
##using TestSetExtensions


#@test isempty(detect_ambiguities(ViscousFlow))
include("fields.jl")
include("points.jl")
include("generatedfields.jl")



#@testset ExtendedTestSet "All tests" begin
#    @includetests ARGS
#end

#if isempty(ARGS)
#    include("../docs/make.jl")
#end
