using CartesianGrids
using Test

const GROUP = get(ENV, "GROUP", "All")

if GROUP == "All" || GROUP == "Fields"
  include("fields.jl")
end

if GROUP == "All" || GROUP == "DDF"
  include("ddf.jl")
end

if GROUP == "All" || GROUP == "Points"
  include("points.jl")
end

if GROUP == "All" || GROUP == "GeneneratedFields"
  include("generatedfields.jl")
end
