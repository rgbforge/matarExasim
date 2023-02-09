__precompile__()

module Mesh

export squaremesh, cubemesh, gmshcall, mesh1D
export readmesh, writemesh

include("squaremesh.jl");
include("cubemesh.jl");
include("gmshcall.jl");
include("readmesh.jl");
include("writemesh.jl");
include("mesh1D.jl")

end
