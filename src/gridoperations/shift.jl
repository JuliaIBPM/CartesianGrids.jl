# Grid interpolation operations

"""
    grid_interpolate!(q::Edges{Dual},w::Nodes{Dual})

Interpolate the dual nodal data `w` to the edges of the dual
cells, and return the result in `q`.

# Example

```jldoctest
julia> w = Nodes(Dual,(8,6));

julia> w[3,4] = 1.0;

julia> q = Edges(Dual,w);

julia> grid_interpolate!(q,w)
Edges{Dual,8,6,Float64} data
u (in grid orientation)
6×7 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.5  0.5  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
v (in grid orientation)
5×8 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.5  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.5  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
```
"""
function grid_interpolate!(q::Edges{Dual, NX, NY}, w::Nodes{Dual,NX, NY}) where {NX, NY}
    grid_interpolate!(q.u,w)
    grid_interpolate!(q.v,w)
    q
end

# This operation is not necessarily desirable, since its meaning is ambiguous
grid_interpolate(nodes::Nodes{Dual,NX,NY}) where {NX,NY} = grid_interpolate!(Edges(Dual, nodes), nodes)

"""
    grid_interpolate!(q::Edges{Primal},w::Nodes{Primal})

Interpolate the primal nodal data `w` to the edges of the primal cells,
and return the result in `q`.
"""
function grid_interpolate!(q::Edges{Primal, NX, NY}, w::Nodes{Primal,NX, NY}) where {NX, NY}
    grid_interpolate!(q.u,w)
    grid_interpolate!(q.v,w)
    q
end


"""
    grid_interpolate!((wx::Nodes,wy::Nodes),q::Edges)

Interpolate the edge data `q` (of either dual or primal
type) to the dual or primal nodes, and return the result in `wx` and `wy`. `wx`
holds the shifted `q.u` data and `wy` the shifted `q.v` data.

# Example

```jldoctest
julia> q = Edges(Primal,(8,6));

julia> q.u[3,2] = 1.0;

julia> wx = Nodes(Dual,(8,6)); wy = deepcopy(wx);

julia> grid_interpolate!((wx,wy),q);

julia> wx
Nodes{Dual,8,6,Float64} data
Printing in grid orientation (lower left is (1,1))
6×8 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.5  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.5  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0

julia> wy
Nodes{Dual,8,6,Float64} data
Printing in grid orientation (lower left is (1,1))
6×8 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
```
"""
function grid_interpolate!(out::Tuple{Nodes{C, NX, NY},Nodes{C, NX, NY}}, q::Edges{D,NX, NY}) where {C<:CellType, D<:CellType, NX, NY}
    grid_interpolate!(out[1],q.u)
    grid_interpolate!(out[2],q.v)
    out
end

"""
    grid_interpolate!(w::Nodes,q::Edges)

Interpolate the edge data `q` (of either dual or primal
type) to the dual or primal nodes, and return the result in `w`, which
represents a sum of the interpolations of each component of `q`.
"""
#function grid_interpolate!(out::Nodes{C, NX, NY}, q::Edges{D,NX, NY}) where {C<:CellType, D<:CellType, NX, NY}
#    u = zero(out)
#    grid_interpolate!((u,out),q)
#    out .+= u
#end
function grid_interpolate!(nodes::Nodes{Primal, NX, NY},edges::Edges{Primal, NX, NY}) where {NX, NY}

    u, v = edges.u, edges.v
    view(nodes,1:NX-1,1:NY-1) .=  view(u,2:NX,1:NY-1) .+ view(u,1:NX-1,1:NY-1) .+
                                  view(v,1:NX-1,2:NY) .+ view(v,1:NX-1,1:NY-1)
    #@inbounds for y in 1:NY-1, x in 1:NX-1
    #    nodes[x,y] = - u[x,y] + u[x+1,y] - v[x,y] + v[x,y+1]
    #end
    nodes .*= 0.5
    nodes
end
function grid_interpolate!(nodes::Nodes{Primal, NX, NY},edges::Edges{Dual, NX, NY}) where {NX, NY}

    u, v = edges.u, edges.v
    view(nodes,1:NX-1,1:NY-1) .=  view(u,1:NX-1,2:NY) .+ view(u,1:NX-1,1:NY-1) .+
                                  view(v,2:NX,1:NY-1) .+ view(v,1:NX-1,1:NY-1)
    #@inbounds for y in 1:NY-1, x in 1:NX-1
    #    nodes[x,y] = - u[x,y] + u[x+1,y] - v[x,y] + v[x,y+1]
    #end
    nodes .*= 0.5
    nodes
end

function grid_interpolate!(nodes::Nodes{Dual, NX, NY},edges::Edges{Dual, NX, NY}) where {NX, NY}

    u, v = edges.u, edges.v
    view(nodes,2:NX-1,2:NY-1) .= view(u,2:NX-1,2:NY-1) .+ view(u,1:NX-2,2:NY-1) .+
                                 view(v,2:NX-1,2:NY-1) .+ view(v,2:NX-1,1:NY-2)
    #@inbounds for y in 2:NY-1, x in 2:NX-1
    #    nodes[x,y] = - u[x-1,y] + u[x,y] - v[x,y-1] + v[x,y]
    #end
    nodes .*= 0.5
    nodes
end

function grid_interpolate!(nodes::Nodes{Dual, NX, NY},edges::Edges{Primal, NX, NY}) where {NX, NY}

    u, v = edges.u, edges.v
    view(nodes,2:NX-1,2:NY-1) .= view(u,2:NX-1,1:NY-2) .+ view(u,2:NX-1,2:NY-1) .+
                               view(v,2:NX-1,2:NY-1) .+ view(v,1:NX-2,2:NY-1)

    #@inbounds for y in 2:NY-1, x in 2:NX-1
    #    nodes[x,y] = - u[x-1,y] + u[x,y] - v[x,y-1] + v[x,y]
    #end
    nodes .*= 0.5
    nodes
end


"""
    grid_interpolate!(dq::EdgeGradient{Primal/Dual},q::Edges{Primal/Dual})

Interpolate the primal (dual) edge data `q` to primal (dual) tensor positions
and hold it in `dq`.
"""
function grid_interpolate!(dq::EdgeGradient{Primal, Dual, NX, NY}, q::Edges{Primal,NX, NY}) where {NX, NY}
    grid_interpolate!(dq.dudx,q.u)
    grid_interpolate!(dq.dudy,q.u)
    grid_interpolate!(dq.dvdx,q.v)
    grid_interpolate!(dq.dvdy,q.v)
    return dq
end

function grid_interpolate!(dq::EdgeGradient{Dual, Primal, NX, NY}, q::Edges{Dual,NX, NY}) where {NX, NY}
    grid_interpolate!(dq.dudx,q.u)
    grid_interpolate!(dq.dudy,q.u)
    grid_interpolate!(dq.dvdx,q.v)
    grid_interpolate!(dq.dvdy,q.v)
    return dq
end

"""
    grid_interpolate!(q::Edges{Primal/Dual},dq::EdgeGradient{Primal/Dual})

Interpolate the primal (dual) tensor data `dq` to primal (dual) edge positions
and hold it in `q`.
"""
function grid_interpolate!(edges::Edges{Primal, NX, NY},
                     nodes::EdgeGradient{Primal, Dual,NX, NY}) where {NX, NY}

    dudx, dudy, dvdx, dvdy = nodes.dudx, nodes.dudy, nodes.dvdx, nodes.dvdy
    u, v = edges.u, edges.v

    view(u,2:NX-1,1:NY-1) .= view(dudx,1:NX-2,1:NY-1) .+ view(dudx,2:NX-1,1:NY-1) .+
                                view(dudy,2:NX-1,1:NY-1) .+ view(dudy,2:NX-1,2:NY)
    view(v,1:NX-1,2:NY-1) .= view(dvdx,1:NX-1,2:NY-1) .+ view(dvdx,2:NX,2:NY-1) .+
                                view(dvdy,1:NX-1,1:NY-2) .+ view(dvdy,1:NX-1,2:NY-1)
    #@inbounds for y in 1:NY-1, x in 2:NX-1
    #    nodes[x,y] = - u[x-1,y] + u[x,y] - v[x,y] + v[x,y+1]
    #end
    edges .*= 0.5
    edges
end
function grid_interpolate!(edges::Edges{Dual, NX, NY},
                     nodes::EdgeGradient{Dual, Primal,NX, NY}) where {NX, NY}

    dudx, dudy, dvdx, dvdy = nodes.dudx, nodes.dudy, nodes.dvdx, nodes.dvdy
    u, v = edges.u, edges.v

    view(u,1:NX-1,2:NY-1) .= view(dudx,1:NX-1,2:NY-1) .+ view(dudx,2:NX,2:NY-1) .+
                                view(dudy,1:NX-1,1:NY-2) .+ view(dudy,1:NX-1,2:NY-1)
    view(v,2:NX-1,1:NY-1) .= view(dvdx,1:NX-2,1:NY-1) .+ view(dvdx,2:NX-1,1:NY-1) .+
                                view(dvdy,2:NX-1,1:NY-1) .+ view(dvdy,2:NX-1,2:NY)

    #@inbounds for y in 1:NY-1, x in 2:NX-1
    #    nodes[x,y] = - u[x-1,y] + u[x,y] - v[x,y] + v[x,y+1]
    #end
    edges .*= 0.5
    edges
end

# (Dual/Primal) edges to (Primal/Dual) edges. These require some rethinking,
# since they lead to broadened stencils.

function grid_interpolate!(dual::XEdges{Dual, NX, NY},
                           primal::XEdges{Primal, NX, NY}) where {NX, NY}

    view(dual,1:NX-1,2:NY-1) .= 0.25.*(view(primal,1:NX-1,2:NY-1) .+ view(primal,2:NX,2:NY-1) .+
                                       view(primal,1:NX-1,1:NY-2) .+ view(primal,2:NX,1:NY-2))
    dual
end

function grid_interpolate!(dual::YEdges{Dual, NX, NY},
                           primal::YEdges{Primal, NX, NY}) where {NX, NY}

    view(dual,2:NX-1,1:NY-1) .= 0.25.*(view(primal,2:NX-1,1:NY-1) .+ view(primal,1:NX-2,1:NY-1) .+
                                       view(primal,2:NX-1,2:NY)   .+ view(primal,1:NX-2,2:NY))
    dual
end

function grid_interpolate!(primal::XEdges{Primal, NX, NY},
                           dual::XEdges{Dual, NX, NY}) where {NX, NY}

    view(primal,2:NX-1,1:NY-1) .= 0.25.*(view(dual,2:NX-1,1:NY-1) .+ view(dual,1:NX-2,1:NY-1) .+
                                         view(dual,2:NX-1,2:NY)   .+ view(dual,1:NX-2,2:NY))
    primal
end

function grid_interpolate!(primal::YEdges{Primal, NX, NY},
                           dual::YEdges{Dual, NX, NY}) where {NX, NY}

    view(primal,1:NX-1,2:NY-1) .= 0.25.*(view(dual,1:NX-1,2:NY-1) .+ view(dual,2:NX,2:NY-1) .+
                                         view(dual,1:NX-1,1:NY-2) .+ view(dual,2:NX,1:NY-2))
    primal
end

"""
    grid_interpolate!(v::Edges{Dual/Primal},q::Edges{Primal/Dual})

Interpolate the primal (resp. dual) edge data `q` to the
edges of the dual (resp. primal) cells, and return the result in `v`.

# Example

```jldoctest
julia> q = Edges(Primal,(8,6));

julia> q.u[3,2] = 1.0;

julia> v = Edges(Dual,(8,6));

julia> grid_interpolate!(v,q)
Edges{Dual,8,6,Float64} data
u (in grid orientation)
6×7 Array{Float64,2}:
 0.0  0.0   0.0   0.0  0.0  0.0  0.0
 0.0  0.0   0.0   0.0  0.0  0.0  0.0
 0.0  0.0   0.0   0.0  0.0  0.0  0.0
 0.0  0.25  0.25  0.0  0.0  0.0  0.0
 0.0  0.25  0.25  0.0  0.0  0.0  0.0
 0.0  0.0   0.0   0.0  0.0  0.0  0.0
v (in grid orientation)
5×8 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
```
"""


function grid_interpolate!(dual::Edges{Dual, NX, NY},
                primal::Edges{Primal, NX, NY}) where {NX, NY}

    grid_interpolate!(dual.u,primal.u)
    grid_interpolate!(dual.v,primal.v)
    #uₚ = primal.u
    #view(dual.u,1:NX-1,2:NY-1) .= 0.25.*(view(uₚ,1:NX-1,2:NY-1) .+ view(uₚ,2:NX,2:NY-1) .+
    #                                   view(uₚ,1:NX-1,1:NY-2) .+ view(uₚ,2:NX,1:NY-2))

    #vₚ = primal.v
    #view(dual.v,2:NX-1,1:NY-1) .= 0.25.*(view(vₚ,2:NX-1,1:NY-1) .+ view(vₚ,1:NX-2,1:NY-1) .+
    #                                   view(vₚ,2:NX-1,2:NY)   .+ view(vₚ,1:NX-2,2:NY))

    # @inbounds for y in 2:NY-1, x in 1:NX-1
    #     dual.u[x,y] = (uₚ[x,y] + uₚ[x+1,y] + uₚ[x,y-1] + uₚ[x+1,y-1])/4
    # end
    #
    # vₚ = primal.v
    # @inbounds for y in 1:NY-1, x in 2:NX-1
    #     dual.v[x,y] = (vₚ[x,y] + vₚ[x-1,y] + vₚ[x,y+1] + vₚ[x-1,y+1])/4
    # end
    dual
end

function grid_interpolate(primal::Edges{Primal, NX, NY}) where {NX, NY}
    grid_interpolate!(Edges(Dual, (NX, NY)), primal)
end

function grid_interpolate!(primal::Edges{Primal, NX, NY},
                dual::Edges{Dual, NX, NY}) where {NX, NY}
    grid_interpolate!(primal.u,dual.u)
    grid_interpolate!(primal.v,dual.v)
    #uₚ = dual.u
    #view(primal.u,2:NX-1,1:NY-1) .= 0.25.*(view(uₚ,2:NX-1,1:NY-1) .+ view(uₚ,1:NX-2,1:NY-1) .+
    #                                     view(uₚ,2:NX-1,2:NY)   .+ view(uₚ,1:NX-2,2:NY))
    #vₚ = dual.v
    #view(primal.v,1:NX-1,2:NY-1) .= 0.25.*(view(vₚ,1:NX-1,2:NY-1) .+ view(vₚ,2:NX,2:NY-1) .+
    #                                     view(vₚ,1:NX-1,1:NY-2) .+ view(vₚ,2:NX,1:NY-2))
    # @inbounds for y in 1:NY-1, x in 2:NX-1
    #     primal.u[x,y] = (uₚ[x,y] + uₚ[x-1,y] + uₚ[x,y+1] + uₚ[x-1,y+1])/4
    # end
    #
    # vₚ = dual.v
    # @inbounds for y in 2:NY-1, x in 1:NX-1
    #     primal.v[x,y] = (vₚ[x,y] + vₚ[x+1,y] + vₚ[x,y-1] + vₚ[x+1,y-1])/4
    # end
    primal
end

function grid_interpolate(dual::Edges{Dual, NX, NY}) where {NX, NY}
    grid_interpolate!(Edges(Primal, dual), dual)
end

"""
    grid_interpolate!(v::Nodes{Dual/Primal},q::Nodes{Primal/Dual})

Interpolate the primal (resp. dual) node data `q` to the
edges of the dual (resp. primal) nodes, and return the result in `v`.
"""
function grid_interpolate!(primal::Nodes{Primal, NX, NY},
                           dual::Nodes{Dual, NX, NY}) where {NX, NY}
    view(primal,1:NX-1,1:NY-1) .= 0.25.*(view(dual,1:NX-1,1:NY-1) .+ view(dual,2:NX,1:NY-1) .+
                                         view(dual,1:NX-1,2:NY)   .+ view(dual,2:NX,2:NY))
    primal
end

function grid_interpolate!(dual::Nodes{Dual, NX, NY},
                           primal::Nodes{Primal, NX, NY}) where {NX, NY}
    view(dual,2:NX-1,2:NY-1) .= 0.25.*(view(primal,1:NX-2,1:NY-2) .+ view(primal,2:NX-1,1:NY-2) .+
                                       view(primal,1:NX-2,2:NY-1) .+ view(primal,2:NX-1,2:NY-1))
    dual
end




# I don't like this one. It is ambiguous what type of nodes are being shifted to.
nodeshift(edges::Edges{Primal,NX,NY}) where {NX,NY} = grid_interpolate!((Nodes(Dual, edges),Nodes(Dual, edges)),edges)
