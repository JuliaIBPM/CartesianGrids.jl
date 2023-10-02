using LinearAlgebra
import LinearAlgebra:dot, norm

export integrate

#=
Note that all of the inner product operations defined here are based on a
domain filled with dual cells, in which the dual nodes are at their centers, and surrounded
by a layer of padding (i.e. ghost) cells.
=#

"""
    dot(p1::Nodes{Dual},p2::Nodes{Dual}) -> Real

Computes the inner product between two sets of dual node data on the same grid.
"""
function dot(p1::Nodes{Dual,NX,NY},p2::Nodes{Dual,NX,NY}) where {NX,NY}
  # remember that sizes NX and NY include the ghost cells
  dims = node_inds(Dual,(NX,NY))
  p1_int = view(p1,2:dims[1]-1,2:dims[2]-1)
  p2_int = view(p2,2:dims[1]-1,2:dims[2]-1)
  return dot(p1_int,p2_int)/((NX-2)*(NY-2))

end

"""
    dot(p1::Nodes{Primal},p2::Nodes{Primal}) -> Real

Computes the inner product between two sets of primal node data on the same grid.
"""
function dot(p1::Nodes{Primal,NX,NY},p2::Nodes{Primal,NX,NY}) where {NX,NY}

  dims = node_inds(Primal,(NX,NY))

  # interior
  tmp = dot(view(p1,2:dims[1]-1,2:dims[2]-1),view(p2,2:dims[1]-1,2:dims[2]-1))

  # boundaries
  tmp += 0.5*dot(view(p1,1,2:dims[2]-1),      view(p2,1,2:dims[2]-1))
  tmp += 0.5*dot(view(p1,dims[1],2:dims[2]-1),view(p2,dims[1],2:dims[2]-1))
  tmp += 0.5*dot(view(p1,2:dims[1]-1,1),      view(p2,2:dims[1]-1,1))
  tmp += 0.5*dot(view(p1,2:dims[1]-1,dims[2]),view(p2,2:dims[1]-1,dims[2]))

  # corners -- use dot to ensure it works for complex types
  tmp += 0.25*dot([p1[1,1],p1[dims[1],1],p1[1,dims[2]],p1[dims[1],dims[2]]],
      [p2[1,1],p2[dims[1],1],p2[1,dims[2]],p2[dims[1],dims[2]]])
  #tmp += 0.25*(p1[1,1]*p2[1,1]             + p1[dims[1],1]*p2[dims[1],1] +
  #             p1[1,dims[2]]*p2[1,dims[2]] + p1[dims[1],dims[2]]*p2[dims[1],dims[2]])

  return tmp/((NX-2)*(NY-2))
end

"""
    dot(p1::XEdges{Dual},p2::XEdges{Dual}) -> Real

Computes the inner product between two sets of dual x-edge component data on the same grid.
"""
function dot(p1::XEdges{Dual,NX,NY},p2::XEdges{Dual,NX,NY}) where {NX,NY}

  udims = xedge_inds(Dual,(NX,NY))

  # interior
  tmp = dot(view(p1,2:udims[1]-1,2:udims[2]-1),view(p2,2:udims[1]-1,2:udims[2]-1))

  # boundaries
  tmp += 0.5*dot(view(p1,1,       2:udims[2]-1),view(p2,1,       2:udims[2]-1))
  tmp += 0.5*dot(view(p1,udims[1],2:udims[2]-1),view(p2,udims[1],2:udims[2]-1))

  return tmp/((NX-2)*(NY-2))
end

"""
    dot(p1::YEdges{Dual},p2::YEdges{Dual}) -> Real

Computes the inner product between two sets of dual y-edge component data on the same grid.
"""
function dot(p1::YEdges{Dual,NX,NY},p2::YEdges{Dual,NX,NY}) where {NX,NY}

  vdims = yedge_inds(Dual,(NX,NY))

  # interior
  tmp = dot(view(p1,2:vdims[1]-1,2:vdims[2]-1),view(p2,2:vdims[1]-1,2:vdims[2]-1))

  # boundaries
  tmp += 0.5*dot(view(p1,2:vdims[1]-1,1),view(p2,2:vdims[1]-1,1))
  tmp += 0.5*dot(view(p1,2:vdims[1]-1,vdims[2]),view(p2,2:vdims[1]-1,vdims[2]))

  return tmp/((NX-2)*(NY-2))
end



"""
    dot(p1::XEdges{Primal},p2::XEdges{Primal}) -> Real

Computes the inner product between two sets of primal x-edge component data on the same grid.
"""
function dot(p1::XEdges{Primal,NX,NY},p2::XEdges{Primal,NX,NY}) where {NX,NY}

  udims = xedge_inds(Primal,(NX,NY))

  # interior
  tmp = dot(view(p1,2:udims[1]-1,2:udims[2]-1),view(p2,2:udims[1]-1,2:udims[2]-1))

  # boundaries
  tmp += 0.5*dot(view(p1,2:udims[1]-1,1),view(p2,2:udims[1]-1,1))
  tmp += 0.5*dot(view(p1,2:udims[1]-1,udims[2]),view(p2,2:udims[1]-1,udims[2]))

  return tmp/((NX-2)*(NY-2))
end

"""
    dot(p1::YEdges{Primal},p2::YEdges{Primal}) -> Real

Computes the inner product between two sets of primal y-edge component data on the same grid.
"""
function dot(p1::YEdges{Primal,NX,NY},p2::YEdges{Primal,NX,NY}) where {NX,NY}

  vdims = yedge_inds(Primal,(NX,NY))

  # interior
  tmp = dot(view(p1,2:vdims[1]-1,2:vdims[2]-1),view(p2,2:vdims[1]-1,2:vdims[2]-1))

  # boundaries
  tmp += 0.5*dot(view(p1,1,2:vdims[2]-1),view(p2,1,2:vdims[2]-1))
  tmp += 0.5*dot(view(p1,vdims[1],2:vdims[2]-1),view(p2,vdims[1],2:vdims[2]-1))

  return tmp/((NX-2)*(NY-2))
end

"""
    dot(p1::Edges{Dual/Primal},p2::Edges{Dual/Primal}) -> Real

Computes the inner product between two sets of edge data on the same grid.
"""
dot(p1::Edges{C,NX,NY},p2::Edges{C,NX,NY}) where {C<:CellType,NX,NY} =
      dot(p1.u,p2.u) + dot(p1.v,p2.v)

"""
    dot(p1::EdgeGradient{Dual/Primal},p2::EdgeGradient{Dual/Primal}) -> Real

Computes the inner product between two sets of edge gradient data on the same grid.
"""
dot(p1::EdgeGradient{C,D,NX,NY},p2::EdgeGradient{C,D,NX,NY}) where {C<:CartesianGrids.CellType,D<:CartesianGrids.CellType,NX,NY} =
      dot(p1.dudx,p2.dudx) + dot(p1.dudy,p2.dudy) + dot(p1.dvdx,p2.dvdx) + dot(p1.dvdy,p2.dvdy)

######## NORMS ###########

"""
    norm(p::GridData) -> Real

Computes the L2 norm of data on a grid.
"""
norm(p::GridData) = sqrt(dot(p,p))

######## INTEGRALS ###########


# This function computes an integral by just taking the inner product with
# another set of cell data uniformly equal to 1
"""
    integrate(p::GridData) -> Real

Computes a numerical quadrature of node data.
"""
function integrate(p::T) where {T<:GridData}
  p2 = T()
  fill!(p2,1) # fill it with ones
  return dot(p2,p) # order matters for complex types
end
