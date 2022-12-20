export coordinates, PhysicalGrid, limits, origin, cellsize, volume, test_cputime,
         optimize_gridsize

"""
    coordinates(w::GridData;[dx=1.0],[I0=(1,1)])

Return a tuple of the ranges of the physical coordinates in each direction for grid
data `w`. If `w` is of `Nodes` type, then it returns a tuple of the form
`xg,yg`. If `w` is of `Edges` or `NodePair` type, then it returns a tuple of
the form `xgu,ygu,xgv,ygv`.

The optional keyword argument `dx` sets the grid spacing; its default is `1.0`. The
optional keyword `I0` accepts a tuple of integers to set the index pair of the
primal nodes that coincide with the origin. The default is `(1,1)`.

# Example

```jldoctest
julia> w = Nodes(Dual,(12,22));

julia> xg, yg = coordinates(w,dx=0.1)
(-0.05:0.1:1.05, -0.05:0.1:2.0500000000000003)
```
"""
function coordinates end

for (gridtype,ctype,dnx,dny,shiftx,shifty) in @generate_scalarlist(SCALARLIST)
   @eval coordinates(w::$gridtype{$ctype,NX,NY,T};dx::Float64=1.0,I0::Tuple{Int,Int}=(1,1)) where {NX,NY,T} =
    dx.*((1-I0[1]-$shiftx):(NX-$dnx-I0[1]-$shiftx),
         (1-I0[2]-$shifty):(NY-$dny-I0[2]-$shifty))

end

for (gridtype,ctype,dunx,duny,dvnx,dvny,shiftux,shiftuy,shiftvx,shiftvy) in @generate_collectionlist(VECTORLIST)
   @eval coordinates(w::$gridtype{$(ctype...),NX,NY,T};dx::Float64=1.0,I0::Tuple{Int,Int}=(1,1)) where {NX,NY,T} =
    dx.*((1-I0[1]-$shiftux):(NX-$dunx-I0[1]-$shiftux),
         (1-I0[2]-$shiftuy):(NY-$duny-I0[2]-$shiftuy),
         (1-I0[1]-$shiftvx):(NX-$dvnx-I0[1]-$shiftvx),
         (1-I0[2]-$shiftvy):(NY-$dvny-I0[2]-$shiftvy))


end

struct PhysicalGrid{ND}
  N :: NTuple{ND,Int}
  I0 :: NTuple{ND,Int}
  Δx :: Float64
  xlim :: NTuple{ND,Tuple{Real,Real}}
  nthreads_opt :: Int
end

"""
    PhysicalGrid(xlim::Tuple{Real,Real},ylim::Tuple{Real,Real},Δx::Float64)

Constructor to set up a grid connected to physical space. The region to be
discretized by the grid is defined by the limits `xlim` and `ylim`, and the
cell spacing (uniform and indentical in each direction) is specified by `Δx`.
The constructor uses this information to determine the number of
cells in each direction, expanding the given range if necessary to accommodate
an integer number. It also pads each side with a ghost cell.
It also determines the indices corresponding to the corner
of the cell to which the physical origin corresponds. Note that the corner
corresponding to the lowest limit in each direction has indices (1,1).
"""
function PhysicalGrid(xlim::Tuple{Real,Real},
                      ylim::Tuple{Real,Real},Δx::Float64;nthreads_max=MAX_NTHREADS)


  #= set grid spacing and the grid position of the origin
  In case the physical limits are not consistent with an integer number of dual cells, based on
  the given Δx, we adjust them outward a bit in all directions. We also seek to place the
  origin on the corner of a cell.
  =#
  xmin, xmax = xlim
  ymin, ymax = ylim
  @assert xmax >= xmin && ymax >= ymin "Maximum limits must exceed minimum limits"

  #NX, i0, xlimnew = _find_efficient_1d_grid(xmin,xmax,Δx)
  #NY, j0, ylimnew = _find_efficient_1d_grid(ymin,ymax,Δx)
  #NX0, i0, xlim = _set_1d_grid(xmin,xmax,Δx)
  #NY0, j0, ylim = _set_1d_grid(ymin,ymax,Δx)

  # Create a nominal grid based on the desired dimensions
  NX0, i0, xlimnew = _set_1d_grid(xmin,xmax,Δx)
  NY0, j0, ylimnew = _set_1d_grid(ymin,ymax,Δx)

  # Expand this grid and find the optimal number of threads
  NX, NY, nt, cput_opt = optimize_gridsize(NX0,NY0,nthreads_max=nthreads_max,nsamp=3)
  NX, i0, xlimnew = _expand_1d_grid(NX,NX0,xlimnew...,Δx)
  NY, j0, ylimnew = _expand_1d_grid(NY,NY0,ylimnew...,Δx)

  PhysicalGrid((NX,NY),(i0,j0),Δx,(xlimnew,ylimnew),nt)
end

function _set_1d_grid(xmin::Real,xmax::Real,Δx::Float64)
  NL, NR = floor(Int,xmin/Δx), ceil(Int,xmax/Δx)
  return NR-NL+2, 1-NL, (Δx*NL, Δx*NR)
end

function _expand_1d_grid(N::Int,Nold::Int,xminold::Real,xmaxold::Real,Δx::Float64)
  NLold, NRold = floor(Int,xminold/Δx), ceil(Int,xmaxold/Δx)
  dN = (N - Nold)÷2
  NL, NR = NLold-dN, NRold+dN
  return NR-NL+2,1-NL,(Δx*NL, Δx*NR)
end

function _factor_prime(N::Integer)
    # Determine if N can be factorized into small primes
    # Returns the list of powers and the remaining factor
    # after factorization.
    Nr = N
    blist = [2,3,5,7,11,13]
    pow = zero(blist)
    for (i,b) in enumerate(blist)
        while mod(Nr,b) == 0
            pow[i] += 1
            Nr = Int(Nr/b)
        end
    end
    return pow, Nr
end

function _find_efficient_1d_grid(xmin::Real,xmax::Real,Δx::Float64)
    # Based on the provided grid dimensions and grid spacing,
    # find the minimally larger grid that has a number of cells
    # that can be factorized into small primes for efficient FFT calculations.
    N, i0, xlimnew = _set_1d_grid(xmin,xmax,Δx)
    pow, Nr = _factor_prime(N)
    while Nr > 1 || sum(pow[end-1:end]) > 1
        xminnew, xmaxnew = xlimnew
        xminnew -= Δx
        xmaxnew += Δx
        N, i0, xlimnew = _set_1d_grid(xminnew,xmaxnew,Δx)
        pow, Nr = _factor_prime(N)
    end
    return N, i0, xlimnew
end

"""
    test_cputime(nx,ny,nthreads;[nsamp=1]) -> Float64

Evaluate a sample problem (solution of Poisson problem) with
the given size of grid `nx` x `ny` and the provided number of threads `nthreads`.
Returns the computational time. The optional argument `nsamp` can be
used to perform an average timing over multiple samples.
"""
function test_cputime(nx,ny,nthreads;nsamp=1)
    w = Nodes(Dual,(nx,ny))
    w .= rand(Float64,size(w))
    L = plan_laplacian(w,with_inverse=true,nthreads=nthreads)
    ldiv!(w,L,w) # to compile the function
    cput = 0.0
    for n in 1:nsamp
        out = @timed CartesianGrids.ldiv!(w,L,w)
        cput += out.time
    end
    return cput/nsamp
end

function optimize_nthreads(nx,ny;nthreads_max = MAX_NTHREADS, kwargs...)
    nt0 = 1
    cput_opt = test_cputime(nx,ny,nt0;kwargs...)
    nt_opt = nt0
    for nt in 2:nthreads_max
        cput = test_cputime(nx,ny,nt;kwargs...)
        if cput < cput_opt
            cput_opt = cput
            nt_opt = nt
        end
    end
    return nt_opt, cput_opt
end

"""
    optimize_gridsize(nx0,ny0[;region_size=4,nthreads_max=length(cpu_info()),nsamp=1])

Given a nominal grid size (`nx0` x `ny0`), determine the optimal grid size
and optimal number of threads (if multithreading is allowed) that minimizes
the compute time.
"""
function optimize_gridsize(nx0,ny0;region_size=4,nthreads_max=MAX_NTHREADS,kwargs...)
    nt0 = 1
    cput0 = test_cputime(nx0,ny0,nt0;kwargs...)
    nx_opt, ny_opt, nt_opt = nx0, ny0, nt0
    cput_opt = cput0

    ny = ny0
    for nx in nx0:2:nx0+2region_size
        nt, cput = optimize_nthreads(nx,ny;nthreads_max=nthreads_max,kwargs...)
        if cput < cput_opt
            cput_opt = cput
            nx_opt = nx
            nt_opt = nt
        end
    end
    nx = nx_opt
    for ny in ny0:2:ny0+2region_size
        nt, cput = optimize_nthreads(nx,ny;nthreads_max=nthreads_max,kwargs...)
        if cput < cput_opt
            cput_opt = cput
            ny_opt = ny
            nt_opt = nt
        end
    end
    return nx_opt, ny_opt, nt_opt, cput_opt
end

### Utilities ###

"""
    size(g::PhysicalGrid,d::Int) -> Int

Return the number of cells in direction `d` in grid `g`.
"""
Base.size(g::PhysicalGrid,d::Int) = g.N[d]

"""
    size(g::PhysicalGrid) -> Tuple

Return a tuple of the number of cells in all directions in grid `g`.
"""
Base.size(g::PhysicalGrid) = g.N

"""
    length(g::PhysicalGrid,d::Int) -> Int

Return the total number of cells in grid `g`.
"""
Base.length(g::PhysicalGrid) = prod(size(g))

"""
    limits(g::PhysicalGrid,d::Int) -> Tuple

Return the minimum and maximum physical dimensions in direction `d` for grid `g`.
"""
limits(g::PhysicalGrid,d::Int) = g.xlim[d]

"""
    volume(g::PhysicalGrid) -> Float64

Return the volume (or area) of the physical grid `g`.
"""
volume(g::PhysicalGrid{ND}) where {ND} = mapreduce(dims -> dims[2]-dims[1],*,[limits(g,d) for d in 1:ND])


"""
    coordinates(w::Nodes/Edges,g::PhysicalGrid) -> Range

Return coordinate data range for type of `w`.
"""
coordinates(w,g::PhysicalGrid) = coordinates(w,dx=g.Δx,I0=g.I0)

"""
    origin(g::PhysicalGrid) -> Tuple{Int,Int}

Return a tuple of the indices of the primal node that corresponds to the
physical origin of the coordinate system used by `g`. Note that these
indices need not lie inside the range of indices occupied by the grid.
For example, if the range of physical coordinates occupied by the grid
is (1.0,3.0) x (2.0,4.0), then the origin is not inside the grid.
"""
origin(g::PhysicalGrid) = g.I0

"""
    cellsize(g::PhysicalGrid) -> Float64

Return the grid cell size of grid `g`
"""
cellsize(g::PhysicalGrid) = g.Δx

"""
    optimal_nthreads(g::PhysicalGrid) -> Int

Return the optimal number of threads for grid `g`.
"""
optimal_nthreads(g::PhysicalGrid) = g.nthreads_opt
