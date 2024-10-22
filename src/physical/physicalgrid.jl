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
  nthreads :: Int
end

"""
    PhysicalGrid(xlim::Tuple{Real,Real},ylim::Tuple{Real,Real},Δx::Float64;[opt_type=:none,optimize_threads=false,nthreads_max=1])

Constructor to set up a grid connected to physical space. The region to be
discretized by the grid is defined by the limits `xlim` and `ylim`, and the
cell spacing (uniform and indentical in each direction) is specified by `Δx`.
The constructor uses this information to determine the number of
cells in each direction, expanding the given range if necessary to accommodate
an integer number. It also pads each side with a ghost cell.
It also determines the indices corresponding to the corner
of the cell to which the physical origin corresponds. Note that the corner
corresponding to the lowest limit in each direction has indices (1,1).

There are a few optional arguments devoted to optimization of the grid size.
The `nthreads_max` sets the number of FFT compute threads to use and 
can be set to a value up to the total number available on the
architecture. It defaults to 1. 

The keyword `opt_type` can be set to `:threads`, `:prime` (default), or `:none`.
If `:none`, then the grid is set as close to the specified range as possible.
If `:prime`, then the grid is expanded in each direction to a number that is
a product of primes (and therefore efficient in an FFT). If `:threads`, then
the grid is tested with a representative calculation on various grid sizes
to identify one that has the minimum cpu time. If `optimize_threads = true`,
then the number of threads is also varied (between 1 and `nthreads_max`).
"""
function PhysicalGrid(xlim::Tuple{Real,Real},
                      ylim::Tuple{Real,Real},Δx::Float64;opt_type=:prime,optimize_threads=false,nthreads_max=DEFAULT_NTHREADS)


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

  nthreads = nthreads_max

  # Expand this grid and find the optimal number of threads
  if opt_type == :threads
    NX, NY, cput_opt = optimize_gridsize(NX0,NY0,optimize_threads=optimize_threads,nthreads_max=nthreads_max,nsamp=3)
    NX, i0, xlimnew = _expand_1d_grid(NX,NX0,xlimnew...,Δx)
    NY, j0, ylimnew = _expand_1d_grid(NY,NY0,ylimnew...,Δx)
    nthreads = convert(Int64,FFTW.get_num_threads())
  elseif opt_type == :prime
    NX, i0, xlimnew = _find_efficient_1d_grid(xmin,xmax,Δx)
    NY, j0, ylimnew = _find_efficient_1d_grid(ymin,ymax,Δx)
    #NX0, i0, xlim = _set_1d_grid(xmin,xmax,Δx)
    #NY0, j0, ylim = _set_1d_grid(ymin,ymax,Δx)
  elseif opt_type == :none
    NX, NY = NX0, NY0
  end

  PhysicalGrid((NX,NY),(i0,j0),Δx,(xlimnew,ylimnew),nthreads)
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
    leftside = false
    while Nr > 1 || sum(pow[end-1:end]) > 1
        xminnew, xmaxnew = xlimnew
        if leftside
            xminnew -= Δx
        else
            xmaxnew += Δx
        end
        #xminnew -= Δx
        #xmaxnew += Δx
        N, i0, xlimnew = _set_1d_grid(xminnew,xmaxnew,Δx)
        pow, Nr = _factor_prime(N)
        leftside = !leftside
    end
    return N, i0, xlimnew
end

"""
    test_cputime(nx,ny,nthreads_max;[nsamp=1][,testtype=:laplacian][,kwargs]) -> Float64, Float64

Evaluate a sample FFT-based problem with
the given size of grid `nx` x `ny` and the provided maximum number of threads `nthreads_max`.
Returns the mean and standard deviation of the computational time. The optional argument `nsamp` can be
used to perform an average timing over multiple samples. The test type is
specified with the `testtype` optional argument. The default test
is inversion of a Laplacian (`:laplacian`). Other options are `:intfact` and `:helmholtz`.
"""
function test_cputime(nx,ny,nthreads_max;optimize=true,nsamp=1,testtype=:laplacian,kwargs...)
    if testtype == :helmholtz
      w = Nodes(Dual,(nx,ny),dtype=ComplexF64)
      w .= rand(ComplexF64,size(w))
    else
      w = Nodes(Dual,(nx,ny),dtype=Float64)
      w .= rand(Float64,size(w))
    end
    op = _cputime_test_operator(w,optimize,nthreads_max,Val(testtype);kwargs...)
    ldiv!(w,op,w) # to compile the function
    cput = zeros(Float64,nsamp)
    for n in 1:nsamp
        out = @timed ldiv!(w,op,w)
        cput[n] = out.time
    end
    return mean(cput), std(cput)
end

_cputime_test_operator(w::GridData,optimize,nthreads_max,::Val{:laplacian}) = plan_laplacian(w,with_inverse=true,optimize=optimize,nthreads=nthreads_max)

function _cputime_test_operator(w::GridData,optimize,nthreads_max,::Val{:intfact};a=-1.0)
  L = plan_laplacian(w,with_inverse=true,optimize=optimize,nthreads=nthreads_max)
  return exp(L,a,w)
end

_cputime_test_operator(w::GridData,optimize,nthreads_max,::Val{:helmholtz};α=0.1) =
      plan_helmholtz(w,α,with_inverse=true,optimize=optimize,nthreads=nthreads_max)


"""
    optimize_gridsize(nx0,ny0[;region_size=4,optimize_threads=true,nthreads_max=length(cpu_info()),nsamp=1])

Given a nominal grid size (`nx0` x `ny0`), determine the optimal grid size
that minimizes the compute time. Optional arguments are the `optimize_threads` flag
and the maximum number of threads `nthreads_max` (if multithreading is allowed) and the number of samples to take
of the cpu time for each trial. Returns optimal `nx`, `ny`, and the corresponding CPU time.
"""
function optimize_gridsize(nx0,ny0;region_size=4,optimize_threads=true,nthreads_max=DEFAULT_NTHREADS,kwargs...)
    cput0_mean, cput0_std = test_cputime(nx0,ny0,nthreads_max;optimize=optimize_threads,kwargs...)
    nx_opt, ny_opt = nx0, ny0
    cput_mean_opt = cput0_mean
    cput_std_opt = cput0_std

    ny = ny0
    for nx in nx0:2:nx0+2region_size
        cput_mean, cput_std = test_cputime(nx,ny,nthreads_max;optimize=optimize_threads,kwargs...)
        if cput_mean < cput_mean_opt
            cput_mean_opt = cput_mean
            cput_std_opt = cput_std
            nx_opt = nx
        end
    end
    nx = nx_opt
    for ny in ny0:2:ny0+2region_size
        cput_mean, cput_std = test_cputime(nx,ny,nthreads_max;optimize=optimize_threads,kwargs...)
        if cput_mean < cput_mean_opt
            cput_mean_opt = cput_mean
            cput_std_opt = cput_std
            ny_opt = ny
        end
    end
    cput_mean, cput_std = test_cputime(nx_opt,ny_opt,nthreads_max;optimize=optimize_threads,kwargs...)
    return nx_opt, ny_opt, cput_mean_opt
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
    nthreads(g::PhysicalGrid) -> Int

Return the maximum number of threads allowed for grid `g`.
"""
Threads.nthreads(g::PhysicalGrid) = g.nthreads

# Extend FFT-based operations
plan_laplacian(g::PhysicalGrid;kwargs...) = plan_laplacian(size(g);nthreads=g.nthreads,kwargs...)
plan_laplacian!(g::PhysicalGrid;kwargs...) = plan_laplacian!(size(g);nthreads=g.nthreads,kwargs...)

plan_intfact(a::Real,g::PhysicalGrid;kwargs...) = plan_intfact(a,size(g);nthreads=g.nthreads,kwargs...)
plan_intfact!(a::Real,g::PhysicalGrid;kwargs...) = plan_intfact!(a,size(g);nthreads=g.nthreads,kwargs...)

plan_helmholtz(g::PhysicalGrid,α::Number;kwargs...) = plan_helmholtz(size(g),α;nthreads=g.nthreads,kwargs...)
plan_helmholtz!(g::PhysicalGrid,α::Number;kwargs...) = plan_helmholtz!(size(g),α;nthreads=g.nthreads,kwargs...)
