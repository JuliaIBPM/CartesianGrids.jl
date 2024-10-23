# Implicit diffusion

_implicit_diffusion_integrand(x, y,ξ, b) = besseli(x,2ξ/(b+4))besseli(y,2ξ/(b+4))/(b+4)

function implicit_diffusion(i,j,a::Float64,quad)
    x, w = quad
    return dot(w, _implicit_diffusion_integrand.(i,j,x,1.0/a))/a
end

"""
    plan_implicit_diffusion(a::Real,dims::Tuple,[fftw_flags=FFTW.ESTIMATE])

Constructor to set up the forward and inverse operators of `I - a*L` where
`L` is the discrete Laplacian (not scaled by grid spacing) and
`a` is a real-valued parameter. This can then be applied with the `*` or `\` operation on
data of the appropriate size.

The `dims` argument can be replaced with data of type `ScalarGridData` to
specify the size of the domain.
"""
function plan_implicit_diffusion end

"""
    plan_implicit_diffusion!(a::Real,dims::Tuple,[fftw_flags=FFTW.ESTIMATE][,nthreads=length(Sys.cpu_info())])

Same as [`plan_implicit_diffusion`](@ref), but the resulting operator performs an in-place
operation on data. The number of threads `threads` defaults to the number of
logical CPU cores on the system.
"""
function plan_implicit_diffusion! end


struct ImplicitDiffusion{NX, NY, inplace}
    a::Float64
    conv::Union{CircularConvolution{NX, NY},Nothing}
end

for (lf,inplace) in ((:plan_implicit_diffusion,false),
                     (:plan_implicit_diffusion!,true))

    @eval function $lf(a::Real,dims::Tuple{Int,Int};fftw_flags = FFTW.ESTIMATE, optimize = false, nthreads = DEFAULT_NTHREADS)
        NX, NY = dims

        # Find the minimum number of Gauss points
        ng = 5
        quad = gausslaguerre(ng)
        Ilast = implicit_diffusion(0,0,a,quad)
        err = 1e10
        while err > 10*eps(Float64)
            ng += 1
            quad = gausslaguerre(ng)
            I = implicit_diffusion(0,0,a,quad)
            err = abs(I - Ilast)
            Ilast = I
        end

        # Find the maximum radius
        Nmax = 0
        while abs(implicit_diffusion(Nmax,0,a,quad)) > eps(Float64)
            Nmax += 1
        end
        Nmax

        # Create the table of values
        qtab = [max(x,y) <= Nmax ? implicit_diffusion(x, y, a, quad) : 0.0 for x in 0:NX-1, y in 0:NY-1]

        ImplicitDiffusion{NX, NY, $inplace}(a,CircularConvolution(qtab, fftw_flags,optimize=optimize,nthreads=nthreads))
      end

      # Base the size on the dual grid associated with any grid data, since this
      # is what the efficient grid size in PhysicalGrid has been established with
      @eval $lf(a::Real,::GridData{NX,NY}; fftw_flags = FFTW.ESTIMATE, optimize=false, nthreads = DEFAULT_NTHREADS) where {NX,NY} =
          $lf(a,(NX,NY), fftw_flags = fftw_flags, optimize=optimize, nthreads = nthreads)


end

for (datatype) in (:Nodes, :XEdges, :YEdges)
  @eval function mul!(out::$datatype{T,NX, NY},
                     E::ImplicitDiffusion{MX, MY, inplace},
                     s::$datatype{T, NX, NY}) where {T <: CartesianGrids.CellType, NX, NY, MX, MY, inplace}

      laplacian!(out,s)
      out .= s .- E.a.*out
      out
  end


  @eval function ldiv!(out::$datatype{T,NX, NY},
                     E::ImplicitDiffusion{MX, MY, inplace},
                     s::$datatype{T, NX, NY}) where {T <: CartesianGrids.CellType, NX, NY, MX, MY, inplace}

      mul!(out.data, E.conv, s.data)
      out
  end


end

for (op) in (:mul!,:ldiv!)
  @eval function $op(out::Edges{C,NX,NY},E::ImplicitDiffusion,s::Edges{C,NX,NY}) where {C,NX,NY}
    $op(out.u,E,s.u)
    $op(out.v,E,s.v)
    out
  end

  @eval function $op(out::EdgeGradient{C,D,NX,NY},E::ImplicitDiffusion,s::EdgeGradient{C,D,NX,NY}) where {C,D,NX,NY}
    $op(out.dudx,E,s.dudx)
    $op(out.dvdx,E,s.dvdx)
    $op(out.dudy,E,s.dudy)
    $op(out.dvdy,E,s.dvdy)
    out
  end

end

*(E::ImplicitDiffusion{MX,MY,false},s::G) where {MX,MY,G<:GridData} =
  mul!(G(), E, s)

*(E::ImplicitDiffusion{MX,MY,true},s::GridData) where {MX,MY} =
    mul!(s, E, deepcopy(s))


\(E::ImplicitDiffusion{MX,MY,false},s::G) where {MX,MY,G<:GridData} =
  ldiv!(G(), E, s)

\(E::ImplicitDiffusion{MX,MY,true},s::GridData) where {MX,MY} =
    ldiv!(s, E, deepcopy(s))
