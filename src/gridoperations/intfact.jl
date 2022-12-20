import Base: exp
export exp!

# Integrating factor

intfact(x, a) = exp(-2a)besseli(x,2a)

intfact(x, y,a) = exp(-4a)besseli(x,2a)besseli(y,2a)

intfact(x, y, z, a) = exp(-6a)besseli(x,2a)besseli(y,2a)besseli(z,2a)


"""
    plan_intfact(a::Real,dims::Tuple,[fftw_flags=FFTW.ESTIMATE])

Constructor to set up an operator for evaluating the integrating factor with
real-valued parameter `a`. This can then be applied with the `*` or `\` operation on
data of the appropriate size.

Note that `a` can be positive, negative, or zero. However, if `a` is negative,
then only the `\` operation is actually correct; the `*` operation merely
returns the identity to avoid excessive (and noisy) calculation. Similarly, if
`a` is positive, the `\` operation returns the identity. Thus, these operations are
not inverses of one another. If `a` is zero, both operations return the identity.

The `dims` argument can be replaced with data of type `ScalarGridData` to
specify the size of the domain.

# Example

```jldoctest
julia> w = Nodes(Dual,(6,6));

julia> w[4,4] = 1.0;

julia> E = plan_intfact(1.0,(6,6))
Integrating factor with parameter 1.0 on a (nx = 6, ny = 6) grid

julia> E*w
Nodes{Dual,6,6,Float64} data
Printing in grid orientation (lower left is (1,1))
6×6 Array{Float64,2}:
 0.00268447   0.00869352  0.0200715   0.028765    0.0200715   0.00869352
 0.00619787   0.0200715   0.0463409   0.0664124   0.0463409   0.0200715
 0.00888233   0.028765    0.0664124   0.0951774   0.0664124   0.028765
 0.00619787   0.0200715   0.0463409   0.0664124   0.0463409   0.0200715
 0.00268447   0.00869352  0.0200715   0.028765    0.0200715   0.00869352
 0.000828935  0.00268447  0.00619787  0.00888233  0.00619787  0.00268447
```
"""
function plan_intfact end

"""
    plan_intfact!(a::Real,dims::Tuple,[fftw_flags=FFTW.ESTIMATE][,nthreads=length(Sys.cpu_info())])

Same as [`plan_intfact`](@ref), but the resulting operator performs an in-place
operation on data. The number of threads `threads` defaults to the number of
logical CPU cores on the system.
"""
function plan_intfact! end

abstract type IFExpSign end
abstract type PosExp <: IFExpSign end
abstract type NegExp <: IFExpSign end
abstract type ZeroExp <: IFExpSign end
NonNegExp = Union{PosExp,ZeroExp}
NonPosExp = Union{NegExp,ZeroExp}


struct IntFact{NX, NY, PA<:IFExpSign, inplace}
    a::Float64
    conv::Union{CircularConvolution{NX, NY},Nothing}
end

for (lf,inplace) in ((:plan_intfact,false),
                     (:plan_intfact!,true))

    @eval function $lf(a::Real,dims::Tuple{Int,Int};fftw_flags = FFTW.ESTIMATE, nthreads = MAX_NTHREADS)
        NX, NY = dims

        if a == 0
          return IntFact{NX, NY, ZeroExp, $inplace}(0.0,nothing)
        elseif a < 0
          a_internal = abs(convert(Float64,a))
          signType = NegExp
        else
          a_internal = convert(Float64,a)
          signType = PosExp
        end

        #qtab = [intfact(x, y, a) for x in 0:NX-1, y in 0:NY-1]
        Nmax = 0
        while abs(intfact(Nmax,0,a_internal)) > eps(Float64)
          Nmax += 1
        end
        qtab = [max(x,y) <= Nmax ? intfact(x, y, a_internal) : 0.0 for x in 0:NX-1, y in 0:NY-1]
        #IntFact{NX, NY, a, $inplace}(Nullable(CircularConvolution(qtab, fftw_flags)))
        IntFact{NX, NY, signType, $inplace}(a_internal,CircularConvolution(qtab, fftw_flags,nthreads=nthreads))
      end

      # Base the size on the dual grid associated with any grid data, since this
      # is what the efficient grid size in PhysicalGrid has been established with
      @eval $lf(a::Real,::GridData{NX,NY}; fftw_flags = FFTW.ESTIMATE, nthreads = MAX_NTHREADS) where {NX,NY} =
          $lf(a,(NX,NY), fftw_flags = fftw_flags, nthreads = nthreads)


end

function Base.show(io::IO, E::IntFact{NX, NY, signType, inplace}) where {NX, NY, signType, inplace}
    nodedims = "(nx = $NX, ny = $NY)"
    atxt = signType == NegExp ? "$(-E.a)" : "$(E.a)"
    isinplace = inplace ? "In-place integrating factor" : "Integrating factor"
    print(io, "$isinplace with parameter $atxt on a $nodedims grid")
end

"""
    exp(L::Laplacian,a[,Nodes(Dual)])

Create the integrating factor exp(L*a). The default size of the operator is
the one appropriate for dual nodes; another size can be specified by supplying
grid data in the optional third argument. Note that, if `L` contains a factor,
it scales the exponent with this factor.
"""
exp(L::Laplacian{NX,NY},a,prototype=Nodes(Dual,(NX,NY))) where {NX,NY} = plan_intfact(L.factor*a,prototype)

"""
    exp!(L::Laplacian,a[,Nodes(Dual)])

Create the in-place version of the integrating factor exp(L*a).
"""
exp!(L::Laplacian{NX,NY},a,prototype=Nodes(Dual,(NX,NY))) where {NX,NY} = plan_intfact!(L.factor*a,prototype)



for (datatype) in (:Nodes, :XEdges, :YEdges)
  @eval function mul!(out::$datatype{T,NX, NY},
                     E::IntFact{MX, MY, PosExp, inplace},
                     s::$datatype{T, NX, NY}) where {T <: CellType, NX, NY, MX, MY, inplace}

      mul!(out.data, E.conv, s.data)
      out
  end

  @eval function mul!(out::$datatype{T,NX, NY},
                     E::IntFact{MX, MY, NegExp, inplace},
                     s::$datatype{T, NX, NY}) where {T <: CellType, NX, NY, MX, MY, inplace}

      out .= deepcopy(s)
  end

  @eval function ldiv!(out::$datatype{T,NX, NY},
                     E::IntFact{MX, MY, PosExp, inplace},
                     s::$datatype{T, NX, NY}) where {T <: CellType, NX, NY, MX, MY, inplace}

      out .= deepcopy(s)
  end

  @eval function ldiv!(out::$datatype{T,NX, NY},
                     E::IntFact{MX, MY, NegExp, inplace},
                     s::$datatype{T, NX, NY}) where {T <: CellType, NX, NY, MX, MY, inplace}

      mul!(out.data, E.conv, s.data)
      out
  end


  @eval function mul!(out::$datatype{T,NX, NY},
                     E::IntFact{MX, MY, ZeroExp, inplace},
                     s::$datatype{T, NX, NY}) where {T <: CellType, NX, NY, MX, MY, inplace}
      out .= deepcopy(s)
  end

  @eval function ldiv!(out::$datatype{T,NX, NY},
                     E::IntFact{MX, MY, ZeroExp, inplace},
                     s::$datatype{T, NX, NY}) where {T <: CellType, NX, NY, MX, MY, inplace}
      out .= deepcopy(s)
  end

end

for (op) in (:mul!,:ldiv!)
  @eval function $op(out::Edges{C,NX,NY},E::IntFact,s::Edges{C,NX,NY}) where {C,NX,NY}
    $op(out.u,E,s.u)
    $op(out.v,E,s.v)
    out
  end

  @eval function $op(out::EdgeGradient{C,D,NX,NY},E::IntFact,s::EdgeGradient{C,D,NX,NY}) where {C,D,NX,NY}
    $op(out.dudx,E,s.dudx)
    $op(out.dvdx,E,s.dvdx)
    $op(out.dudy,E,s.dudy)
    $op(out.dvdy,E,s.dvdy)
    out
  end

end




#=
function mul!(out::Nodes{T,NX, NY},
                   E::IntFact{MX, MY, a, inplace},
                   s::Nodes{T, NX, NY}) where {T <: CellType, NX, NY, MX, MY, a, inplace}

    mul!(out.data, E.conv, s.data)
    out
end

function mul!(out::Nodes{T,NX, NY},
                   E::IntFact{MX, MY, 0.0, inplace},
                   s::Nodes{T, NX, NY}) where {T <: CellType, NX, NY, MX, MY, inplace}
    out .= deepcopy(s)
end
=#

*(E::IntFact{MX,MY,signType,false},s::G) where {MX,MY,signType,G<:GridData} =
  mul!(G(), E, s)

*(E::IntFact{MX,MY,signType,true},s::GridData) where {MX,MY,signType} =
    mul!(s, E, deepcopy(s))


\(E::IntFact{MX,MY,signType,false},s::G) where {MX,MY,signType,G<:GridData} =
  ldiv!(G(), E, s)

\(E::IntFact{MX,MY,signType,true},s::GridData) where {MX,MY,signType} =
    ldiv!(s, E, deepcopy(s))
