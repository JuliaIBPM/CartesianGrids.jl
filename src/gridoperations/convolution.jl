import Base: *

"""
    CircularConvolution{M, N, T}

A preplanned, circular convolution operator on an M × N matrix of data of type T

# Fields
- `Ĝ`: DFT coefficients of the convolution kernel
- `F`: preplanned rFFT operator
- `F⁻¹`: preplanned irFFT operator
- `nthreads` : optimized number of threads to use, if appropriate
- `paddedSpace`: scratch space to zero-pad the input matrix
- `Â`: scratch space to store the DFT coefficients of the zero-padded input matrix

# Constructors:

- `CircularConvolution(G::Matrix{T})`

# Example:
```jldoctest
julia> G = repeat(1.0:3,1,4)
3×4 Array{Float64,2}:
 1.0  1.0  1.0  1.0
 2.0  2.0  2.0  2.0
 3.0  3.0  3.0  3.0

julia> C = CircularConvolution(G)
Circular convolution on a 3 × 4 matrix of data type Float64

julia> C*reshape(1:12, 3, 4)
3×4 Array{Int64,2}:
 164  164  164  164
 130  130  130  130
 148  148  148  148
```
"""
struct CircularConvolution{M, N, T, K, KI}
    Ĝ::Matrix{ComplexF64}
    F::K
    F⁻¹::KI
    nthreads::Int

    paddedSpace::Matrix{T}
    Â::Matrix{ComplexF64}
end

function Base.show(io::IO, c::CircularConvolution{M, N, T}) where {M, N, T}
    print(io, "Circular convolution on a $M × $N matrix of data type $T")
end

function CircularConvolution(G::AbstractMatrix{T},fftw_flags = FFTW.ESTIMATE; dtype = Float64, optimize = false, nthreads = DEFAULT_NTHREADS) where {T}
  nt_opt = optimize ? _optimize_convolution_threads(G,nthreads,fftw_flags,dtype) : nthreads
  #=
  if optimize
    
    # find the optimal number of threads, up to `nthreads`
    C = _circular_convolution(G,fftw_flags,dtype,1)
    cput_mean_opt, cput_std_opt = test_cputime_convolution(_circular_convolution(G,fftw_flags,dtype,1),nsamp=3)
    nt_opt = 1
    for nt in 2:nthreads
      cput_mean, cput_std = test_cputime_convolution(_circular_convolution(G,fftw_flags,dtype,nt),nsamp=3)
      if cput_mean < cput_mean_opt
        cput_mean_opt = cput_mean
        nt_opt = nt
      end
    end
  end
  =#
  C = _circular_convolution(G,fftw_flags,dtype,nt_opt)
end

function _optimize_convolution_threads(G,max_nthreads,fftw_flags,dtype; nsamp = 3)

  C = _circular_convolution(G,fftw_flags,dtype,1)
  cput_mean_opt, cput_std_opt = test_cputime_convolution(_circular_convolution(G,fftw_flags,dtype,1),nsamp=nsamp)
  nt_opt = 1
  for nt in 2:max_nthreads
    cput_mean, cput_std = test_cputime_convolution(_circular_convolution(G,fftw_flags,dtype,nt),nsamp=nsamp)
    if cput_mean < cput_mean_opt
      cput_mean_opt = cput_mean
      nt_opt = nt
    end
  end
  return nt_opt
end

function _circular_convolution(G,fftw_flags,dtype,nthreads)
  FFTW.set_num_threads(nthreads)

  M, N = size(G)
  #paddedSpace = Matrix{Float64}(undef, 2M-1, 2N-1)
  paddedSpace = Matrix{dtype}(undef, 2M, 2N)

  if dtype == ComplexF64
    F = FFTW.plan_fft(paddedSpace, flags = fftw_flags)
  else
    F = FFTW.plan_rfft(paddedSpace, flags = fftw_flags)
  end

  mirror!(paddedSpace, G)
  Ĝ = F * paddedSpace

  Â = similar(Ĝ)
  #F⁻¹ = FFTW.plan_irfft(Â, 2M - 1, flags = fftw_flags)
  if dtype == ComplexF64
    F⁻¹ = FFTW.plan_ifft(Â, flags = fftw_flags)
  else
    F⁻¹ = FFTW.plan_irfft(Â, 2M, flags = fftw_flags)
  end

  CircularConvolution{M, N, dtype, typeof(F), typeof(F⁻¹)}(Ĝ, F, F⁻¹, nthreads, paddedSpace, Â)
end

function test_cputime_convolution(C::CircularConvolution{M,N,T};nsamp=1) where {M,N,T}
  prototype = rand(T,(M,N))

  mul!(prototype,C,prototype) # to compile the function
  cput = zeros(Float64,nsamp)
  for n in 1:nsamp
      out = @timed mul!(prototype,C,prototype)
      cput[n] = out.time
  end
  return mean(cput), std(cput)
end

### ARITHMETIC ###

function mul!(out, C::CircularConvolution{M, N, T}, B) where {M, N, T}
    FFTW.set_num_threads(C.nthreads)

    MB, NB = size(B)
    #@assert size(out) == size(B) == (M, N)
    @assert size(out) == (MB, NB)

    inds = CartesianIndices((MB,NB))
    fill!(C.paddedSpace, 0)
    copyto!(C.paddedSpace, inds, B, inds)
    mul!(C.Â, C.F, C.paddedSpace)

    C.Â .*= C.Ĝ

    mul!(C.paddedSpace, C.F⁻¹, C.Â)

    #copyto!(out, inds, C.paddedSpace, CartesianIndices((M:2M-1,N:2N-1)))
    copyto!(out, inds, C.paddedSpace, CartesianIndices((M+1:M+MB,N+1:N+NB)))

end

C::CircularConvolution * B = mul!(similar(B), C, B)

### UTILITIES ###

function mirror!(A, a::AbstractArray{T,2}) where {T}
    Nr, Nc = size(a)
    #A[1:Nr-1, 1:Nc-1] .= a[Nr:-1:2, Nc:-1:2]
    #A[1:Nr-1, Nc:end] .= a[Nr:-1:2, 1:Nc]
    #A[Nr:end, 1:Nc-1] .= a[1:Nr, Nc:-1:2]
    #A[Nr:end, Nc:end] .= a
    A .= 0
    A[2:Nr, 2:Nc] .= a[Nr:-1:2, Nc:-1:2]
    A[2:Nr, Nc+1:end] .= a[Nr:-1:2, 1:Nc]
    A[Nr+1:end, 2:Nc] .= a[1:Nr, Nc:-1:2]
    A[Nr+1:end, Nc+1:end] .= a[1:Nr, 1:Nc]
    A
end

function mirror(a::AbstractArray{T,2}) where {T}
    Nr, Nc = size(a)
    mirror!(zeros(T, 2Nr-1, 2Nc-1), a)
end
