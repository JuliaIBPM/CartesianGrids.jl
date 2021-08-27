using LinearAlgebra

#if VERSION < v"0.7-"
#  import Base: A_mul_B!
#  mul!(x,B,y) = A_mul_B!(x,B,y)
#end

@testset "Point-Field Routines" begin

  @testset "Point creation" begin
    @test_throws AssertionError VectorData([1,2,3],[1,2])

    @test_throws AssertionError TensorData([1,2,3],[1,2],[2,3],[4,5])

    f = ScalarData(10)
    f .= rand(10)

    ft = TensorData(f)
    ft[25] = 4
    @test ft.dvdx[5] == 4

    f2 = zero(f)
    @test typeof(f2) == typeof(f)

  end

  @testset "Point operations" begin

    f = ScalarData(10)
    f .= rand(10)

    g = -f
    @test g[4] == -f[4]

    g = ScalarData(f)
    g .= rand(10)

    h = f + g

    @test h[8] == f[8]+g[8]

    h = 2.0*g

    @test h[4] == 2.0*g[4]


    Y = VectorData(4)
    Y .= rand(length(Y))
    X = 2*Y

    @test X.u[1] == 2*Y.u[1] && X.v[1] == 2*Y.v[1]

    X = Y/2

    @test X.u[3] == Y.u[3]/2 && X.v[3] == Y.v[3]/2

    fill!(Y,0.0)

    X = Y + (1,2)
    @test X.v[4] == 2.0

    X2 = (1,2) + Y
    @test X2 == X

    X = Y - (1,2)
    @test X.u[4] == -1.0

    X = (1,2) - Y
    @test X.u[4] == 1.0

    Z = 2.0 × X
    @test typeof(Z) <: VectorData
    @test Z.u[1] == -4.0

    X = TensorData(Y)
    fill!(X,1)
    Z = X + X
    @test Z.dvdx[4] == X.dvdx[4] + X.dvdx[4]

    Z = (1,2) ⋅ X
    @test typeof(Z) <: VectorData
    @test Z.u[3] == 3.0

    Xt = TensorData(Y)
    transpose!(Xt,X)
    @test Xt.dudx == X.dudx
    @test Xt.dudy == X.dvdx
    @test Xt.dvdx == X.dudy
    @test Xt.dvdy == X.dvdy
    @test Xt == transpose(X)


  end

  @testset "Hadamard products" begin

    fcs = ScalarData(5,dtype=ComplexF64)
    fill!(fcs,2im)
    fcs .= rand(5)*im

    a = 2.0
    frs = TensorData(fcs,dtype=Float64)
    fill!(frs,a)

    out = similar(frs,element_type=ComplexF64)

    product!(out,frs,fcs)

    @test out[3] == frs[3]*fcs[3]

    out = frs ∘ fcs

    @test out[2] == frs[2]*fcs[2]

  end

  @testset "Cartesian product" begin

    N = 100
    u = VectorData(N,dtype=Float64)
    u.u .= randn(length(u.u))

    v = VectorData(N,dtype=Float64)
    v.u .= randn(length(v.u))

    uv = u*v
    @test uv.dudx == u.u ∘ v.u
    @test uv.dudy == u.v ∘ v.u
    @test uv.dvdx == u.u ∘ v.v
    @test uv.dvdy == u.v ∘ v.v


  end

  @testset "Cross and dot products" begin

    N = 100
    u = VectorData(N,dtype=Float64)
    u .= randn(length(u))

    v = VectorData(N,dtype=Float64)
    v .= randn(length(v))

    uv = pointwise_dot(u,v)
    @test uv == u.u ∘ v.u + u.v ∘ v.v

    uv = cross(u,v)
    @test uv == u.u ∘ v.v - u.v ∘ v.u

    A = TensorData(N,dtype=Float64)
    A .= randn(length(A))

    Au = pointwise_dot(A,u)
    @test Au.u == A.dudx∘u.u + A.dvdx∘u.v
    @test Au.v == A.dudy∘u.u + A.dvdy∘u.v

    Au = pointwise_dot(u,A)
    @test Au.u == A.dudx∘u.u + A.dudy∘u.v
    @test Au.v == A.dvdx∘u.u + A.dvdy∘u.v


  end

  n = 10
  x = 0.5 .+ 0.2*rand(n)
  y = 0.5 .+ 0.2*rand(n)
  X = VectorData(x,y)

  nx = 12; ny = 12

  dx = 0.1
  H = Regularize(X,dx)
  H̃ = Regularize(X,dx,filter=true)
  Hs = Regularize(X,dx,issymmetric=true)


  @testset "Regularize vector to primal edges" begin

    f = VectorData(X)
    f.u .= rand(n)
    f.v .= rand(n)

    q = Edges(Primal,(nx,ny))

    H(q,f)
    @test sum(f.u) ≈ sum(q.u)*dx*dx
    @test sum(f.v) ≈ sum(q.v)*dx*dx

  end

  @testset "Regularize vector to dual edges" begin

  f = VectorData(X)
  f.u .= rand(n)
  f.v .= rand(n)

  p = Edges(Dual,(nx,ny))

  H(p,f)
  @test sum(f.u) ≈ sum(p.u)*dx*dx
  @test sum(f.v) ≈ sum(p.v)*dx*dx

  end

  @testset "Regularize vector to dual and primal nodes" begin

  f = VectorData(X)
  f.u .= rand(n)
  f.v .= rand(n)

  w = NodePair(Dual,(nx,ny))

  H(w,f)
  @test sum(f.u) ≈ sum(w.u)*dx*dx
  @test sum(f.v) ≈ sum(w.v)*dx*dx

  w = NodePair(Primal,(nx,ny))

  H(w,f)
  @test sum(f.u) ≈ sum(w.u)*dx*dx
  @test sum(f.v) ≈ sum(w.v)*dx*dx

  end

  @testset "Regularize tensor to edge gradient" begin

  f = TensorData(X)
  f.dudx .= rand(n)
  f.dudy .= rand(n)
  f.dvdx .= rand(n)
  f.dvdy .= rand(n)

  gradq = EdgeGradient(Dual,(nx,ny))

  H(gradq,f)

  @test sum(f.dudx) ≈ sum(gradq.dudx)*dx*dx
  @test sum(f.dudy) ≈ sum(gradq.dudy)*dx*dx
  @test sum(f.dvdx) ≈ sum(gradq.dvdx)*dx*dx
  @test sum(f.dvdy) ≈ sum(gradq.dvdy)*dx*dx

  gradq = EdgeGradient(Primal,(nx,ny))

  H(gradq,f)

  @test sum(f.dudx) ≈ sum(gradq.dudx)*dx*dx
  @test sum(f.dudy) ≈ sum(gradq.dudy)*dx*dx
  @test sum(f.dvdx) ≈ sum(gradq.dvdx)*dx*dx
  @test sum(f.dvdy) ≈ sum(gradq.dvdy)*dx*dx


  end

  @testset "Regularize scalar to primal nodes" begin

  f = ScalarData(X)
  f .= rand(n)

  w = Nodes(Primal,(nx,ny))

  H(w,f)
  @test sum(f) ≈ sum(w)*dx*dx

  end

  @testset "Regularize scalar to dual nodes" begin

  f = ScalarData(X)
  f .= rand(n)

  w = Nodes(Dual,(nx,ny))

  H(w,f)
  @test sum(f) ≈ sum(w)*dx*dx

  end

  @testset "Matrix representation dispatch" begin

  u = Nodes(Primal,(nx,ny))
  w = XEdges(Primal,u)
  dq = EdgeGradient(Primal,w)
  dq2 = EdgeGradient(Primal,w)

  s = NodePair(Dual,w)
  s2 = NodePair(Dual,w)

  q = Edges(Primal,w)
  q2 = Edges(Primal,w)

  f = ScalarData(X)
  h = VectorData(X)

  Hmat = RegularizationMatrix(H,f,w)
  Hmatn = RegularizationMatrix(H,f,u)
  Hmate = RegularizationMatrix(H,h,q)

  Emat = InterpolationMatrix(H,w,f)
  Ematn = InterpolationMatrix(H,u,f)
  Emate = InterpolationMatrix(H,q,h)

  @test_throws MethodError mul!(w,Hmat,h)
  @test_throws MethodError mul!(q,Hmate,f)
  @test_throws MethodError mul!(h,Emat,w)
  @test_throws MethodError mul!(f,Emate,q)

  # None of these should throw an error
  mul!(w,Hmat,h.u)
  mul!(u,Hmatn,f)
  mul!(h.u,Emat,w)
  mul!(h.u,Emat,q.u)
  mul!(f,Emat,q.u)
  mul!(h.u,Ematn,u)
  mul!(f,Ematn,u)

  Hmat*h.u
  Hmate*h
  Emat*w
  Emat*q.u

  @test_throws MethodError Hmat*h
  @test_throws MethodError Emate*w


  end

  @testset "Matrix representation" begin

  f = ScalarData(X)
  w = Nodes(Dual,(nx,ny))
  Hmat = RegularizationMatrix(H,f,w)
  Emat = InterpolationMatrix(H,w,f)

  f .= rand(n)

  w2 = Nodes(Dual,(nx,ny))
  mul!(w,Hmat,f)
  H(w2,f)
  @test w ≈ w2

  # Test that scalar data of different data representation can be used with matrix
  # Here, Hmat has the second parameter ScalarData with its DT parameter
  # different from that of X.u (which is a SubArray)
  mul!(w,Hmat,X.u)
  @test typeof(Hmat*X.u) == typeof(w2)

  @test_throws MethodError mul!(f,Hmat,w)

  f2 = ScalarData(f)
  mul!(f,Emat,w)
  H(f2,w)
  @test f ≈ f2

  w .= rand(nx,ny)
  Ẽmat = InterpolationMatrix(H̃,w,f)

  mul!(f,Ẽmat,w)
  H̃(f2,w)
  @test f ≈ f2

  w = Nodes(Primal,(nx,ny))
  Hmat = RegularizationMatrix(H,f,w)
  Emat = InterpolationMatrix(H,w,f)
  Ẽmat = InterpolationMatrix(H̃,w,f)


  w2 = Nodes(Primal,(nx,ny))
  mul!(w,Hmat,f)
  H(w2,f)
  @test w ≈ w2

  w .= rand(Float64,size(w))
  f2 = ScalarData(f)
  mul!(f,Emat,w)
  H(f2,w)
  @test f ≈ f2

  f2 = ScalarData(f)
  mul!(f,Ẽmat,w)
  H̃(f2,w)
  @test f ≈ f2

  f = VectorData(X)
  f.u .= rand(n)
  f.v .= rand(n)

  p = Edges(Dual,(nx,ny))
  Hmat = RegularizationMatrix(H,f,p)
  Emat = InterpolationMatrix(H,p,f)
  Ẽmat = InterpolationMatrix(H̃,p,f)

  p2 = Edges(Dual,(nx,ny))
  mul!(p,Hmat,f)
  H(p2,f)
  @test p.u ≈ p2.u && p.v ≈ p2.v

  p.u .= rand(Float64,size(p.u))
  p.v .= rand(Float64,size(p.v))
  f2 = VectorData(f)
  mul!(f,Emat,p)
  H(f2,p)
  @test f.u ≈ f2.u && f.v ≈ f2.v

  mul!(f,Ẽmat,p)
  H̃(f2,p)
  @test f.u ≈ f2.u && f.v ≈ f2.v

  p = NodePair(Dual,(nx,ny))
  p2 = deepcopy(p)
  Hmat = RegularizationMatrix(H,f,p)
  Emat = InterpolationMatrix(H,p,f)
  Ẽmat = InterpolationMatrix(H̃,p,f)

  mul!(p,Hmat,f)
  H(p2,f)
  @test p.u ≈ p2.u && p.v ≈ p2.v

  p.u .= rand(Float64,size(p.u))
  p.v .= rand(Float64,size(p.v))
  mul!(f,Emat,p)
  H(f2,p)
  @test f.u ≈ f2.u && f.v ≈ f2.v

  mul!(f,Ẽmat,p)
  H̃(f2,p)
  @test f.u ≈ f2.u && f.v ≈ f2.v

  p = NodePair(Primal,(nx,ny))
  p2 = deepcopy(p)
  Hmat = RegularizationMatrix(H,f,p)
  Emat = InterpolationMatrix(H,p,f)
  Ẽmat = InterpolationMatrix(H̃,p,f)
  mul!(p,Hmat,f)
  H(p2,f)
  @test p.u ≈ p2.u && p.v ≈ p2.v

  p.u .= rand(Float64,size(p.u))
  p.v .= rand(Float64,size(p.v))
  mul!(f,Emat,p)
  H(f2,p)
  @test f.u ≈ f2.u && f.v ≈ f2.v

  mul!(f,Ẽmat,p)
  H̃(f2,p)
  @test f.u ≈ f2.u && f.v ≈ f2.v

  f = TensorData(X)
  f.dudx .= rand(n)
  f.dudy .= rand(n)
  f.dvdx .= rand(n)
  f.dvdy .= rand(n)
  f2 = TensorData(f)

  p = EdgeGradient(Dual,(nx,ny))
  p2 = deepcopy(p)
  Hmat = RegularizationMatrix(H,f,p)
  Emat = InterpolationMatrix(H,p,f)
  Ẽmat = InterpolationMatrix(H̃,p,f)

  mul!(p,Hmat,f)
  H(p2,f)
  @test p.dudx ≈ p2.dudx && p.dudy ≈ p2.dudy && p.dvdx ≈ p2.dvdx && p.dvdy ≈ p2.dvdy

  p.dudx .= rand(Float64,size(p.dudx))
  p.dudy .= rand(Float64,size(p.dudy))
  p.dvdx .= rand(Float64,size(p.dvdx))
  p.dvdy .= rand(Float64,size(p.dvdy))
  mul!(f,Emat,p)
  H(f2,p)
  @test f.dudx ≈ f2.dudx && f.dudy ≈ f2.dudy && f.dvdx ≈ f2.dvdx && f.dvdy ≈ f2.dvdy

  mul!(f,Ẽmat,p)
  H̃(f2,p)
  @test f.dudx ≈ f2.dudx && f.dudy ≈ f2.dudy && f.dvdx ≈ f2.dvdx && f.dvdy ≈ f2.dvdy

  p = EdgeGradient(Primal,(nx,ny))
  p2 = deepcopy(p)
  Hmat = RegularizationMatrix(H,f,p)
  Emat = InterpolationMatrix(H,p,f)
  Ẽmat = InterpolationMatrix(H̃,p,f)

  mul!(p,Hmat,f)
  H(p2,f)
  @test p.dudx ≈ p2.dudx && p.dudy ≈ p2.dudy && p.dvdx ≈ p2.dvdx && p.dvdy ≈ p2.dvdy

  p.dudx .= rand(Float64,size(p.dudx))
  p.dudy .= rand(Float64,size(p.dudy))
  p.dvdx .= rand(Float64,size(p.dvdx))
  p.dvdy .= rand(Float64,size(p.dvdy))
  mul!(f,Emat,p)
  H(f2,p)
  @test f.dudx ≈ f2.dudx && f.dudy ≈ f2.dudy && f.dvdx ≈ f2.dvdx && f.dvdy ≈ f2.dvdy

  mul!(f,Ẽmat,p)
  H̃(f2,p)
  @test f.dudx ≈ f2.dudx && f.dudy ≈ f2.dudy && f.dvdx ≈ f2.dvdx && f.dvdy ≈ f2.dvdy

  Hs(p2,f)
  Hsmat, Esmat = RegularizationMatrix(Hs,f,p)
  mul!(p,Hsmat,f)
  @test p.dudx ≈ p2.dudx && p.dudy ≈ p2.dudy && p.dvdx ≈ p2.dvdx && p.dvdy ≈ p2.dvdy


  end

### COMPLEX ROUTINES

@testset "Complex point data" begin

  f = ScalarData(10)
  f .= rand(10)

  fc = ScalarData(f,dtype=ComplexF64)
  ftc = TensorData(f,dtype=ComplexF64)

  a = 1.0 + 2.0im
  fc[5] = a

  @test fc[5] == a

  fvc = VectorData(f,dtype=ComplexF64)
  b = (-2.0+0im,5.0im)
  gvc = fvc + b
  @test gvc.u[5] == b[1]
  @test gvc.v[5] == b[2]

  gvc *= 2im
  @test gvc.u[5] == 2im*b[1]
  @test gvc.v[5] == 2im*b[2]


end


end
