using FFTW

import LinearAlgebra: norm, dot, mul!

_size(::CartesianGrids.Laplacian{NX,NY}) where {NX,NY} = NX,NY

function randomize!(f::ScalarGridData;offset=0)
    f[offset:end-offset,offset:end-offset] .= randn(size(f[offset:end-offset,offset:end-offset]))
    return f
end

@testset "Grid Routines" begin

  # size
  nx = 12; ny = 12

  # sample point
  i = 5; j = 7
  cellzero = Nodes(Dual,(nx,ny))
  nodezero = Nodes(Primal,cellzero)
  facezero = Edges(Primal,cellzero)
  dualfacezero = Edges(Dual,cellzero)

  @test typeof(cellzero) <: ScalarGridData

  @test typeof(nodezero) <: ScalarGridData

  @test typeof(facezero) <: VectorGridData

  @test typeof(dualfacezero) <: VectorGridData

  @test typeof(facezero.u) <: XEdges{Primal}
  @test typeof(facezero.v) <: YEdges{Primal}

  @test typeof(dualfacezero.u) <: XEdges{Dual}
  @test typeof(dualfacezero.v) <: YEdges{Dual}

  @test celltype(cellzero) == Dual
  @test celltype(facezero) == Primal
  @test griddatatype(dualfacezero) == Edges

  @test griddatatype(NodePair(Dual,dualfacezero)) == NodePair
  @test griddatatype(EdgeGradient(Dual,cellzero)) == EdgeGradient

  cellzero2 = typeof(cellzero)()
  @test typeof(cellzero2) == typeof(cellzero)

  cellzero2 = zero(cellzero)
  @test typeof(cellzero2) == typeof(cellzero)

  cellunit = deepcopy(cellzero)
  cellunit[i,j] = 1.0

  nodeunit = deepcopy(nodezero)
  nodeunit[i,j] = 1.0

  facexunit = deepcopy(facezero)
  facexunit.u[i,j] = 1.0

  faceyunit = deepcopy(facezero)
  faceyunit.v[i,j] = 1.0

  dualfacexunit = deepcopy(dualfacezero)
  dualfacexunit.u[i,j] = 1.0

  dualfaceyunit = deepcopy(dualfacezero)
  dualfaceyunit.v[i,j] = 1.0



  @testset "Basic array operations" begin
    w = zero(cellunit)
    w .= cellunit
    @test w[i,j] == 1.0
    q = similar(facexunit)
    q .= facexunit
    @test q.u[i,j] == 1.0
    @test iszero(q.v)

    # providing a vector of data to GridData
    data = zeros(Float64,length(w))
    w2 = typeof(w)(data)
    w2[1,1] = 2.0
    @test vec(w2) == data

    data = zeros(Float64,length(q))
    q2 = typeof(q)(data)
    @test all(q2 .== 0.0)
    @test vec(q2) == data

    q2.u .= 2.0
    @test all(data[1:length(q2.u)] .== 2.0)


  end

  @testset "Arithmetic" begin

    w = 2*cellunit
    @test w[i,j] == 2
    w2 = w + w
    @test w2[i,j] == 4

    q = Edges(Dual,w)
    q.u .= 1.0
    q2 = 2*q
    @test all(q2.u .== 2.0)
    q2 .= q + q2
    @test all(q2.u .== 3.0)

    t = EdgeGradient(Dual,w)
    t .= 1.0
    t2 = 2*t
    @test all(t2 .== 2.0)
    t2 = -t
    @test all(t2 .== -1.0)


  end

  @testset "Inner products and norms" begin
    w = zero(cellunit)
    i0, j0 = rand(2:nx-1), rand(2:ny-1)
    w[i0,j0] = 1.0
    @test norm(w)*sqrt((nx-2)*(ny-2)) == 1.0
    w .= 1.0
    @test norm(w) == 1.0

    @test 2*w == w*2

    p = Nodes(Primal,w)
    p .= 1.0
    @test norm(p) == 1.0
    p2 = deepcopy(p)
    @test dot(p,p2) == 1.0
    @test norm(p-p2) == 0.0

    q = Edges(Dual,w)
    q.u .= 1.0
    q2 = deepcopy(q)
    @test dot(q,q2) == 1.0

    q = Edges(Primal,w)
    q.u .= 1.0
    q2 = deepcopy(q)
    @test dot(q,q2) == 1.0

    @test integrate(w) == 1.0

    @test integrate(p) == 1.0

    q .= 1
    @test norm(2*q) == sqrt(8)

    v = Edges(Primal,(100,200))
    v.u[3:end-3,3:end-3] .= rand(size(v.u[3:end-3,3:end-3])...)
    v.v[3:end-3,3:end-3] .= rand(size(v.v[3:end-3,3:end-3])...)

    dq = EdgeGradient(Primal,v)

    grid_interpolate!(dq,v)

    dq2 = similar(dq)
    dq2.dudx[3:end-3,3:end-3] .= rand(size(dq2.dudx[3:end-3,3:end-3])...)
    dq2.dudy[3:end-3,3:end-3] .= rand(size(dq2.dudy[3:end-3,3:end-3])...)
    dq2.dvdx[3:end-3,3:end-3] .= rand(size(dq2.dvdx[3:end-3,3:end-3])...)
    dq2.dvdy[3:end-3,3:end-3] .= rand(size(dq2.dvdy[3:end-3,3:end-3])...)

    v2 = similar(v)
    grid_interpolate!(v2,dq2)

    @test isapprox(dot(dq2,dq),dot(v2,v),atol=100*eps(1.0))

  end

  @testset "Dual cell center data Laplacian" begin
    lapcell = laplacian(cellunit)
    @test lapcell[i,j] == -4.0
    lapcell[i,j] = 0.0
    @test lapcell[i+1,j] == lapcell[i-1,j] == lapcell[i,j-1] == lapcell[i,j+1] == 1.0
    lapcell[i+1,j] = lapcell[i-1,j] = lapcell[i,j-1] = lapcell[i,j+1] = 0.0
    @test iszero(lapcell)
  end

  @testset "Dual cell center data curl" begin
    q = curl(cellunit)
    @test q.u[i,j-1] == 1.0 && q.u[i,j] == -1.0
    q.u[i,j-1] = q.u[i,j] = 0.0
    @test iszero(q.u)
    @test q.v[i-1,j] == -1.0 && q.v[i,j] == 1.0
    q.v[i-1,j] = q.v[i,j] = 0.0
    @test iszero(q.v)
  end

  @testset "Dual cell node gradient" begin
    q = grad(nodeunit)
    @test q.u[i,j] == 1.0 && q.u[i+1,j] == -1.0
    q.u[i,j] = q.u[i+1,j] = 0.0
    @test iszero(q.u)
    @test q.v[i,j] == 1.0 && q.v[i,j+1] == -1.0
    q.v[i,j] = q.v[i,j+1] = 0.0
    @test iszero(q.v)
  end

  @testset "Face data curl" begin
    cellcurl = curl(facexunit)
    @test cellcurl[i,j] == -1.0 && cellcurl[i,j+1] == 1.0
    cellcurl[i,j] = cellcurl[i,j+1] = 0.0
    @test iszero(cellcurl)
    cellcurl = curl(faceyunit)
    @test cellcurl[i,j] == 1.0 && cellcurl[i+1,j] == -1.0
    cellcurl[i,j] = cellcurl[i+1,j] = 0.0
    @test iszero(cellcurl)
  end

  @testset "Face data divergence" begin
    nodediv = divergence(facexunit)
    @test nodediv[i,j] == -1.0 && nodediv[i-1,j] == 1.0
    nodediv[i,j] = nodediv[i-1,j] = 0.0
    @test iszero(nodediv)
    nodediv = divergence(faceyunit)
    @test nodediv[i,j] == -1.0 && nodediv[i,j-1] == 1.0
    nodediv[i,j] = nodediv[i,j-1] = 0.0
    @test iszero(nodediv)
  end

  @testset "Face data Laplacian" begin
    lap = laplacian(facexunit)
    @test lap.u[i,j] == -4.0
    lap.u[i,j] = 0.0
    @test lap.u[i+1,j] == lap.u[i-1,j] == lap.u[i,j-1] == lap.u[i,j+1] == 1.0
    lap.u[i+1,j] = lap.u[i-1,j] = lap.u[i,j-1] = lap.u[i,j+1] = 0.0
    @test iszero(lap.u)
    @test iszero(lap.v)

    lap = laplacian(faceyunit)
    @test lap.v[i,j] == -4.0
    lap.v[i,j] = 0.0
    @test lap.v[i+1,j] == lap.v[i-1,j] == lap.v[i,j-1] == lap.v[i,j+1] == 1.0
    lap.v[i+1,j] = lap.v[i-1,j] = lap.v[i,j-1] = lap.v[i,j+1] = 0.0
    @test iszero(lap.u)
    @test iszero(lap.v)
  end

  @testset "Dual face data divergence" begin
    celldiv = divergence(dualfacexunit)
    @test celldiv[i,j] == 1.0 && celldiv[i+1,j] == -1.0
    celldiv[i,j] = celldiv[i+1,j] = 0.0
    @test iszero(celldiv)
    celldiv = divergence(dualfaceyunit)
    @test celldiv[i,j] == 1.0 && celldiv[i,j+1] == -1.0
    celldiv[i,j] = celldiv[i,j+1] = 0.0
    @test iszero(celldiv)
  end

  @testset "Face data shift to dual face" begin
    shiftx = Edges(Dual,cellzero)
    grid_interpolate!(shiftx,facexunit)
    @test shiftx.u[i,j] == shiftx.u[i-1,j] == shiftx.u[i,j+1] == shiftx.u[i-1,j+1] == 0.25
    shiftx.u[i,j] = shiftx.u[i-1,j] = shiftx.u[i,j+1] = shiftx.u[i-1,j+1] = 0.0
    @test iszero(shiftx.u)
    @test iszero(shiftx.v)
    shifty = Edges(Dual,cellzero)
    grid_interpolate!(shifty,faceyunit)
    @test shifty.v[i,j] == shifty.v[i,j-1] == shifty.v[i+1,j] == shifty.v[i+1,j-1] == 0.25
    shifty.v[i,j] = shifty.v[i,j-1] = shifty.v[i+1,j] = shifty.v[i+1,j-1] = 0.0
    @test iszero(shifty.u)
    @test iszero(shifty.v)
  end

  @testset "Dual cell center data shift to dual face" begin
    w = Edges(Dual,cellzero)
    grid_interpolate!(w,cellunit)
    @test w.u[i,j] == w.u[i-1,j] == 0.5
    w.u[i,j] = w.u[i-1,j] = 0.0
    @test iszero(w.u)
    @test w.v[i,j] == w.v[i,j-1] == 0.5
    w.v[i,j] = w.v[i,j-1] = 0.0
    @test iszero(w.v)
  end

  @testset "Face data shift to dual cell center" begin
    cellx = Nodes(Dual,cellzero)
    celly = Nodes(Dual,cellzero)
    grid_interpolate!((cellx,celly),facexunit)
    @test cellx[i,j] == 0.5 && cellx[i,j+1] == 0.5
    cellx[i,j] = cellx[i,j+1] = 0.0
    @test iszero(cellx)
    @test iszero(celly)

    celltmp = Nodes(Dual,cellzero)
    grid_interpolate!(celltmp,facexunit)
    @test celltmp[i,j] == 0.5 && celltmp[i,j+1] == 0.5
    celltmp[i,j] = celltmp[i,j+1] = 0.0
    @test iszero(celltmp)

    cellx = Nodes(Dual,cellzero)
    celly = Nodes(Dual,cellzero)
    grid_interpolate!((cellx,celly),faceyunit)
    @test celly[i,j] == 0.5 && celly[i+1,j] == 0.5
    celly[i,j] = celly[i+1,j] = 0.0
    @test iszero(cellx)
    @test iszero(celly)

    celltmp = Nodes(Dual,cellzero)
    grid_interpolate!(celltmp,faceyunit)
    @test celltmp[i,j] == 0.5 && celltmp[i+1,j] == 0.5
    celltmp[i,j] = celltmp[i+1,j] = 0.0
    @test iszero(celltmp)

  end

  @testset "Nonlinear operations" begin
    usq = Nodes(Primal,facexunit)
    gridwise_dot!(usq,facexunit,facexunit)
    @test usq[i,j] == 0.5
    @test gridwise_dot(facexunit,facexunit) == usq

    magsq!(usq,facexunit)
    @test usq[i,j] == 0.5
    @test magsq(facexunit) == usq

    magsq!(usq,faceyunit)
    @test usq[i,j] == 0.5

    usq = Nodes(Dual,facexunit)
    magsq!(usq,dualfacexunit)
    @test usq[i,j] == 0.5

    magsq!(usq,dualfaceyunit)
    @test usq[i,j] == 0.5

    magu = Nodes(Primal,facexunit)
    mag!(magu,facexunit)
    @test magu[i,j] == sqrt(0.5)
    @test mag(facexunit) == magu

    magu = Nodes(Dual,facexunit)
    mag!(magu,dualfacexunit)
    @test magu[i,j] == sqrt(0.5)
    @test mag(dualfacexunit) == magu


  end

  @testset "div curl" begin
    @test iszero(divergence(curl(cellunit)))
    @test iszero(curl(grad(nodeunit)))
  end



  v = Edges(Primal,(100,200))
  offset = 4
  randomize!(v.u,offset=offset)
  randomize!(v.v,offset=offset)

  u = Edges(Primal,(100,200))
  randomize!(u.u,offset=offset)
  randomize!(u.v,offset=offset)

  vd = Edges(Dual,(100,200))
  randomize!(v.u,offset=offset)
  randomize!(v.v,offset=offset)

  ud = Edges(Dual,(100,200))
  randomize!(u.u,offset=offset)
  randomize!(u.v,offset=offset)

  f1 = Nodes(Primal,u)
  f2 = Nodes(Primal,u)
  randomize!(f1,offset=offset)
  randomize!(f2,offset=offset)

  w = Nodes(Dual,u)
  randomize!(w,offset=offset)

  dq = EdgeGradient(Primal,v)
  dv = EdgeGradient(Primal,v)
  dqd = EdgeGradient(Dual,v)
  dvd = EdgeGradient(Dual,v)

  lhsn = Nodes(Primal,f1)
  rhsn = Nodes(Primal,f1)
  lhsnd = Nodes(Dual,f1)
  rhsnd = Nodes(Dual,f1)

  lhse = Edges(Primal,f1)
  rhse = Edges(Primal,f1)
  lhsed = Edges(Dual,f1)


  qtmp = Edges(Primal,u)
  qtmpd = Edges(Dual,u)
  qtmp1d = Edges(Dual,u)
  qtmp2d = Edges(Dual,u)
  ftmp = Nodes(Primal,u)
  wtmp = Nodes(Dual,u)
  utmp = EdgeGradient(Primal,u)
  utmpd = EdgeGradient(Dual,u)


  @testset "Commutativity of operators" begin

    # Laplacian as divergence of the gradient (edge data)
    divergence!(lhse,grad!(dv,v))
    rhse = laplacian(v)
    @test isapprox(norm(lhse-rhse),0.0,atol=100.0*eps())


    # Node laplacian of the edge-to-node interpolation =
    # edge-to-node interpolation of the edge laplacian
    lhsn .= laplacian(grid_interpolate!(ftmp,v))
    rhsn .= 0.0
    grid_interpolate!(rhsn,laplacian(v))
    @test isapprox(norm(lhsn-rhsn),0.0,atol=100.0*eps())

    # Edge laplacian of the node-to-edge interpolation =
    # node-to-edge interpolation of the node laplacian
    lhse .= 0
    rhse .= 0
    qtmp .= 0
    grid_interpolate!(lhse,laplacian(f1))
    rhse .= laplacian(grid_interpolate!(qtmp,f1))
    @test isapprox(norm(lhse-rhse),0.0,atol=100.0*eps())

    # Curl of the edge laplacian = node laplacian of the curl
    lhsnd .= curl(laplacian(v))
    rhsnd .= laplacian(curl(v))
    @test isapprox(norm(lhsnd-rhsnd),0.0,atol=100.0*eps())


  end

  @testset "Discrete product rules" begin

    # Laplacian of curl = curl of Laplacian
    curllapv = curl(laplacian(v))
    lapcurlv = laplacian(curl(v))

    @test isapprox(norm(curllapv-lapcurlv),0.0,atol=100.0*eps())

    # grad(f1*f2) = f1*grad(f2) + grad(f1)*f2
    lhse .= 0
    rhse .= 0
    qtmp .= 0
    lhse .= grad(f1∘f2)
    rhse .= grad(f1)∘ grid_interpolate!(qtmp,f2) + grid_interpolate!(qtmp,f1) ∘ grad(f2)
    @test isapprox(norm(lhse-rhse),0.0,atol=100.0*eps())

    # div(f1*u) = grad(f1).u + f1*(div u)
    lhsn .= 0
    rhsn .= 0
    qtmp .= 0
    lhsn .= divergence(grid_interpolate!(qtmp,f1)∘u)
    grid_interpolate!(ftmp,grad(f1)∘u)
    rhsn .= ftmp + f1∘divergence(u)
    @test isapprox(norm(lhsn-rhsn),0.0,atol=100.0*eps())

    # div(u*w) = grad(w).u + w*(div u)
    lhsnd .= 0
    rhsnd .= 0
    qtmp1d .= 0
    qtmp2d .= 0
    lhsnd .= divergence(grid_interpolate!(qtmp1d,u)∘grid_interpolate!(qtmp2d,w))
    qtmp1d .= 0
    grid_interpolate!(rhsnd,grid_interpolate!(qtmp1d,u)∘grad(w))
    wtmp .= 0
    rhsnd .+= w∘grid_interpolate!(wtmp,divergence(u))
    @test isapprox(norm(lhsnd-rhsnd),0.0,atol=100.0*eps())


    # div(uv) = u.grad v + (div u)v
    divuv = zero(u)
    uv = EdgeGradient(Primal,u)
    divergence!(divuv,u*v)

    dv .= 0.0
    utmp .= 0.0
    dq .= 0
    ugradv = zero(u)
    product!(dq,transpose(grid_interpolate!(utmp,u)),grad!(dv,v))
    grid_interpolate!(ugradv,dq)

    lhse .= 0
    directional_derivative!(lhse,v,u)
    @test isapprox(norm(lhse - ugradv),0.0,atol=100.0*eps())

    qtmp .= 0.0
    vdivu = zero(u)
    product!(vdivu,grid_interpolate!(qtmp,divergence(u)),v)

    @test isapprox(norm(divuv - ugradv - vdivu),0.0,atol=100.0*eps())

    # Now on duals
    divuv = zero(ud)
    uv = EdgeGradient(Primal,ud)
    divergence!(divuv,ud*vd)

    dvd .= 0.0
    utmpd .= 0.0
    dqd .= 0.0
    ugradv = zero(ud)
    product!(dqd,transpose(grid_interpolate!(utmpd,ud)),grad!(dvd,vd))
    grid_interpolate!(ugradv,dqd)

    lhsed .= 0
    directional_derivative!(lhsed,vd,ud)
    @test isapprox(norm(lhsed - ugradv),0.0,atol=100.0*eps())

    qtmpd .= 0.0
    vdivu = zero(ud)
    product!(vdivu,grid_interpolate!(qtmpd,divergence(ud)),vd)

    @test isapprox(norm(divuv - ugradv - vdivu),0.0,atol=100.0*eps())



  end

  L = plan_laplacian(nx,ny;with_inverse=true)


  @testset "Laplacian of the LGF" begin
    ψ = L\cellunit
    lapψ = L*ψ
    @test lapψ[i,j]≈1.0
    lapψ[1:end .∉ i,:]
    @test isapprox(maximum(abs.(lapψ[1:end .∉ i,:])),0.0;atol=10.0*eps()) &&
            isapprox(maximum(abs.(lapψ[:,1:end .∉ j])),0.0;atol=10.0*eps())

    ψ = L\nodeunit
    lapψ = L*ψ
    @test lapψ[i,j]≈1.0
    @test isapprox(maximum(abs.(lapψ[1:end .∉ i,:])),0.0;atol=10.0*eps()) &&
            isapprox(maximum(abs.(lapψ[:,1:end .∉ j])),0.0;atol=10.0*eps())

    ψ = L\facexunit.u
    lapψ = L*ψ
    @test lapψ[i,j]≈1.0
    @test isapprox(maximum(abs.(lapψ[1:end .∉ i,:])),0.0;atol=10.0*eps()) &&
            isapprox(maximum(abs.(lapψ[:,1:end .∉ j])),0.0;atol=10.0*eps())

    ψ = L\faceyunit.v
    lapψ = L*ψ
    @test lapψ[i,j]≈1.0
    @test isapprox(maximum(abs.(lapψ[1:end .∉ i,:])),0.0;atol=10.0*eps()) &&
            isapprox(maximum(abs.(lapψ[:,1:end .∉ j])),0.0;atol=10.0*eps())


    ψ = L\dualfacexunit.u
    lapψ = L*ψ
    @test lapψ[i,j]≈1.0
    @test isapprox(maximum(abs.(lapψ[1:end .∉ i,:])),0.0;atol=10.0*eps()) &&
            isapprox(maximum(abs.(lapψ[:,1:end .∉ j])),0.0;atol=10.0*eps())

    ψ = L\dualfaceyunit.v
    lapψ = L*ψ
    @test lapψ[i,j]≈1.0
    @test isapprox(maximum(abs.(lapψ[1:end .∉ i,:])),0.0;atol=10.0*eps()) &&
            isapprox(maximum(abs.(lapψ[:,1:end .∉ j])),0.0;atol=10.0*eps())

    L2 = plan_laplacian(nodeunit,with_inverse=true)
    ψ = L2\nodeunit
    lapψ = L2*ψ
    @test lapψ[i,j]≈1.0


    L2_2 = plan_laplacian(Edges(Primal,nodeunit),with_inverse=true)
    @test _size(L2_2) == _size(L2)

  end

  @testset "Laplacian of the LGF with factor" begin
    for Lscale in [plan_laplacian(nx,ny;with_inverse=true,factor=2.0), 2.0 * L, L * 2.0]
      ψ = Lscale\cellunit
      lapψ = Lscale*ψ
      @test lapψ[i,j]≈1.0

      ψ = Lscale*cellunit
      @test ψ[i,j]≈-8.0 && ψ[i+1,j]≈2.0

      mul!(ψ,Lscale,cellunit)
      @test ψ[i,j]≈-8.0 && ψ[i-1,j]≈2.0
    end
  end


  @testset "LGF for Helmholtz equation" begin
    alpha = 0.02
    LH(i,j,f::Function,α) = im*α*f(i,j,α)-(f(i-1,j,α)+f(i+1,j,α)+f(i,j+1,α)+f(i,j-1,α)-4*f(i,j,α))

    i0, j0 = rand(1:100), rand(0:100)
    @test abs(LH(i0,j0,CartesianGrids.lgf_helmholtz,alpha)) < 100.0*eps()

    @test isapprox(real(LH(0,0,CartesianGrids.lgf_helmholtz,alpha)),1.0;atol=100.0*eps())


  end

  @testset "Implicit diffusion" begin
    a = rand()
    A = plan_implicit_diffusion(a,(nx,ny))

    v = A\cellunit
    dudt = (v - cellunit)/a
    lapv = similar(v)
    laplacian!(lapv,v)

    @test norm(dudt-lapv) < 1000.0*eps()

    u2 = A*v
    @test u2[i,j] ≈ 1.0

    v = A\nodeunit
    dudt = (v - nodeunit)/a
    lapv = similar(v)
    laplacian!(lapv,v)

    @test norm(dudt[2:end-1,2:end-1]-lapv[2:end-1,2:end-1]) < 1000.0*eps()

    u2 = A*v
    @test u2[i,j] ≈ 1.0

    v = A\facexunit.u
    dudt = (v - facexunit.u)/a
    lapv = similar(v)
    laplacian!(lapv,v)

    @test norm(dudt[2:end-1,2:end-1]-lapv[2:end-1,2:end-1]) < 1000.0*eps()

    u2 = A*v
    @test u2[i,j] ≈ 1.0

    v = A\faceyunit.v
    dudt = (v - faceyunit.v)/a
    lapv = similar(v)
    laplacian!(lapv,v)

    @test norm(dudt[2:end-1,2:end-1]-lapv[2:end-1,2:end-1]) < 1000.0*eps()

    u2 = A*v
    @test u2[i,j] ≈ 1.0

    v = A\dualfacexunit.u
    dudt = (v - dualfacexunit.u)/a
    lapv = similar(v)
    laplacian!(lapv,v)

    @test norm(dudt[2:end-1,2:end-1]-lapv[2:end-1,2:end-1]) < 1000.0*eps()

    u2 = A*v
    @test u2[i,j] ≈ 1.0

    v = A\dualfaceyunit.v
    dudt = (v - dualfaceyunit.v)/a
    lapv = similar(v)
    laplacian!(lapv,v)

    @test norm(dudt[2:end-1,2:end-1]-lapv[2:end-1,2:end-1]) < 1000.0*eps()

    u2 = A*v
    @test u2[i,j] ≈ 1.0

  end

end

##### COMPLEX GRID #####

@testset "Complex Grid Routines" begin

# size
nx = 12; ny = 12

# sample point
i = 5; j = 7

cellzero = Nodes(Dual,(nx,ny),dtype=ComplexF64)
nodezero = Nodes(Primal,cellzero)
facezero = Edges(Primal,cellzero)
dualfacezero = Edges(Dual,cellzero)

a = 1.0+2.0im

cellunit = deepcopy(cellzero)
cellunit[i,j] = a

nodeunit = deepcopy(nodezero)
nodeunit[i,j] = a

facexunit = deepcopy(facezero)
facexunit.u[i,j] = a

faceyunit = deepcopy(facezero)
faceyunit.v[i,j] = a

dualfacexunit = deepcopy(dualfacezero)
dualfacexunit.u[i,j] = a

dualfaceyunit = deepcopy(dualfacezero)
dualfaceyunit.v[i,j] = a

@testset "Basic array operations" begin
  w = zero(cellunit)
  w .= cellunit
  @test w[i,j] == a
  q = similar(facexunit)
  q .= facexunit
  @test q.u[i,j] == a
  @test iszero(q.v.data)
end

@testset "Basic complex operations" begin
  w = conj(cellunit)
  @test typeof(w) == typeof(cellunit)
  @test w[i,j] == conj(a)

  w = real(cellunit)
  @test typeof(w) <: Nodes{Dual,nx,ny,Float64}
  @test real(w[i,j]) == real(a) && imag(w[i,j]) == 0.0

  w = imag(cellunit)
  @test typeof(w) <: Nodes{Dual,nx,ny,Float64}
  @test real(w[i,j]) == imag(a) && imag(w[i,j]) == 0.0

end

@testset "Inner products and norms" begin
  w = zero(cellunit)
  i0, j0 = rand(2:nx-1), rand(2:ny-1)
  w[i0,j0] = 1.0im
  @test norm(w)*sqrt((nx-2)*(ny-2)) == 1.0
  w .= 1.0im
  @test norm(w) == 1.0

  @test 2*w == w*2

  p = Nodes(Primal,w)
  p .= 1.0im
  @test norm(p) == 1.0
  p2 = deepcopy(p)
  @test dot(p,p2) == 1.0
  @test norm(p-p2) == 0.0

  q = Edges(Dual,w)
  q.u .= 1.0im
  q2 = deepcopy(q)
  @test dot(q,q2) == 1.0

  q = Edges(Primal,w)
  q.u .= 1.0im
  q2 = deepcopy(q)
  @test dot(q,q2) == 1.0

  @test integrate(w) == 1.0im

  @test integrate(p) == 1.0im

  q .= 1.0im
  @test norm(2*q) == sqrt(8)


end

@testset "Dual cell center data Laplacian" begin
  lapcell = laplacian(cellunit)
  @test lapcell[i,j] == -4.0*a
  lapcell[i,j] = 0.0
  @test lapcell[i+1,j] == lapcell[i-1,j] == lapcell[i,j-1] == lapcell[i,j+1] == a
  lapcell[i+1,j] = lapcell[i-1,j] = lapcell[i,j-1] = lapcell[i,j+1] = 0.0
  @test iszero(lapcell)
end

@testset "Dual cell center data curl" begin
  q = curl(cellunit)
  @test q.u[i,j-1] == 1.0*a && q.u[i,j] == -1.0*a
  q.u[i,j-1] = q.u[i,j] = 0.0
  @test iszero(q.u)
  @test q.v[i-1,j] == -1.0*a && q.v[i,j] == 1.0*a
  q.v[i-1,j] = q.v[i,j] = 0.0
  @test iszero(q.v)
end

@testset "Dual cell node gradient" begin
  q = grad(nodeunit)
  @test q.u[i,j] == 1.0*a && q.u[i+1,j] == -1.0*a
  q.u[i,j] = q.u[i+1,j] = 0.0
  @test iszero(q.u)
  @test q.v[i,j] == 1.0*a && q.v[i,j+1] == -1.0*a
  q.v[i,j] = q.v[i,j+1] = 0.0
  @test iszero(q.v)
end

@testset "Face data curl" begin
  cellcurl = curl(facexunit)
  @test cellcurl[i,j] == -1.0*a && cellcurl[i,j+1] == 1.0*a
  cellcurl[i,j] = cellcurl[i,j+1] = 0.0
  @test iszero(cellcurl)
  cellcurl = curl(faceyunit)
  @test cellcurl[i,j] == 1.0*a && cellcurl[i+1,j] == -1.0*a
  cellcurl[i,j] = cellcurl[i+1,j] = 0.0
  @test iszero(cellcurl)
end

@testset "Face data divergence" begin
  nodediv = divergence(facexunit)
  @test nodediv[i,j] == -1.0*a && nodediv[i-1,j] == 1.0*a
  nodediv[i,j] = nodediv[i-1,j] = 0.0
  @test iszero(nodediv)
  nodediv = divergence(faceyunit)
  @test nodediv[i,j] == -1.0*a && nodediv[i,j-1] == 1.0*a
  nodediv[i,j] = nodediv[i,j-1] = 0.0
  @test iszero(nodediv)
end

@testset "Face data Laplacian" begin
  lap = laplacian(facexunit)
  @test lap.u[i,j] == -4.0*a
  lap.u[i,j] = 0.0
  @test lap.u[i+1,j] == lap.u[i-1,j] == lap.u[i,j-1] == lap.u[i,j+1] == 1.0*a
  lap.u[i+1,j] = lap.u[i-1,j] = lap.u[i,j-1] = lap.u[i,j+1] = 0.0
  @test iszero(lap.u)
  @test iszero(lap.v)

  lap = laplacian(faceyunit)
  @test lap.v[i,j] == -4.0*a
  lap.v[i,j] = 0.0
  @test lap.v[i+1,j] == lap.v[i-1,j] == lap.v[i,j-1] == lap.v[i,j+1] == 1.0*a
  lap.v[i+1,j] = lap.v[i-1,j] = lap.v[i,j-1] = lap.v[i,j+1] = 0.0
  @test iszero(lap.u)
  @test iszero(lap.v)
end

@testset "Dual face data divergence" begin
  celldiv = divergence(dualfacexunit)
  @test celldiv[i,j] == 1.0*a && celldiv[i+1,j] == -1.0*a
  celldiv[i,j] = celldiv[i+1,j] = 0.0
  @test iszero(celldiv)
  celldiv = divergence(dualfaceyunit)
  @test celldiv[i,j] == 1.0*a && celldiv[i,j+1] == -1.0*a
  celldiv[i,j] = celldiv[i,j+1] = 0.0
  @test iszero(celldiv)
end

@testset "Face data shift to dual face" begin
  shiftx = Edges(Dual,cellzero)
  grid_interpolate!(shiftx,facexunit)
  @test shiftx.u[i,j] == shiftx.u[i-1,j] == shiftx.u[i,j+1] == shiftx.u[i-1,j+1] == 0.25*a
  shiftx.u[i,j] = shiftx.u[i-1,j] = shiftx.u[i,j+1] = shiftx.u[i-1,j+1] = 0.0
  @test iszero(shiftx.u)
  @test iszero(shiftx.v)
  shifty = Edges(Dual,cellzero)
  grid_interpolate!(shifty,faceyunit)
  @test shifty.v[i,j] == shifty.v[i,j-1] == shifty.v[i+1,j] == shifty.v[i+1,j-1] == 0.25*a
  shifty.v[i,j] = shifty.v[i,j-1] = shifty.v[i+1,j] = shifty.v[i+1,j-1] = 0.0
  @test iszero(shifty.u)
  @test iszero(shifty.v)
end

@testset "Dual cell center data shift to dual face" begin
  w = Edges(Dual,cellzero)
  grid_interpolate!(w,cellunit)
  @test w.u[i,j] == w.u[i-1,j] == 0.5*a
  w.u[i,j] = w.u[i-1,j] = 0.0
  @test iszero(w.u)
  @test w.v[i,j] == w.v[i,j-1] == 0.5*a
  w.v[i,j] = w.v[i,j-1] = 0.0
  @test iszero(w.v)
end

@testset "Face data shift to dual cell center" begin
  cellx = Nodes(Dual,cellzero)
  celly = Nodes(Dual,cellzero)
  grid_interpolate!((cellx,celly),facexunit)
  @test cellx[i,j] == 0.5*a && cellx[i,j+1] == 0.5*a
  cellx[i,j] = cellx[i,j+1] = 0.0
  @test iszero(cellx)
  @test iszero(celly)

  cellx = Nodes(Dual,cellzero)
  celly = Nodes(Dual,cellzero)
  grid_interpolate!((cellx,celly),faceyunit)
  @test celly[i,j] == 0.5*a && celly[i+1,j] == 0.5*a
  celly[i,j] = celly[i+1,j] = 0.0
  @test iszero(cellx)
  @test iszero(celly)
end

@testset "div curl" begin
  @test iszero(divergence(curl(cellunit)))
  @test iszero(curl(grad(nodeunit)))
end

L = plan_laplacian(nx,ny;with_inverse=true,dtype=ComplexF64)

@testset "Laplacian of the LGF" begin
  ψ = L\cellunit
  lapψ = L*ψ
  @test lapψ[i,j]≈1.0*a
  @test isapprox(maximum(abs.(lapψ[1:end .∉ i,:])),0.0;atol=100.0*eps()) &&
          isapprox(maximum(abs.(lapψ[:,1:end .∉ j])),0.0;atol=100.0*eps())

end

Lscale = plan_laplacian(nx,ny;with_inverse=true,factor=2.0,dtype=ComplexF64)

@testset "Laplacian of the LGF with factor" begin
   ψ = Lscale\cellunit
   lapψ = Lscale*ψ
   @test lapψ[i,j]≈1.0*a

   ψ = Lscale*cellunit
   @test ψ[i,j]≈-8.0*a && ψ[i+1,j]≈2.0*a

   mul!(ψ,Lscale,cellunit)
   @test ψ[i,j]≈-8.0*a && ψ[i-1,j]≈2.0*a


end

α = 0.07
LH = plan_helmholtz(nx,ny,α;with_inverse=true)
LH2 = plan_helmholtz(nx,ny,α;factor=2.0,with_inverse=true)


@testset "Helmholtz of the LGF" begin

  ψ = LH\cellunit
  helmψ = LH*ψ
  @test helmψ[i,j]≈1.0*a
  @test isapprox(maximum(abs.(helmψ[1:end .∉ i,:])),0.0;atol=100.0*eps()) &&
          isapprox(maximum(abs.(helmψ[:,1:end .∉ j])),0.0;atol=100.0*eps())

  ψ2 = LH2\cellunit
  @test ψ2[i,j] ≈ ψ[i,j]/2.0
  helmψ2 = LH2*ψ2
  @test helmψ2[i,j]≈1.0*a
  @test isapprox(maximum(abs.(helmψ2[1:end .∉ i,:])),0.0;atol=100.0*eps()) &&
          isapprox(maximum(abs.(helmψ2[:,1:end .∉ j])),0.0;atol=100.0*eps())

end

end

@testset "Fields" begin
    @testset "Hadamard Product" begin
        edges_p  = Edges(Primal,(30,40))
        edges_p.u .= rand(Float64,size(edges_p.u))

        # Should be safe for the output to be the same as the input
        edges_p2 = edges_p ∘ edges_p
        product!(edges_p, edges_p, edges_p)
        @test edges_p2.u == edges_p.u
        @test edges_p2.v == edges_p.v

        edges_d  = Edges{Dual, 30, 40, Float64}()
        @test_throws MethodError (edges_p ∘ edges_d)
    end

    @testset "Cartesian product of vectors" begin

      edges_p  = Edges(Primal,(30,40))
      edges_p.u .= rand(Float64,size(edges_p.u))

      edges_q  = Edges(Primal,(30,40))
      edges_q.u .= rand(Float64,size(edges_q.u))

      tensors_pq = edges_p * edges_q

      pt = EdgeGradient(Primal,Dual,(30,40))
      qt = EdgeGradient(Primal,Dual,(30,40))

      grid_interpolate!(pt,edges_p)
      grid_interpolate!(qt,edges_q)

      @test tensors_pq.dudx[12,25] == pt.dudx[12,25]*qt.dudx[12,25]
      @test tensors_pq.dvdx[12,25] == pt.dudy[12,25]*qt.dvdx[12,25]
      @test tensors_pq.dudy[12,25] == pt.dvdx[12,25]*qt.dudy[12,25]
      @test tensors_pq.dvdy[12,25] == pt.dvdy[12,25]*qt.dvdy[12,25]


    end

    @testset "Discrete Laplacian" begin
        s = Nodes(Dual,(30,40))
        s[3:end-2, 3:end-2] .= rand(26, 36)

        L = plan_laplacian(30, 40)

        @test L*s ≈ -curl(curl(s))

        @test_throws MethodError (L \ s)

        L = plan_laplacian(30, 40, with_inverse = true) #, fftw_flags = FFTW.PATIENT)
        @test L \ (L*s) ≈ s

        L! = plan_laplacian!(30, 40, with_inverse = true) #, fftw_flags = FFTW.PATIENT)
        sold = deepcopy(s)
        L! \ (L!*s)
        @test s ≈ sold
    end

    @testset "Integrating factor" begin
        s = Nodes(Dual,(30,40))
        s[15,15] = 1.0

        E1 = plan_intfact(1,s)
        E2 = plan_intfact(2,s)

        @test E1*(E1*s) ≈ E2*s

        E! = plan_intfact!(2,s)
        s2 = deepcopy(s)
        E! * s

        @test s ≈ E2*s2

        L = plan_laplacian(s,factor=2)
        EL = exp(L)
        @test EL*s ≈ E2*s

        L2 = plan_laplacian(s)
        EL2 = exp(2*L2)
        @test EL2*s ≈ E2*s

        p = XEdges(Primal,s)
        p[15,15] = 1.0
        @test E1*(E1*p) ≈ E2*p

        q = Edges(Primal,s)
        q.u[15,15] = 1.0
        @test E1*(E1*q) ≈ E2*q

        EmL = exp(-1*L)
        @test EmL\s == EL*s

    end

    @testset "Discrete Divergence" begin
        s = Nodes(Dual,(5,4))
        s .= rand(5, 4)

        @test norm(divergence(curl(s))) ≈ 0 atol=eps()

        s = Nodes(Primal,s)
        q′ = Edges(Primal,s)
        q′.u .= reshape(1:15, 5, 3)
        q′.v .= reshape(1:16, 4, 4)

        # Not sure if this is the behavior we want yet
        # Currently, the ghost cells are not affected
        # by the divergence operator
        #s .= 1.0
        divergence!(s, q′)
        @test s == [ 5.0  5.0  5.0
                     5.0  5.0  5.0
                     5.0  5.0  5.0
                     5.0  5.0  5.0 ]
    end

    @testset "Discrete Curl" begin
        s = Nodes(Dual,(5,4))
        s .= reshape(1:20, 4, 5)'

        q = curl(s)

        @test q.u == [ 1.0  1.0  1.0
                       1.0  1.0  1.0
                       1.0  1.0  1.0
                       1.0  1.0  1.0
                       1.0  1.0  1.0 ]

        @test q.v == [ -4.0  -4.0  -4.0  -4.0
                       -4.0  -4.0  -4.0  -4.0
                       -4.0  -4.0  -4.0  -4.0
                       -4.0  -4.0  -4.0  -4.0 ]
    end

    @testset "Shifting Primal Edges to Dual Edges" begin

        q = Edges(Primal,(5,4))
        q.u .= reshape(1:15, 5, 3)
        q.v .= reshape(1:16, 4, 4)
        Qq = Edges(Dual,q)

        grid_interpolate!(Qq,q)
        @test Qq.u == [ 0.0  4.0  9.0  0.0
                        0.0  5.0  10.0 0.0
                        0.0  6.0  11.0 0.0
                        0.0  7.0  12.0 0.0 ]

        @test Qq.v == [ 0.0  0.0   0.0
                        3.5  7.5  11.5
                        4.5  8.5  12.5
                        5.5  9.5  13.5
                        0.0  0.0   0.0 ]

    end

    @testset "Shifting Dual Edges to Primal Edges" begin

        q = Edges(Dual,(5,4))
        q.u .= reshape(1:16, 4, 4)
        q.v .= reshape(1:15, 5, 3)
        v = Edges(Primal,q)
        grid_interpolate!(v,q)

        @test v.u == [ 0.0  0.0   0.0
                       3.5  7.5  11.5
                       4.5  8.5  12.5
                       5.5  9.5  13.5
                       0.0  0.0   0.0 ]

        @test v.v == [ 0.0  4.0   9.0  0.0
                        0.0  5.0  10.0  0.0
                        0.0  6.0  11.0  0.0
                        0.0  7.0  12.0  0.0 ]

    end

    @testset "Shifting Dual Nodes to Dual Edges" begin

        w = Nodes(Dual,(5,4))
        w .= reshape(1:20, 5, 4)

        Ww = Edges(Dual,w)
        grid_interpolate!(Ww,w)

        @test Ww.u == [ 0.0  6.5  11.5  0.0
                        0.0  7.5  12.5  0.0
                        0.0  8.5  13.5  0.0
                        0.0  9.5  14.5  0.0 ]
        @test Ww.v == [ 0.0   0.0   0.0
                        4.5   9.5  14.5
                        5.5  10.5  15.5
                        6.5  11.5  16.5
                        0.0   0.0   0.0 ]
    end

    @testset "Physical grid" begin

        g = PhysicalGrid((-1.0,3.0),(-2.0,3.0),0.02,nthreads_max=1)
        #=
        @test size(g) == (208,256)
        @test size(g,1) == 208
        @test size(g,2) == 256
        @test length(g) == 208*256
        @test origin(g) == (54,103)

        @test limits(g,1) == (-1.06,3.06)
        @test limits(g,2) == (-2.04,3.04)
        =#
        @test cellsize(g) == 0.02
        @test Threads.nthreads(g) == 1

        g = PhysicalGrid((-1.0,3.0),(-2.0,3.0),0.02,opt_type=:prime)
        xc, yc = coordinates(Nodes(Primal,size(g)),g)
        i0 = findall(x -> x ≈ 0.0,xc)
        j0 = findall(y -> y ≈ 0.0,yc)
        @test length(i0) == 1 && length(j0) == 1

    end
end
