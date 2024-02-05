using ForwardDiff
FD = ForwardDiff

@testset "Regularization accepting Dual type numbers" begin
    Δx = 0.2
    Lx = 1.0
    xlim = (-Lx,Lx)
    ylim = (-Lx,Lx)
    g = PhysicalGrid(xlim,ylim,Δx)
    
    n = 5
    x = 0.5 .+ 0.2*rand(n)
    y = 0.5 .+ 0.2*rand(n)

    X = zeros(length(x)+length(y))
    for i in eachindex(x)
        X[2i-1] = x[i]
        X[2i] = y[i]
    end

    cfg = FD.JacobianConfig(Regularize, X)
    Xdual = cfg.duals
    FD.seed!(Xdual, X)
    seeds = cfg.seeds
    FD.seed!(Xdual, X, 1, seeds)
    xdual = Xdual[1:2:end]
    ydual = Xdual[2:2:end]

    Hdual = Regularize(xdual,ydual,cellsize(g),I0=origin(g),issymmetric=true)
    H = Regularize(x,y,cellsize(g),I0=origin(g),issymmetric=true)
    @test FD.value.(H.x) == H.x
    @test FD.value.(H.y) == H.y
end 
