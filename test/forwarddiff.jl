using ForwardDiff
FD = ForwardDiff

@testset "Regularization accepting Dual type numbers" begin
    Δx = 0.2
    Lx = 1.0
    xlim = (-Lx,Lx)
    ylim = (-Lx,Lx)
    g = PhysicalGrid(xlim,ylim,Δx)
    w = Nodes(Primal,size(g))
    xg, yg = coordinates(w,g)
    x = collect(xg)
    y = collect(yg)

    X = zeros(length(x)+length(y))
    for i in eachindex(x)
        X[2i-1] = x[i]
        X[2i] = y[i]
    end
    cfg = FD.JacobianConfig(f, X)
    Xdual = cfg.duals
    xdual = Xdual[1:2:end]
    ydual = Xdual[2:2:end]

    H = Regularize(xdual,ydual,cellsize(g),I0=origin(g),issymmetric=true)
end 
