Δx = 0.01
xlim = (-2,2)
ylim = (-2,2)
gr = PhysicalGrid(xlim,ylim,Δx)
w = Nodes(Dual,size(gr));
q = Edges(Primal,w);
p = Nodes(Primal,w);

L = plan_laplacian(w,with_inverse=true);

n = 300
θ = range(0,2π,length=n+1)
rad = 1.0
x, y = rad*cos.(θ[1:n]), rad*sin.(θ[1:n])
X = VectorData(x,y);
f = ScalarData(X);
τ = VectorData(X);

reg = Regularize(X,cellsize(gr),I0=origin(gr))
