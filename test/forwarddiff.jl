using ForwardDiff
FD = ForwardDiff

Δx = 0.2
Lx = 1.0
xlim = (0.0,Lx)
ylim = (0.0,Lx)
g = PhysicalGrid(xlim,ylim,Δx)
    
n = 5
x = 0.5 .+ 0.2*rand(n)
y = 0.5 .+ 0.2*rand(n)

X = zeros(2n)
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

function get_ddf_type(::DDF{ddftype,OVERDX}) where {ddftype,OVERDX}
    return ddftype
end

@testset "Grid operators accepting FD.Dual numbers" begin
    Hdual = Regularize(xdual,ydual,cellsize(g),I0=origin(g),issymmetric=true)
    H = Regularize(x,y,cellsize(g),I0=origin(g),issymmetric=true)

    @testset "Derivative of DDF for FD.Dual numbers" begin
        Hdual_ddf = Hdual.ddf(xdual,ydual)
        H_ddf = H.ddf(x,y)
        ddftype = get_ddf_type(H.ddf)
        ddfx = GradDDF(1,ddftype=ddftype,dx=1.0)
        ddfy = GradDDF(2,ddftype=ddftype,dx=1.0)
        ddfxval = ddfx(x,y)
        ddfyval = ddfy(x,y)

        for i=1:n
            @test FD.partials.(Hdual_ddf,2i-1)[i,:] == ddfxval[i,:]
        end
        for i=1:n
            @test FD.partials.(Hdual_ddf,2i)[:,i] == ddfyval[:,i]
        end
    end 

    @testset "Regularize point sources of type FD.Dual numbers to grid" begin
        @test FD.value.(Hdual.x) == H.x
        @test FD.value.(Hdual.y) == H.y

        wdual = Nodes(Dual,size(g),dtype=Real)
        Xdvec = VectorData(xdual,ydual)
        sdual = ScalarData(Xdvec)
        sdual.data .= xdual
        Hdual(wdual,sdual)

        w = Nodes(Dual,size(g))
        Xvec = VectorData(x,y)
        s = ScalarData(Xvec)
        s.data .= x
        H(w,s)

        @test FD.value.(wdual.data) == w.data

        wmat_dual = Nodes(Dual,size(g),dtype=Real)
        Hmat_dual, Emat_dual = RegularizationMatrix(Hdual,sdual,wmat_dual)
        mul!(wmat_dual,Hmat_dual,sdual)
        @test wdual == wmat_dual

        wmat = Nodes(Dual,size(g))
        Hmat, Emat = RegularizationMatrix(H,s,wmat)
        mul!(wmat,Hmat,s)
        @test wmat == w

        
    end

    @testset "Matrix representation" begin
        # Regularize point sources of type FD.Dual onto the grid
        wdual = Nodes(Dual,size(g),dtype=Real)
        Xvec = VectorData(xdual,ydual)
        sdual = ScalarData(Xvec)
        Hmat_dual = RegularizationMatrix(Hdual,sdual,wdual)

        sdual.data .= xdual
        wdual2 = Nodes(Dual,size(g),dtype=Real)
        mul!(wdual,Hmat_dual,sdual)
        Hdual(wdual2,sdual)
        @test FD.value.(wdual.data) ≈ FD.value.(wdual2.data)

        parmat = FD.partials.(wdual.data)
        idx = findfirst(x -> x != 0, wdual.data)
        npar = length(parmat[idx])
        for k=1:npar
            @test FD.partials.(wdual.data,k) ≈ FD.partials.(wdual2.data,k)
        end

        # Interpolate grid data of type FD.Dual onto the point sources
        Emat_dual = InterpolationMatrix(Hdual,wdual,sdual)
        sdual2 = ScalarData(sdual)
        mul!(sdual,Emat_dual,wdual)
        Hdual(sdual2,wdual)
        @test FD.value.(sdual.data) ≈ FD.value.(sdual2.data)

        parmat = FD.partials.(sdual.data)
        idx = findfirst(x -> x != 0, sdual.data)
        npar = length(parmat[idx])
        for k=1:npar
            @test FD.partials.(sdual.data,k) ≈ FD.partials.(sdual2.data,k)
        end

    end

    @testset "Inverse Laplacian for FD.Dual numbers" begin
        wdual = Nodes(Dual,size(g),dtype=Real)
        Xdvec = VectorData(xdual,ydual)
        sdual = ScalarData(Xdvec)
        sdual.data .= xdual
        Hdual(wdual,sdual)

        w = Nodes(Dual,size(g))
        Xvec = VectorData(x,y)
        s = ScalarData(Xvec)
        s.data .= x
        H(w,s)

        L = plan_laplacian(size(w),with_inverse=true)
        linvd = L\wdual
        linv = L\w
        parmat = FD.partials.(linvd.data)
        idx = findfirst(x -> x != 0, linvd.data)
        npar = length(parmat[idx])
        wdpar = deepcopy(wdual)
        lap_linvd = L*linvd

        @test FD.value.(linvd.data) == linv.data
        @test FD.value.(lap_linvd.data) ≈ FD.value.(wdual.data)
        for k=1:npar
            linvdpar = FD.partials.(linvd.data,k)
            wdpar.data .= FD.partials.(wdual.data,k)
            linvpar = L\wdpar
            @test linvdpar == linvpar.data
            @test FD.partials.(lap_linvd.data,k) ≈ FD.partials.(wdual.data,k)
        end
    end

end



