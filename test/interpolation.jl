@testset "Interpolatable field" begin    

    i, j = 10, 20
    g = PhysicalGrid((-1.0,1.0),(-2.0,3.0),0.05,nthreads_max=1)
    w = Nodes(Dual,size(g))
    w[i,j] = 1.0
    x, y = coordinates(w,g)

    wi = interpolatable_field(x,y,w)
    @test wi(x[i],y[j]) ≈ 1.0


end