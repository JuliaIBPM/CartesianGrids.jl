

const TOL = 100*eps(1.0)

Δx = 0.01
x = -15.0:Δx:15.0;

@testset "DDF moments" begin

for kernel in (:Roma,:Yang3,:Goza,:Witchhat,:M3,:M4prime)
  @eval ddf = DDF(ddftype=CartesianGrids.$kernel)
  @eval dddf1 = GradDDF(1,ddftype=CartesianGrids.$kernel)
  @eval dddf2 = GradDDF(2,ddftype=CartesianGrids.$kernel)

  @test isapprox(sum(ddf.(x)*Δx),1.0,atol=TOL)
  @test isapprox(sum(x.*ddf.(x)*Δx),0.0,atol=TOL)
  @test isapprox(sum(dddf1.(x)*Δx),0.0,atol=TOL)
  if kernel != :Witchhat
    @test isapprox(sum(x.*dddf1.(x)*Δx),-1.0,atol=TOL)
  end

  x0, y0 = rand(), rand()
  @test isapprox(ddf(x0,y0),ddf(x0)*ddf(y0),atol=TOL)
  @test isapprox(dddf1(x0,y0),dddf1(x0)*ddf(y0),atol=TOL)
  @test isapprox(dddf2(x0,y0),ddf(x0)*dddf1(y0),atol=TOL)

end


end
