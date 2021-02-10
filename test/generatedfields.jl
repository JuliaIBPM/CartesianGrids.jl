using Random

@testset "SpatialFields" begin

σ = rand()
g = Gaussian(σ,0,1)
@test g(σ) ≈ exp(-1)/sqrt(π*σ^2)

@test radius(g) == σ
@test center(g) == 0
@test strength(g) == 1

g = EmptySpatialField()
@test g(rand(2)...) == 0.0

g = SpatialGaussian(σ,σ,0.0,0.5,1)
@test g(0,0.5+σ) ≈ g(σ,0.5) ≈ g(-σ,0.5) ≈ g(0,0.5-σ) ≈ exp(-1)/(π*σ^2)

u = 1
v = 0
gc = SpatialGaussian(σ,σ,0.0,0.5,1,u,v)
t = 1
@test gc(u*t,0.5+σ+v*t,t) ≈ gc(σ+u*t,0.5+v*t,t) ≈ gc(-σ+u*t,0.5+v*t,t) ≈ gc(0+u*t,0.5-σ+v*t,t) ≈ exp(-1)/(π*σ^2)



g = EmptySpatialField()
for x in [-0.5,0,0.5], y in [-0.5,0,0.5]
  g += SpatialGaussian(0.2,0.5,x,y,1)
end
@test g(0.5,0) ≈ 5.5357580598289235

g2 = -g
@test g2(0.5,0) ≈ -5.5357580598289235


gr = PhysicalGrid((-3.0,3.0),(-2.0,2.0),0.03)
w = Nodes(Dual,size(gr))
xg, yg = coordinates(w,gr)

gfield = GeneratedField(w,g,gr)

@test gfield()[104,70] == g(xg[104],yg[70])
@test datatype(gfield) == typeof(w)

wfield = PulseField(gfield,0.5,0.1)

@test maximum(wfield(0.5)) ≈ maximum(gfield()) ≈ 5.532960088678624
@test maximum(wfield(2)) ≈ 0.0
@test datatype(wfield) == datatype(gfield) <: ScalarGridData


q = Edges(Primal,size(gr))
gauss = SpatialGaussian(0.5,0,0,1)
gaussfield = GeneratedField(q,gauss,EmptySpatialField(),gr)
q .= gaussfield()

ffield = PulseField(gaussfield,0.5,0.1)

@test maximum(ffield(0.5).u) ≈ maximum(q.u) ≈ 1.272094144652253
@test maximum(ffield(2).u) ≈ 0.0
@test maximum(ffield(0.5).v) ≈ maximum(q.v) ≈ 0.0
@test maximum(ffield(2).v) ≈ maximum(q.v) ≈ 0.0

@test datatype(ffield) == datatype(gaussfield) <: VectorGridData

gq = EdgeGradient(Dual,size(gr))
tfield = GeneratedField(gq,[g,gauss,EmptySpatialField(),EmptySpatialField()],gr)
gq .= tfield()
@test maximum(gq.dudx) ≈ 5.532960088678624
@test maximum(gq.dudy) ≈ 1.2732395447351625
@test maximum(gq.dvdx) == maximum(gq.dvdy) == 0.0

ffield = PulseField(tfield,0.5,0.1)
@test datatype(ffield) <: TensorGridData



end

@testset "Derivatives of gaussians" begin

  σ = 0.2
  x0 = 0
  A = 1
  dg = DGaussian(σ,x0,A)
  x = 0.1
  @test dg(x) == -2*A*x/sqrt(π)/σ^3*exp(-x^2/σ^2)

  σx = 0.5
  σy = 0.5
  x0 = 0
  y0 = 0
  A = 1
  dgaussx = SpatialGaussian(σx,σy,x0,y0,A,deriv=1)
  dgaussy = SpatialGaussian(σx,σy,x0,y0,A,deriv=2)
  x = 0.1
  y = 0.2
  @test dgaussx(0.1,0.2) == dgaussy(0.2,0.1) ≈ -2*A*x/π/σx^3/σy*exp(-x^2/σx^2)*exp(-y^2/σy^2)

end
