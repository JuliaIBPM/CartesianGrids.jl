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

end
