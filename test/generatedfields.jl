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
