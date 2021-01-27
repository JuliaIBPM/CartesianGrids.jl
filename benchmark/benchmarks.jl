using BenchmarkTools, CartesianGrids


BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true

SUITE = BenchmarkGroup()
SUITE["field ops"] = BenchmarkGroup()
SUITE["ddf"] = BenchmarkGroup()
SUITE["regularization"] = BenchmarkGroup()

include("setupproblem.jl")

w .= randn(size(w));
f .= randn(size(f));

SUITE["field ops"]["inverse Laplacian"] = @benchmarkable L\w;
SUITE["field ops"]["curl of dual nodes"] = @benchmarkable curl!(q,w);
SUITE["field ops"]["divergence of primal edges"] = @benchmarkable divergence!(p,q);

for kernel in (:Roma,:Yang3,:M4prime)
  name = string(kernel)
  @eval SUITE["ddf"][$name] = @benchmarkable evaluate_ddf(DDF(ddftype=CartesianGrids.$kernel))
end

SUITE["regularization"]["create node regularization matrix"] = @benchmarkable RegularizationMatrix(reg,f,w);
SUITE["regularization"]["create node interpolation matrix"] = @benchmarkable InterpolationMatrix(reg,w,f);

Rn = RegularizationMatrix(reg,f,w);
En = InterpolationMatrix(reg,w,f);
SUITE["regularization"]["evaluate node regularization"] = @benchmarkable Rn*f;
SUITE["regularization"]["evaluate node interpolation"] = @benchmarkable En*w;


SUITE["regularization"]["create edge regularization matrix"] = @benchmarkable RegularizationMatrix(reg,τ,q);
SUITE["regularization"]["create edge interpolation matrix"] = @benchmarkable InterpolationMatrix(reg,q,τ);

Re = RegularizationMatrix(reg,τ,q);
Ee = InterpolationMatrix(reg,q,τ);
SUITE["regularization"]["evaluate edge regularization"] = @benchmarkable Re*τ;
SUITE["regularization"]["evaluate edge interpolation"] = @benchmarkable Ee*q;


run(SUITE)
