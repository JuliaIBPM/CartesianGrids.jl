# This is copied from RigidBodyTools. Need to make this a separate package.

export EldredgeRamp, ColoniusRamp, Sinusoid

using ForwardDiff

import Base: +, *, -, >>, <<, show

"""
An abstract type for real-valued functions of one variable.
"""
abstract type Profile end

#=
Profiles
=#


"""
    ConstantProfile(c::Number)

Create a profile consisting of a constant `c`.

# Example

```jldoctest
julia> p = RigidBodyMotions.ConstantProfile(1.0)
Constant (2.3)
```
"""
struct ConstantProfile <: Profile
    c::Number
end

function show(io::IO, p::ConstantProfile)
    print(io, "Constant ($(p.c))")
end

(p::ConstantProfile)(t) = p.c

struct DerivativeProfile{P} <: Profile
    p::P
end

function show(io::IO, ṗ::DerivativeProfile)
    print(io, "d/dt ($(ṗ.p))")
end

(ṗ::DerivativeProfile)(t) = ForwardDiff.derivative(ṗ.p, t)

"""
    d_dt(p::Profile)

Take the time derivative of `p` and return it as a new profile.

# Example

```jldoctest
julia> s = Plates.RigidBodyMotions.Sinusoid(π)
Sinusoid (ω = 3.14)

julia> s.([0.0, 0.5, 0.75])
3-element Array{Float64,1}:
 0.0
 1.0
 0.707107

julia> c = Plates.RigidBodyMotions.d_dt(s)
d/dt (Sinusoid (ω = 3.14))

julia> c.([0.0, 0.5, 0.75])
3-element Array{Float64,1}:
  3.14159
  1.92367e-16
 -2.22144
```
"""
d_dt(p::Profile) = DerivativeProfile(p)

struct ScaledProfile{N <: Real, P <: Profile} <: Profile
    s::N
    p::P
end
function show(io::IO, p::ScaledProfile)
    print(io, "$(p.s) × ($(p.p))")
end

"""
    s::Number * p::Profile

Returns a scaled profile with `(s*p)(t) = s*p(t)`

# Example

```jldoctest
julia> s = Sinusoid(π)
Sinusoid (ω = 3.14)

julia> 2s
2 × (Sinusoid (ω = 3.14))

julia> (2s).([0.0, 0.5, 0.75])
3-element Array{Float64,1}:
 0.0
 2.0
 1.41421
```
"""
s::Number * p::Profile = ScaledProfile(s, p)

"""
    -(p₁::Profile, p₂::Profile)

```jldoctest
julia> s = Plates.RigidBodyMotions.Sinusoid(π)
Sinusoid (ω = 3.14)

julia> 2s
2 × (Sinusoid (ω = 3.14))

julia> (2s).([0.0, 0.5, 0.75])
3-element Array{Float64,1}:
 0.0
 2.0
 1.41421

julia> s = Plates.RigidBodyMotions.Sinusoid(π);

julia> s.([0.0, 0.5, 0.75])
3-element Array{Float64,1}:
 0.0
 1.0
 0.707107

julia> (-s).([0.0, 0.5, 0.75])
3-element Array{Float64,1}:
 -0.0
 -1.0
 -0.707107

julia> (s - s).([0.0, 0.5, 0.75])
3-element Array{Float64,1}:
 0.0
 0.0
 0.0
```
"""
-(p::Profile) = ScaledProfile(-1, p)
(p::ScaledProfile)(t) = p.s*p.p(t)

struct ShiftedProfile{N <: Real, P <: Profile} <: Profile
    Δt::N
    p::P
end
function show(io::IO, p::ShiftedProfile)
    print(io, "$(p.p) >> $(p.Δt)")
end

(p::ShiftedProfile)(t) = p.p(t - p.Δt)

"""
    p::Profile >> Δt::Number

Shift the profile in time so that `(p >> Δt)(t) = p(t - Δt)`

# Example

```jldoctest
julia> s = Plates.RigidBodyMotions.Sinusoid(π);

julia> s >> 0.5
Sinusoid (ω = 3.14) >> 0.5

julia> (s >> 0.5).([0.0, 0.5, 0.75])
3-element Array{Float64,1}:
 -1.0
  0.0
  0.707107

julia> (s << 0.5).([0.0, 0.5, 0.75])
3-element Array{Float64,1}:
  1.0
  1.22465e-16
 -0.707107
```
"""
p::Profile >> Δt::Number = ShiftedProfile(Δt, p)
p::Profile << Δt::Number = ShiftedProfile(-Δt, p)

struct AddedProfiles{T <: Tuple} <: Profile
    ps::T
end
function show(io::IO, Σp::AddedProfiles)
    println(io, "AddedProfiles:")
    for p in Σp.ps
        println(io, "  $p")
    end
end

"""
    p₁::Profile + p₂::Profile

Add the profiles so that `(p₁ + p₂)(t) = p₁(t) + p₂(t)`.

# Examples

```jldoctest
julia> ramp₁ = Plates.RigidBodyMotions.EldredgeRamp(5)
logcosh ramp (aₛ = 5.0)

julia> ramp₂ = Plates.RigidBodyMotions.ColoniusRamp(5)
power series ramp (n = 5.0)

julia> ramp₁ + ramp₂
AddedProfiles:
  logcosh ramp (aₛ = 5.0)
  power series ramp (n = 5.0)


julia> ramp₁ + (ramp₂ + ramp₁) == ramp₁ + ramp₂ + ramp₁
true

```
"""
+(p::Profile, Σp::AddedProfiles) = AddedProfiles((p, Σp.ps...))
+(Σp::AddedProfiles, p::Profile) = AddedProfiles((Σp.ps..., p))
function +(Σp₁::AddedProfiles, Σp₂::AddedProfiles)
    AddedProfiles((Σp₁..., Σp₂...))
end

-(p₁::Profile, p₂::Profile) = p₁ + (-p₂)

+(p::Profile...) = AddedProfiles(p)

function (Σp::AddedProfiles)(t)
    f = 0.0
    for p in Σp.ps
        f += p(t)
    end
    f
end


struct MultipliedProfiles{T <: Tuple} <: Profile
    ps::T
end
function show(io::IO, Πp::MultipliedProfiles)
    println(io, "MultipliedProfiles:")
    for p in Πp.ps
        println(io, "  $p")
    end
end
*(p::Profile, Πp::MultipliedProfiles) = MultipliedProfiles((p, Πp.ps...))
*(Πp::MultipliedProfiles, p::Profile) = MultipliedProfiles((Πp.ps..., p))
function *(Πp₁::MultipliedProfiles, Πp₂::MultipliedProfiles)
    MultipliedProfiles((Πp₁..., Πp₂...))
end

function (Πp::MultipliedProfiles)(t)
    f = 1.0
    for p in Πp.ps
        f *= p(t)
    end
    f
end


## Functional forms of profiles

"""
    Gaussian(σ,x0,A)

Construct a 1-d Gaussian function centered at `x0` with standard deviation `σ`
and amplitude `A`. The resulting function can be evaluated at any real-valued
number.

# Example

```jldoctest
julia> g = Gaussian(0.2,1)
Gaussian(0.2, 1, 2.8209479177387813)

julia> g(0.2)
1.0377687435514866
```
"""
struct Gaussian <: Profile
  σ :: Real
  A :: Real
  fact :: Float64
  Gaussian(σ,A) = new(σ,A,A/sqrt(π*σ^2))
end
show(io::IO, s::Gaussian) = print(io,
      "Gaussian (σ = $(round(s.σ, digits=2)), A = $(round(s.A, digits=2)))")

@inline _gaussian(r;tol=6.0) = abs(r) < tol ? exp(-r^2) : 0.0

radius(g::Gaussian) = g.σ
strength(g::Gaussian) = g.A

(g::Gaussian)(x) = g.fact*_gaussian(x/radius(g))

DGaussian(σ,A) = d_dt(Gaussian(σ,A))



struct Sinusoid <: Profile
    ω::Float64
end
(s::Sinusoid)(t) = sin(s.ω*t)
show(io::IO, s::Sinusoid) = print(io, "Sinusoid (ω = $(round(s.ω, digits=2)))")

struct EldredgeRamp <: Profile
    aₛ::Float64
end
(r::EldredgeRamp)(t) = 0.5(log(2cosh(r.aₛ*t)) + r.aₛ*t)/r.aₛ
show(io::IO, r::EldredgeRamp) = print(io, "logcosh ramp (aₛ = $(round(r.aₛ, digits=2)))")

struct ColoniusRamp <: Profile
    n::Int
end
function (r::ColoniusRamp)(t)
    Δt = t + 0.5
    if Δt ≤ 0
        0.0
    elseif Δt ≥ 1
        Δt - 0.5
    else
        f = 0.0
        for j = 0:r.n
            f += binomial(r.n + j, j)*(r.n - j + 1)*(1 - Δt)^j
        end
        f*Δt^(r.n + 2)/(2r.n + 2)
    end
end
show(io::IO, r::ColoniusRamp) = print(io, "power series ramp (n = $(round(r.n, digits=2)))")
