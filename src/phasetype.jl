## Definition of a phase-type distribution, and related functions.
using Distributions
using Statistics: cor, cov, median, std, quantile

import Base: minimum, maximum
import Distributions: cdf, insupport, logpdf, pdf, rand
import Statistics: mean, var

macro check_args(D, cond)
    quote
        if !($(esc(cond)))
            throw(ArgumentError(string(
                $(string(D)), ": the condition ", $(string(cond)),
                " is not satisfied.")))
        end
    end
end

struct PhaseType <: ContinuousUnivariateDistribution
    # Defining properties
    π # entry probabilities
    T # transition probabilities

    # Derived properties
    t # exit probabilities
    p # number of phases

    function PhaseType(π, T, t, p)
        @check_args(PhaseType, all(π .>= zero(π[1])))
        @check_args(PhaseType, isapprox(sum(π), 1.0, atol=1e-4))
        @check_args(PhaseType, p == length(π) && p == length(t) &&
            all(p .== size(T)))
        @check_args(PhaseType, all(t .>= zero(t[1])))
        @check_args(PhaseType, all(isapprox(t, -T*ones(p))))
        new(π, T, t, p)
    end
end

PhaseType(π, T) = PhaseType(π, T, -T*ones(length(π)), length(π))
PhaseType(π, T, t) = PhaseType(π, T, t, length(π))

minimum(d::PhaseType) = 0
maximum(d::PhaseType) = Inf
insupport(d::PhaseType, x::Real) = x > 0 && x < Inf

pdf(d::PhaseType, x::Real) = transpose(d.π) * exp(d.T * x) * d.t
logpdf(d::PhaseType, x::Real) = log(pdf(d, x))
cdf(d::PhaseType, x::Real) = 1 - transpose(d.π) * exp(d.T * x) * ones(d.p)

mean(d::PhaseType) = -transpose(d.π) * inv(d.T) * ones(d.p)

# My new functions
PHmoment(d::PhaseType, n::Int64) = factorial(n) * transpose(d.π) * (-inv(d.T))^n * ones(d.p)
var(d::PhaseType) = PHmoment(d,2) - (mean(d))^2

# Random samples
function rand(rng::AbstractRNG, d::PhaseType)
    # auxilary variables
    a = d.π
    A = d.T
    N = length(a)

    cummInitial = cumsum(a, dims=1)
    sojourn = -1.0./diag(A)
    nextpr = diagm(sojourn) * A
    nextpr = nextpr - diagm(diag(nextpr))
    nextpr = hcat(nextpr, 1.0 .- sum(nextpr,dims=2))
    nextpr = cumsum(nextpr, dims=2)

    time = 0

    # draw initial distribution
    r =  rand()
    state = 1
    while cummInitial[state] <= r
        state += 1
    end

    # play state transitions
    while state <= N
        time += - log(rand()) * sojourn[state]
        r = rand()
        nstate = 1
        while nextpr[state, nstate] <= r
            nstate += 1
        end
        state = nstate
    end
    return time
end

iscoxian(d::PhaseType) = all(isapprox.(d.T[diagm(0 => ones(d.p),
    1 => ones(d.p-1)) .< 1], 0))

function reverse_coxian(π::AbstractArray{Float64}, T::AbstractArray{Float64},
        t::AbstractArray{Float64})
    # Get the time-reversed distribution
    Δ_v = Diagonal((-π' * inv(T))')
    πStar = (t' * Δ_v)'
    TStar = inv(Δ_v) * T' * Δ_v

    # Relabel the states from 1,..,p as p,...,1
    πCanon = reverse(πStar)
    λ = -reverse(diag(TStar))
    TCanon = diagm(0 => -λ, 1 => λ[1:end-1])
    (πCanon, TCanon, -TCanon*ones(length(π)))
end

function reverse_coxian(dist::PhaseType)
    PhaseType(reverse_coxian(dist.π, dist.T, dist.t) ...)
end

function sort_into_canonical_form(π::AbstractArray{Float64},
        T::AbstractArray{Float64}, t::AbstractArray{Float64})
    p = length(π)
    λ = -diag(T)
    π = π[:]

    while true
        swapped = false
        for i = 1:(p-1)
            if λ[i] > λ[i+1]
                w = λ[i+1]/λ[i]
                π[i] += (1-w)*π[i+1]
                π[i+1] *= w
                λ[i:i+1] = λ[i+1:-1:i]
                swapped = true
            end
        end

        if swapped == false
            break
        end
    end

    TSort = diagm(0 => -λ, 1 => λ[1:end-1])
    (π, TSort, -TSort*ones(p))
end

function sort_into_canonical_form(dist::PhaseType)
    PhaseType(sort_into_canonical_form(dist.π, dist.T, dist.t) ...)
end
