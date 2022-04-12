using LinearAlgebra
using Statistics
using Parameters

include("algo_util.jl")

abstract type FirstOrder <: DescentDirectionMethod end
function solve(M::FirstOrder, f, ∇f, x0, max_iters; num_eval_termination=true)
    init!(M, f, ∇f, x0)
    x_hist = [x0]
    x, i = x0, 0 
    while i < max_iters 
        x = step!(M, f, ∇f, x)
        push!(x_hist, x)
        i += 1
        if num_eval_termination && (COUNTERS[string(∇f)]*2 == max_iters) # 2 calls per iteration
            break
        end
    end
    return x, x_hist
end

# Vanilla Gradient Descent
@with_kw mutable struct GradientDescent <: FirstOrder 
    α = 1e-3
end
init!(M::GradientDescent, f, ∇, x) = M
function step!(M::GradientDescent, f, ∇f, x)
    @unpack α = M
    return x - α * ∇f(x)
end

# GD + Momentum
@with_kw mutable struct GDMomentum <: FirstOrder
    α = 1e-3
    β = 0.9
    v = nothing
end
function init!(M::GDMomentum, f, ∇f, x)
    M.v = zeros(length(x))
    return M
end
function step!(M::GDMomentum, f, ∇f, x)
    @unpack α, β, v = M
    v[:] = β * v - α * ∇f(x)
    return x + v
end


# GD + Nesterov Momentum
@with_kw mutable struct GDMomentumNesterov <: FirstOrder
    α = 1e-3
    β = 0.9
    v = nothing
end
function init!(M::GDMomentumNesterov, f, ∇f, x)
    M.v = zeros(length(x))
    return M
end
function step!(M::GDMomentumNesterov, f, ∇f, x)
    @unpack α, β, v = M
    v[:] = β * v - α * ∇f(x + β * v)
    return x + v
end

# Adam
@with_kw mutable struct Adam <: FirstOrder
    α = 1e-3
    β1 = 0.9
    β2 = 0.999
    ϵ = 1e-8
    k = 0
    v = nothing
    s = nothing
end
function init!(M::Adam, f, ∇f, x)
    M.v = zeros(length(x))
    M.s = zeros(length(x))
    return M
end
function step!(M::Adam, f, ∇f, x) 
    @unpack α, β1, β2, ϵ, k, v, s = M
    g = ∇f(x)
    v[:] = v * β1 + (1 - β1) * g
    s[:] = s * β2 + (1 - β2) * g .* g
    k += 1
    v̂ = v ./ (1 - β1 ^ k)
    ŝ = s ./ (1 - β2 ^ k)
    return x - α * v̂ ./ (sqrt.(ŝ) .+ ϵ)
end


# Gradient Descent + Backtracking Line Search
@with_kw mutable struct GDApproxLineSearch <: FirstOrder
    α = 3e-3
    approx_line_search = backtracking_line_search
end
init!(M::GDApproxLineSearch, f, ∇, x) = M
function step!(M::GDApproxLineSearch, f, ∇f, x)
    @unpack α, approx_line_search = M
    g = ∇f(x)
    α = approx_line_search(f, ∇f, x, -g, α)
    return x - α * g 
end
