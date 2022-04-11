using Parameters
include("helpers.jl")

abstract type ZerothOrder end
abstract type HookeJeevesMethod <: ZerothOrder end

function solve(M::HookeJeevesMethod, f, x0, max_iters)
    init!(M, f, x0)
    x, y, terminate = x0, f(x0), false

    while !terminate && (COUNTERS[string(f)] < max_iters - M.evals_per_iter)
        x, y, terminate = step!(M, f, x, y)
    end
    return x    
end

# Hooke Jeeves
@with_kw mutable struct HookeJeeves <: HookeJeevesMethod
    α = 1e-2
    ϵ = 1e-4
    γ = 0.5
    n = nothing
    evals_per_iter = nothing
end
function init!(M::HookeJeeves, f, x)
    M.n = length(x)
    M.evals_per_iter = 2 * M.n
end
function step!(M::HookeJeeves, f, x, y)
    @unpack α, ϵ, γ, n = M
    improved, terminate = false, false

    x_best, y_best = x, y 
    xs_new = [x + sgn * α * basis(i,n) for i in 1:n for sgn in (-1, +1)]
    ys_new = [f(x_new) for x_new in xs_new]
    y_best_iter, y_best_iter_idx = findmin(ys_new)

    if y_best_iter < y_best
        x_best = xs_new[y_best_iter_idx]
        y_best = y_best_iter
        improved = true
    end

    M.α *= (!improved ? γ : 1)
    terminate = (M.α <= ϵ ? true : false)
    return x_best, y_best, terminate
end