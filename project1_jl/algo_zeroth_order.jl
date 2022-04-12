include("helpers.jl")
include("algo_util.jl")

abstract type ZerothOrder end

function solve(M::ZerothOrder, f, x0, max_iters; num_eval_termination=true)
    init!(M, f, x0)
    x_hist = [x0]
    x, y, terminate = x0, f(x0), false
    while !terminate 
        x, y, terminate = step!(M, f, x, y)
        push!(x_hist, x)
        if num_eval_termination && (COUNTERS[string(f)] >= max_iters - M.evals_per_iter)
            break
        end
    end 
    return x, x_hist
end

# Coordinate Descent
Base.@kwdef mutable struct CoordinateDescentAcceleration <: ZerothOrder
    ϵ = 1e-4
    approx_line_search = backtracking_line_search
    n = nothing
end
function init!(M::CoordinateDescentAcceleration, f, x)
    M.n = length(x)
end
function step!(M::CoordinateDescentAcceleration, f, ∇f, x)
    ϵ, n, approx_line_search = M.ϵ, M.n, M.approx_line_search
    terminate = false

    x_old = copy(x)
    for i in 1:n
        d = basis(i,n)
        α = backtracking_line_search(f, ∇f, x, d, 1e-2)
        x += α * d
    end
    α = backtracking_line_search(f, ∇f, x, x - x_old, 1e-2)
    x += α * (x - x_old)

    if abs(norm(x - x_old)) <= ϵ
        terminate = true
    end
    return x, terminate
end

function solve(M::CoordinateDescentAcceleration, f, ∇f, x0, max_iters; num_eval_termination=true)
    init!(M, f, x0)
    x_hist = [x0]
    x, terminate = x0, false
    while !terminate 
        x, terminate = step!(M, f, ∇f, x)
        push!(x_hist, x)
        if num_eval_termination && (COUNTERS[string(f)] >= max_iters - M.evals_per_iter)
            break
        end
    end
    return x, x_hist
end

# Hooke Jeeves
Base.@kwdef mutable struct HookeJeeves <: ZerothOrder
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
    α, ϵ, γ, n = M.α, M.ϵ, M.γ, M.n
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

# Hooke Jeeves Dynamic w/ Eager execution
Base.@kwdef mutable struct HookeJeevesDynamic <: ZerothOrder
    α = 1e-2
    ϵ = 1e-4
    γ = 0.5
    n = nothing
    evals_per_iter = nothing
    D = nothing # directions to search in, need to store it for permutating order
end
function init!(M::HookeJeevesDynamic, f, x)
    M.n = length(x)
    M.evals_per_iter = 2 * M.n
    M.D = [sgn * basis(i, M.n) for i in 1:M.n for sgn in (-1, +1)]
end
function step!(M::HookeJeevesDynamic, f, x, y, idx_best_prev)
    α, ϵ, γ = M.α, M.ϵ, M.γ
    improved, terminate, idx_best = false, false, 1

    x_best, y_best = x, y 
    d_best_prev = M.D[idx_best_prev]
    M.D = pushfirst!(deleteat!(M.D, idx_best_prev), d_best_prev)
    xs_new = [x + α * d for d in M.D]

    for (idx, x_new) in enumerate(xs_new)
        y_new = f(x_new)
        if y_new < y_best
            x_best, y_best = x_new, y_new
            improved = true
            idx_best = idx
            break
        end
    end

    M.α *= (!improved ? γ : 1)
    terminate = (M.α <= ϵ ? true : false)
    return x_best, y_best, terminate, idx_best
end

function solve(M::HookeJeevesDynamic, f, x0, max_iters, num_eval_termination=true)
    init!(M, f, x0)
    x_hist = [x0]
    x, y, terminate, idx = x0, f(x0), false, 1

    while !terminate && (COUNTERS[string(f)] < max_iters - M.evals_per_iter)
        x, y, terminate, idx = step!(M, f, x, y, idx)
        push!(x_hist, x)

        if num_eval_termination && (COUNTERS[string(f)] >= max_iters - M.evals_per_iter)
            break
        end
    end
    return x, x_hist
end

