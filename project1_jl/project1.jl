#=
        project1.jl -- This is where the magic happens!

    All of your code must either live in this file, or be `include`d here.
=#

#=
    If you want to use packages, please do so up here.
    Note that you may use any packages in the julia standard library
    (i.e. ones that ship with the julia language) as well as Statistics
    (since we use it in the backend already anyway)
=#

# Example:
# using LinearAlgebra

#=
    If you're going to include files, please do so up here. Note that they
    must be saved in project1_jl and you must use the relative path
    (not the absolute path) of the file in the include statement.

    [Good]  include("somefile.jl")
    [Bad]   include("/pathto/project1_jl/somefile.jl")
=#

# Example
# include("myfile.jl")
include("algo_zeroth_order.jl")
include("algo_first_order.jl")
include("algo_util.jl")


"""
    optimize(f, g, x0, n, prob)

Arguments:
    - `f`: Function to be optimized
    - `∇f`: Gradient function for `f`
    - `x0`: (Vector) Initial position to start from
    - `n`: (Int) Number of evaluations allowed. Remember `g` costs twice of `f`
    - `prob`: (String) Name of the problem. So you can use a different strategy for each problem. E.g. "simple1", "secret2", etc.

Returns:
    - The location of the minimum
"""
function optimize(f, ∇f, x0, n, prob)
    # method = GradientDescent(3e-3)
    # method = GDMomentum(α=3e-4, β=0.9)
    method = GDMomentumNesterov(α=3e-4, β=0.8)
    # method = Adam(α=3e-4, v_decay=0.6)
    # method = GDApproxLineSearch()
    x = solve(method, f, ∇f, x0, 9)
    
    # x = hooke_jeeves(f, x0, 0.5)

    # println("Pre: $(round.(x0; digits=3)) -> $(round(f(x0); digits=3))")
    # println("Pos: $(round.(x; digits=3)) -> $(round(f(x); digits=3)) \n")
    return x
end