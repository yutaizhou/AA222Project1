using LinearAlgebra

abstract type DescentDirectionMethod end

basis(i, n) = [k == i ? 1.0 : 0.0 for k in 1:n]

diff_complex(f, x; h = 1e-20) = imag(f(x .+ h * im)) / h

function backtracking_line_search(f, ∇f, x, d, α; p=0.5, β=1e-4)
    y, g = f(x), ∇f(x)
    while f(x + α * d) > y + β * α * (g ⋅ d)
        α *= p
    end
    return α
end



