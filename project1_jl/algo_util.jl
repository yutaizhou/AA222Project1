using LinearAlgebra

abstract type DescentDirectionMethod end

diff_complex(f, x; h = 1e-20) = imag(f(x .+ h * im)) / h
