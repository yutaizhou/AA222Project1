using Parameters

basis(i, n) = [k == i ? 1.0 : 0.0 for k in 1:n]

function hooke_jeeves(f, x, α; ϵ=1e-4, decay=0.5)
    y, n = f(x), length(x)
    while α > ϵ
        improved = false
        x_best, y_best = x, y
        xs_new = [x + sgn * α * basis(i,n) for i in 1:n for sgn in (-1, +1)]
        ys_new = [f(x_new) for x_new in xs_new]
        y_best_iter, y_best_iter_idx = findmin(ys_new)
        
        if y_best_iter < y_best
            x_best = xs_new[y_best_iter_idx]
            y_best = y_best_iter
            improved = true
        end

        x, y = x_best, y_best
        if !improved
            α *= decay
        end

    end
    return x

end