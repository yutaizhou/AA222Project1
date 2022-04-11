
using Dates
using DrWatson
using Plots
include("algo_zeroth_order.jl")
include("algo_first_order.jl")
include("helpers.jl")
include("simple.jl")

time_now = Dates.format(now(), "Y-mm-dd-HH:MM:SS")
outputdir(args...) = projectdir("output", args...)
outputdir_subpath = outputdir(time_now)
mkdir(outputdir_subpath)

function rosenbrock_plot(x1, x2)
    return (1.0 - x1)^2 + 100.0 * (x2 - x1^2)^2
end

function get_x_hist(prob_name, f, ∇f, x0, n)
    x_hist = []
    method = nothing
    if     prob_name == "simple1"
        method = GDMomentumNesterov(α=3e-4, β=0.8)
        _, x_hist = solve(method, f, ∇f, x0, n/2; num_eval_termination = false)

    elseif prob_name == "simple2"
        method = HookeJeeves(α=1.0)
        _, x_hist = solve(method, f, x0, n; num_eval_termination = false)

    elseif prob_name == "simple3"
        method = GDMomentumNesterov(α=1e-3, β=0.9)
        _, x_hist = solve(method, f, ∇f, x0, n; num_eval_termination = false)
    end
    return x_hist, string(typeof(method))
end


for (prob_name, (f, ∇f, x_init_fn, n)) in PROBS
    method_name = ""
    data = []
    for _ in 1:3
        x0 = x_init_fn()
        x_hist, method_name = get_x_hist(prob_name, f, ∇f, x0, n)
        f_hist = [f(x) for x in x_hist]
        push!(data, (iteration=collect(1: length(f_hist)), x=x_hist, y=f_hist))
    end

    # Convergence Plot
    title = "$(uppercasefirst(string(f))) w/ $(method_name)"
    plot(data[1].iteration, data[1].y, title=title, xlabel = "Iteration", ylabel = "f(x)", label="Init 1")
    for i in 2:3
        plot!(data[i].iteration, data[i].y, label="Init $i")
    end
    savefig(outputdir(outputdir_subpath, "converge_$(prob_name)_$(method_name).png"))

    # Contour Plot
    if prob_name == "simple1"
        xr = -2:0.1:2
        yr = -2:0.1:2
        contour(xr, yr, rosenbrock_plot,
            levels = [10,25,50,100,200,250,300], colorbar = false, c=cgrad(:viridis, rev = true), legend = false, title=title,
            xlims =(-2,2), ylims =(-2,2), xlabel = "x₁", ylabel = "x₂", aspectratio = :equal, clim =(2,500)
            )

        plot!([data[1].x[i][1] for i = 1:length(data[1].x)], [data[1].x[i][2] for i = 1:length(data[1].x)], color = :black, arrow=(:closed, 0.2))
        for i in 2:3
            plot!([data[i].x[j][1] for j = 1:length(data[i].x)], [data[i].x[j][2] for j = 1:length(data[i].x)], color = :black, arrow=(:closed, 0.2))
        end
        
        savefig(outputdir(outputdir_subpath, "contour_rosenbrock_$(method_name).png"))

    end
end


