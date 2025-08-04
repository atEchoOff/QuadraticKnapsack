using QuadraticKnapsack
using LinearAlgebra

# Define a vector a
a = Float64[1., 2., 3., 4.]

# and a scalar b
b = 2.

# and a minimizer
minimizer! = QuadraticKnapsackMinimizer{Float64}(zeros(Float64, 4))

# and create a buffer to store the solution
x = zeros(Float64, 4)

# In this example, no upper bounds are reached.
minimizer!(x, a, b)
@show a
@show b
@show x
println(repeat("-", 50))

@assert abs(dot(a, x) - b) < 100 * eps()
@assert all(0. .<= x .<= 1.)

# In this example, b <= 0, so all components should be 0.
b = -1.
minimizer!(x, a, b)
@show a
@show b
@show x
println(repeat("-", 50))

@assert all(x .== 0.)

# In this example, some upper bounds are reached, but the problem is feasible.
a = Float64[1., 2., 3., 4.]
b = 9.

minimizer!(x, a, b)
@show a
@show b
@show x
println(repeat("-", 50))

@assert abs(dot(a, x) - b) < 100 * eps()
@assert all(0. .<= x .<= 1.)

# In this example, a has negative components. Corresponding elements of theta should be 0.
a = Float64[-1., 0., 1., 2.]
b = 2.

minimizer!(x, a, b)
@show a
@show b
@show x
println(repeat("-", 50))

@assert abs(dot(a, x) - b) < 100 * eps()
@assert all(0. .<= x .<= 1.)
@assert x[1] == 0.
@assert x[2] == 0.

# Now we apply nontrivial upper bounds
upper_bounds = Float64[.5, .05, .75, .25]
a = Float64[1., 2., 3., 4.]
b = 1.5

minimizer!(x, a, b; upper_bounds=upper_bounds)
@show a
@show upper_bounds
@show b
@show x
println(repeat("-", 50))

@assert abs(dot(a, x) - b) < 100 * eps()
@assert all(0. .<= x .<= upper_bounds)

# Finally, we apply a weighting
upper_bounds = Float64[.5, .05, .75, .25]
a = Float64[1., 2., 3., 4.]
b = 1.5
w = Float64[4., 3., 2., 1.]

minimizer!(x, a, b; upper_bounds=upper_bounds, w=w)
@show a
@show upper_bounds
@show w
@show b
@show x
println(repeat("-", 50))

@assert abs(dot(a, x) - b) < 100 * eps()
@assert all(0. .<= x .<= upper_bounds)

# Finally, lets make a huge problem
PROBLEM_SIZE = 1000000
println("Running and benchmarking with problem size $PROBLEM_SIZE")
minimizer! = QuadraticKnapsackMinimizer{Float64}(zeros(Float64, PROBLEM_SIZE))
x = zeros(Float64, PROBLEM_SIZE)
upper_bounds = rand(PROBLEM_SIZE)
a = rand(PROBLEM_SIZE)
w = rand(PROBLEM_SIZE)
b = dot(a, upper_bounds) / 2 # ensure feasibility

minimizer!(x, a, b; upper_bounds=upper_bounds, w=w)

@assert abs(dot(a, x) - b) < 1e-10 # a bit of leeway for a larger problem
@assert all(0. .<= x .<= upper_bounds)

using BenchmarkTools
@benchmark minimizer!(x, a, b; upper_bounds=upper_bounds, w=w)