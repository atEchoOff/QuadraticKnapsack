module QuadraticKnapsack

export QuadraticKnapsackMinimizer

using LinearAlgebra # just for dot

"""
    QuadraticKnapsackMinimizer{Type}(buffer[, tol = 100 * eps()])

A continuous quadratic knapsack problem solver of eltype `Type`. Given internal `buffer` whose size is larger than that of the knapsack problem dimension. Given optional tolerance `tol` to stop iteration (default `100 * eps()`).
"""
struct QuadraticKnapsackMinimizer{Ttol}
    tol::Ttol
    a_over_w::Vector{Float64}
    direction::Int64

    function QuadraticKnapsackMinimizer{Ttol}(buffer::Vector{Ttol}; tol::Ttol = 100 * eps()) where {Ttol}
        return new{Ttol}(tol, buffer, 1)
    end
end

# Override indexing floats, makes code for QKL simpler
import Base.getindex
function Base.getindex(x::Float64, i::Int64)
    return x
end

"""
    minimizer(x, a, b[, upper_bounds=1., w=1., tol=100 * eps(), maxit=200])

Solve the continuous quadratic knapsack problem: min x^T diag(w) x subject to a^T x >= b and 0 <= x <= upper_bounds.

Output returned in `x`. Given vector `a` and scalar `b`. Elementwise upper bounds on `x` are `upper_bounds`, defaulted to one. Objective function weights given by `w`, defaulted to one. Maximum number of iterations set to `maxit`, defaulted to 200. 

Note that 200 is a very loose upper bound. 
"""
function (s::QuadraticKnapsackMinimizer{Ttol})(x::Vector{Ttol}, a::Vector{Ttol}, b::Ttol; upper_bounds=one(Ttol), w=one(Ttol), maxit=200) where {Ttol}
    # Use local knapsack memory
    a_over_w = @view s.a_over_w[1:length(a)]

    if b <= 0.0
        # x = 0 is the optimal solution for the minimization
        fill!(x, zero(eltype(x)))
        return x
    end

    # Check for potential infeas while calculating a / w
    worst_possible = zero(eltype(a))

    for i in eachindex(a)
        if a[i] > 0
            a_over_w[i] = a[i] / w[i]
            worst_possible += a[i] * upper_bounds[i]
        else
            a_over_w[i] = zero(eltype(a))
        end
    end

    if worst_possible < b
        # FIXME keep infeas check?
        x .= upper_bounds .* (a .> 0)
        return x
    end

    # Start the Newton iteration
    lambdak = zero(eltype(a))
    itercount = 0 # for sanity check
    
    for _ in range(0, maxit)
        # Clip current solution within feasible domain
        x .= lambdak .* a_over_w
        x .= clamp.(x, 0., upper_bounds)

        f_val = dot(a, x) - b

        if abs(f_val) / max(1, norm(a)) < s.tol
            break
        end

        # Faster derivative computation
        deriv = zero(eltype(a))
        for i in eachindex(a)
            if x[i] < upper_bounds[i]
                deriv += a[i] * a_over_w[i]
            end
        end

        # Compute the next root
        lambdak -= f_val / deriv

        itercount += 1 # for sanity check
    end

    # FIXME report itercount?
    if itercount > 6
        println("The itercount was $itercount")
    end

    return x
end

end # module