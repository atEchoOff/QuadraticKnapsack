module QuadraticKnapsack

export QuadraticKnapsackMinimizer
export QuadraticKnapsackStats

using LinearAlgebra: dot, norm

"""
    QuadraticKnapsackMinimizer{Type}(buffer[, tol = 100 * eps()])

A continuous quadratic knapsack problem solver of eltype `Type`. Given internal `buffer` whose size is the knapsack problem dimension. Given optional tolerance `tol` to stop iteration (default `100 * eps()`).
"""
struct QuadraticKnapsackMinimizer{Ttol}
    tol::Ttol
    a_over_w::Vector{Float64}
    direction::Int64

    function QuadraticKnapsackMinimizer{Ttol}(buffer::Vector{Ttol}; tol::Ttol = 100 * eps()) where {Ttol}
        return new{Ttol}(tol, buffer, 1)
    end
end

mutable struct QuadraticKnapsackStats
    infeasible::Bool
    itercount::Int64

    function QuadraticKnapsackStats()
        return new(false, 0)
    end
end

"""
    minimizer(x, a, b[, upper_bounds=ones(Ttol, length(buffer)), w=ones(Ttol, length(buffer)), tol=100 * eps(), maxit=200, stats=nothing])

Solve the continuous quadratic knapsack problem: min x^T diag(w) x subject to a^T x >= b and 0 <= x <= upper_bounds.

Output returned in `x`. Given vector `a` and scalar `b`. Elementwise upper bounds on `x` are `upper_bounds`, defaulted to one. Objective function weights given by `w`, defaulted to one. Maximum number of iterations set to `maxit`, defaulted to 200. 

If given stats::QuadraticKnapsackStats, it will contain infeasible::Bool for whether or not the problem is infeasible, and itercount::Int64, for the total number of iterations taken.

In the event of infeasibility, the "best possible" (though infeasible) solution is returned.

Note that 200 is a very loose upper bound. 
"""
function (s::QuadraticKnapsackMinimizer{Ttol})(x::Vector{Ttol}, a::Vector{Ttol}, b::Ttol; upper_bounds=ones(Ttol, length(s.a_over_w))::Vector{Ttol}, w=ones(Ttol, length(s.a_over_w))::Vector{Ttol}, maxit=200::Int64, stats=nothing::Union{Nothing, QuadraticKnapsackStats}) where {Ttol}
    # Use local knapsack memory
    a_over_w = @view s.a_over_w[1:length(a)]

    if b <= 0.0
        # x = 0 is the optimal solution for the minimization
        fill!(x, zero(eltype(x)))

        if !isnothing(stats)
            stats.infeasible = false
            stats.itercount = 0
        end
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
        # Infeasibility detected, return "best possible" (infeasible) solution.
        x .= upper_bounds .* (a .> 0)

        if !isnothing(stats)
            stats.infeasible = true
            stats.itercount = 0
        end
        return x
    end

    # Start the Newton iteration
    lambdak = zero(eltype(a))
    itercount = 1
    
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

    if !isnothing(stats)
        stats.infeasible = false
        stats.itercount = itercount
    end

    return x
end

end # module