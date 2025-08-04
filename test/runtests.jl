using QuadraticKnapsack
using Test
using LinearAlgebra

@testset "infeasible" begin
    minimizer! = QuadraticKnapsackMinimizer{Float64}(zeros(Float64, 4))
    x = zeros(Float64, 4)

    stats = QuadraticKnapsackStats()

    a = Float64[1., 2., 3., 4.]
    b = 11.

    minimizer!(x, a, b; stats=stats)

    @test all(x .== 1.)

    @test stats.infeasible == true
    @test stats.itercount == 0
end

@testset "trivial solution" begin
    minimizer! = QuadraticKnapsackMinimizer{Float64}(zeros(Float64, 4))
    x = zeros(Float64, 4)

    stats = QuadraticKnapsackStats()

    a = Float64[1., 2., 3., 4.]
    b = -1.

    minimizer!(x, a, b; stats=stats)

    @test all(x .== 0.)

    @test stats.infeasible == false
    @test stats.itercount == 0
end

@testset "inactive upper bounds" begin
    minimizer! = QuadraticKnapsackMinimizer{Float64}(zeros(Float64, 4))
    x = zeros(Float64, 4)

    stats = QuadraticKnapsackStats()

    a = Float64[1., 2., 3., 4.]
    b = 2.

    minimizer!(x, a, b; stats=stats)

    @test abs(dot(a, x) - b) < 100 * eps()
    @test all(0. .<= x .< 1.)

    @test stats.infeasible == false
    @test stats.itercount == 1
end

@testset "active upper bounds" begin
    minimizer! = QuadraticKnapsackMinimizer{Float64}(zeros(Float64, 4))
    x = zeros(Float64, 4)

    stats = QuadraticKnapsackStats()

    a = Float64[1., 2., 3., 4.]
    b = 9.

    minimizer!(x, a, b; stats=stats)

    @test abs(dot(a, x) - b) < 100 * eps()
    @test all(0. .<= x .<= 1.)

    @test stats.infeasible == false
    @test stats.itercount == 3
end

@testset "negative components" begin
    minimizer! = QuadraticKnapsackMinimizer{Float64}(zeros(Float64, 4))
    x = zeros(Float64, 4)

    stats = QuadraticKnapsackStats()

    a = Float64[-1., 0., 1., 2.]
    b = 2.

    minimizer!(x, a, b; stats=stats)

    @test abs(dot(a, x) - b) < 100 * eps()
    @test all(0. .<= x .<= 1.)
    @test x[1] == 0.
    @test x[2] == 0.

    @test stats.infeasible == false
    @test stats.itercount == 1
end

@testset "nontrivial upper bounds" begin
    minimizer! = QuadraticKnapsackMinimizer{Float64}(zeros(Float64, 4))
    x = zeros(Float64, 4)

    stats = QuadraticKnapsackStats()

    upper_bounds = Float64[.5, .05, .75, .25]
    a = Float64[1., 2., 3., 4.]
    b = 1.5

    minimizer!(x, a, b; upper_bounds=upper_bounds, stats=stats)

    @test abs(dot(a, x) - b) < 100 * eps()
    @test all(0. .<= x .<= upper_bounds)

    @test stats.infeasible == false
    @test stats.itercount == 2
end

@testset "weighted objective" begin
    minimizer! = QuadraticKnapsackMinimizer{Float64}(zeros(Float64, 4))
    x = zeros(Float64, 4)

    stats = QuadraticKnapsackStats()

    upper_bounds = Float64[.5, .05, .75, .25]
    a = Float64[1., 2., 3., 4.]
    b = 1.5
    w = Float64[4., 3., 2., 1.]

    minimizer!(x, a, b; upper_bounds=upper_bounds, w=w, stats=stats)

    @test abs(dot(a, x) - b) < 100 * eps()
    @test all(0. .<= x .<= upper_bounds)

    @test stats.infeasible == false
    @test stats.itercount == 3
end