# QuadraticKnapsack

This package provides a tool to efficiently solve the continuous quadratic knapsack problem:
$$\min_{\begin{matrix}a^T x \geq b\\ 0 \leq x \leq u\end{matrix}} x^T \text{diag}(w) x$$
This problem is instrumental to quadratic knapsack limiting. Both quadratic knapsack limiting and a proof for the algorithm for the continuous quadratic knapsack problem are discussed in:

    @misc{christner2025entropystablenodaldiscontinuous,
        title={Entropy Stable Nodal Discontinuous Galerkin Methods via Quadratic Knapsack Limiting}, 
        author={Brian Christner and Jesse Chan},
        year={2025},
        eprint={2507.14488},
        archivePrefix={arXiv},
        primaryClass={math.NA},
        url={https://arxiv.org/abs/2507.14488}, 
    }

The following code provides the solution to the quadratic knapsack problem, with given dimension size `dim`, vector `a`, scalar `b`, weights `w` and upper bounds `upper_bounds`:

```
minimizer! = QuadraticKnapsackMinimizer{Float64}(zeros(Float64, dim))
x = zeros(Float64, dim)

minimizer!(x, a, b; upper_bounds=upper_bounds, w=w, stats=stats)
```

Information on feasibility and iteration count can be returned by optionally passing in an object of type `QuadraticKnapsackStats` to the minimizer. 