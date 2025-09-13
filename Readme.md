# Operations Research Project: Transportation Problem Solver

This project implements and compares two different approaches to solving the Transportation Problem: a library-based solver using PuLP and a custom implementation of the Revised Simplex Method.

## Overview

The Transportation Problem is a classic optimization problem in operations research that deals with finding the most cost-effective way to transport goods from multiple supply sources to multiple demand destinations.

## Features

- **Random Instance Generation**: Creates transportation problem instances with customizable parameters
- **Dual Solver Approach**:
  - PuLP-based solver using CBC optimizer
  - Custom Revised Simplex Method implementation with Big M method
- **Performance Comparison**: Benchmarks both solvers across different problem sizes
- **Standard LP Formulation**: Converts transportation problems to standard linear programming form

## Problem Formulation

The transportation problem is formulated as:

**Minimize**: ∑∑ c_ij \* x_ij (total transportation cost)

**Subject to**:

- ∑ x_ij = s_i (supply constraints)
- ∑ x_ij = d_j (demand constraints)
- x_ij ≥ 0 (non-negativity constraints)

Where:

- `c_ij` = cost of transporting one unit from source i to destination j
- `x_ij` = amount transported from source i to destination j
- `s_i` = supply at source i
- `d_j` = demand at destination j

## Requirements

```
pip install pulp numpy
```

## Usage

Simply run the main script to execute the performance comparison experiment:

```bash
python main.py
```

The program will:

1. Generate transportation problem instances of varying sizes (5, 10, 20, 30, 50, 100, 150, 200 nodes)
2. Solve each instance using both PuLP and the custom Revised Simplex Method
3. Display the optimal objective values and execution times for comparison

## Implementation Details

### Key Components

1. **`generateInstance()`**: Creates random transportation problem instances with balanced supply and demand
2. **`solver()`**: PuLP-based solver using the CBC optimizer
3. **`formulate()`**: Converts transportation problem to standard LP form
4. **`revisedSimplex()`**: Custom implementation of the Revised Simplex Method with Big M technique
5. **`experiment()`**: Runs performance benchmarking across different problem sizes

### Data Structures

- **`Problem`**: Dataclass containing LP problem data (c, A, b, vars)
- **`Result`**: Dataclass containing solution results (optimal status, objective value, variable values)

### Algorithm Features

- **Big M Method**: Handles artificial variables in the Revised Simplex Method
- **Optimality Detection**: Checks reduced costs for optimality conditions
- **Infeasibility Detection**: Identifies infeasible problems through artificial variable analysis
- **Unboundedness Detection**: Recognizes unbounded problems during the simplex iterations

## Output

For each experiment, the program displays:

- Problem parameters (number of nodes, max cost, max supply/demand)
- Optimal objective values from both solvers
- Execution times for performance comparison

## Technical Notes

- The custom Revised Simplex implementation uses matrix operations for efficiency
- Numerical stability is maintained through tolerance checks (1e-8)
- The Big M value is set to 1e6 by default for artificial variable penalties
- Supply and demand are automatically balanced during instance generation

## Educational Purpose

This project serves as an educational tool for understanding:

- Linear programming formulations
- The Simplex Method and its revised version
- Transportation problem modeling
- Performance comparison between different optimization approaches
