import pulp as plp
import random
import numpy as np
from dataclasses import dataclass

# encapsulate LP problem data and result for better organization
@dataclass
class Problem:
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    vars: list

@dataclass
class Result:
    optimal: bool
    status: str
    objective_val: float
    variable_vals: dict

# generates instances for the transportation problem
# supply_node: number of supply nodes
# demand_node: number of demand nodes
# max_cost: maximum cost for each edge
# max_supply_demand: maximum supply/demand for each node
def generateInstance(supply_node, demand_node, max_cost, max_supply_demand):
    supply_set = []
    demand_set = []
    cost_matrix = []
    
    # fill supply set with random values
    for i in range(supply_node):
        supply_set.append(random.randint(1, max_supply_demand))
    
    # fill demand set with random values
    for i in range(demand_node):
        demand_set.append(random.randint(1, max_supply_demand))
    
    supply_sum = sum(supply_set)
    demand_sum = sum(demand_set)

    # make sure supply and demand are equal
    while (supply_sum < demand_sum):
        rand_index = random.randint(0, demand_node - 1)
        if (demand_set[rand_index] > 0):
            demand_set[rand_index] -= 1
        supply_sum = sum(supply_set)
        demand_sum = sum(demand_set)
    
    while (demand_sum < supply_sum):
        rand_index = random.randint(0, supply_node - 1)
        if (supply_set[rand_index] > 0):
            supply_set[rand_index] -= 1
        supply_sum = sum(supply_set)
        demand_sum = sum(demand_set)
        
    # fill cost matrix with random values
    for i in range(supply_node):
        cost_matrix.append([])
        for j in range(demand_node):
            cost_matrix[i].append(random.randint(1, max_cost))
    
    return supply_set, demand_set, cost_matrix

# pulp solver for the transportation problem
def solver(supply_set, demand_set, cost_matrix):
    problem = plp.LpProblem("TransportationProblem", plp.LpMinimize)
    decision_vars = plp.LpVariable.dicts("X", [(i, j) for i in range(len(supply_set)) for j in range(len(demand_set))], lowBound=0, cat='Continuous')
    
    problem += plp.lpSum([decision_vars[(i, j)] * cost_matrix[i][j] for i in range(len(supply_set)) for j in range(len(demand_set))])
    
    # Supply constraints
    for i in range(len(supply_set)):
        problem += plp.lpSum([decision_vars[(i, j)] for j in range(len(demand_set))]) == supply_set[i]
    
    # Demand constraints
    for j in range(len(demand_set)):
        problem += plp.lpSum([decision_vars[(i, j)] for i in range(len(supply_set))]) == demand_set[j]
        
    problem.solve(plp.PULP_CBC_CMD(msg=False))
    
    solution = {
        "objective_value": plp.value(problem.objective),
        "decision_vars": {k: plp.value(v) for k, v in decision_vars.items()}
    }
    return solution
        

# formulates the transportation problem into standard LP form
# supply_set: list of supply values for each node
# demand_set: list of demand values for each node
# cost_matrix: matrix of costs for each edge
def formulate(supply_set, demand_set, cost_matrix):
    num_supply = len(supply_set)
    num_demand = len(demand_set)
    
    vars = [f"x{i + 1}{j + 1}" for i in range(num_supply) for j in range(num_demand)]
    
    # objective function coefficients
    c = np.array(cost_matrix).flatten()
    
    # supply constraints
    supply_constraints = []
    for i in range(num_supply):
        constraint = np.zeros(num_supply * num_demand)
        for j in range(num_demand):
            constraint[i * num_demand + j] = 1
        supply_constraints.append(constraint)
    
    # demand constraints
    demand_constraints = []
    for j in range(num_demand):
        constraint = np.zeros(num_supply * num_demand)
        for i in range(num_supply):
            constraint[i * num_demand + j] = 1
        demand_constraints.append(constraint)
    
    # combine all constraints
    A = np.array(supply_constraints + demand_constraints)
    b = np.array(supply_set + demand_set)
    
    return c, A, b, vars

# revised simplex method. It uses Big M method to handle artificial variables
def revisedSimplex(lp: Problem, M=1e6):
    # get dimensions of the matrix. m = number of constraints, n = number of variables
    m, n = lp.A.shape

    # artificial variables for equality constraints
    art_indices = list(range(n, n + m))
    
    # add artificials to the objective function. Identity matrix for artificals
    A_bigM = np.hstack((lp.A, np.eye(m)))
    c_bigM = np.hstack((lp.c, np.ones(m) * M)) # add Big M penalties for artificial variables
    vars_bigM = lp.vars + [f"a_{i + 1}" for i in range(m)]
    
    # Initialize basis with artificial variables
    basis = art_indices.copy()
    non_basis = [i for i in range(n + m) if i not in basis]

    # Compute initial basis matrix and its inverse
    B = A_bigM[:, basis]
    B_inv = np.linalg.inv(B)
    
    # Compute initial basic solution
    x = np.zeros(n + m)
    x[basis] = B_inv @ lp.b

    while True:
        # compute dual variables
        c_B = c_bigM[basis]
        y = c_B @ B_inv

        # compute reduced costs for non-basic variables
        A_N = A_bigM[:, non_basis]
        c_N = c_bigM[non_basis]
        reduced_costs = c_N - y @ A_N

        # optimality check
        if np.all(reduced_costs >= -1e-8):
            break

        # entering variable
        entering_idx = np.argmin(reduced_costs)
        entering_var = non_basis[entering_idx]

        # compute direction vector
        d = B_inv @ A_bigM[:, entering_var]

        # Check for unboundedness
        if np.all(d <= 0):
            return Result(False, "Unbounded", 0.0, {})

        # min ratio test
        ratios = []
        for i in range(len(d)):
            if d[i] > 1e-8:
                ratios.append(x[basis[i]] / d[i])
            else:
                ratios.append(np.inf)
                
        # leaving variable
        leaving_idx = np.argmin(ratios)
        leaving_var = basis[leaving_idx]

        # update basis and non-basic variables
        basis[leaving_idx] = entering_var
        non_basis[entering_idx] = leaving_var

        # update B_inv
        B = A_bigM[:, basis]
        B_inv = np.linalg.inv(B)

        # update basic solution
        x = np.zeros(n + m)
        x[basis] = B_inv @ lp.b

    # Check for artificial variables still in the basis
    for i in art_indices:
        if i in basis and abs(x[i]) > 1e-8:
            return Result(False, "Infeasible", 0.0, {})

    # Compute objective value
    obj_val = c_bigM[:n] @ x[:n]
    sol = {vars_bigM[i]: x[i] for i in range(len(vars_bigM)) if "a" not in vars_bigM[i]}

    return Result(True, "Optimal", obj_val, sol)
        
def experiment():
    for i in range(5):
        number_of_nodes = np.random.randint(3, 6)
        max_cost = np.random.randint(10, 50)
        max_supply_demand = np.random.randint(10, 50)
        print(f"\033[33mExperiment {i + 1}:\033[0m")
        print(f"Number of Nodes: {number_of_nodes}")
        print(f"Max Cost: {max_cost}")
        print(f"Max Supply/Demand: {max_supply_demand}", end="\n\n")
        
        supply_set, demand_set, cost_matrix = generateInstance(number_of_nodes, number_of_nodes, max_cost, max_supply_demand)

        c, A, b, vars = formulate(supply_set, demand_set, cost_matrix)

        lp = Problem(c, A, b, vars)

        solverResult = solver(supply_set, demand_set, cost_matrix)
        manualResult = revisedSimplex(lp)

        printSolverResult(solverResult, number_of_nodes, number_of_nodes)
        printManualResult(manualResult, number_of_nodes, number_of_nodes)


def printSolverResult(result, supply_nodes, demand_nodes):
    print("Solver Result:")
    obj_val = result["objective_value"]
    decision_vars = result["decision_vars"]
    
    print("Optimal Objective Value:", obj_val)
    for i in range(supply_nodes):
        for j in range(demand_nodes):
            print(f"x{i + 1}{j + 1} = {decision_vars[(i, j)]}", end="\t")
        print()
    print()


def printManualResult(result, supply_nodes, demand_nodes):
    print("Manual Result:")
    obj_val = result.objective_val
    var_vals = result.variable_vals
    
    print("Optimal Objective Value:", obj_val)
    for i in range(supply_nodes):
        for j in range(demand_nodes):
            print(f"x{i + 1}{j + 1} = {var_vals[f'x{i + 1}{j + 1}']}", end="\t")
        print()
    print()

experiment()