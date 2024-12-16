import pulp as plp
import random

# generates instances for the transportation problem
# supply_node: number of supply nodes
# demand_node: number of demand nodes
# max_cost: maximum cost for each edge
# max_supply_demand: maximum supply/demand for each node
def generateInstance(supply_node, demand_node, max_cost, max_supply_demand):
    supply_set = []
    demand_set = []
    cost_matrix = []
    
    if supply_node < demand_node:
        # fill supply set with random values
        for i in range(supply_node):
            supply_set.append(random.randint(1, max_supply_demand))
        
        supply_sum = sum(supply_set)
        
        for i in range(demand_node):
            demand_set.append(random.randint(min(max_supply_demand, supply_sum), min(max_supply_demand, supply_sum)))
            supply_sum -= demand_set[i]
    else:
        for i in range(demand_node):
            demand_set.append(random.randint(1, max_supply_demand))
        
        demand_sum = sum(demand_set)
        
        for i in range(supply_node):
            supply_set.append(random.randint(min(max_supply_demand, demand_sum), min(max_supply_demand, demand_sum)))
            demand_sum -= supply_set[i]

    # fill cost matrix with random values
    for i in range(supply_node):
        cost_matrix.append([])
        for j in range(demand_node):
            cost_matrix[i].append(random.randint(1, max_cost))
    print(supply_set)
    print(demand_set)
    print(cost_matrix)
    
    return [supply_set, demand_set, cost_matrix]


def solver(supply_set, demand_set, cost_matrix):
    problem = plp.LpProblem("TransportationProblem", plp.LpMinimize)
    decision_vars = plp.LpVariable.dicts("X", [(i, j) for i in range(len(supply_set)) for j in range(len(demand_set))], lowBound=0, cat='Continuous')
    problem += plp.lpSum([decision_vars[(i, j)] * cost_matrix[i][j] for i in range(len(supply_set)) for j in range(len(demand_set))])
    
    # Supply constraints
    for i in range(len(supply_set)):
        problem += plp.lpSum([decision_vars[(i, j)] for j in range(len(demand_set))]) == supply_set[i], f"Supply_Constraint_{i}"
    
    # Demand constraints
    for j in range(len(demand_set)):
        problem += plp.lpSum([decision_vars[(i, j)] for i in range(len(supply_set))]) == demand_set[j], f"Demand_Constraint_{j}"
        
    problem.solve()
    
    solution = {
        "objective": plp.value(problem.objective),
        "decision_vars": {k: plp.value(v) for k, v in decision_vars.items()}
    }
        
    print(problem)
    print()
    print()
    print()
    print(solution)
    

def revisedSimplex():
    pass
    
sets = generateInstance(3, 3, 10, 10)
supply_set = sets[0]
demand_set = sets[1]
cost_matrix = sets[2]
solver(supply_set, demand_set, cost_matrix)