import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# Create the model
model = pyo.ConcreteModel()

# Sets
model.Products = pyo.Set(initialize=['Bread', 'Cake'])
model.Ingredients = pyo.Set(initialize=['Sugar', 'Flour'])

# Parameters
model.Profit = pyo.Param(model.Products, initialize={'Bread': 4, 'Cake': 8})
model.IngredientQuantity = pyo.Param(model.Products, model.Ingredients, initialize={
    ('Bread', 'Sugar'): 0.25, ('Bread', 'Flour'): 1,
    ('Cake', 'Sugar'): 1.0, ('Cake', 'Flour'): 0.5
})
model.AvailableIngredients = pyo.Param(model.Ingredients, initialize={'Sugar': 8, 'Flour': 10})
scalar_ProductionCapacity = 11

# Variables
model.Quantity = pyo.Var(model.Products, domain=pyo.NonNegativeIntegers)

# Objective function
def objective_rule(model):
    return sum(model.Profit[p] * model.Quantity[p] for p in model.Products)
model.Objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

# Constraints
def ingredient_constraint_rule(model, i):
    return sum(model.IngredientQuantity[p, i] * model.Quantity[p] for p in model.Products) <= model.AvailableIngredients[i]
model.IngredientConstraint = pyo.Constraint(model.Ingredients, rule=ingredient_constraint_rule)

def capacity_constraint_rule(model):
    return sum(model.Quantity[p] for p in model.Products) <= scalar_ProductionCapacity
model.CapacityConstraint = pyo.Constraint(rule=capacity_constraint_rule)

# Solve the model
solver = pyo.SolverFactory('gurobi')
solver.solve(model)

# Print the optimal solution
print("Optimal Solution:")
for p in model.Products:
    print(p, ": ", model.Quantity[p]())
print("Objective Value: ", model.Objective())
