import cplex
import numpy as np

nodes = 6
classes = 2
# Create an instance of a linear problem to solve
problem = cplex.Cplex()

# We want to find a maximum of our objective function
problem.objective.set_sense(problem.objective.sense.minimize)

# The names of our variables
dv_names = []
for k in range(classes):
    for i in range(nodes):
        for j in range(nodes):
            dv_names += (["x%s%s_%s" % (str(i), str(j), str(k))])
print dv_names

# The objective function. More precisely, the coefficients of the objective
# function. Note that we are casting to floats.
#objective = [5.0, 2.0, -1.0]
pre_objective = [[10000 for x in xrange(6)]for x in xrange(6)]
pre_objective[0][2] = 10
pre_objective[0][3] = 12
pre_objective[1][2] = 8
pre_objective[1][3] = 10
pre_objective[2][3] = 3
pre_objective[2][4] = 25
pre_objective[2][5] = 20
pre_objective[3][2] = 3
pre_objective[3][4] = 25
pre_objective[3][5] = 20
of= np.reshape(pre_objective,(36,1))
of= np.concatenate((of, of), axis=0)
objective = [item for sublist in of for item in sublist]

lower_bounds = [0] * 72
upper_bounds = [cplex.infinity] * 72

problem.variables.add(obj = objective,
                      lb = lower_bounds,
                      ub = upper_bounds,
                      names = dv_names)

constraints = []
constraint_senses = []
rhs = []
constraint_names = []
#TODO flow constraint

def index_finder(i,j,k):
    return j + (i * nodes) + (k * nodes * nodes)

flow = [2000,3000,-1500,-1000,-1500,-1000,150,200,-100,-50,-150,-50]
for i in range(nodes):
    for k in range(classes):
        c1 = [0] * (nodes*nodes*classes)
        for j in range(nodes):
            c1[index_finder(i,j,k)] = 1
            c1[index_finder(j,i,k)] = -1
        # append constraint to different lists
        constraints.append([dv_names, c1])
        constraint_senses.append("E")
        # Why use flow from node i?
        rhs.append(flow[index_finder(k,i,0)])
        constraint_names.append("flow_x%s_%s" % (i,k))

print len(constraints)
#capacity constraint
pre_capacity = [[0 for x in xrange(6)]for x in xrange(6)]
pre_capacity[0][2] = 1700
pre_capacity[0][3] = 500
pre_capacity[1][2] = 2000
pre_capacity[1][3] = 1200
pre_capacity[2][3] = 250
pre_capacity[2][4] = 1000
pre_capacity[2][5] = 800
pre_capacity[3][2] = 250
pre_capacity[3][4] = 800
pre_capacity[3][5] = 300
c= np.reshape(pre_capacity,(36,1))
c= np.concatenate((c, c), axis=0)
capacity= [item for sublist in c for item in sublist]

for i in range(nodes):
    for j in range(nodes):
        c2 = [0] * 72
        for k in range(classes):
            c2[index_finder(i,j,k)] = 1
        #append constraint to different lists
        constraints.append([dv_names,c2])
        constraint_senses.append("L")
        rhs.append(capacity[index_finder(i,j,k)])
        constraint_names.append("cap_x%s%s_%s" % (i, j, k))

problem.linear_constraints.add(lin_expr = constraints,
                               senses = constraint_senses,
                               rhs = rhs,
                               names = constraint_names)

problem.solve()
print(problem.solution.get_values())
print problem.solution.get_objective_value()