import cplex
import os
import csv
import numpy as np

#Paths to csv data files
pwd = os.getcwd() + '/data/'
aircraft_data_file = pwd + 'aircraft.csv'
demand_data_file = pwd + 'demand.csv'
distance_data_file = pwd + 'distance.csv'
lf_data_file = pwd + 'loadfactor.csv'
pso_data_file = pwd + 'pso.csv'
yield_data_file = pwd + 'yield.csv'

def read_aircraft_data(filename):
    aircraft_dict = {}
    with open(filename, 'rU') as csv_file:
        reader = csv.reader(csv_file)
        next(reader,None)  #skip header
        for row in reader:
            aircraft_dict[row[0]] = [float(i) for i in row[1:]]
    return aircraft_dict

def read_csv(filename):
    """Given a filename, reads the csv file and returns a corresponding matrix"""
    #TODO: EDIT SUCH THAT READS ALL AIRPORTS, NOT ONLY EUROPEAN ONES: REMOVE COMMENTED LINES
    matrix = []
    with open(filename, 'rU') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)   #skip header
        for row in reader:
            if matrix == []:
                matrix = [row[1:21]]
                #matrix = [row[1:]]
            else:
                matrix = np.append(matrix,[row[1:21]], axis=0)
                #matrix = np.append(matrix,[row[1:]], axis=0)
    matrix = matrix.astype(float)
    return matrix[:20,:]
    #return matrix

#read all csv files into matrices
distance = read_csv(distance_data_file)
demand = read_csv(demand_data_file)
lf = read_csv(lf_data_file)
pso = read_csv(pso_data_file)
yield_matrix = read_csv(yield_data_file)
aircraft_dict = read_aircraft_data(aircraft_data_file)

#_______________________________________________________________________________________________________________________
#HELPER FUNCTIONS

def total_operating_cost(i,j,k):
    """Given origin, destination and aircraft, calculates the total operating cost using fixed cost, time based costs
    and fuel costs"""
    fixed_op_cost = aircraft_dict['Cx'][k]
    time_based_cost = aircraft_dict['Ct'][k] * (distance[i][j]/aircraft_dict['Speed'][k])
    fuel_cost = aircraft_dict['Cf'][k] * distance[i][j]
    total_cost = fixed_op_cost + time_based_cost + fuel_cost
    #discount for flights departing or arriving at hub
    if hub(i) == 0 or hub(j) == 0:
        return total_cost * 0.7
    else:
        return total_cost

def turnaround(destination, aircraft_type):
    """ Calculates the required turnaround time for an aircraft type at a certain destination"""
    if destination == 0:
        return max(60, 2*aircraft_dict['TAT'][aircraft_type])/60
    else:
        return aircraft_dict['TAT'][aircraft_type]/60

def hub(airport):
    """returns 0 if airport is hub, else 1"""
    if airport == 0:
        return 0
    else:
        return 1
#______________________________________________________________________________________________________________________
#PARAMETERS
num_fleet = 3
#num_fleet = 2
#TODO CHANGE TO LEN(FLEET) WHEN CONSIDERING MORE TYPES
nodes = 20 #TODO CHANGE TO ALL DESTINATIONS
#nodes = 2
BT = 10
subsidy_airports = [16,17,18,19]
destinations_by_index = ['London','Paris','Amsterdam','Frankfurt','Madrid','Barcelona','Munich','Rome','Dublin',
                         'Stockholm','Lisbon','Berlin','Helsinki','Warsaw','Edinburgh','Bucharest','Heraklion',
                         'Reykjavik','Palermo','Madeira','New York','Atlanta','Los Angeles','Chicago']
#______________________________________________________________________________________________________________________

#make new cplex problem and define as maximization problem
problem = cplex.Cplex()
problem.objective.set_sense(problem.objective.sense.maximize)

#VARIABLE NAMES
#dv's: xijk, wijk, zijk, (ack)  #TODO ADD AC
dv_names = []
#x_ij k of size: nodes*nodes
#w_ij k of size: nodes*nodes
#z_ij k of size: nodes*nodes*num_fleet
for dv in ['x', 'w', 'z', 'b']:
    if dv == 'z':
        for k in range(num_fleet):
            for i in range(nodes):
                for j in range(nodes):
                    dv_names += (["%s%s_%s_%s" % (str(dv),str(i), str(j), str(k))])
    elif dv == 'x' or dv == 'w':   #dv = x or w
        for i in range(nodes):
            for j in range(nodes):
                dv_names += (["%s%s_%s" % (str(dv),str(i), str(j))])
    elif dv == 'b':
        for s in subsidy_airports:
            dv_names += (['%s%s' % (str(dv), str(s))])

def index_finder(v,origin,destination,k=0):
    """Given decision variable name, OD leg and AC type k, gives the index for the coefficient vertices"""
    base_case = destination + (origin * nodes) + (k * nodes * nodes)
    if v == 'x':
        return base_case
    elif v == 'w':
        #return (nodes*nodes*num_fleet) + base_case
        return nodes*nodes + base_case
    elif v == 'z':
        #return 2*(nodes*nodes*num_fleet) + base_case
        return 2 * (nodes * nodes) + base_case
    else:
        print('wrong variable name in index finder')

def binary_index_finder(n):
    #last
    if s == 19:
        return len(dv_names) - 1
    elif s == 18:
        return len(dv_names) - 2
    elif s == 17:
        return len(dv_names) - 3
    #first
    elif s == 16:
        return len(dv_names) - 4

#OBJECTIVE FUNCTION
#TODO: CHECK COEFFICIENTS
#TODO: ADD AC DV
objective = [0] * (len(dv_names))
for dv in ['x', 'w', 'z', 'b']:
    #assign bonus subsidy coefficient
    if dv == 'b':
        for s in subsidy_airports:
            objective[binary_index_finder(s)] = 15000
    else:
        for i in range(nodes):
            for j in range(nodes):
                if dv in ['x', 'w']:
                    objective[index_finder(dv, i, j)] = distance[i][j] * yield_matrix[i][j]
                elif dv == 'z':
                    for k in range(num_fleet):
                        objective[index_finder(dv, i, j, k)] = -1 * total_operating_cost(i, j, k)
                else:
                    print 'objective function was given wrong variable name, aka not being x, w, z'

#LOWER BOUNDS
lower_bounds = [0] * len(dv_names)
#UPPER BOUNDS
upper_bounds = [cplex.infinity] * (len(dv_names) - 4) + [1,1,1,1] #binary upperbounds included

problem.variables.add(obj = objective,
                      lb = lower_bounds,
                      ub = upper_bounds,
                      types= ["I"] * len(dv_names),
                      names = dv_names)
#______________________________________________________________________________________________________________________
#CONSTRAINTS
constraints = []
constraint_senses = []
rhs = []
constraint_names = []

#flow <= demand
for i in range(nodes):
    for j in range(nodes):
        c1 = [0] * len(dv_names)
        c1[index_finder('x', i, j)] = 1
        c1[index_finder('w',i,j)] = 1
        constraints.append([dv_names,c1])
        constraint_senses.append("L")
        rhs.append(demand[i][j])
        constraint_names.append("flow_%s_%s" % (i, j))

#transfer only if hub is not origin or destination
for i in range(nodes):
    for j in range(nodes):
        c2 = [0] * len(dv_names)
        c2[index_finder('w', i, j)] = 1
        #append constraint to different lists
        constraints.append([dv_names, c2])
        constraint_senses.append("L")
        if i != 0 and j!= 0:
           rhs.append(demand[i][j])
        else:
           rhs.append(0)
        constraint_names.append("transfer_%s_%s" % (i, j))

#capacity constraint
for i in range(nodes):
    for j in range(nodes):
        c3 = [0] * len(dv_names)
        c3[index_finder('x', i, j)] = 1
        for m in range(nodes):
            c3[index_finder('w', i, m)] += 1 - hub(j)
            c3[index_finder('w', m, j)] += 1 - hub(i)
        for k in range(num_fleet):
            c3[index_finder('z', i, j, k)] = -1 * aircraft_dict['Seats'][k] * lf[i][j]
        constraints.append([dv_names, c3])
        constraint_senses.append("L")
        rhs.append(0)
        constraint_names.append("capacity_%s%s" % (i, j))

#balance between incoming and outgoing flights per node
#TODO: CHECK CORRECTNESS .LP
for i in range(nodes):
    for k in range(num_fleet):
        c4= [0] * len(dv_names)
        for j in range(nodes):
            if i != j:
                c4[index_finder('z',i,j,k)] = 1
                c4[index_finder('z',j,i,k)] = -1
        constraints.append([dv_names, c4])
        constraint_senses.append('E')
        rhs.append(0)
        constraint_names.append("balance_%s_%s" % (i, k))


#aircraft use
#TODO: change when using aircraft as dv
for k in range(num_fleet):
    c5 =[0] * len(dv_names)
    for i in range(nodes):
        for j in range(nodes):
            c5[index_finder('z',i,j,k)] = (distance[i][j]/aircraft_dict['Speed'][k]) + turnaround(j,k)
    constraints.append([dv_names,c5])
    constraint_senses.append('L')
    rhs.append(BT * aircraft_dict['Amount'][k] * 7)
    constraint_names.append("ACuse_%s" % k)

#range constraint
for i in range(nodes):
    for j in range(nodes):
        for k in range(num_fleet):
            c6 = [0] * len(dv_names)
            c6[index_finder('z',i,j,k)] = 1
            constraints.append([dv_names, c6])
            constraint_senses.append('L')
            if distance[i][j] <= aircraft_dict['Range'][k]:
                rhs.append(cplex.infinity)
            else:
                rhs.append(0)
            constraint_names.append("range_%s_%s_%s" % (i,j,k))

#subsidy constraints
#Sum of offered seats to subsidy destinations should be larger or equal to its binary dv * 200
for s in subsidy_airports:
    c9 = [0] * len(dv_names)
    for k in range(num_fleet):
        c9[index_finder('z',0,s,k)] = aircraft_dict['Seats'][k]
        c9[binary_index_finder(s)] = -200
    constraints.append([dv_names,c9])
    constraint_senses.append('G')
    rhs.append(0)
    constraint_names.append("subsidy_hub_%s" % s)

for s in subsidy_airports:
    c10 = [0] * len(dv_names)
    for k in range(num_fleet):
        c10[index_finder('z',s,0,k)] = aircraft_dict['Seats'][k]
        c10[binary_index_finder(s)] = -200
    constraints.append([dv_names,c10])
    constraint_senses.append('G')
    rhs.append(0)
    constraint_names.append("subsidy_%s_hub" % s)


# #make sure at least one flight
# c7=[0] * len(dv_names)
# for i in range(nodes):
#     for j in range(nodes):
#         for k in range(num_fleet):
#             c7[index_finder('z',i,j,k)] = 1
# # constraints.append([dv_names, c7])
# # constraint_senses.append('G')
# # rhs.append(1)
# # constraint_names.append("larger than 1")

#no flights to itself allowed
# for i in range(nodes):
#     for k in range(num_fleet):
#         c8 = [0] * len(dv_names)
#         c8[index_finder('z',i,i,k)] = 1
#         # constraints.append([dv_names, c8])
#         # constraint_senses.append('E')
#         # rhs.append(0)
#         # constraint_names.append("not itself_%s_%s" % (i,k))

problem.linear_constraints.add(lin_expr = constraints,
                               senses = constraint_senses,
                               rhs = rhs,
                               names = constraint_names)

problem.parameters.timelimit.set(60.0)

problem.solve()
problem.write("test.lp")
print problem.solution.get_status()
print(problem.solution.get_values())

#___________________________CHECKS________________________________________________________________________________
# for index, variable in enumerate(problem.solution.get_values()):
#     if variable != 0:
#         print dv_names[index], variable

# #CHECK AC HOURS
# print 'k, amount, total_flights, total_operating_hours'
# for k in range(num_fleet):
#     total_operating_hours = 0
#     total_flights = 0
#     for i in range(nodes):
#         for j in range(nodes):
#             if problem.solution.get_values()[index_finder('z',i,j,k)] != 0:
#                 total_operating_hours += ((distance[i][j] / aircraft_dict['Speed'][k]) + turnaround(j, k))* \
#                                     problem.solution.get_values()[index_finder('z',i,j,k)]
#                 total_flights += problem.solution.get_values()[index_finder('z', i,j,k)]
#     print k, aircraft_dict['Amount'][k], total_flights, total_operating_hours


#Total flows per combination
for i in range(nodes):
    for j in range(nodes):
        total_flow_pair = 0
        total_flow_pair += problem.solution.get_values()[index_finder('x',i,j)]
        total_flow_pair += problem.solution.get_values()[index_finder('w',i,j)]
        if total_flow_pair != 0:
            print i,",",j,",",total_flow_pair

#objective value
solution = problem.solution.get_values()
print "objective value minus lease:"
lease_cost = 0
for index, ac in enumerate(aircraft_dict['Amount']):
    lease_cost += ac * aircraft_dict['Lease'][index]

print problem.solution.get_objective_value() - lease_cost

# Check revenue and cost values
revenue = 0
cost = 0
for i in range(nodes):
    for j in range(nodes):
        revenue += solution[index_finder('x',i,j)] * distance[i][j] * yield_matrix[i][j]
        revenue += solution[index_finder('w',i,j)] * distance[i][j] * yield_matrix[i][j]
        for k in range(num_fleet):
            cost += solution[index_finder('z',i,j,k)] * total_operating_cost(i,j,k)
for b in subsidy_airports:
    revenue += solution[binary_index_finder(b)] * 15000
cost += lease_cost

print "lease cost: %s" % lease_cost
print "yield is %s (including subsidy)" % str(revenue)
print "cost (including lease) is %s" % str(cost)

destinations_by_index = ['London','Paris','Amsterdam','Frankfurt','Madrid','Barcelona','Munich','Rome','Dublin','Stockholm','Lisbon','Berlin','Helsinki','Warsaw','Edinburgh','Bucharest','Heraklion','Reykjavik','Palermo','Madeira']

# def print_tables():
#     print 'from, to, x, w, z AC1, z AC2, z AC3'
#     for i in range(nodes):
#         for j in range(nodes):
#             solution = problem.solution.get_values()
#             x_value = solution[index_finder('x',i,j)]
#             w_value = solution[index_finder('w',i,j)]
#             z_value_0 = solution[index_finder('z',i,j,0)]
#             z_value_1 = solution[index_finder('z',i,j,1)]
#             z_value_2 = solution[index_finder('z',i,j,2)]
#             if x_value or w_value or z_value_0 or z_value_1 or z_value_2 != 0:
#                 print "%s , %s, %s, %s, %s, %s, %s" % (destinations_by_index[i], destinations_by_index[j], str(x_value),str(w_value),str(z_value_0),str(z_value_1),str(z_value_2))
#
# print_tables()