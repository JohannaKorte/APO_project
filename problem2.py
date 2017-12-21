import cplex
import csv
import numpy as np
import os
#_______________________________________________________________________________________________________________________
# CSV --> matrices and dicts
#Paths to csv data files
pwd = os.getcwd() + '/data/'
aircraft_data_file = pwd + 'aircraft.csv'
demand_data_file = pwd + 'demand.csv'
distance_data_file = pwd + 'distance.csv'
lf_data_file = pwd + 'loadfactor.csv'
pso_data_file = pwd + 'pso.csv'
yield_data_file = pwd + 'yield.csv'
airport_data_file = pwd + 'airport_data.csv'

def read_aircraft_data(filename):
    aircraft_dict = {}
    with open(filename, 'rU') as csv_file:
        reader = csv.reader(csv_file)
        next(reader,None)  #skip header
        for row in reader:
            aircraft_dict[row[0]] = row[1:]
    return aircraft_dict

def read_csv(filename):
    """Given a filename, reads the csv file and returns a corresponding matrix"""
    matrix = []
    with open(filename, 'rU') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)   #skip header
        for row in reader:
            if matrix == []:
                matrix = [row[1:]]
            else:
                matrix = np.append(matrix,[row[1:]], axis=0)
    matrix = matrix.astype(float)
    return matrix

def dict_to_float(n):
    for key in n:
        if key == 'ICAO Code':
            n[key] = n[key]
        else:
            n[key] = [float(i) for i in n[key]]
    return n

#read all csv files into matrices
distance = read_csv(distance_data_file)
demand = read_csv(demand_data_file)
lf = read_csv(lf_data_file)
pso = read_csv(pso_data_file)
yield_matrix = read_csv(yield_data_file)
aircraft_dict = dict_to_float(read_aircraft_data(aircraft_data_file))
airport_dict = dict_to_float(read_aircraft_data(airport_data_file))
#______________________________________________________________________________________________________________________
# Calculatable variables

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
num_fleet = len(aircraft_dict['Seats'])
nodes = 24
BT = 10
destinations_by_index = ['London','Paris','Amsterdam','Frankfurt','Madrid','Barcelona','Munich','Rome','Dublin',
                         'Stockholm','Lisbon','Berlin','Helsinki','Warsaw','Edinburgh','Bucharest','Heraklion',
                         'Reykjavik','Palermo','Madeira','New York','Atlanta','Los Angeles','Chicago']
america = [20,21,22,23]
europe = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
long_haul_AC = [3,4] # equals to AC4 and AC5, because of 0-indexing
adding_cost = 2000
terminating_cost = 8000
subsidy = 15000
subsidy_airports = [16, 17, 18, 19]
#______________________________________________________________________________________________________________________
# Index finders

def index_finder(v,origin,destination,k=0):
    """Given decision variable name, OD leg and AC type k, gives the index for the coefficient vertices"""
    base_case = destination + (origin * nodes) + (k * nodes * nodes)
    if v == 'x':
        return base_case
    elif v == 'w':
        return nodes*nodes + base_case
    elif v == 'z':
        return 2 * (nodes * nodes) + base_case
    else:
        print('wrong variable name in index finder')

def binary_index_finder(n):
    #last
    if n == 19:
        return len(dv_names) - 1
    elif n == 18:
        return len(dv_names) - 2
    elif n == 17:
        return len(dv_names) - 3
    #first
    elif n == 16:
        return len(dv_names) - 4

def ac_index_finder(v, n):
    if v == 'ac':
        return 2*nodes*nodes + nodes*nodes*num_fleet + n
    elif v == 'n_add':
        return 2*nodes*nodes + nodes*nodes*num_fleet + num_fleet + n
    elif v == 'n_term':
        return 2*nodes*nodes + nodes*nodes*num_fleet + 2*num_fleet + n



#___________________________________Variable_Names______________________________________________________________________
# Initialize cplex problem
problem = cplex.Cplex()
problem.objective.set_sense(problem.objective.sense.maximize)

#VARIABLE NAMES
#   dv's: xij, wij, zij_k, ac_k, n_add_k, n_term_k b_k
#   x_ij of size: nodes*nodes                  -- Direct flow from i to j
#   w_ij of size: nodes*nodes                  -- Transfer flow from i to j
#   z_ij k of size: nodes*nodes*num_fleet      -- Frequency from i to j with type k
#   ack of size: num_fleet                     -- Total number of aircraft of type k
#   nk of size: num_fleet                      -- change in aircraft of certain type k
#   bs of size: num subsidy airports           -- Binary variable subsidy per airport s
dv_names = []
for dv in ['x', 'w', 'z', 'ac', 'n_add', 'n_term', 'b']:
    if dv == 'z':
        for k in range(num_fleet):
            for i in range(nodes):
                for j in range(nodes):
                    dv_names += (["%s%s_%s_%s" % (str(dv),str(i), str(j), str(k))])
    elif dv == 'x' or dv == 'w':   #dv = x or w
        for i in range(nodes):
            for j in range(nodes):
                dv_names += (["%s%s_%s" % (str(dv),str(i), str(j))])
    elif dv == 'ac' or dv == 'n_add' or dv == 'n_term':
        for k in range(num_fleet):
            dv_names+= (["%s%s" % (str(dv), str(k))])
    elif dv == 'b':
        for s in subsidy_airports:
            dv_names += (['%s%s' % (str(dv), str(s))])

# ___________________________________Objective_Function_&_Bounds________________________________________________________
objective = [0] * (len(dv_names))
for dv in ['x', 'w', 'z', 'ac', 'n_add', 'n_term', 'b']:
    if dv == 'b':
        for s in subsidy_airports:
            objective[binary_index_finder(s)] = subsidy
    elif dv == 'ac':
        for k in range(num_fleet):
            objective[ac_index_finder(dv,k)] = -1 * aircraft_dict['Lease'][k]
    elif dv == 'n_add':
        for k in range(num_fleet):
            objective[ac_index_finder(dv,k)] = -1 * adding_cost
    elif dv == 'n_term':
        for k in range(num_fleet):
            objective[ac_index_finder(dv,k)] = -1 * terminating_cost
    else:
        for i in range(nodes):
            for j in range(nodes):
                if dv in ['x', 'w']:
                    objective[index_finder(dv, i, j)] = distance[i][j] * yield_matrix[i][j]
                elif dv == 'z':
                    for k in range(num_fleet):
                        objective[index_finder(dv, i, j, k)] = -1 * total_operating_cost(i, j, k)
                else:
                    print dv
                    print 'objective function was given wrong variable name, aka not being x, w, z,b,ac'

#LOWER BOUNDS
lower_bounds = [0] * len(dv_names)
#UPPER BOUNDS
upper_bounds = [cplex.infinity] * (len(dv_names) - 4) + [1,1,1,1] #binary upperbounds included

problem.variables.add(obj = objective,
                      lb = lower_bounds,
                      ub = upper_bounds,
                      types= ["I"] * len(dv_names),
                      names = dv_names)

#_____________________________________Constraints______________________________________________________________________
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
for k in range(num_fleet):
    #TODO: TEST
    c5 =[0] * len(dv_names)
    for i in range(nodes):
        for j in range(nodes):
            c5[index_finder('z',i,j,k)] = (distance[i][j]/aircraft_dict['Speed'][k]) + turnaround(j,k)
            c5[ac_index_finder('ac',k)] = -1 * BT * 7
    constraints.append([dv_names,c5])
    constraint_senses.append('L')
    rhs.append(0)
    constraint_names.append("AC_usage_%s" % str(k))

#range constraint
#TODO: TEST
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
            constraint_names.append("range_%s_%s_%s" % (str(i),str(j),str(k)))

#runway constraint
#TODO: TEST
for k in range(num_fleet):
    for i in range(nodes):
        for j in range(nodes):
            c7 = [0] * len(dv_names)
            c7[index_finder('z',i,j,k)] = 1
            constraints.append([dv_names,c7])
            constraint_senses.append('L')
            if aircraft_dict['Runway'] <= airport_dict['Runway']:
                rhs.append(cplex.infinity)
            else:
                rhs.append(0)
            constraint_names.append("runway_%s_%s_%s" % (str(i),str(j),str(k)))

#no direct flights constraint
#can only go to US from hub
#TESTED
for i in range(nodes):
    for j in range(nodes):
        c8 = [0] * len(dv_names)
        c8[index_finder('x',i,j)] = 1
        constraints.append([dv_names,c8])
        constraint_senses.append('L')
        # from non hub to america not allowed, hub(i) == 1 --> i is non-hub
        if hub(i) == 1 and j in america:
            rhs.append(0)
        elif hub(j) == 1 and i in america:
            rhs.append(0)
        else:
            rhs.append(cplex.infinity)
        constraint_names.append("no_direct_eu_us_%s_%s" % (str(i),str(j)))

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
    constraint_names.append("subsidy_hub_%s" % str(s))

for s in subsidy_airports:
    c10 = [0] * len(dv_names)
    for k in range(num_fleet):
        c10[index_finder('z',s,0,k)] = aircraft_dict['Seats'][k]
        c10[binary_index_finder(s)] = -200
    constraints.append([dv_names,c10])
    constraint_senses.append('G')
    rhs.append(0)
    constraint_names.append("subsidy_%s_hub" % str(s))

#Total AC amount constraint
#TODO: TEST
#AC_k = initial + n_add - n_term -->
#AC_k - n_add + n_term = initial
for k in range(num_fleet):
    c11 = [0] * len(dv_names)
    c11[ac_index_finder('ac',k)] = 1
    c11[ac_index_finder('n_add',k)] = -1
    c11[ac_index_finder('n_term',k)] = 1
    constraints.append([dv_names,c11])
    constraint_senses.append('E')
    rhs.append(aircraft_dict['Amount'][k])
    constraint_names.append("AC_amount_%s" % str(k))

#US capacity constraint
#Todo: TEST
# for i in america:
#     for j in america:
#         c12 = [0] * len(dv_names)
#         for k in range(num_fleet):
#             c12[index_finder('z',i,j,k)] = aircraft_dict['Seats'][k] #* lf[i][j]
#         constraints.append([dv_names,c12])
#         constraint_senses.append('L')
#         rhs.append(7500)
#         constraint_names.append("US_capacity_%s_%s" % (str(i),str(j)))
for j in america:
    i = 0
    c12_1 = [0] * len(dv_names)
    for k in range(num_fleet):
        c12_1[index_finder('z',i,j,k)] = aircraft_dict['Seats'][k] #* lf[i][j]
    constraints.append([dv_names,c12_1])
    constraint_senses.append('L')
    rhs.append(7500)
    constraint_names.append("US_capacity_%s_%s" % (str(i),str(j)))

for i in america:
    j = 0
    c12_2 = [0] * len(dv_names)
    for k in range(num_fleet):
        c12_2[index_finder('z', 0, j, k)] = aircraft_dict['Seats'][k]  * lf[i][j]
    constraints.append([dv_names, c12_2])
    constraint_senses.append('L')
    rhs.append(7500)
    constraint_names.append("US_capacity_%s_%s" % (str(i), str(j)))

#long-haul AC constraint
#TESTED
for i in range(nodes):
    for j in range(nodes):
        for k in long_haul_AC:
            c13 = [0] * len(dv_names)
            c13[index_finder('z',i,j,k)] = 1
            constraints.append([dv_names,c13])
            constraint_senses.append('L')
            if hub(i) == 0 and j in america:
                rhs.append(cplex.infinity)
            elif hub(j) == 0 and i in america:
                rhs.append(cplex.infinity)
            else:
                rhs.append(0)
            constraint_names.append("Long_haul_%s_%s_%s" % (i,j,k))

#_____________________________________Solve_LP______________________________________________________________________

problem.linear_constraints.add(lin_expr = constraints,
                               senses = constraint_senses,
                               rhs = rhs,
                               names = constraint_names)

problem.parameters.timelimit.set(120.0)

problem.solve()
problem.write("Problem2.lp")
print problem.solution.get_status()
solution = problem.solution.get_values()
print problem.solution.get_objective_value()
np.save('solution_prob2', solution)

#Save solution for use in problem 3
frequencies = []
for i in range(nodes):
    destinations = []
    for j in range(nodes):
        total_freq = 0
        for k in range(num_fleet):
            total_freq += solution[index_finder('z',i,j,k)]
        destinations.append(total_freq)
    frequencies.append(destinations)
np.save('frequencies_prob2',frequencies)
print frequencies
print len(frequencies)

#____________________________________KPI_______________________________________________________________________
# for index, variable in enumerate(solution):
#     if variable != 0:
#         print dv_names[index], variable

def kpi(solution, nodes, num_fleet):
    # Initialization
    cost = 0
    lease_cost = 0
    revenue = 0
    subsidy_total = 0
    acquisition_fees = 0
    termination_fees = 0
    ask = 0
    rpk = 0
    total_flights = 0
    total_seats = 0
    total_flow = 0

    #Revenue & operational cost
    for i in range(nodes):
        for j in range(nodes):
            revenue += solution[index_finder('x', i, j)] * distance[i][j] * yield_matrix[i][j]
            revenue += solution[index_finder('w', i, j)] * distance[i][j] * yield_matrix[i][j]
            for k in range(num_fleet):
                cost += solution[index_finder('z', i, j, k)] * total_operating_cost(i, j, k)

    #Add problem specific factors
    for k in range(num_fleet):
        lease_cost += solution[ac_index_finder('ac', k)] * aircraft_dict['Lease'][k]

    for ap in subsidy_airports:
        subsidy_total += solution[binary_index_finder(ap)] * subsidy
        revenue += solution[binary_index_finder(ap)] * subsidy

    for k in range(num_fleet):
        acquisition_fees += solution[ac_index_finder('n_add', k)] * adding_cost
        termination_fees += solution[ac_index_finder('n_term', k)] * terminating_cost

    profit = revenue - cost - lease_cost - acquisition_fees - termination_fees

    #ASK
    #frequency * #seats * distance
    for i in range(nodes):
        for j in range(nodes):
            for k in range(num_fleet):
                ask += solution[index_finder('z',i,j,k)] * distance[i][j] * aircraft_dict['Seats'][k]

    #RPK
    # flow * distance
    for i in range(nodes):
        for j in range(nodes):
            rpk += solution[index_finder('x',i,j)] * distance[i][j]
            rpk += solution[index_finder('w',i,j)] * distance[i][j]

    #RASK
    rask_1 = (revenue - subsidy_total) / ask
    rask_2 = (revenue) / ask

    #CASK
    cask = (cost + lease_cost + acquisition_fees + termination_fees) / ask

    # Load Factor
    for i in range(nodes):
        for j in range(nodes):
            for k in range(num_fleet):
                total_flights += solution[index_finder('z', i, j, k)]
                total_seats += solution[index_finder('z', i, j, k)] * aircraft_dict['Seats'][k]
            total_flow += solution[index_finder('x', i, j)]
            total_flow += solution[index_finder('w', i, j)]
    av_lf = total_flow / total_seats

    #Print results
    print "Revenue (incl subsidy):          %s" % (revenue + subsidy_total)
    print "Total subsidies:                 %s" % subsidy_total
    print "Cost (incl lease and fees):      %s" % (cost + lease_cost + acquisition_fees + termination_fees)
    print "Lease cost:                      %s" % lease_cost
    print "Acquisition fees:                %s" % acquisition_fees
    print "Termination fees:                %s" % termination_fees
    print "Profit:                          %s" % profit
    print "_________________________________________________________________\n"
    print "ASK:                             %s" % ask
    print "RPK:                             %s" % rpk
    print "RASK (excl. subsidy):            %s" % rask_1
    print "RASK (incl. subsidy):            %s" % rask_2
    print "CASK:                            %s" % cask
    print "Load Factor                      %s" % av_lf
    print "_________________________________________________________________\n"
    print "Total seats:                     %s" % total_seats
    print "Total flow:                      %s" % total_flow
    print "Total flights:                   %s" % total_flights


print kpi(solution,nodes,num_fleet)