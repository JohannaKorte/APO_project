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
    matrix = []
    with open(filename, 'rU') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)   #skip header
        for row in reader:
            if matrix == []:
                matrix = [row[1:21]]
            else:
                matrix = np.append(matrix,[row[1:21]], axis=0)
    matrix = matrix.astype(float)
    return matrix[:20,:]

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
nodes = 20
BT = 10
destinations_by_index = ['London','Paris','Amsterdam','Frankfurt','Madrid','Barcelona','Munich','Rome','Dublin',
                         'Stockholm','Lisbon','Berlin','Helsinki','Warsaw','Edinburgh','Bucharest','Heraklion',
                         'Reykjavik','Palermo','Madeira','New York','Atlanta','Los Angeles','Chicago']
#______________________________________________________________________________________________________________________

#make new cplex problem and define as maximization problem
problem = cplex.Cplex()
problem.objective.set_sense(problem.objective.sense.maximize)

#VARIABLE NAMES
#dv's: xijk, wijk, zijk, (ack)
dv_names = []
#x_ij k of size: nodes*nodes
#w_ij k of size: nodes*nodes
#z_ij k of size: nodes*nodes*num_fleet
for dv in ['x', 'w', 'z']:
    if dv == 'z':
        for k in range(num_fleet):
            for i in range(nodes):
                for j in range(nodes):
                    dv_names += (["%s%s_%s_%s" % (str(dv),str(i), str(j), str(k))])
    else:   #dv = x or w
        for i in range(nodes):
            for j in range(nodes):
                dv_names += (["%s%s_%s" % (str(dv),str(i), str(j))])

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

#OBJECTIVE FUNCTION
#Number of aircraft is not added as decision variable, thus to arrive at the actual objective value including the
#lease costs, this needs to be subtracted after running.
objective = [0] * (len(dv_names))
for dv in ['x', 'w', 'z']:
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
upper_bounds = [cplex.infinity] * len(dv_names)

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

problem.linear_constraints.add(lin_expr = constraints,
                               senses = constraint_senses,
                               rhs = rhs,
                               names = constraint_names)

problem.parameters.timelimit.set(6000.0)

problem.solve()
problem.write("problem1_1.lp")
print problem.solution.get_status()
print(problem.solution.get_values())

solution = problem.solution.get_values()

#___________________________KPI_______________________________________________________________________________
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

    # Lease cost
    for k in range(num_fleet):
        lease_cost += aircraft_dict['Amount'][k] * aircraft_dict['Lease'][k]

    profit = revenue - cost - lease_cost

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
            for k in range(num_fleet):
                rpk += solution[index_finder('x',i,j,k)] * distance[i][j]
                rpk += solution[index_finder('w',i,j,k)] * distance[i][j]

    #RASK
    rask_1 = revenue / ask
    rask_2 = (revenue + subsidy_total) / ask

    #CASK
    cask = (cost + lease_cost + acquisition_fees + termination_fees) / ask

    #Load Factor
    for i in range(nodes):
        for j in range(nodes):
            for k in range(num_fleet):
                total_flights += solution[index_finder('z',i,j,k)]
                total_seats += solution[index_finder('z',i,j,k)] * aircraft_dict['Seats'][k]
            total_flow += solution[index_finder('x',i,j)]
            total_flow += solution[index_finder('w',i,j)]
    av_lf = total_flow / total_seats

    #Print results
    print "Revenue (incl subsidy):          %s" % str(revenue)
    print "Total subsidies:                 %s" % str(subsidy_total)
    print "Cost (incl lease and fees):      %s" % str(cost + lease_cost + acquisition_fees + termination_fees)
    print "Lease cost:                      %s" % str(lease_cost)
    print "Acquisition fees:                %s" % str(acquisition_fees)
    print "Termination fees:                %s" % str(termination_fees)
    print "Profit:                          %s" % str(profit)
    print "_________________________________________________________________\n"
    print "ASK:                             %s" % str(ask)
    print "RPK:                             %s" % str(rpk)
    print "RASK (excl. subsidy):            %s" % str(rask_1)
    print "RASK (incl. subsidy):            %s" % str(rask_2)
    print "CASK:                            %s" % str(cask)
    print "Load Factor                      %s" % str(av_lf)
    print "_________________________________________________________________\n"
    print "Total seats:                     %s" % str(total_seats)
    print "Total flow:                      %s" % str(total_flow)
    print "Total flights:                   %s" % str(total_flights)
    print "_________________________________________________________________\n"
    print "AC1:                             %s" % str(aircraft_dict['Amount'][0])
    print "AC2:                             %s" % str(aircraft_dict['Amount'][1])
    print "AC3:                             %s" % str(aircraft_dict['Amount'][2])
    print "AC4:                             %s" % str(aircraft_dict['Amount'][3])
    print "AC5:                             %s" % str(aircraft_dict['Amount'][4])


def print_tables():
    print 'source, target, x, w'
    for i in range(nodes):
        for j in range(nodes):
            x_value = solution[index_finder('x', i, j)]
            w_value = solution[index_finder('w', i, j)]
            # z_value_0 = solution[index_finder('z',i,j,0)]
            # z_value_1 = solution[index_finder('z',i,j,1)]
            # z_value_2 = solution[index_finder('z',i,j,2)]
            # z_value_3 = solution[index_finder('z',i,j,3)]
            # z_value_4 = solution[index_finder('z', i, j, 4)]
            # z_value = z_value_0+z_value_1+z_value_2+z_value_3+z_value_4
            if x_value or w_value != 0:
                print "%s , %s, %s, %s" % (i, j, str(x_value), str(w_value))

print kpi(solution,20,3)
print_tables()

