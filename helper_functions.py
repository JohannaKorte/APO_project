import os
import csv
import numpy as np

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
        # for row in reader:
        #     aircraft_dict[row[0]] = [float(i) for i in row[1:]]
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

