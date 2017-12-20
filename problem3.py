import numpy as np
import math
import cplex
#______________________Load_Initial_Solution________________________________________
initial_freq = np.load('frequencies_prob2.npy')

#TODO: Load all files and make dictionaries & matrices

#______________________Parameters_________________________________________________________________________
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
swiss_hub = ...
#_____________________________________________________________________________________
#todo: frequency als matrix (alle k' bij elkaar optellen dus)

def main(start_frequency, num_iterations):
    reachable_demand = demand
    frequency = start_frequency
    i = 0
    while i < num_iterations:
        #For all OD pairs:
        for i in range(nodes):
            for j in range(nodes):
                #calculate market share
                ms = market_share(frequency,i,j)
                #update demand
                reachable_demand[i][j] = reachable_demand * ms
        #Run program iteration
        frequency = iteration(reachable_demand)

def market_share(frequency,i,j):
    "given OD pair, calculates market share"
    #get direct frequencies
    f_dir = frequency[i][j]
    f_swiss_dir = swiss_frequency[i][j]

    #Both fly direct
    if f_dir > 0 and f_swiss_dir > 0:
        #direct competition
        a,b=1.7,1.7
    else:
        f_swiss_indir = min(swiss_frequency[i][swiss_hub], swiss_frequency[swiss_hub][j])
        f_indir = min(frequency[i][0],frequency[0][j])

        if f_dir > 0:
            # We direct, they indirect
            if f_swiss_indir > 0:
                # competitive advantage
                a,b = 1, 1.7
                fs1 = f_dir
                fs2 = f_indir

        if f_swiss_dir > 0:
            #We indirect, they direct
            if f_indir > 0:
                #competitive disadvantage
                a,b=1.7,1
                fs1 = f_indir
                fs2 = f_swiss_dir

        if f_dir == 0 and f_swiss_dir == 0:
            #both not direct
            if f_indir > 0 and f_swiss_indir > 0:
                #both indirect
                #direct competition
                a,b=1.7,1.7
                fs1 = f_swiss_indir
                fs2 = f_indir

    if a == 0 and b == 0:
        return 1
    else:
        #return marketshare
        return (pow(fs1,a)/(pow(fs1,a) + pow(fs2,b)))

def iteration(reachable_demand):
    """ Takes demand matrix, outputs frequency array"""
    #Put cplex program here
