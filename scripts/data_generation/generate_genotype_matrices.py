import os
import sys
module_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../")
sys.path.append(module_directory)
import numpy as np
import argparse
import demographies
import psutil
import time
import pickle  

process = psutil.Process()
initial_memory = process.memory_info().rss
start = time.perf_counter()

# Define defualt values for the arguments for debugging purposes
defaults = {
    'demography':"split",
    'output_path':"../data/test_output/",
    'N': 100,
    'n': 20,
    'L': 100,
    'tau': 4e-4,
    'K':5,
    'm':0.01
}


# Define arguments to read from the terminal
parser = argparse.ArgumentParser()
#sys.argv = []
parser.add_argument("--output_path",type = str, default=defaults["output_path"]) # The directory to save the genotype matrix
parser.add_argument("--demography",type = str, default=defaults["demography"]) # name of the demography to simulate from
parser.add_argument("--N",type = int, default=defaults["N"]) # population size in each subpopulation
parser.add_argument("--n",type = int, default=defaults["n"]) # number of individuals sampled from each subpopulation
parser.add_argument("--L",type = int, default=defaults["L"]) # number of loci sampled -> number of independent msprime replicate simulations to run
## population split parameters
parser.add_argument("--tau",type = float, default=defaults["tau"]) # time since the split ~ Fst
## stepping stones parameters
parser.add_argument("--K",type = int, default=defaults["K"]) # number of demes
parser.add_argument("--m",type = float, default=defaults["m"]) #migration rate out of a single deme


if __name__ == "__main__":
    # Parse the command line arguments, removing the script name from the list
    args = parser.parse_args()

    # Use the parsed arguments
    output_path = args.output_path
    demography = args.demography
    N = args.N
    n = args.n
    L = args.L
    # Use a standard chrom_length and mutation rate
    chrom_length = 100
    mu=1e-4

    if demography == "split":
        tau = args.tau
        filename = "tau=%.2e" % tau
    if demography == "SteppingStones_1d" or demography == "SteppingStones_2d":
        K = args.K
        m = args.m
        filename = "m=%.2e"%(m)
    
############### Running the simulation for a specified demography    
    if demography == "split":
        dem,samples = demographies.pop_split(N,tau,n)
    elif demography == "SteppingStones_1d":
        dem,samples = demographies.Stepping_Stones_1d(N,K,m,n)
    elif demography == "SteppingStones_2d":
        dem,samples = demographies.Stepping_Stones_2d(N,K,m,n)

    geno = demographies.Create_Genotypes(dem,samples,chrom_length,L,mu)

    with open(os.path.join(output_path,"genotypes.pkl"),"wb") as file:
        pickle.dump(geno,file)

    del geno

    
end = time.perf_counter()
final_memory = process.memory_info().rss
time_elapsed = (end-start)/60
memory_used = (final_memory-initial_memory) / 1048576
print("time taken eigen = %.2f minutes" % time_elapsed) 
print("memory used eigen= %.2f megabytes" % memory_used) 