#%%
import numpy as np
import argparse
from utils import generate_eigen_df
import psutil
import time
import pickle
import os

#%%
process = psutil.Process()
initial_memory = process.memory_info().rss
start = time.perf_counter()

defaults = {
    'output_path': "../data/test_output/",
    'n_min': 2,
    'n_max': 100,
    'n_num': 5,
    'n_space': "lin",
    'L_min': 50,
    'L_max': 3000,
    'L_num': 4,
    'L_space': "geom",
    'p': 1,
    'demography': 'split'
}

# input_file_path = "../data/test_output/tau=1.00e-03.npy"

parser = argparse.ArgumentParser()
parser.add_argument("--output_path",type = str, default=defaults["output_path"]) # The directory to save the dataframes
parser.add_argument("--n_min",type = int, default=defaults["n_min"]) # minimum number of individuals to sample
parser.add_argument("--n_max",type = int, default=defaults["n_max"]) # maximum number of individuals to sample
parser.add_argument("--n_num",type = int, default=defaults["n_num"]) # number of different sample sizes between min and max
parser.add_argument("--n_space",type = str, default=defaults["n_space"]) # whetehr to take steps in linspace or geomspace
parser.add_argument("--L_min",type = int, default=defaults["L_min"]) # minimum number of loci to sample
parser.add_argument("--L_max",type = int, default=defaults["L_max"]) # maximum number of loci to sample
parser.add_argument("--L_num",type = int, default=defaults["L_num"]) # number of different sample sizes between min and max
parser.add_argument("--L_space",type = str, default=defaults["L_space"]) # number of different sample sizes between min and max
parser.add_argument("--p",type = int, default=defaults["p"]) # name of the demography to simulate from
parser.add_argument("--demography",type = str, default=defaults["demography"]) # name of the demography to sample from

if __name__ == "__main__":
    # Parse the command line arguments, removing the script name from the list
    args = parser.parse_args()

    # Use the parsed arguments
    output_path = args.output_path
    n_min = args.n_min
    n_max = args.n_max
    n_num = args.n_num
    n_space = args.n_space

    L_min = args.L_min
    L_max = args.L_max
    L_num = args.L_num
    L_space = args.L_space
    p = args.p
    demography = args.demography

    if n_space == "lin":
        n_list = np.linspace(n_min, n_max, n_num).astype(int)
    elif n_space == "geom":
        n_list = np.geomspace(n_min,n_max,n_num).astype(int)

    if L_space == "lin":
        L_list = np.linspace(L_min,L_max,L_num).astype(int)
    elif L_space == "geom":
        L_list = np.geomspace(L_min,L_max,L_num).astype(int)


with open(os.path.join(output_path,"genotypes.pkl"), 'rb') as file:
    genotypes = pickle.load(file)

if demography == "cont":
    with open(os.path.join(output_path,"positions.pkl"), 'rb') as file:
        positions = pickle.load(file)
else:
    positions = np.array([])

if demography == "split":
    K=2
elif demography == "SteppingStones_1d":
    K = int(output_path.split("/")[-3].split("=")[1])
elif demography == "cont":
    K = int(output_path.split("/")[-3].split(",")[0].split("=")[1])
elif demography == "SteppingStones_2d":
    K = int(output_path.split("/")[-3].split("=")[1])**2

if demography == "cont":
    spectrum_df,eigenvector_df,sample_positions = generate_eigen_df(genotypes,n_list,L_list,K,p,positions)
    with open(os.path.join(output_path,"sample_positions.pkl"),"wb") as file:
        pickle.dump(np.array(sample_positions),file)
else:
    spectrum_df,eigenvector_df = generate_eigen_df(genotypes,n_list,L_list,K,p,positions)


with open(os.path.join(output_path,"eigenvalues.pkl"),"wb") as file:
    pickle.dump(spectrum_df,file)
with open(os.path.join(output_path,"eigenvectors.pkl"),"wb") as file:
    pickle.dump(eigenvector_df,file)


end = time.perf_counter()
final_memory = process.memory_info().rss
time_elapsed = (end-start)/60
memory_used = (final_memory-initial_memory) / 1048576
print("time taken eigen = %.2f minutes" % time_elapsed) 
print("memory used eigen= %.2f megabytes" % memory_used) 