#%%
import os
import sys
import time
import psutil
import argparse
import pickle
import numpy as np
module_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../")
sys.path.append(module_directory)
from utils import clustered_sampling_vector
from utils import sample_genotype_matrix_msprime
from utils import standardize

#%%
defaults = {
    'demography': "SteppingStones_1d",
    'K': 5,
    'x_num': 20,
    'n_min': 2,
    'n_max': 100,
    'n_num': 50,
    'n_space': 'lin',
    'a_num': 3,
    'L': 5000
}
parser = argparse.ArgumentParser()
parser.add_argument("--demography",type = str, default=defaults["demography"])
parser.add_argument("--K",type = int, default=defaults["K"])
parser.add_argument("--x_num",type = int, default=defaults["x_num"])
parser.add_argument("--n_min",type = int, default=defaults["n_min"])
parser.add_argument("--n_max",type = int, default=defaults["n_max"])
parser.add_argument("--n_num",type = int, default=defaults["n_num"])
parser.add_argument("--n_space",type = str, default=defaults["n_space"])
parser.add_argument("--a_num",type = int, default=defaults["a_num"])
parser.add_argument("--L",type = int, default=defaults["L"])

args = parser.parse_args()
demography = args.demography
K = args.K
x_num = args.x_num
n_min = args.n_min
n_max = args.n_max
n_num = args.n_num
n_space = args.n_space
a_num = args.a_num
L = args.L
x_list = np.geomspace(0.01,1,x_num)
a_list = np.arange(1,a_num+1)
if n_space == "lin":
    n_list = np.linspace(n_min, n_max, n_num).astype(int)
elif n_space == "geom":
    n_list = np.geomspace(n_min,n_max,n_num).astype(int)
if demography == "SteppingStones_1d":
    d=1
elif demography == "SteppingStones_2d":
    d=2
num_vals = 20
m_list = os.listdir(f"../../data/SS_{d}d/K={K}")

output_path = f"../../data/biased_samples/d={d}/K={K}"
if not os.path.isdir(output_path):
    os.makedirs(output_path)

#%%
# output_path = "../../data/test/"
# d = 1
# K = 5
# x_num = 20
# a_num = 2
# n_max = 100
# n_list = np.linspace(2,100,3)
# n_num = 3
# x_list = np.geomspace(0.01,1,x_num)
# a_list = np.arange(1,a_num+1)
# L=3000
# num_vals = 20

#%%
process = psutil.Process()
initial_memory = process.memory_info().rss
start = time.perf_counter()

for m in m_list:
    with open(f"../../data/SS_{d}d/K={K}/{m}/genotypes.pkl","rb") as file:
        geno = pickle.load(file)
    eigenvalue_matrix = np.zeros((n_num,x_num,a_num,n_max*K))
    for i,n in enumerate(n_list):
        for j,x in enumerate(x_list):
            for k,a in enumerate(a_list):
                try:
                    sample_vector = clustered_sampling_vector(d,K,a,smaller_sample=x)
                    geno_sample = sample_genotype_matrix_msprime(geno,n,L,K**d,sample_vector=sample_vector,ploidy=1)
                    if len(np.shape(geno_sample)) != 0:
                        stand_sample = standardize(geno_sample)
                        cov = (1/L)*np.matmul(stand_sample,stand_sample.T)
                        vals = np.linalg.eigvalsh(cov)
                        vals = np.sort(vals)[::-1][:num_vals]
                        eigenvalue_matrix[i,j,k,:min(len(vals),num_vals)] = vals
                        with open(os.path.join(output_path,m),"wb") as file:
                            pickle.dump(eigenvalue_matrix,file)
                except:
                    continue

end = time.perf_counter()
final_memory = process.memory_info().rss
time_elapsed = (end-start)/60
memory_used = (final_memory-initial_memory) / 1048576
print("time taken eigen = %.2f minutes" % time_elapsed) 
print("memory used eigen= %.2f megabytes" % memory_used) 

# %%
