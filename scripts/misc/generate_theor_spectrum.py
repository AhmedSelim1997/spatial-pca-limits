#%%
import os
import sys
module_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../")
sys.path.append(module_directory)
import numpy as np
import pandas as pd
import argparse
from utils import calculate_theor_spectrum
from demographies import Stepping_Stones_1d,Stepping_Stones_2d
import psutil
import time
import pickle
import os
from tqdm import tqdm

#%%
process = psutil.Process()
initial_memory = process.memory_info().rss
start = time.perf_counter()

defaults = {
    'output_path': "../../data/theoretical_spectrum/SS_1d/",
    'demography': 'SS_1d',
    'K': '5',
    'n': '20',
    'L': '5000',
    'm_min': '0.0001',
    'm_max': '0.5',
    'm_num': '100',
    'N':'1000'
}

# input_file_path = "../data/test_output/tau=1.00e-03.npy"

parser = argparse.ArgumentParser()
parser.add_argument("--output_path",type = str, default=defaults["output_path"]) # The directory to save the dataframes
parser.add_argument("--demography",type = str, default=defaults["demography"]) # name of the demography to sample from
parser.add_argument("--K",type = int, default=defaults["K"]) # number of demes
parser.add_argument("--n",type = int, default=defaults["n"]) # number of samples per deme in simulations
parser.add_argument("--L",type = int, default=defaults["L"]) # number of replicate simulations
parser.add_argument("--m_min",type = float, default=defaults["m_min"]) # minimum migration rate
parser.add_argument("--m_max",type = float, default=defaults["m_max"]) # maximum migration rate
parser.add_argument("--m_num",type = int, default=defaults["m_num"]) # number of different values for migration rate
parser.add_argument("--N",type = int, default=defaults["N"]) # number of different values for migration rate


args = parser.parse_args()
output_path = args.output_path
demography = args.demography
K=args.K
n=args.n
L=args.L
m_min=args.m_min
m_max=args.m_max
m_num=args.m_num
N=args.N

#%%
if demography == "SS_1d":
    dem_function = Stepping_Stones_1d
    p=K-1
elif demography == "SS_2d":
    dem_function = Stepping_Stones_2d
    p=K**2-1
m_list = np.round(np.geomspace(m_min,m_max,m_num),5)

#%%
vals_columns = ["m"] + [f"l_{x+1}" for x in range(p)]
vecs_columns = ["m"] + [f"v_{x+1}" for x in range(p)]
vals_df = pd.DataFrame(columns = vals_columns)
vecs_df = pd.DataFrame(columns = vecs_columns)

for m in tqdm(m_list, desc="Processing", unit="item"):
    dem,samples = dem_function(N,K,m,n)
    vals,vecs = calculate_theor_spectrum(dem,sample_sizes = n,n_reps = n,L_reps=L)
    order = np.argsort(vals)[::-1]
    vals = vals[order][:p]
    vecs = vecs[:,order][:,:p]
    vals_row = [[m] + list(vals)]
    vecs_row = np.insert(vecs,0,np.array([m]*(p+1)),axis=1)
    
    vals_row = pd.DataFrame(vals_row,columns = vals_columns)
    vecs_row = pd.DataFrame(vecs_row,columns = vecs_columns)
    vals_df = vals_df.append(vals_row,ignore_index=True)
    vecs_df = vecs_df.append(vecs_row,ignore_index=True)

with open(os.path.join(output_path,f"vals_K={K}.pkl"),"wb") as f:
    pickle.dump(vals_df,f)
with open(os.path.join(output_path,f"vecs_K={K}.pkl"),"wb") as f:
    pickle.dump(vecs_df,f)

end = time.perf_counter()
final_memory = process.memory_info().rss
time_elapsed = (end-start)/60
memory_used = (final_memory-initial_memory) / 1048576
print("time taken eigen = %.2f minutes" % time_elapsed) 
print("memory used eigen= %.2f megabytes" % memory_used) 