#%%
import os
import sys
module_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../")
sys.path.append(module_directory)
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
from utils import parse_full_output_slim
from utils import standardize
from utils import TWp

#%%
parser = argparse.ArgumentParser()
parser.add_argument("--output_path",type = str)
parser.add_argument("--L_thinning",type = int)
args = parser.parse_args()
output_path = args.output_path
L_thinning = args.L_thinning
#%%
full_output_path = os.path.join(output_path,"full.txt")
genotypes, positions = parse_full_output_slim(full_output_path,L_thinning)

#%%
n,L = genotypes.shape
geno_stand = standardize(genotypes)
cov = 1/L*np.matmul(geno_stand,geno_stand.T)
vals,vecs = np.linalg.eigh(cov)
order = np.argsort(vals)[::-1]
vals = vals[order]
p_values = [TWp(val,n,L,i) for i,val in enumerate(vals)]
vecs = vecs[:,order]
#%%
fig,ax = plt.subplots(1,2,figsize = (10,5))
ax[0].scatter(range(len(vals)),vals)
ax[1].scatter(range(len(p_values)),-np.log10(p_values))

ax[0].set_title("eigenvalues")
ax[1].set_title("-log10(p_values)")
fig.savefig(os.path.join(output_path,"vals_full_data.png"))
#%%
fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(vecs[:,0],vecs[:,1])
fig.savefig(os.path.join(output_path,"vecs_full_data.png"))
#%%
fig,axes=plt.subplots(2,2,figsize = (10,10),sharex=True,sharey=True)
for i in range(2):
    for j in range(2):
        ax = axes[i,j]
        idx = ((i+1)//2)*2 + j
        ax.scatter(positions[:,0],positions[:,1],c = vecs[:,idx],cmap="seismic")
        ax.set_title(f"PC{idx+1} map")
fig.savefig(os.path.join(output_path,"PC_maps_full_data.png"))
#%%
with open(os.path.join(output_path,"positions.pkl"),"wb") as file:
    pickle.dump(np.array(positions),file)
with open(os.path.join(output_path,"genotypes.pkl"),"wb") as file:
    pickle.dump(np.array(genotypes),file)
# %%
