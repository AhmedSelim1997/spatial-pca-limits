#%%
import numpy as np
import msprime
import pandas as pd
from demographies import Stepping_Stones_1d
from tqdm import tqdm

#%%
def Branch_Length(demography,samples,reps):
    #demography and samples are outputs from the stepping stones function
    #Chrom_length: set to one for simple simulations
    #L: number of unlinked snps to simulate
    #mu: mutation rate per nuceotide per generation
    trees = msprime.sim_ancestry(
    samples = samples,
    demography = demography,
    num_replicates=reps,
    ploidy=1
    )
    branch_lengths = []
    for tree in trees:
        sample_branch_length = []
        for sub_tree in tree.trees():
            sample_branch_length.append(sub_tree.total_branch_length)
        branch_lengths.append(np.mean(sample_branch_length))
    T = np.mean(branch_lengths)
    return T

# %%
reps = 3000
N=1000
K=5
m_list=np.round(np.geomspace(0.001,0.5,100),5)
n_list = np.linspace(2,100,50).astype(int)

df = pd.DataFrame(np.zeros((len(n_list),len(m_list))),index = n_list,columns = m_list)
for m in tqdm(m_list):
    for n in n_list:
        dem,samples = Stepping_Stones_1d(N,K,m,n)
        T = Branch_Length(dem,samples,reps=reps)
        df.loc[n,m] = T

df.to_pickle("./T_1d.pkl")
# %%
