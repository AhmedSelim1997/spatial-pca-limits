#%%
import numpy as np
import msprime
import time
import os

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)


def coor(index,K):
    x=index%K
    y=index//K
    return x,y

def mig_barrier_graph(K,m,x,thickness,N,n):
    # The landscape will be a K x K square grid of demes exchanging migrants at rate:
        # 1- m on the sides
        # 2- m*x in the middle (the migration barrier)
    # thickness is the thickness of the barrier as a proportion of the size of the landscape
    demography = msprime.Demography.isolated_model(initial_size = [N]*(K**2))
    samples = {pop.name:n for pop in demography.populations}
    M = demography.migration_matrix
    a = np.floor(0.5*K*(1-thickness))
    b = K-a
    barrier_indices = list(range(int(a),int(b)))
    for i in range(K**2):
        for j in range(K**2):
            x1,y1 = coor(i,K)
            x2,y2 = coor(j,K)
            if (abs(x1-x2)==1 and abs(y1-y2)==0) or (abs(y1-y2)==1 and abs(x1-x2)==0):
                if (x1 in barrier_indices) or (x2 in barrier_indices):
                    M[i,j] = x*m/4
                else:
                    M[i,j] = m/4
    return demography,samples


def mutation_function(tree,mu): ##Define mu here rather than down
    mut_tree = msprime.sim_mutations(tree,rate=mu, model ='binary',discrete_genome=False)
    return mut_tree


def Create_Genotypes(demography,samples,L,chrom_length=1,mu=2e-3):
    #demography and samples are outputs from the stepping stones function
    #Chrom_length: set to one for simple simulations
    #S: number of unlinked snps to simulate
    #mu: mutation rate per nuceotide per generation
    trees = msprime.sim_ancestry(
        samples = samples,
        demography = demography,
        sequence_length=chrom_length,
        num_replicates=L,
        ploidy = 1
    )
    mutations_reps = map(mutation_function, trees,[mu]*L)
    halotypes = []
    for tree in mutations_reps:
        H = tree.genotype_matrix()
        p,n = H.shape
        if p==0:
            continue
        else:
            idx = np.random.choice(np.arange(p), 1)
            h = H[idx, :]
        halotypes.append(h)
    H = np.vstack(halotypes)
    genotypes = H.T

    return genotypes

# %%
os.chdir("../simulations/")
from fast_spatial_caol import accelerate_demography
n=10
K=8
m=0.5
x=1.0
thickness = 0.2
N=500
L=1500
chrom_length=100
mu=1e-4
dem,samples = mig_barrier_graph(K,m,x,thickness,N,n)
dem,samples = accelerate_demography(dem,samples,alpha=2)
geno = Create_Genotypes(dem,samples,L)

#%%
x_list = np.round(np.linspace(0.1,1,50),3)
n=20
L=10000
K=8
m=0.5
thickness = 0.2
N=500
start_counter = 0
for x in x_list[start_counter:]:
    start_sim = time.perf_counter()
    dem,samples = mig_barrier_graph(K,m,x,thickness,N,n)
    dem,samples = accelerate_demography(dem,samples,alpha=3)
    geno = Create_Genotypes(dem,samples,L)
    end_sim = time.perf_counter()
    print("time taken for x=%.2f is %.2f seconds" % (x,round(end_sim-start_sim, 2)))
    np.save("../../data/sim_output/mig_barrier_grid_fast/x=%.2f"%x,geno)
    del geno

# %%
