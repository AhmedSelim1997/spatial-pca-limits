#%%
import numpy as np
import msprime

def coor(index,K):
    x=index%K
    y=index//K
    return x,y

########################## Defining the demographies
########################## So far, I have 3 demographies: population split - 1d circular stepping stones - 2d stepping stones on a grid
def pop_split(N,tau,n):
    #x: N_pop1/N_t (the proportion of the total population that goes to pop 1)
    #tau: Time Since split
    #n: number of samples to draw from each population 
    demography = msprime.Demography()
    demography.add_population(name="main", initial_size= N)
    demography.add_population(name="split1", initial_size= int(0.5*N))
    demography.add_population(name="split2", initial_size= int(0.5*N))
    demography.add_population_split(time=N*tau, derived=["split1", "split2"], ancestral="main")
    samples = {"split1":n,"split2":n}
    return demography, samples

def Stepping_Stones_1d(N,K,m,n):
    #N: Number of individuals per deme
    #K: Number of demes in the chain
    #m: probability of obtaining the new individual from a nearby deme
    #n: sample size drawn from each deme
    demography = msprime.Demography.stepping_stone_model([N]*K, migration_rate= m/2)
    samples = {pop.name:n for pop in demography.populations}
    return demography, samples

def Stepping_Stones_2d(N,K,m,n):
    #N: Number of individuals per deme
    #d: Number of demes per side (total number of demes is d*d)
    #m: probability of obtaining the new individual from a nearby deme (backward migration rate)
    #n: sample size drawn from each deme
    demography = msprime.Demography.isolated_model(initial_size = [N]*(K**2))
    samples = {pop.name:n for pop in demography.populations}
    M = demography.migration_matrix
    for i in range(K**2):
        for j in range(K**2):
            x1,y1 = coor(i,K)
            x2,y2 = coor(j,K)
            if (abs(x1-x2)==1 and abs(y1-y2)==0) or (abs(y1-y2)==1 and abs(x1-x2)==0):
                M[i,j] = m/4
    return demography, samples

def mutation_function(tree,mu): ##Define mu here rather than down
    mut_tree = msprime.sim_mutations(tree,rate=mu, model ='binary',discrete_genome=False)
    return mut_tree

def Create_Genotypes(demography,samples,chrom_length,L,mu):
    #demography and samples are outputs from the stepping stones function
    #Chrom_length: set to one for simple simulations
    #S: number of unlinked snps to simulate
    #mu: mutation rate per nuceotide per generation
    ## Also calculates mean total branch length
    ts_list = msprime.sim_ancestry(
        samples = samples,
        demography = demography,
        sequence_length=chrom_length,
        num_replicates=L,
        ploidy = 1
    )
    n_tot=list(samples.values())[0]
    K=demography.num_populations
    mutations_reps = map(mutation_function, ts_list,[mu]*L)
    halotypes = []
    branch_lengths = np.zeros((L,n_tot))
    for i,ts in enumerate(mutations_reps):
        for j,n in enumerate(range(1,n_tot+1)):
            simplified_ts = ts.simplify(np.concatenate([np.arange(k*n_tot,k*n_tot+n) for k in range(K)]))
            branch_lengths[i,j] = np.mean([tree.total_branch_length for tree in simplified_ts.trees()])
        H = ts.genotype_matrix()
        p,n = H.shape
        if p==0:
            continue
        else:
            idx = np.random.choice(np.arange(p), 1)
            h = H[idx, :]
        halotypes.append(h)
    H = np.vstack(halotypes)
    genotypes = H.T
    T = np.mean(branch_lengths,axis=0)
    return genotypes,T


# %%
