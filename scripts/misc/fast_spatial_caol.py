#%%
import numpy as np
import msprime
import time

def coor(index,K):
    x=index%K
    y=index//K
    return x,y

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

def accelerate_demography(demography,samples,alpha = 3):
    M = demography.migration_matrix
    K = demography.num_populations
    # Calculating the time beyond which switch to a panmictic population
    # This time is derived from the second eigenvalue of the markov chain that defines the migration
    ## Adjust the migration matrix by adding diagonal elements that will cause the rows to sum to 1 (to become a proper markov chain)
    M_adjusted = M+np.diag(1-np.sum(M,axis =1))
    z = np.sort(np.real(np.linalg.eigvals(M_adjusted)))[-2]
    t_scatter = int(-alpha/(np.log10(z)))
    # t_scatter = 100

    # Add a subpopulation to move all demes into beyond t_scatter, with total size = sum of sizes of subpopulations
    Ntot = int(np.sum([population.initial_size for population in demography.populations]))
    demography.add_population(name = "panmictic",initial_size=Ntot)
    samples["panmictic"] = 0 #This subpopulation should have zero samples
    M = np.pad(M,((0,1),(0,1)),mode='constant') # Specifies no migraiton between the panmictic population and other populations

    for i in range(K):
        demography.add_mass_migration(time = t_scatter,source = i,dest = "panmictic",proportion=1.0)    
    demography.add_symmetric_migration_rate_change(time = t_scatter, populations = list(range(K)),rate = 0)
    return demography,samples


# %%
# N=500
# K=5
# m=0.1
# n=10
# demography,samples = Stepping_Stones_2d(N,K,m,n)
# faster_demography,samples2 = accelerate_demography(demography,samples)
# %%
