#%%
import os
os.chdir(os.path.dirname(os.path.abspath("__file__")))
import numpy as np
import random
from TracyWidom import TracyWidom
import pandas as pd
import math

#%%
def parse_full_output_slim(full_output,L_thinning):
    # Input: the file generate by outputFull() method in SLiM
    # Output: (nxL) Genotype matrix and (x,y) positions of all individuals in the simulation
    with open(full_output,"r") as f:
        y=f.read()
    raw_mut_data = y.split(":\n")[2].split("\n")[:-1]
    raw_positions = y.split(":\n")[3].split("\n")[:-1]
    raw_genomes = y.split(":\n")[4].split("\n")[:-1]
    N=len(raw_genomes)
    L=len(raw_mut_data)

    genomes = (np.array(genome.split("A ")[1].split()).astype(int) for genome in raw_genomes)

    genotype_matrix = np.zeros((N,L))
    positions = np.zeros((N,2))
    for i,genome in enumerate(genomes):
        genotype_matrix[i,genome] = 1
        positions[i,:] = raw_positions[i//2].split(" ")[-2:]
        positions = positions.astype(float)
    if L>L_thinning:
        genotype_matrix = genotype_matrix[:,::int(L/L_thinning)] ## Thin the genotype matrix to reduce its size
    return genotype_matrix,positions


def sample_genotype_matrix_slim(genotype_matrix,positions,n,L,K,margin=0.05):
    # Divides the landscape into KxK patches with some margin between the patches
    # Samples n random individuals from each patch
    # Samples L variance loci after sampling the individuals

    margin = margin/K ## So that grid squares are not directly adjecent 
    n_samples_indices = []

    ## Creating Grid boundaries
    left_boundaries = (np.linspace(0,1,K+1)+margin)[:-1].reshape(-1,1)
    right_boundaries = (np.linspace(0,1,K+1)-margin)[1:].reshape(-1,1)
    grid_boundaries = np.hstack((left_boundaries,right_boundaries)) ## a Kx2 matrix of the boundaries of each square patch

    ## Sampling individuals from the grid
    for i,x_boundaries in enumerate(grid_boundaries):
        for j,y_boundaries in enumerate(grid_boundaries):
            samples_in_x_boundaries = np.bitwise_and(positions[:,0] > x_boundaries[0],positions[:,0] < x_boundaries[1])
            samples_in_y_boundaries = np.bitwise_and(positions[:,1] > y_boundaries[0],positions[:,1] < y_boundaries[1])
            samples_in_boundaries = np.where(np.bitwise_and(samples_in_x_boundaries,samples_in_y_boundaries))[0]
            if len(samples_in_boundaries)>n:
                n_samples_indices = n_samples_indices + list(np.random.choice(samples_in_boundaries, size=n, replace=False))
            else: 
                n_samples_indices = n_samples_indices + list(samples_in_boundaries)
                print(f"patch ({i},{j}) has only {len(samples_in_boundaries)} individuals, while n={n}")


    sample_matrix = genotype_matrix[n_samples_indices,:]
    sample_positions = positions[n_samples_indices,:]
    variant_sites = np.where(~np.all(sample_matrix == 0, axis=0))[0]
    sample_matrix = sample_matrix[:,variant_sites]

    non_fixed_sites = np.where(~np.all(sample_matrix == 1, axis=0))[0]
    sample_matrix = sample_matrix[:,non_fixed_sites]

    ## Sampling SNPs
    L_sample_indices = random.sample(range(sample_matrix.shape[1]),L)
    sample_matrix = sample_matrix[:,L_sample_indices]

    return sample_matrix,sample_positions

def sample_genotype_matrix_msprime(genotype,n,L,K,sample_vector=[],ploidy=1):
    ## This function selects n random individuals from each subpopulation (out of a total of K subpopulations) and L loci from a genotype matrix.
    ## Assumes that the original genotype matrix has same number of individuals in each subpopulation.
    # n: number of individuals to sample from each subpopulation (rows)
    # L: number of loci to sample (columns)
    # K: Number of subpopulations
    # Sample_vector: A vector that specifies the relative sample size from each deme (multiplied later by n to give the total sample size)
    if len(sample_vector) == 0:
        sample_vector = [1]*K
    sample_vector = ((np.array(sample_vector)/max(sample_vector))*n).astype(int)
    sample_end_indices=np.cumsum(sample_vector)
    sample_end_indices = np.insert(sample_end_indices,0,0) 

    genotype = np.array(genotype)
    n_total,L_total = genotype.shape
    n_tot_per_deme = int(n_total/K)
    if n > n_tot_per_deme:
        return 0
    sample_geno = np.zeros((sample_end_indices[-1],L_total))

    ## Choosing n random individuals from each subpopulation
    for i in range(K):
        n_sample = np.array(random.sample(range(1,n_tot_per_deme), sample_vector[i])) ## accounting for biased sampling
        if len(n_sample) == 0:
            continue
        sample_geno_deme = genotype[n_tot_per_deme*i:n_tot_per_deme*(i+1),:][n_sample,:]
        sample_geno[sample_end_indices[i]:sample_end_indices[i+1],:] = sample_geno_deme

    ## Choosing SNPs that still vary for the subset of individuals   
    sample_geno = sample_geno[:,np.mean(np.array(sample_geno), axis = 0) != 0]
    if ploidy == 1:
        sample_geno = sample_geno[:,np.mean(np.array(sample_geno), axis = 0) != 1]
    elif ploidy == 2:
        sample_geno = sample_geno[:,np.mean(np.array(sample_geno), axis = 0) != 2]
    if L<np.shape(sample_geno)[1]:
        L_sample = np.array(random.sample(range(1,np.shape(sample_geno)[1]), L))
        sample_geno = sample_geno[:,L_sample]
        return sample_geno
    else:
        return 0
    
def clustered_sampling_vector(d,K,p,smaller_sample=0):
    ## This function constructs a sample vector for the relative sample sizes for each deme, by by taking samples from the two ends of the space
    # d: dimensionality of space, either 1 or 2
    # K: number of demes (or number of demes per row in the case of 2d)
    # p: number of demes (or columns of demes) to cluster from each end
    # smaller_sample: the relative number of samples to be taken from the middle demes
    if d == 1:
        sample_vector = np.full((1,K),smaller_sample)[0,:]
        mid_sample = math.ceil(K/2)
        full_sample_indices = list(range(0,p)) + list(range(mid_sample,mid_sample+p))
        sample_vector[full_sample_indices]=1
    elif d==2:
        sample_vector = np.full((1,K**2),smaller_sample)[0,:]
        full_sample_indices = []
        for i in range(p):
            full_sample_indices = full_sample_indices + list(np.linspace(i,K**2-(K-i),K))
            full_sample_indices = full_sample_indices + list(np.linspace(K-i-1,K**2-i-1,K))
        full_sample_indices = np.array(full_sample_indices).astype(int)
        sample_vector[full_sample_indices]=1
    return sample_vector

def standardize(matrix):
    matrix = matrix - np.mean(matrix,axis=0)
    matrix = np.divide(matrix,np.std(matrix,axis=0))
    return matrix

tw = TracyWidom(beta=1)
def TWp(eigenvalue,n,L,r):
    # eigenvalue: The raw eigenvalue
    # n: the total number of individuals sampled
    # L: the number of SNPs sampled
    # r: the rank of the eigenvalue (pth largest eigenvalue)
    if n>L:
        return np.nan

    try:
        n=n-1-r ##actual sample size is n-1
        gamma = n/L
        mean = (1+np.sqrt(gamma))**2
        std = ((1+np.sqrt(gamma))**(4/3))*gamma**(-1/6)
        l = (eigenvalue-mean)/std
        p = 1-tw.cdf(l)
        return p
    except:
        return np.nan
    
def spectrum_with_largest_vector(geno_sample,q):
    ## For a given genotype matrix, calculate the spectrum and largest p eigenvectors for a subsample of n individuals per subpopulation and L SNPs
    stand_sample = standardize(geno_sample)
    n,L = geno_sample.shape
    cov = 1/L*np.matmul(stand_sample,stand_sample.T)
    vals,vecs = np.linalg.eig(cov)
    order = np.argsort(vals)[::-1]
    largest_vals = vals[order[:q]]
    largest_vecs = vecs[:,order[:q]]
    return largest_vals,largest_vecs  

def generate_eigen_df(genotype_matrix,n_list,L_list,K,q,positions = np.array([])):
    ## For each n and L, calculate the first q eigenvalues and their respective p-value on a TW distribution
    eigenvalue_columns = ["n","L"] + ["l_%d"%(i+1) for i in range(q)] + ["p_%d"%(i+1) for i in range(q)]
    eigenvector_columns = ["n","L"] + ["v_%d"%(i+1) for i in range(q)]
    eigenvalue_df = pd.DataFrame(columns = eigenvalue_columns)
    eigenvectors_df = pd.DataFrame(columns = eigenvector_columns)
    if positions.shape != (0,):
        all_sample_positions = np.zeros((len(n_list),len(L_list),max(n_list)*K**2,2))
    for i,n in enumerate(n_list):
        for j,L in enumerate(L_list):
            if n < L:
                try:
                    if positions.shape == (0,): ## Positions will not be declared for msprime simulations
                        geno_sample = sample_genotype_matrix_msprime(genotype_matrix,n,L,K)
                    else: ## Positions will be declared, and thus will be an input for SLiM simulations
                        geno_sample,sample_positions = sample_genotype_matrix_slim(genotype_matrix,positions,n,L,K,margin=0.05)
                        all_sample_positions[i,j,:len(sample_positions),:] = sample_positions
                    vals,vecs = spectrum_with_largest_vector(geno_sample,q)
                    n_tot,L = geno_sample.shape
                    vals = np.round(vals,4)
                    p_values = []
                    for r,val in enumerate(vals):
                        p_values.append(TWp(val,n_tot,L,r))
                    p_values = np.round(-np.log10(np.array(p_values)+1e-10),5)
                    record = [int(n),int(L)] + list(vals) + list(p_values)
                    eigenvalue_df.loc[len(eigenvalue_df.index)] = record

                    temp_df = pd.DataFrame(np.hstack((np.ones((n_tot,1),int)*n,np.hstack((np.ones((n_tot,1),int)*L,vecs)))),columns = eigenvector_columns)
                    eigenvectors_df = pd.concat([eigenvectors_df, temp_df], ignore_index=True)
                except:
                    continue

    eigenvalue_df = eigenvalue_df.applymap(np.real)
    eigenvectors_df = eigenvectors_df.applymap(np.real)
    if positions.shape == (0,): 
        return eigenvalue_df,eigenvectors_df
    else:
        return eigenvalue_df,eigenvectors_df,all_sample_positions
# %%
