#%%
import os
os.chdir(os.path.dirname(os.path.abspath("__file__")))
import numpy as np
import random
from TracyWidom import TracyWidom
import pandas as pd
import math
import msprime

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
    # matrix = np.divide(matrix,np.std(matrix,axis=0))
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



## Calculating theoretical eigenvalues and eigenvectors
def calculate_coal_times(demography,n_reps,L_reps,ploidy=1):
    K = demography.num_populations

    ts_list = msprime.sim_ancestry(
    samples = {pop.name:n_reps for pop in demography.populations},
    demography = demography,
    num_replicates=L_reps,
    ploidy=ploidy
)
        
    K= len(demography.populations)
    t_same=np.zeros((K,L_reps))
    t_diff = np.zeros((K,K,L_reps))

    # with future.Executor() as executor:
    #     mutations_reps = executor.map(mutation_function, ts_list)
    for k,ts in enumerate(ts_list): ## k is a counter for the current simulation rep
        for i in range(K): ## i is a counter for subpopulation 1
            t_same[i,k] = ts.diversity(sample_sets = ts.samples(population=i),mode = "branch")/2 ## we divide by 2 because we want the coalescent time not the total branch legnth
            for j in range(i,K): ## j is a counter for subpopulation 2
                t_diff[i,j,k] = ts.divergence(sample_sets = [ts.samples(population=i),ts.samples(population=j)],mode = "branch")/2
                t_diff[j,i,k] = ts.divergence(sample_sets = [ts.samples(population=i),ts.samples(population=j)],mode = "branch")/2

    t_same_mean = np.mean(t_same,axis=1)
    t_diff_mean = np.mean(t_diff,axis=2)
    for i in range(K):
        t_diff_mean[i,i] = t_same_mean[i]
    return t_diff_mean


def calculate_theor_spectrum(demography,sample_sizes,n_reps=10,L_reps=5000,ploidy=1):
    # demography: THe demographic history that gave rise to the sample set, as an msprime demography object
    # sample_sizes: number of samples obtained from each subpopulation (biased sampling will distort the theoretical PC plot)
    # This function performs 2 steps:
        # 1- calculates pairwise coalescent times between individuals drawn from eahc pair of subpopulations (through replicated msprime simulations)
        # 2- uses these coal times and McVeans equation to build a theoretical covariance matrix, whose spectrum is calculated
    # n_reps: number of samples in each subpopulation, used for simulation pruposes only
    # L_reps: number of repeated simulations to perform, to get a good average of coal times

    ## 1- calculating tij matrix through simulations
    coal_times = calculate_coal_times(demography,n_reps,L_reps,chrom_length=100,ploidy=1)
    K = demography.num_populations
    if type(sample_sizes)== int: ## The case where the same number of samples is drawn from each subpopulation to form the covariance matrix
        expanded_caol_times = np.kron(coal_times,np.ones((n_reps,n_reps)))  ## expanding the matrix of coal times for more than one sample per subpopulation
        expanded_caol_times = expanded_caol_times  - np.diag(np.diag(expanded_caol_times))
        t_ave = np.mean(expanded_caol_times[expanded_caol_times!=0]) ## average pairwise caol times between any two samples ## forming a matrix of all elements equal to t_ave
        theor_cov = t_ave - expanded_caol_times
        sample_sizes = [sample_sizes]*K

    else:
        assert coal_times.shape[0] == len(sample_sizes)
        tot_sample_size = sum(sample_sizes)
        t_ave_subpop = [] # ith element is the average coal time between any individual from subpop i and any other individual
        for i in range(K):
            weighted_ave_vector =[]
            for j,n in enumerate(sample_sizes):
                if i == j: ## This part is to set self coal time to zero, in order not to inflate the average
                    temp_vec = [coal_times[i,j]]*n
                    temp_vec[0] = 0
                    weighted_ave_vector = weighted_ave_vector+temp_vec
                else:
                    weighted_ave_vector = weighted_ave_vector+[coal_times[i,j]]*n
            t_ave_subpop.append(np.mean(weighted_ave_vector))
        t_ave = np.mean(t_ave_subpop)
## 2- Calculatung theoretical covariance matrix using pairwise coalescent times

        theor_cov = np.zeros((tot_sample_size,tot_sample_size))
        subpop_start_indices = [0] + list(np.cumsum(sample_sizes))
        for i in range(K):
            for j in range(i,K):
                M = t_ave_subpop[i] + t_ave_subpop[j] - t_ave - coal_times[i,j]
                temp_matrix = np.ones((sample_sizes[i],sample_sizes[j]))*M
                if i == j: ## This part is to set self coal time to zero
                    temp_matrix = temp_matrix + np.diag([coal_times[i,i]]*sample_sizes[i])
                theor_cov[subpop_start_indices[i]:subpop_start_indices[i+1],subpop_start_indices[j]:subpop_start_indices[j+1]] = temp_matrix
                theor_cov[subpop_start_indices[j]:subpop_start_indices[j+1],subpop_start_indices[i]:subpop_start_indices[i+1]] = temp_matrix.T

    
    vals,vecs = np.linalg.eig(theor_cov)
    # vals = vals/t_ave
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[order]

    sample_sizes_sum = [0] + list(np.cumsum(sample_sizes))
    vec_ave = np.vstack([np.mean(vecs[sample_sizes_sum[i]:sample_sizes_sum[i+1],:],axis=0) for i in range(K)])

    return vals,np.real(vec_ave)

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