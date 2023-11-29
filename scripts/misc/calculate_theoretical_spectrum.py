#%% ## This script calculates the theoretical eigenvalues and eigenvectors of a:
    # 1- general demography
    # 2 - with a general sampling scheme
#%%
import msprime
import numpy as np
import matplotlib.pyplot as plt
module_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../")
sys.path.append(module_directory)
from demographies import Stepping_Stones_1d
#%%
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

    if type(sample_sizes)== int: ## The case where the same number of samples is drawn from each subpopulation to form the covariance matrix
        expanded_caol_times = np.kron(coal_times,np.ones((n_reps,n_reps)))  ## expanding the matrix of coal times for more than one sample per subpopulation
        expanded_caol_times = expanded_caol_times  - np.diag(np.diag(expanded_caol_times))
        t_ave = np.mean(expanded_caol_times[expanded_caol_times!=0]) ## average pairwise caol times between any two samples ## forming a matrix of all elements equal to t_ave
        theor_cov = t_ave - expanded_caol_times

    else:
        assert coal_times.shape[0] == len(sample_sizes)
        K = len(sample_sizes)
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
    vals = vals/t_ave

    return vals,vecs


#%%
n=20
L=5000
K=5
m=0.01
N=1000
dem,samples = Stepping_Stones_1d(N,K,m,n)
vals,vecs = calculate_theor_spectrum(dem,sample_sizes = n,n_reps = n,L_reps=L)


