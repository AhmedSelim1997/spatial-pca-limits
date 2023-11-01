#%% ## This script calculates the theoretical eigenvalues and eigenvectors of a:
    # 1- general demography
    # 2 - with a general sampling scheme
#%%
import msprime
import numpy as np
import concurrent.futures as future
import matplotlib.pyplot as plt
import networkx as nx
import itertools as it
import time
import os

# Get the absolute path of the current script
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)


## This is just a test demography, but should work with any general demography
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

def setup_graph(
    n_rows=5,
    n_columns=8,
    barrier_startpt=2.5,
    barrier_endpt=5.5,
    anisotropy_scaler=1.0,
    barrier_w=0.1,
    corridor_w=0.5,
    n_samples_per_node=10,
    barrier_prob=1,
    corridor_left_prob=1,
    corridor_right_prob=1,
    sample_prob=1.0,
):
    """Setup graph (triangular lattice) for simulation
    Arguments
    ---------
    n_rows : int
        number of rows in the lattice
    n_columns : int
        number of rows in the lattice
    barrier_startpt : float
        geographic position of the starting pt of the barrier from left to right
    barrier_endpt : float
        geographic position of the starting pt of the barrier from left to right
    anisotropy_scaler : float
        scaler on horizontal edge weights to create an anisotropic simulation
    barrier_w : float
        migration value for nodes in the barrier
    corridor_w : float
        migration value for nodes in the corridor
    n_samples_per_node : int
        total number of samples in an node
    barrier_prob : float
        probability of sampling an individual in the barrier
    corridor_left_prob : float
        probability of sampling a individual in the left corridor
    corridor_right_prob : float
        probability of sampling an individual in the right corridor
    sample_prob : float
        probability of sampling a node
    Returns
    -------
    tuple of graph objects
    """
    n_samples_per_node = int(n_samples_per_node/2)
    graph = nx.generators.lattice.triangular_lattice_graph(
        n_rows - 1, 2 * n_columns - 2, with_positions=True
    )
    graph = nx.convert_node_labels_to_integers(graph)
    pos_dict = nx.get_node_attributes(graph, "pos")
    for i in graph.nodes:

        # node position
        x, y = graph.nodes[i]["pos"]

        if x <= barrier_startpt:
            graph.nodes[i]["sample_size"] = 2 * np.random.binomial(
                n_samples_per_node, corridor_left_prob
            )
        elif x >= barrier_endpt:
            graph.nodes[i]["sample_size"] = 2 * np.random.binomial(
                n_samples_per_node, corridor_right_prob
            )
        else:
            graph.nodes[i]["sample_size"] = 2 * np.random.binomial(
                n_samples_per_node, barrier_prob
            )

        # sample a node or not
        graph.nodes[i]["sample_size"] = graph.nodes[i][
            "sample_size"
        ] * np.random.binomial(1, sample_prob)

    # assign edge weights
    for i, j in graph.edges():
        x = np.mean([graph.nodes[i]["pos"][0], graph.nodes[j]["pos"][0]])
        y = np.mean([graph.nodes[i]["pos"][1], graph.nodes[j]["pos"][1]])
        if x <= barrier_startpt:
            graph[i][j]["w"] = corridor_w
        elif x >= barrier_endpt:
            graph[i][j]["w"] = corridor_w
        else:
            graph[i][j]["w"] = barrier_w

        # if horizontal edge
        if graph.nodes[i]["pos"][1] == graph.nodes[j]["pos"][1]:
            graph[i][j]["w"] = anisotropy_scaler * graph[i][j]["w"]

    grid = np.array(list(pos_dict.values()))
    edge = np.array(graph.edges)
    edge += 1  # 1 indexed nodes for feems

    # create sample coordinates array
    sample_sizes_dict = nx.get_node_attributes(graph, "sample_size")
    pops = [[i] * int(sample_sizes_dict[i] / 2) for i in graph.nodes]
    pops = list(it.chain.from_iterable(pops))
    coord = grid[pops, :]
    return (graph, coord, grid, edge)



#%%
def mutation_function(tree): ##Define mu here rather than down
    mu=1e-4
    mut_tree = msprime.sim_mutations(tree,rate=mu, model ='binary',discrete_genome=False)
    return mut_tree

def calculate_coal_times(demography,n_reps,L_reps,chrom_length,mu=1e-4,ploidy=1):
    K = demography.num_populations

    ts_list = msprime.sim_ancestry(
    samples = {pop.name:n_reps for pop in demography.populations},
    demography = demography,
    sequence_length=chrom_length,
    num_replicates=L_reps,
    ploidy=ploidy
)
        
    K= len(demography.populations)
    t_same=np.zeros((K,L_reps))
    t_diff = np.zeros((K,K,L_reps))

    # with future.Executor() as executor:
    #     mutations_reps = executor.map(mutation_function, ts_list)
    mutations_reps = map(mutation_function, ts_list)
    for k,ts in enumerate(mutations_reps): ## k is a counter for the current simulation rep
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
n=5
L=500
K=8
m=0.5
thickness = 0.2
N=1
x=0.5
dem,samples = mig_barrier_graph(K,m,x,thickness,N=1,n=20)
vals,vecs = calculate_theor_spectrum(dem,sample_sizes = n,n_reps = n,L_reps=L)


#%%
path_vals = "../../data/theor_eigen/mig_barrier_theor_spectrum_grid/eigenvalues/"
path_vecs = "../../data/theor_eigen/mig_barrier_theor_spectrum_grid/eigenvectors/"
x_list = np.round(np.linspace(0.1,1,50),3)
L=5000
K=8
m=0.5
thickness = 0.2
N=1
n = 10
start_counter = 25
for x in x_list[start_counter:]:
    start_sim = time.perf_counter()
    dem,samples = mig_barrier_graph(K,m,x,thickness,N=1,n=n)
    vals,vecs = calculate_theor_spectrum(dem,sample_sizes = n,n_reps = n,L_reps=L)
    np.save(path_vals+"x=%.2f"%x,vals)
    np.save(path_vecs+"x=%.2f"%x,vecs)

    end_sim = time.perf_counter()
    print("time taken for x=%.2f is %.2f seconds" % (x,round(end_sim-start_sim, 2)))
    del vals
    del vecs

#%%
# graph, coord, grid, edge = setup_graph(n_samples_per_node=10,barrier_w=0.1)
# K = len(graph.nodes)
# mig_mat = nx.adj_matrix(graph, weight="w").toarray().tolist()
# demography = msprime.Demography.isolated_model(initial_size = [1]*K)
# demography.migration_matrix = np.array(mig_mat)

# # sample sizes per node
# sample_sizes = 10

# low_mig = calculate_theor_spectrum(demography,sample_sizes)
# np.save("low_mig",low_mig)

# #%%
# graph, coord, grid, edge = setup_graph(n_samples_per_node=10,barrier_w=0.5)
# K = len(graph.nodes)
# mig_mat = nx.adj_matrix(graph, weight="w").toarray().tolist()
# demography = msprime.Demography.isolated_model(initial_size = [1]*K)
# demography.migration_matrix = np.array(mig_mat)

# # sample sizes per node
# sample_sizes = 10

# high_mig = calculate_theor_spectrum(demography,sample_sizes)
# np.save("high_mig",high_mig)

#%%
# K=5
# m=1e-2
# n=10
# demography,samples = Stepping_Stones_2d(N,K,m,n)

# coal_times = calculate_coal_times(demography,n_reps,chrom_length,L_reps,mu,ploidy=1)
