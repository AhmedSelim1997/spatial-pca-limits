#%%
from feems.utils import prepare_graph_inputs
from feems import SpatialGraph, Viz
import numpy as np
import networkx as nx
import itertools as it
import matplotlib.pyplot as plt
from feems.cross_validation import run_cv
import random
import os
import time
# Get the absolute path of the current script
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)


#%%
def select_sample_demes(genotype,n,S,K,ploidy=1):
    genotype = np.array(genotype)
    n_total,S_total = genotype.shape
    n_tot_per_deme = int(n_total/K)
    if n > n_tot_per_deme:
        return 0
    sample_geno = np.zeros((n*K,S_total))
    for i in range(K):
        n_sample = np.array(random.sample(range(1,n_tot_per_deme), n))
        sample_geno_deme = genotype[n_tot_per_deme*i:n_tot_per_deme*(i+1),:][n_sample,:]
        sample_geno[n*i:n*(i+1),:] = sample_geno_deme

    ##Choosing SNPs that still vary for the subset of individuals   
    sample_geno = sample_geno[:,np.mean(np.array(sample_geno), axis = 0) != 0]
    if ploidy == 1:
        sample_geno = sample_geno[:,np.mean(np.array(sample_geno), axis = 0) != 1]
    elif ploidy == 2:
        sample_geno = sample_geno[:,np.mean(np.array(sample_geno), axis = 0) != 2]
    if S<np.shape(sample_geno)[1]:
        S_sample = np.array(random.sample(range(1,np.shape(sample_geno)[1]), S))
        sample_geno = sample_geno[:,S_sample]
        return sample_geno
    else:
        return 0

def setup_graph(
    n_rows=8,
    n_columns=8,
    barrier_startpt=3.5,
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


# %%
def x_estimate_matrix(geno,K=64,n_list = np.arange(2,21,2).astype(int),L_list = np.geomspace(400,8000,10).astype(int),strictness=0.5):
    ## Outputs a 3d matrix with:
        # rows: number of samples
        # columns: number of SNPs
        # layers:
            #layer 1: x estimate
            #layer 2: strict x estiamte (defined as the ratio between the smallest and largest 25% migration rates)
            #layer 3: optimal lambda
    ## Strictness parameter: take the ratio between the highest and lowest (strictness)% of edge values
    n_max = 2*np.shape(geno)[0]/K
    graph, coord, grid, edges = setup_graph(n_samples_per_node=n_max,barrier_w=0.1)
    output_matrix = np.full((len(n_list),len(L_list),3),np.nan)
    edges_theor = []
    for i,j in graph.edges():
        edges_theor.append(graph[i][j]["w"])
    edges_theor = np.array(edges_theor)
    barrier_indices = np.where(edges_theor==0.1)
    corr_indices = np.where(edges_theor==0.5)
    num_barrier_edges = int(len(barrier_indices[0])*strictness)
    num_corr_edges = int(len(corr_indices[0])*strictness)

    for i,n in enumerate(n_list):
        for j,L in enumerate(L_list):
            if K*n>L: ## More samples than snps
                continue
            try:
                sample_geno = select_sample_demes(geno,n,L,K)
            except:
                continue
            if len(np.shape(sample_geno)) == 0:
                continue
            graph, coord, grid, edges = setup_graph(n_samples_per_node=n*2,barrier_w=0.1)
            sp_graph_sample = SpatialGraph(sample_geno, coord, grid, edges, scale_snps=True)
            # run cross-validation
            try:
                lamb_grid = np.geomspace(1e-6, 1e3, 20)[::-1]
                cv_err = run_cv(sp_graph_sample, lamb_grid, n_folds=sp_graph_sample.n_observed_nodes, factr=1e10,outer_verbose=False)
                mean_cv_err = np.mean(cv_err, axis=0)
                lamb = float(lamb_grid[np.argmin(mean_cv_err)])
                # Fitting the Graph
                sp_graph_sample.fit(lamb = lamb)
            except:
                try:
                    lamb = 10
                    sp_graph_sample.fit(lamb = 10.0)
                except:
                    continue

            edges_estimate = sp_graph_sample.w
            barrier_estimate = edges_estimate[barrier_indices]
            corr_estimate = edges_estimate[corr_indices]
            x_estimate = np.mean(barrier_estimate)/np.mean(corr_estimate)
            x_estimate_strict = np.mean(np.sort(barrier_estimate)[:num_barrier_edges])/np.mean(np.sort(corr_estimate)[::-1][:num_corr_edges])

            output_matrix[i,j,0] = x_estimate
            output_matrix[i,j,1] = x_estimate_strict
            output_matrix[i,j,2] = lamb

    return output_matrix


# %%
path = "../../data/sim_output/mig_barrier_grid/"
path_out = "../../data/misc/square_grid_x_estimates/"
files = os.listdir(path)
start_counter = 2
for file in files[start_counter:]:
    start = time.perf_counter()
    geno = np.load(path+file)
    output = x_estimate_matrix(geno)
    end = time.perf_counter()
    x = float(file[2:-4])
    np.save(path_out+file[:-4],output)
    print("time taken for %.2f is %.2f seconds" % (x,round(end-start, 2)))
    del output
    



# # %%
# graph, coord, grid, edges = setup_graph(n_samples_per_node=40,barrier_w=0.1)
# geno = np.load("../../data/sim_output/mig_barrier_1/x=0.10.npy")
# sp_graph = SpatialGraph(geno, coord, grid, edges, scale_snps=True)

# # Lambda CV
# # reverse the order of lambdas and alphas for warmstart
# lamb_grid = np.geomspace(1e-6, 1e3, 20)[::-1]

# # run cross-validation
# cv_err = run_cv(sp_graph, lamb_grid, n_folds=sp_graph.n_observed_nodes, factr=1e10)

# # average over folds
# mean_cv_err = np.mean(cv_err, axis=0)

# # argmin of cv error
# lamb_cv = float(lamb_grid[np.argmin(mean_cv_err)])
# sp_graph.fit(lamb = lamb_cv)
# fig = plt.figure(dpi=300)
# ax = fig.add_subplot(1, 1, 1)  
# v = Viz(ax, sp_graph, edge_width=.5, 
#         edge_alpha=1, edge_zorder=100, sample_pt_size=20, 
#         obs_node_size=7.5, sample_pt_color="black", 
#         cbar_font_size=10)
# v.draw_edges(use_weights=True)
# v.draw_obs_nodes(use_ids=False) 
# v.draw_edge_colorbar()
# edges_theor = []
# for i,j in graph.edges():
#     edges_theor.append(graph[i][j]["w"])
# edges_theor = np.array(edges_theor)
# barrier_indices = np.where(edges_theor==0.1)
# corr_indices = np.where(edges_theor==0.5)

# edges_estimate = sp_graph.w
# barrier_estimate = edges_estimate[barrier_indices]
# corr_estimate = edges_estimate[corr_indices]

# x_estimate = np.mean(barrier_estimate)/np.mean(corr_estimate)

# num_barrier_edges = int(len(barrier_indices[0])*0.25)
# num_corr_edges = int(len(corr_indices[0])*0.25)

# x_estimate_strict = np.mean(np.sort(barrier_estimate)[:num_barrier_edges])/np.mean(np.sort(corr_estimate)[::-1][:num_corr_edges])

# %%
