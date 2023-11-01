#%%
## Code for "setup_graph" function copied from feems.sim.py

import itertools as it
import networkx as nx
import numpy as np
import msprime
import time
import warnings
warnings.filterwarnings("ignore")

def mig_barr_tri_graph(K,m,x,thickness,N,n):
    demography = msprime.Demography.isolated_model(initial_size = [N]*(K**2))
    samples = {pop.name:n for pop in demography.populations}
    M = demography.migration_matrix
    graph = nx.generators.triangular_lattice_graph(K-1,2*K-2)
    a = np.floor(0.5*K*(1-thickness))
    b = K-a
    barrier_indices = list(range(int(a),int(b)))
    for i,j in graph.edges:
        x1 = i[0]
        x2 = j[0]
        if (x1 in barrier_indices) or (x2 in barrier_indices):
            graph[i][j]["w"] = x*m/4
        else:
            graph[i][j]["w"] = m/4

def setup_graph(
    n_rows=8,
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

def mutation_function(tree,mu): ##Define mu here rather than down
    mut_tree = msprime.sim_mutations(tree,rate=mu, model ='binary',discrete_genome=False)
    return mut_tree

def simulate_genotypes(
    graph, chrom_length=1, mu=2e-3, n_e=1, L=1000, n_samples=10
):
    # number of nodes
    K = len(graph.nodes)
    mig_mat = nx.adj_matrix(graph, weight="w").toarray().tolist()
    demography = msprime.Demography.isolated_model(initial_size = [n_e]*K)
    demography.migration_matrix = np.array(mig_mat)

    # sample sizes per node
    sample_sizes = {pop.name:n_samples for pop in demography.populations}
    

    # tree sequences
    ts = msprime.sim_ancestry(
        samples=sample_sizes,
        demography = demography,
        sequence_length=chrom_length,
        num_replicates=L,
        ploidy = 1
    )

    halotypes = []
    # with ProcessPoolExecutor() as exectuor:
    #     mutations_reps = exectuor.map(mutation_function, ts,[mu]*L)
    mutations_reps = map(mutation_function, ts,[mu]*L)
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

#%%
# graph, coord, grid, edge = setup_graph(corridor_w=0.2/4,barrier_w = (0.2/4)*0.1)
# geno = simulate_genotypes(
#     graph, chrom_length=1, mu=2e-3, n_e=1, L=1000, n_samples=10
# )
#%%
x_list = np.round(np.linspace(0.1,1,50),3)
n=20
L=10000
m=0.5
N=1
start_counter = 0
for x in x_list[start_counter:]:
    start_sim = time.perf_counter()
    graph, coord, grid, edge = setup_graph(corridor_w=m/4,barrier_w = (m/4)*x)
    geno = simulate_genotypes(
    graph, chrom_length=1, mu=2e-3, n_e=1, L=L, n_samples=n
)
    end_sim = time.perf_counter()
    print("time taken for x=%.2f is %.2f seconds" % (x,round(end_sim-start_sim, 2)))
    np.save("../../data/sim_output/mig_barrier_feems_grid/x=%.2f"%x,geno)
    del geno

# %%
