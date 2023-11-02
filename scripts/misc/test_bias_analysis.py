#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import sys 
sys.path.append('../code/')
from eigenfunctions import clustered_sampling_vector
from eigenfunctions import select_sample
from eigenfunctions import standerdize

# %%
def explore_biased_sampling(d,K,m,n,L,x,a):
    geno = np.load(f"../data/eigen_analysis/{d}d_SS/K={K}/m={m}/genotypes.npy")
    sample_vector = clustered_sampling_vector(d,K,a,smaller_sample=x)
    geno_sample = select_sample(geno,n,L,K**d,sample_vector=sample_vector,ploidy=1)
    stand_sample = standerdize(geno_sample)
    cov = (1/L)*np.matmul(stand_sample,stand_sample.T)
    vals,vecs = np.linalg.eig(cov)
    order = np.argsort(vals)[::-1]
    vals=vals[order]
    vecs = vecs[:,order]
    counts = (np.array(sample_vector)*n).astype(int)
    name = "hsv"
    cmap = plt.get_cmap(name)
    inx = np.linspace(0,255,K**d).astype(int)
    colors = cmap([inx])[0,:,:3]
    colors_repeated = np.repeat(colors,counts,axis=0)
    populations = [f"pop_{k}" for k in range(K**d)]

    fig,ax = plt.subplots(1,3,figsize = (22,6))
    ax[0].scatter(range(len(vals)),vals,marker='.')
    ax[1].scatter(vecs[:,0],vecs[:,1],c=colors_repeated)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=7) for c in colors]
    ax[1].legend(handles, populations,fontsize="x-small",bbox_to_anchor=(1.15,0.5),loc="center right")

    if d==1:
        ax[1].set_xlim(-0.25, 0.25)
        ax[1].set_ylim(-0.25, 0.25)
    elif d==2:
        ax[1].set_xlim(-0.15, 0.15)
        ax[1].set_ylim(-0.15, 0.15)


    # Drawing the theoretical SS model

    cmap = plt.get_cmap("Wistia")  # Define the colormap
    if d==1:
        radius_large = 1  # Radius of the large circle
        theta = np.linspace(0, 2 * np.pi, K + 1)[:-1]  # Equally spaced angles
        # Plot the large circle
        large_circle = plt.Circle((0, 0), radius_large, color='skyblue', fill=False, linewidth=2)
        ax[2].add_artist(large_circle)
        # Plot the smaller circles
        for i, angle in enumerate(theta):
            x = radius_large * np.cos(angle)
            y = radius_large * np.sin(angle)
            radius_small = 0.15  # Radius of the smaller circle
            # Vary the color based on a number ranging from 0 to 1
            color_value = sample_vector[i]
            color = cmap(color_value)
            small_circle = plt.Circle((x, y), radius_small, color=color)
            ax[2].add_artist(small_circle)
            # Annotate the circles with numbers
            ax[2].annotate(i, (x, y), color='black', ha='center', va='center', fontsize=12, weight='bold')

        # Set equal aspect ratio
        ax[2].set_aspect('equal', adjustable='box')

        # Set the limits
        ax[2].set_xlim(-1.2, 1.2)
        ax[2].set_ylim(-1.2, 1.2)

        # Set the title
        plt.title('1d SS with biased sampling', fontsize=16, weight='bold')

        # Remove the axes
        ax[2].axis('off')

        # Display the plot
        plt.show()

        print(sum(counts)/K)

    elif d==2:
        # Create the circles
        for i in range(K):
            for j in range(K):
                color_value = sample_vector[K * i + j]
                color = cmap(color_value)
                circle = plt.Circle((j, K - i - 1), 0.3, color=color)  # Swap i and j to rotate 90 degrees
                ax[2].add_artist(circle)

        # Connect the circles with lines
        for i in range(K):
            for j in range(K):
                if j < K - 1:  # Connect with the right neighbor
                    ax[2].plot([j + 0.3, j + 1 - 0.3], [K - i - 1, K - i - 1], 'b-')  # Swap i and j to rotate 90 degrees
                if i < K - 1:  # Connect with the lower neighbor
                    ax[2].plot([j, j], [K - i - 1 - 0.3, K - i - 2 + 0.3], 'b-')  # Swap i and j to rotate 90 degrees

        # Set the aspect of the plot to be equal, so the circles are circular
        ax[2].set_aspect('equal', adjustable='box')

        # Set the limits of the plot
        ax[2].set_xlim(-1, K)
        ax[2].set_ylim(-1, K)

    # Show the plot
    plt.show()

# Show the plot
plt.show()




# %%
n=20
L=4000
K=9
x = 0.2
d=1
m= 0.10461
a= 2 #num of demes on edges
explore_biased_sampling(d,K,m,n,L,x,a)

# %%
files_1d = os.listdir("../data/eigen_analysis/1d_SS/K=5/")
m_list_1d = np.sort([float(file[2:]) for file in files_1d])

files_2d = os.listdir("../data/eigen_analysis/2d_SS/K=5/")
m_list_2d = np.sort([float(file[2:]) for file in files_2d])
# %%
print(m_list_1d)
print(m_list_2d)
# %%
