#%%
## This script has a function that rotates a PC plot appropriately to be able to compare 2 PC plots
## It rotates such that the centroid of the first subpopulation occurs on the x=0 axis, with y>0
## It also reflects approporiately so that counting demes occurs counter clockwise
## After the vectors are appropriately rotated and reflected, simply taking the dot product will give the eigenvector inconsistency 
import numpy as np
import matplotlib.pyplot as plt

def standerdize(matrix,ploidy = 1):
    means = np.mean(np.array(matrix), axis = 0)
    if ploidy == 1:
        p = means
        stds = np.sqrt(p*(1-p))
    elif ploidy == 2:
        p=means/2
        stds = np.sqrt(2*p*(1-p))
    matrix = matrix - means
    matrix = matrix/stds
    return matrix

# m1 = 3.40e-04
# m2 = 9.60e-04
# K = 25

# x = np.load("../../data/eigen_output/2d_eigenvectors/m=%.2e,d=%d.npy"%(m1,K))
# y = np.load("../../data/eigen_output/2d_eigenvectors/m=%.2e,d=%d.npy"%(m2,K))
# eigenvectors1 = x[-1,-1,:,:2]
# eigenvectors1 = eigenvectors1[~np.isnan(eigenvectors1).any(axis=1),:] ## remove rows with nan values

# eigenvectors2 = y[-1,-1,:,:2]
# eigenvectors2 = eigenvectors2[~np.isnan(eigenvectors2).any(axis=1),:] ## remove rows with nan values


#%%
def rotate_eigenvectors(eigenvectors, K, index=0):
    n=int(np.shape(eigenvectors)[0]/K)

    centroids_vector_1 = [np.mean(eigenvectors[k*n:(k+1)*n,0]) for k in range(K)]
    centroids_vector_2 = [np.mean(eigenvectors[k*n:(k+1)*n,1]) for k in range(K)]
    centroids_vector = np.vstack((centroids_vector_1,centroids_vector_2)).T ## K*2 vector of centroids

    x1 = np.abs(centroids_vector[index,0])
    x2 = np.abs(centroids_vector[index,1])

    x1_sign = centroids_vector[index,0] > 0
    x2_sign = centroids_vector[index,1] > 0

    if x1_sign and x2_sign: # both positive
        theta = -(np.pi/2-np.arctan(x2/x1))
    elif x1_sign and not x2_sign: # x positive and y negative
        theta = -(np.pi/2+np.arctan(x2/x1))
    elif not x1_sign and x2_sign: # x negative and y postiive
        theta = np.pi/2-np.arctan(x2/x1)
    elif not x1_sign and not x2_sign: # both negative
        theta = np.pi/2+np.arctan(x2/x1)

    Rotation_Matrix = np.array([[np.cos(theta),np.sin(theta)],
                            [-np.sin(theta),np.cos(theta)]])
    Rotated_centroids = np.matmul(Rotation_Matrix,centroids_vector.T).T
    Rotated_eigenvectors = np.matmul(Rotation_Matrix,eigenvectors.T).T
    
    # Reflect if necessary, so that demes are counted clockwise
    if Rotated_centroids[1,0]<0:
        Rotated_centroids[:,0] = -Rotated_centroids[:,0]
        Rotated_eigenvectors[:,0] = -Rotated_eigenvectors[:,0]

    ## output the rotation matrix as well to use it to rotate the whole PC plot
    return Rotated_eigenvectors,Rotated_centroids

#%%
if __name__ == "__main__":

    slow = np.load("../../data/random/feems_grid_slow.npy")
    fast = np.load("../../data/random/feems_grid_faster.npy")
    geno1 = standerdize(slow)
    geno2 = standerdize(fast)
    cov1 = np.matmul(geno1,geno1.T)
    cov2 = np.matmul(geno2,geno2.T)
    vals1,vecs1 = np.linalg.eig(cov1)
    vals2,vecs2 = np.linalg.eig(cov2)
    eigenvectors1 = np.real(vecs1[:,[1,2]])
    eigenvectors2 = np.real(vecs2[:,[1,2]])

    K=40

    #%%
    plt.scatter(eigenvectors1[:,0],eigenvectors1[:,1],color="blue",label = "m=1e-5")
    plt.scatter(eigenvectors2[:,0],eigenvectors2[:,1],color="red",label = "m=1e-2")
    plt.legend()
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    Rotated_eigenvectors_1 = rotate_eigenvectors(eigenvectors1, K=K)
    Rotated_eigenvectors_2 = rotate_eigenvectors(eigenvectors2, K=K)

    #%%
    plt.scatter(Rotated_eigenvectors_1[:,0],Rotated_eigenvectors_1[:,1],color="blue",label = "m=%.2e"%m1)
    plt.scatter(Rotated_eigenvectors_2[:,0],Rotated_eigenvectors_2[:,1],color="red",label = "m=%.2e"%m2)
    plt.legend()
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    #%%
    # fig,ax = plt.subplots(5,5,figsize = (10,10))
    # for i in range(25):
    #     ax[i//5,i%5].scatter(Rotated_eigenvectors_1[i*100:(i+1)*100,0],Rotated_eigenvectors_1[i*100:(i+1)*100,1],c="blue")
    #     ax[i//5,i%5].scatter(Rotated_eigenvectors_2[i*100:(i+1)*100,0],Rotated_eigenvectors_2[i*100:(i+1)*100,1],c="red")
    #     ax[i//5,i%5].set_xlim(-0.04,0.04)
    #     ax[i//5,i%5].set_ylim(-0.04,0.04)

    fig,ax = plt.subplots(8,5,figsize = (20,20))
    for i in range(40):
        ax[i//5,i%5].scatter(Rotated_eigenvectors_1[i*10:(i+1)*10,0],Rotated_eigenvectors_1[i*10:(i+1)*10,1],c="blue")
        ax[i//5,i%5].scatter(Rotated_eigenvectors_2[i*10:(i+1)*10,0],Rotated_eigenvectors_2[i*10:(i+1)*10,1],c="red")
        ax[i//5,i%5].set_xlim(-0.15,0.15)
        ax[i//5,i%5].set_ylim(-0.15,0.15)

    # %%
    print(np.arccos(np.dot(Rotated_eigenvectors_1[:,0],Rotated_eigenvectors_2[:,0])))
    print(np.arccos(np.dot(Rotated_eigenvectors_1[:,1],Rotated_eigenvectors_2[:,1])))


    # %%
