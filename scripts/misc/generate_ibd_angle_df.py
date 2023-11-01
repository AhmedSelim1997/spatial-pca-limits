#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from compute_general_angle import rotate_eigenvectors
from sklearn.preprocessing import normalize
import seaborn as sns
import os

#%%
theor_files_dir = "../../data/theor_eigen/mig_barrier_theor_spectrum_grid/eigenvectors/"
theor_spectrum_dir = "../../theor_eigen/mig_barrier_theor_spectrum_grid/eigenvalues/"
emp_files_dir = "../../data/emp_eigen/mig_barrier_grid/eigenvectors/"
files = os.listdir(theor_files_dir)

x_list = np.sort([round(float(file[2:-4]),2) for file in files])
n_list = np.arange(2,21,2).astype(int)
L_list = np.geomspace(400,8000,10).astype(int)
K = 64

#%%
# full_spectrum = []
# for x in x_list:
#     theor_spectrum =  np.load(theor_spectrum_dir + "x=%.2f.npy"%x)
#     full_spectrum.append(theor_spectrum)
# full_spectrum = np.real(np.array(full_spectrum))
# full_spectrum = pd.DataFrame(full_spectrum,index = x_list, columns = range(1,401))
# full_spectrum = full_spectrum.iloc[:,1:].reset_index() ## Removing the first eigenvalue, and adding a column of index
# full_spectrum = full_spectrum.rename(columns = {"index":"x"})
# some_spectrum = full_spectrum.iloc[:,:6].melt("x", var_name="eigenvalue_rank",value_name= "eigenvalue")
# sns.scatterplot(some_spectrum,x="x",y="eigenvalue",hue = "eigenvalue_rank")


#%%
df = pd.DataFrame(columns = ["x","theta_IBD","n","L","theta_noise"])
IBD = np.real(np.load("../../data/theor_eigen/mig_barrier_theor_spectrum_grid/eigenvectors/x=1.00.npy"))
IBD_eigenvectors = IBD[:,[0,1]]
IBD_eigenvectors_rotated,IBD_centroids_rotated = rotate_eigenvectors(IBD_eigenvectors,K=K)
IBD_centroids_rotated = normalize(IBD_centroids_rotated, axis=0, norm='l2')


for x in x_list:
    theor_vec = np.load(theor_files_dir + "x=%.2f.npy"%x)
    theor_vec = theor_vec[:,[0,1]]
    theor_vec_rotated,theor_centroids_rotated = rotate_eigenvectors(theor_vec,K=K)
    theor_centroids_rotated = np.real(theor_centroids_rotated)
    theor_centroids_rotated = normalize(theor_centroids_rotated, axis=0, norm='l2')

    theta_IBD_1 = np.arccos(np.round(np.dot(theor_centroids_rotated[:,0],IBD_centroids_rotated[:,0]),4))
    theta_IBD_2 = np.arccos(np.round(np.dot(theor_centroids_rotated[:,1],IBD_centroids_rotated[:,1]),4))
    theta_IBD = (theta_IBD_1+theta_IBD_2)/2

    emp_vecs = np.load(emp_files_dir + "x=%.2f.npy"%x)
    for i,n in enumerate(n_list):
        theor_centroids_extended = np.repeat(theor_centroids_rotated,n,axis = 0)
        theor_centroids_extended = normalize(theor_centroids_extended, axis=0, norm='l2')

        for j,L in enumerate(L_list):
            emp_vec = emp_vecs[i,j][:n*K,[0,1]]
            if np.all(np.isnan(emp_vec)):
                continue
            emp_vec_rotated,emp_centroids_rotated = rotate_eigenvectors(emp_vec,K=K)

            theta_noise_1 = np.arccos(np.round(np.dot(theor_centroids_extended[:,0],emp_vec_rotated[:,0]),4))
            theta_noise_2 =np.arccos(np.round(np.dot(theor_centroids_extended[:,1],emp_vec_rotated[:,1]),4))
            theta_noise = (theta_noise_1+theta_noise_2)/2

            temp_array = [x,theta_IBD,n,L,theta_noise]
            df.loc[len(df)] = temp_array
#%%
colors = ["blue","green","orange","purple","brown"]
x_list_test = x_list[1::10]
for i,x_test in enumerate(x_list_test):
    test = np.real(np.load("../../data/theor_eigen/mig_barrier_theor_spectrum_grid/eigenvectors/x=%.2f.npy"%x_test))
    test_eigenvectors = test[:,[0,1]]
    test_eigenvectors_rotated,test_centroids_rotated = rotate_eigenvectors(test_eigenvectors,K=K)
    test_centroids_rotated = normalize(test_centroids_rotated, axis=0, norm='l2')
    plt.scatter(test_centroids_rotated[:,0],test_centroids_rotated[:,1],c=colors[i],label = "x=%.2f"%x_test)
plt.scatter(IBD_centroids_rotated[:,0],IBD_centroids_rotated[:,1],c="red",label = "x=1.00")
plt.legend()
# %%
x_test = 0.98
test = np.real(np.load("../../data/theor_eigen/mig_barrier_theor_spectrum_grid/eigenvectors/x=%.2f.npy"%x_test))
test_eigenvectors = test[:,[0,1]]
test_eigenvectors_rotated,test_centroids_rotated = rotate_eigenvectors(test_eigenvectors,K=K)
test_centroids_rotated = normalize(test_centroids_rotated, axis=0, norm='l2')
plt.scatter(test_centroids_rotated[:,0],test_centroids_rotated[:,1],c="green",label = "x=%.2f"%x_test)
plt.scatter(IBD_centroids_rotated[:,0],IBD_centroids_rotated[:,1],c="red",label = "x=1.00")
plt.legend()

# %%
x_target = x_list[45]
ax = sns.heatmap(df[df["x"]==x_target].iloc[:,2:].pivot_table(index = "n",columns = "L",values = "theta_noise")[::-1])
ax.collections[0].colorbar.set_label('theta_noise')
plt.title("eigenvector inconsistency angles for x=%.2f"%x_target)
plt.show
# %%
