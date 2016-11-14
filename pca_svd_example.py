# -*- coding: utf-8 -*-
# PCA_SVD_Example.py

import numpy as  np
import datetime as dt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

# Globals
__DISPLAY_RAWDATA = True
__DISPLAY_EIGENVECTORS = True
__DISPLAY_REDUCED_DIMENSION = True
__NUM_SAMPLES = 3

# Get seed for random number generator so that
# we do not get the same result each run.
#  If you want to repeat the same result, then comment
# out the following statement. 
np.random.seed(dt.datetime.now().microsecond) 
#np.random.seed(1) 


# Create a 3D set of data for two classes. Each class will
# have __NUM_SAMPLES samples.
# Create class vectore for first class
mu_vec1 = np.array([0, 0, 0]) 
cov_mat1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, __NUM_SAMPLES).T
# Make sure that we have a 3x__NUM_SAMPLES matric
assert class1_sample.shape == (3,__NUM_SAMPLES), "The class1_sample does not have a 3x__NUM_SAMPLES dimenstion."

# Create class vectors for second class
mu_vec2 = np.array([1,1,1])
cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, __NUM_SAMPLES).T
# Make sure that we have a 3x__NUM_SAMPLES matric
assert class2_sample.shape == (3,__NUM_SAMPLES), "The class2_sample does not have a 3x__NUM_SAMPLES dimenstion."


# Display generated data for our two classes
if __DISPLAY_RAWDATA:
    fig1 = plt.figure(figsize=(8,8))
    ax = fig1.add_subplot(111, projection='3d')
    plt.rcParams['legend.fontsize'] = 10
    ax.plot(class1_sample[0,:], class1_sample[1,:], class1_sample[2,:],
            'o', markersize=8, color='blue', alpha=0.5, label='class1')
    ax.plot(class2_sample[0,:], class2_sample[1,:], class2_sample[2,:],
            '^', markersize=8, alpha=0.5, color='red', label='class2')
    
    plt.title('Samples for class 1 and class 2')
    ax.legend(loc='upper right')
    
    plt.show()

print
print ('class1_sample:\n', class1_sample)
print
print ('class2_sample:\n', class2_sample)
print

# Combine both class samples to a 3x(2 * __NUM_SAMPLES) dimensional array
all_samples = np.concatenate((class1_sample, class2_sample), axis=1)
assert all_samples.shape == (3,(2 * __NUM_SAMPLES)), "The matrix has not the dimensions 3x(2 * __NUM_SAMPLES)"

print
print "all_samples = ", all_samples
print


# Compute the d-dimensional mean vector
mean_x = np.mean(all_samples[0,:])
mean_y = np.mean(all_samples[1,:])
mean_z = np.mean(all_samples[2,:])

mean_vector = np.array([[mean_x],[mean_y],[mean_z]])

print
print('Mean Vector:\n', mean_vector)
print
print

# Compute the scattar matrix
scatter_matrix = np.zeros((3,3))
for i in range(all_samples.shape[1]):
    scatter_matrix += (all_samples[:,i].reshape(3,1) - mean_vector).dot(
        (all_samples[:,i].reshape(3,1) - mean_vector).T)
print('Scatter Matrix:\n', scatter_matrix)
print


# Compute covariance matrix
cov_mat = np.cov([all_samples[0,:],all_samples[1,:],all_samples[2,:]])
print('Covariance Matrix:\n', cov_mat)
print
print

# Compute eigen vectors and eigenvalues
# eigenvectors and eigenvalues from the scatter matrix
eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)

# eigenvectors and eigenvalues from the covariance matrix
eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

for i in range(len(eig_val_sc)):
    eigvec_sc = eig_vec_sc[:,i].reshape(1,3).T
    eigvec_cov = eig_vec_cov[:,i].reshape(1,3).T
    assert eigvec_sc.all() == eigvec_cov.all(), 'Eigenvectors are not identical'

    print('Eigenvector {}: \n{}'.format(i+1, eigvec_sc))
    print('Eigenvalue {} from scatter matrix: {}'.format(i+1, eig_val_sc[i]))
    print('Eigenvalue {} from covariance matrix: {}'.format(i+1, eig_val_cov[i]))
    print('Scaling factor: ', eig_val_sc[i]/eig_val_cov[i])
    print((2 * __NUM_SAMPLES) * '-')
    
# Check that the eigenvector-eigenvalue calculation is correct and satisfy the equation
for i in range(len(eig_val_sc)):
    eigv = eig_vec_sc[:,i].reshape(1,3).T
    np.testing.assert_array_almost_equal(scatter_matrix.dot(eigv),
                                         eig_val_sc[i] * eigv,
                                         decimal=6, err_msg='', verbose=True)


# Visualize eigenvectors
if __DISPLAY_EIGENVECTORS:
    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
            self._verts3d = xs, ys, zs
    
        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
            FancyArrowPatch.draw(self, renderer)
    
    fig2 = plt.figure(figsize=(8,8))
    ax = fig2.add_subplot(111, projection='3d')
    
    ax.plot(all_samples[0,:], all_samples[1,:], all_samples[2,:],
            'o', markersize=8, color='green', alpha=0.2)
    ax.plot([mean_x], [mean_y], [mean_z],
            'o', markersize=10, color='red', alpha=0.5)
    for v in eig_vec_sc.T:
        a = Arrow3D([mean_x, v[0]+mean_x],
                    [mean_y, v[1]+mean_y],
                    [mean_z, v[2]+mean_z],
                    mutation_scale=__NUM_SAMPLES, lw=3, arrowstyle="-|>", color="r")
    
        ax.add_artist(a)
    ax.set_xlabel('x_values')
    ax.set_ylabel('y_values')
    ax.set_zlabel('z_values')
    
    plt.title('Eigenvectors')
    
    plt.show()



# Sorting the eigenvectors by decreasing eigenvalues
for ev in eig_vec_sc:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
    # instead of 'assert' because of rounding errors

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i])
             for i in range(len(eig_val_sc))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
for i in eig_pairs:
    print(i[0])


"""
Choosing k eigenvectors with the largest eigenvalues.
For our simple example, where we are reducing a 3-dimensional 
feature space to a 2-dimensional feature subspace, we are 
combining the two eigenvectors with the highest eigenvalues 
to construct our d×k-dimensional eigenvector matrix W
"""
matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1),
                      eig_pairs[1][1].reshape(3,1)))
print('Matrix W:\n', matrix_w)

"""
Transforming the samples onto the new subspace
In the last step, we use the 2×3-dimensional 
matrix W that we just computed to transform our 
samples onto the new subspace via the equation
y=WT×x
"""
transformed = matrix_w.T.dot(all_samples)
assert transformed.shape == (2,(2 * __NUM_SAMPLES)), "The matrix is not 2x(2 * __NUM_SAMPLES) dimensional."

# Show the reduced dimensional space
if __DISPLAY_REDUCED_DIMENSION:
    fig3 = plt.figure(figsize=(8,8))
    rd=fig3.add_subplot(111)
    
    rd.plot(transformed[0,0:__NUM_SAMPLES], transformed[1,0:__NUM_SAMPLES],
            'o', markersize=7, color='blue', alpha=0.5, label='class1')
    rd.plot(transformed[0,__NUM_SAMPLES:(2 * __NUM_SAMPLES)], transformed[1,__NUM_SAMPLES:(2 * __NUM_SAMPLES)],
            '^', markersize=7, color='red', alpha=0.5, label='class2')
    plt.xlim([-5,5])
    plt.ylim([-5,5])
    plt.xlabel('x_values')
    plt.ylabel('y_values')
    plt.legend()
    plt.title('Transformed samples with class labels')
    
    plt.show()
    

