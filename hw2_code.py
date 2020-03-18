# Import neccessary packages
import numpy as np

# Define a 3*3 real symmetric positive definite matrix as a numpy array A and print the matrix
A = np.array([[6, -3, 1], [-3, 2, 0], [1, 0, 4]])
print(A)

# Check if the matrix A is symmetric by definition of symmetric
A.T == A

# Generate eigenvalues and corresponding eigenvectors and print them
eigval, eigvec = np.linalg.eig(A)
print("eigenvalue 1 is " + str(eigval[0]))
print("eigenvalue 2 is " + str(eigval[1]))
print("eigenvalue 3 is " + str(eigval[2]))
print("eigenvector 1 is " + str(eigvec[:, 0]))
print("eigenvector 2 is " + str(eigvec[:, 1]))
print("eigenvector 3 is " + str(eigvec[:, 2]))

# Generate Cholesky Decomposition of A previously defined and print the lower triangular with positive diagonal entries
L = np.linalg.cholesky(A)
print("L is\n" + str(L))

# Genearate a data vector u consisting of three standard normal variables of 100,000 points
mu, sigma = 0, 1 
u = np.random.normal(mu, sigma, (3, 100000))

# Calculate the covariance matrix of u as cov_u
cov_u = np.cov(u)
print("covariance matrix for u is\n" + str(cov_u))

# Generate correlated data v
v = np.dot(L, u)

# Generate the covariance matix of correlated data v as cov_v
cov_v = np.cov(v)
print("covariance matrix for v is\n" + str(cov_v))