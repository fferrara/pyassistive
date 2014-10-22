from util import config, preprocessing, processing
from util.processing import Processing
import numpy as np
import scipy.linalg as LA

def MSI(X, Y):
    try: 
        n, p = X.shape
        n, q = Y.shape
    except ValueError as v:
        return []

    X = X.astype('float32', copy=False)
    X -= X.mean(axis=0)
    X /= np.max(np.abs(X))
    Y = Y.astype('float32', copy=False)
    Y -= Y.mean(axis=0)
    Y /= np.max(np.abs(Y))

    C = np.cov(X.T, Y.T, bias=1)
    CXX = C[:p,:p]
    CYY = C[p:,p:]

    sqx,_ = LA.sqrtm(LA.inv(CXX),False) # SXX^(-1/2)
    sqy,_ = LA.sqrtm(LA.inv(CYY),False) # SYY^(-1/2)

    # build square matrix
    u1 = np.vstack((sqx, np.zeros((sqy.shape[0], sqx.shape[1]))))
    u2 = np.vstack((np.zeros((sqx.shape[0], sqy.shape[1])), sqy))
    U = np.hstack((u1, u2))
    
    R = np.dot(np.dot(U, C), U.T)

    eigvals = LA.eigh(R)[0]
    eigvals /= np.sum(eigvals)
    # Compute index
    return 1 + np.sum(eigvals * np.log(eigvals)) / np.log(eigvals.shape[0])
    
FREQUENCIES = [6.4, 8.] # SX, DX
ref = Processing.generate_references(512, 6.4)

# --> X = vetor(:, 1:2) Y = vetor(:, 3)
print MSI(ref[:, 0:2], ref[:, 2:3])