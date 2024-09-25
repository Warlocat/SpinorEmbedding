# Module to solve AX = B using Krylov subspace method
# The GMRES algorithm is implemented here together with another
# algorithm proposed by Pople in 1979 when solving CPHF equations.
# 
# Pople's paper: https://doi.org/10.1002/qua.560160825
import numpy as np

def GMRES_Pople(AMultX, ADiag, B, maxIter = 20, tol = 1e-14, printLevel = 0):
    """
    Solve AX = B using Pople's algorithm
    A = ADiag*[I - Ap], Ap = I - ADiagInv*A, Bp = ADiagInv*B
    Ap*X = X - ADiagInv*A*X
    [I - Ap] X = Bp
    X = span[Bp, Ap*Bp, Ap^2*Bp, ...]
      = span[Bp, ADiagInv*A*Bp, ADiagInv*A^2*Bp, ...]
    """
    size = B.shape[0]
    if maxIter is None:
        maxIter = size + 1

    Bp = B.copy()
    for ii in range(size):
        Bp[ii] = B[ii]/ADiag[ii]

    X_list = [Bp]
    AX_list = []
    converged = False
    for ii in range(1, maxIter):
        AX = AMultX(X_list[ii-1])
        for jj in range(size):
            AX[jj] = AX[jj]/ADiag[jj]
        AX_list.append(AX)
        for jj in range(ii):
            Xnorm = np.dot(X_list[jj].conj().T, X_list[jj])
            AX = AX - np.dot(X_list[jj].conj().T, AX)/Xnorm * X_list[jj]
        AXnorm = np.sqrt(np.dot(AX.conj().T, AX))/size
        if AXnorm < tol:
            converged = True
            break
        else:
            X_list.append(AX)
    if not converged:
        print("ERROR: GMRES (Pople) did not converge in %d iterations!" % maxIter)
        exit()
    # print("GMRES (Pople) converged in %d iterations!" % (ii+1))
    # print("Final residual norm: ", AXnorm)
    nbasis = len(X_list)
    A_sub = np.zeros((nbasis, nbasis), dtype = B.dtype)
    B_sub = np.zeros((nbasis), dtype = B.dtype)
    for ii in range(nbasis):
        B_sub[ii] = np.dot(X_list[ii].conj().T, Bp)
        for jj in range(nbasis):
            A_sub[ii,jj] = np.dot(X_list[ii].conj().T, AX_list[jj])
    X_sub = np.linalg.solve(A_sub, B_sub)
    X = np.zeros((size), dtype = B.dtype)
    for ii in range(nbasis):
        X = X + X_sub[ii]*X_list[ii]
    if printLevel >= 1:
        print("gmres residual subspace: ", np.linalg.norm(A_sub.dot(X_sub) - B_sub))
        print("gmres residual: ", np.linalg.norm(AMultX(X) - B))
    return X

        
def GMRES(AMultX, ADiag, B, maxIter = 100, tol = 1e-12):
    raise Exception("GMRES is not implemented yet!")
    


    
