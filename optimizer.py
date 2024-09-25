# The unitary optimizer
# Refs: J. Chem. Theory Comput. 2013, 9, 5365, in which they cited 
#       IEEE Transactions on Signal Processing 2008, 56, 1134 
#       Signal Processing 2009, 89, 1704
import numpy as np
from functools import reduce
import math
import scipy

def innerProduct(a, b):
    return np.trace(np.dot(a, b.conj().T)).real/2.0

def searchDirection(grad, grad_last, H_last, iter, updateMethod = "CGFR"):
    if iter == 0 or updateMethod == "SD" or (iter+1)%grad.shape[0] == 0:
        return grad
    if updateMethod == "CGPR":
        gamma = innerProduct(grad, grad)/innerProduct(grad_last, grad_last)
    elif updateMethod == "CGFR":
        gamma = innerProduct(grad, grad - grad_last)/innerProduct(grad_last, grad_last)
    else:
        raise Exception("Unknown update method")
    
    H_new = gamma * H_last + grad
    if innerProduct(H_new, grad) < 0:
        H_new = grad
    return H_new

def stepSizeSearch(H, W, costFuncDeriv, polyOrder = 4, q = 8):
    w, v = np.linalg.eigh(H)
    w = np.abs(w[-1])
    period = 2.0*math.pi/w/q
    np.linalg.eigh(H)
    basis = [np.eye(H.shape[0]), scipy.linalg.expm(-1*period*H/polyOrder)]
    mu_array = np.zeros(polyOrder)
    for ii in range(1, polyOrder):
        mu_array[ii] = ii * period/polyOrder
        if(ii > 1):
            basis.append(np.dot(basis[-1], basis[1]))
    A_array = np.zeros((polyOrder, polyOrder))
    B_array = np.zeros(polyOrder)
    for ii in range(polyOrder):
        for jj in range(polyOrder):
            A_array[ii,jj] = mu_array[ii]**jj
        tmp = np.dot(basis[ii], W)
        B_array[ii] = -2.0 * np.trace(reduce(np.dot, (costFuncDeriv(tmp), tmp.conj().T, H.conj().T))).real
    coeff = np.linalg.solve(A_array, B_array)
    coeff = np.flip(coeff)
    mu = np.roots(coeff)
    mu = mu[mu > 0]
    print("Step size search: ", np.min(mu))
    return np.min(mu)
    
class optimizer:
    def __init__(self, costFunc, costFuncDeriv, Winit, maxIter = 20, externalOrder = 4, gradThreshold = 1e-5, updateMethod = "CGFR", descent = True):
        self.costFunc = costFunc
        self.costFuncDeriv = costFuncDeriv
        self.externalOrder = externalOrder
        self.W = Winit
        self.maxIter = maxIter
        self.gradThreshold = gradThreshold
        self.updateMethod = updateMethod
        self.factor = -1.0 if descent else 1.0
        self.conv = False

    def gradient(self):
        return self.costFuncDeriv(self.W)
    def cost(self):
        return self.costFunc(self.W)

    def optimize(self):
        cost_last = self.cost()
        grad_last = self.gradient()
        H_last = grad_last

        for iter in range(self.maxIter):
            if self.updateMethod == "SD":
                grad = grad_last
            elif self.updateMethod == "CGFR" or self.updateMethod == "CGPR":
                
            stepSize = 1.0
            self.W = np.dot(self.W, scipy.linalg.expm(self.factor*stepSize*H))
            grad_last = grad
            H_last = H
            cost_last = cost
            grad = self.gradient()
            H = searchDirection(grad, grad_last, H_last, iter, self.updateMethod)
            cost = self.cost()
            print("Iteration ", iter, " cost: ", cost)
            print("Gradient norm: ", np.linalg.norm(grad), " cost change: ", cost - cost_last)
            if np.linalg.norm(grad) < self.gradThreshold:
                self.conv = True
                print("Converged at iteration ", iter)
                break
                
        
        if not self.conv:
            print("Optimizer did not converge!")

        return self.W
