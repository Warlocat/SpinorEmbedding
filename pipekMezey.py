import scipy
from ibo import get_iao, SymOrth
from optimizer import optimizer
from pyscf.lo.pipek import atomic_pops as atomic_pops_pyscf
import numpy as np
from functools import reduce

def atomic_pops(mol, mo_coeff, method='iao', iaos = None, aoslices = None, spinor = False):
    ovlp = mol.intor_symmetric("int1e_ovlp") if not spinor else mol.intor("int1e_ovlp_spinor")
    natms = mol.natm
    pop_matrix = []
    if method == 'iao':
        if iaos is None:
            iaos, aoslices = get_iao(mol, mo_coeff, spinor = spinor)
        for aa in range(natms):
            iao_a = iaos[:,aoslices[aa][2]:aoslices[aa][3]]
            pop_matrix.append(reduce(np.dot, (mo_coeff.T.conj(), ovlp, iao_a, iao_a.T.conj(), ovlp, mo_coeff)))
    else:
        raise Exception("Unknown method")
    return pop_matrix

def checkUnitary(UniMat):
    return np.allclose(np.eye(UniMat.shape[0]), np.dot(UniMat.T.conj(), UniMat))

def unpack_uniq_var(v):
    nmo = int(np.sqrt(v.size*2)) + 1
    idx = np.tril_indices(nmo, -1)
    mat = np.zeros((nmo,nmo), dtype=v.dtype)
    mat[idx] = v
    return mat - mat.conj().T

def pack_uniq_var(mat):
    # take the lower triangle of a matrix
    # [m,n] -> [mn]
    nmo = mat.shape[0]
    idx = np.tril_indices(nmo, -1)
    return mat[idx]

def pack_uniq_var_2d(mat):
    # take the lower triangle of a matrix-by-matrix Hessians
    # [m,n,k,j] -> [mn,kj]
    nmo = mat.shape[0]
    idx = np.tril_indices(nmo, -1)
    packed_mat = np.zeros((nmo*(nmo-1)//2,nmo*(nmo-1)//2), dtype=mat.dtype)
    for ii in range(nmo*(nmo-1)//2):
        for jj in range(nmo*(nmo-1)//2):
            packed_mat[ii,jj] = mat[idx[0][ii],idx[1][ii],idx[0][jj],idx[1][jj]]
    return packed_mat

    

def unitaryOptimization(guess, HessMultFunc, HdiagFunc, gradFunc, maxIter = 200, tol = 1e-6, algorithm = '2nd', explicitHess = None):
    if not checkUnitary(guess):
        raise Exception("Initial guess is not unitary")
    else:
        UniMat = guess
    optConv = False
    for iter in range(maxIter):
        if algorithm == '2nd':
            import gmres
            grad = gradFunc(UniMat)
            norm_grad = np.linalg.norm(grad)
            if(explicitHess is not None):
                Hess = explicitHess(UniMat)
                print(np.diag(Hess))
                print(HdiagFunc(UniMat))
                exit()
                Rnew = scipy.linalg.solve(Hess, -1.0*grad)
                print("solve residual: ", np.linalg.norm(Hess.dot(Rnew) + grad))
            else:
                Rnew = gmres.GMRES_Pople(HessMultFunc(UniMat), HdiagFunc(UniMat), -1.0*grad, printLevel=1)

            weight = 1.0
            while True:
                R = unpack_uniq_var(weight*Rnew)
                UniMat_new = np.dot(UniMat, scipy.linalg.expm(R))
                norm_grad_new = np.linalg.norm(gradFunc(UniMat_new))
                if norm_grad_new < norm_grad:
                    break
                else:
                    # break
                    weight *= 0.5
                    if weight < 0.001:
                        break
            norm = np.linalg.norm(R)
            norm_grad = norm_grad_new
            UniMat = UniMat_new
            print("Unitary optimization: Iteration %d, |grad| = %e, |rotation| = %e" % (iter, norm_grad, norm))
            if norm_grad < tol:
                optConv = True
                break
    if not optConv:
        print("Unitary optimization did not converge!")
        exit()
    else:
        print("Unitary optimization converged")
        if not checkUnitary(UniMat):
            print("Warning: Optimized matrix is not unitary!")

    return UniMat

def costFunc(mol, coeff_loc, coeff_canonical, pop_method='iao', order = 4, spinor = False):
    iaos, aoslices = get_iao(mol, coeff_canonical, spinor = spinor)
    pop_matrix = atomic_pops(mol, coeff_loc, pop_method, iaos=iaos, aoslices=aoslices, spinor=spinor)
    cost = 0.0
    for a in range(mol.natm):
        for i in range(pop_matrix[0].shape[1]):
            cost += pop_matrix[a][i,i]**order
    return cost.real

def atomic_init_guess(mol, mo_coeff, spinor = False):
    s = mol.intor_symmetric('int1e_ovlp') if not spinor else mol.intor('int1e_ovlp_spinor')
    c = SymOrth(np.eye(s.shape[0]),s)
    mo = reduce(np.dot, (c.conj().T, s, mo_coeff))
# Find the AOs which have largest overlap to MOs
    idx = np.argsort(np.einsum('pi,pi->p', mo.conj(), mo))
    nmo = mo.shape[1]
    idx = sorted(idx[-nmo:])

    # Rotate mo_coeff, make it as close as possible to AOs
    u, w, vh = np.linalg.svd(mo[idx])
    return np.dot(u, vh).conj().T


def localization_PM(mol, coeff_canonical, pop_method='iao', C_loc_guess = None, order = 4, spinor = False):
    iaos, aoslices = get_iao(mol, coeff_canonical, spinor = spinor)
    pop_matrix = atomic_pops(mol, coeff_canonical, pop_method, iaos=iaos, aoslices=aoslices, spinor=spinor)

    def costFuncPM(UniMat, order = order):
        return costFunc(mol, np.dot(coeff_canonical, UniMat), coeff_canonical, pop_method, order, spinor)
    
    def costFuncDerivPM(UniMat, order = order):
        grad = np.zeros(UniMat.shape, dtype=UniMat.dtype)
        for a in range(mol.natm):
            pop1 = reduce(np.dot, (UniMat.T.conj(), pop_matrix[a], UniMat))
            for m in range(UniMat.shape[0]):
                for n in range(UniMat.shape[1]):
                    grad[m,n] += order*(pop1[n,n]**(order-1) - pop1[m,m]**(order-1))*pop1[n,m]
        return grad
    
    # def optHessDiag(UniMat, order = order):
    #     diag = np.zeros(UniMat.shape)
    #     for a in range(mol.natm):
    #         pop1 = reduce(np.dot, (UniMat.T.conj(), pop_matrix[a], UniMat))
    #         for m in range(UniMat.shape[0]):
    #             for n in range(UniMat.shape[1]):
    #                 if m != n:
    #                     diag[m,n] += order*(pop1[m,m]**(order-1) - pop1[n,n]**(order-1))*(pop1[m,m]-pop1[n,n]) \
    #                                - order*(order-1)*(pop1[n,n]**(order-2) + pop1[m,m]**(order-2)) \
    #                                *(pop1[m,n]*pop1[n,m])

    #     return pack_uniq_var(diag.real)
    
    # def optHessExplicit(UniMat, order = order):
    #     Hess = np.zeros((UniMat.shape[0], UniMat.shape[1], UniMat.shape[0], UniMat.shape[1]), dtype=UniMat.dtype)
    #     for a in range(mol.natm):
    #         pop1 = reduce(np.dot, (UniMat.T.conj(), pop_matrix[a], UniMat))
    #         for m in range(UniMat.shape[0]):
    #             for n in range(UniMat.shape[1]):
    #                 for k in range(UniMat.shape[0]):
    #                     for j in range(UniMat.shape[1]):
    #                         if m == j:
    #                             Hess[m,n,k,j] += order*(order-1)*pop1[m,m]**(order-2)*pop1[m,k]*pop1[m,n]
    #                         if n == j:
    #                             Hess[m,n,k,j] += -order*(order-1)*pop1[n,n]**(order-2)*pop1[n,k]*pop1[m,n] \
    #                                              +order*(pop1[m,m]**(order-1) - pop1[n,n]**(order-1))*pop1[m,k]
    #                         if m == k:
    #                             Hess[m,n,k,j] += -order*(order-1)*pop1[m,m]**(order-2)*pop1[j,m]*pop1[m,n] \
    #                                              -order*(pop1[m,m]**(order-1) - pop1[n,n]**(order-1))*pop1[j,n]
    #                         if n == k:
    #                             Hess[m,n,k,j] += order*(order-1)*pop1[n,n]**(order-2)*pop1[j,n]*pop1[m,n]
    #     return pack_uniq_var_2d(Hess)


    # def optHessMult(UniMat, order = order):
    #     def HessMultFunc(Rvec):
    #         Rmat = unpack_uniq_var(Rvec)
    #         hx = np.zeros(UniMat.shape, dtype=UniMat.dtype)
    #         for a in range(mol.natm):
    #             pop1 = reduce(np.dot, (UniMat.T.conj(), pop_matrix[a], UniMat))
    #             tmp = np.dot(pop1, Rmat) - np.dot(Rmat, pop1)
    #             for m in range(UniMat.shape[0]):
    #                 for n in range(UniMat.shape[1]):
    #                     hx[m,n] += order*(order-1)*(pop1[m,m]**(order-2)*tmp[m,m] - pop1[n,n]**(order-2)*tmp[n,n])*pop1[m,n] \
    #                              + order*(pop1[m,m]**(order-1) - pop1[n,n]**(order-1))*tmp[m,n]
                
    #         return pack_uniq_var(hx)
    #     return HessMultFunc
    
    # def optHessMult2(UniMat, order = order):
    #     def HessMultFunc(Rvec):
    #         Rmat = unpack_uniq_var(Rvec)
    #         hx = np.zeros(UniMat.shape, dtype=UniMat.dtype)
    #         B = np.zeros(UniMat.shape, dtype=UniMat.dtype)
    #         for a in range(mol.natm):
    #             pop1 = reduce(np.dot, (UniMat.T.conj(), pop_matrix[a], UniMat))
    #             pop2 = np.dot(pop_matrix[a], UniMat)
    #             for m in range(UniMat.shape[0]):
    #                 for n in range(UniMat.shape[1]):
    #                     B[m,n] += order*pop1[n,n]**(order-1)*pop2[m,n]
    #         BdU = np.dot(B.conj().T, UniMat)
    #         UdB = np.dot(UniMat.T.conj(), B)
    #         hx = np.dot(BdU, Rmat) + np.dot(Rmat.conj().T, UdB)
    #         return pack_uniq_var(hx)
    #     return HessMultFunc
    
    # def optHessDiag2(UniMat, order = order):
    #     B = np.zeros(UniMat.shape, dtype=UniMat.dtype)
    #     for a in range(mol.natm):
    #         pop1 = reduce(np.dot, (UniMat.T.conj(), pop_matrix[a], UniMat))
    #         pop2 = np.dot(pop_matrix[a], UniMat)
    #         for m in range(UniMat.shape[0]):
    #             for n in range(UniMat.shape[1]):
    #                 B[m,n] += order*pop1[n,n]**(order-1)*pop2[m,n]
    #     BdU = np.dot(B.conj().T, UniMat)
    #     UdB = np.dot(UniMat.T.conj(), B)
    #     diag = np.zeros(UniMat.shape, dtype=UniMat.dtype)
    #     for m in range(UniMat.shape[0]):
    #         for n in range(UniMat.shape[1]):
    #             diag[m,n] = BdU[m,m] + UdB[n,n]

    #     return pack_uniq_var(diag)
    
    def costFuncDeriv_new(UniMat, order = order):
        grad = np.zeros(UniMat.shape, dtype=UniMat.dtype)
        for a in range(mol.natm):
            pop1 = reduce(np.dot, (UniMat.T.conj(), pop_matrix[a], UniMat))
            for m in range(UniMat.shape[0]):
                for n in range(UniMat.shape[1]):
                    grad[m,n] += order*(pop1[n,n]**(order-1) - pop1[m,m]**(order-1))*pop1[n,m]
        return pack_uniq_var(grad)
    def optHessDiag_new(UniMat, order = order):
        diag = np.zeros(UniMat.shape, dtype=UniMat.dtype)
        for a in range(mol.natm):
            pop1 = reduce(np.dot, (UniMat.T.conj(), pop_matrix[a], UniMat))
            for m in range(UniMat.shape[0]):
                for n in range(UniMat.shape[1]):
                    diag[m,n] += order*(order-1)*(pop1[n,n]**(order-2) + pop1[m,m]**(order-2))*pop1[n,m]*pop1[m,n]
        return pack_uniq_var(diag)
    def optHessMult_new(UniMat, order = order):
        def HessMultFunc(Rvec):
            Rmat = unpack_uniq_var(Rvec)
            size = Rmat.shape[0]
            ppm1 = order*(order-1)
            hx = np.zeros(UniMat.shape, dtype=UniMat.dtype)
            for a in range(mol.natm):
                pop1 = reduce(np.dot, (UniMat.T.conj(), pop_matrix[a], UniMat))
                for u in range(size):
                    for v in range(u):
                        for x in range(v+1, size):
                            hx[u,v] += ppm1*pop1[v,v]**(order-2)*pop1[v,u]*pop1[v,x]*Rmat[x,v]
                        for x in range(u+1, size):
                            hx[u,v] -= ppm1*pop1[u,u]**(order-2)*pop1[v,u]*pop1[u,x]*Rmat[x,u]
                            hx[u,v] += order*(0.5*(pop1[v,v]**(order-1) + pop1[x,x]**(order-1)) - pop1[u,u]**(order-1))*pop1[v,x]*Rmat[x,u]
                        for y in range(v):
                            hx[u,v] -= ppm1*pop1[v,v]**(order-2)*pop1[v,u]*pop1[y,v]*Rmat[v,y]
                            hx[u,v] += order*(0.5*(pop1[u,u]**(order-1) + pop1[y,y]**(order-1)) - pop1[v,v]**(order-1))*pop1[y,u]*Rmat[v,y]
                        for y in range(u):
                            hx[u,v] += ppm1*pop1[u,u]**(order-2)*pop1[v,u]*pop1[y,u]*Rmat[u,y]

            return pack_uniq_var(hx)
        return HessMultFunc
    def optHessExplicit_new(UniMat, order = order):
        size = UniMat.shape[0]
        Hess = np.zeros((size,size,size,size), dtype=UniMat.dtype)
        ppm1 = order*(order-1)
        for a in range(mol.natm):
            pop1 = reduce(np.dot, (UniMat.T.conj(), pop_matrix[a], UniMat))
            for u in range(size):
                for v in range(size):
                    for x in range(size):
                        for y in range(size):
                            if v == y:
                                Hess[u,v,x,y] += ppm1*pop1[y,x]*pop1[y,y]**(order-2)*pop1[y,u]
                            if u == y:
                                Hess[u,v,x,y] -= ppm1*pop1[y,x]*pop1[y,y]**(order-2)*pop1[v,y]
                                Hess[u,v,x,y] += order*(0.5*(pop1[x,x]**(order-1) + pop1[v,v]**(order-1)) - pop1[v,v]**(order-1))*pop1[v,x]
                            if v == x:
                                Hess[u,v,x,y] -= ppm1*pop1[y,x]*pop1[x,x]**(order-2)*pop1[x,u]
                                Hess[u,v,x,y] += order*(0.5*(pop1[u,u]**(order-1) + pop1[y,y]**(order-1)) - pop1[u,u]**(order-1))*pop1[y,u]
                            if u == x:
                                Hess[u,v,x,y] += ppm1*pop1[y,x]*pop1[x,x]**(order-2)*pop1[v,x]
        return pack_uniq_var_2d(Hess)

    if C_loc_guess is None:
        guess = atomic_init_guess(mol, coeff_canonical, spinor = spinor)
        # guess = np.eye(coeff_canonical.shape[1], dtype=coeff_canonical.dtype)
    else:
        ovlp = mol.intor_symmetric("int1e_ovlp") if not spinor else mol.intor("int1e_ovlp_spinor")
        guess = reduce(np.dot, (coeff_canonical.T.conj(), ovlp, C_loc_guess))
                
    # Uopt = unitaryOptimization(guess, optHessMult, optHessDiag, costFuncDeriv, explicitHess=optHessExplicit)
    # Uopt = unitaryOptimization(guess, optHessMult, optHessDiag, costFuncDeriv)
    # Uopt = unitaryOptimization(guess, optHessMult2, optHessDiag2, costFuncDeriv)
    # Uopt = unitaryOptimization(guess, optHessMult_new, optHessDiag_new, costFuncDeriv_new, explicitHess=optHessExplicit_new)
    # Uopt = unitaryOptimization(guess, optHessMult_new, optHessDiag_new, costFuncDeriv_new, explicitHess=None)
        
    opt = optimizer(costFuncPM, costFuncDerivPM, guess, externalOrder=order, descent=False)
    Uopt = opt.optimize()
    C_loc = coeff_canonical.dot(Uopt)
    print("Canonical cost value: ", costFunc(mol, coeff_canonical, coeff_canonical, pop_method, order, spinor))
    print("Localized cost value: ", costFunc(mol, C_loc, coeff_canonical, pop_method, order, spinor))
    return C_loc

