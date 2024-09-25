import numpy as np
from numpy import dot as matmul
import scipy
from functools import reduce

def diis_error(f,d,s):
    size = f.shape[0]
    fds = reduce(matmul, (f,d,s))
    fds = fds - fds.T.conj()
    return fds.reshape(size*size)

def myscf_solver(mol, mf, ovlp, h1e, mo_occ, sd_Huzinage, huzinage_factor, ao_keep = None, dm0 = None, 
                 diis_start=None, diis_size=None, max_cycle = None, tol = None, damping_factor = 0.5):
    if(diis_start is None):
        diis_start = mf.diis_start_cycle
    if(diis_size is None):
        diis_size = mf.diis_space
    if(max_cycle is None):
        max_cycle = mf.max_cycle
    if(tol is None):
        tol = mf.conv_tol
    size = h1e.shape[0]
    data_type = h1e.dtype
    conv = False
    errdiis = []
    fockdiis = []
    if(dm0 is None):
        e,c = scipy.linalg.eigh(h1e, ovlp)
        dm0 = mf.make_rdm1(c, mo_occ)
    old_dm = dm0.copy()

    ndiis = 0
    for icycle in range(max_cycle):
        veff = mf.get_veff(mol=mol, dm=old_dm)
        fock = h1e + veff
        sdf_Huzinage = huzinage_factor*matmul(sd_Huzinage, fock)
        fock = fock - sdf_Huzinage - sdf_Huzinage.T.conj()
        ndiis = ndiis + 1
        errdiis.append(diis_error(fock, old_dm, ovlp))
        fockdiis.append(fock.copy())
        if(ndiis > diis_size):
            ndiis = diis_size
            errdiis.pop(0)
            fockdiis.pop(0)
        if(icycle >= diis_start):
            if(icycle == diis_start and mf.verbose >= 3):
                print("DIIS started with DIIS size = %d" % diis_size)
            matdiis = np.zeros((ndiis+1, ndiis+1), dtype=data_type)
            b = np.zeros(ndiis+1, dtype=data_type)
            for i in range(ndiis):
                for j in range(ndiis):
                    matdiis[i,j] = matmul(errdiis[i].T.conj(), errdiis[j])
                matdiis[i,ndiis] = -1.0
                matdiis[ndiis,i] = -1.0
            matdiis[ndiis,ndiis] = 0.0
            b[ndiis] = -1.0
            c_diis = scipy.linalg.solve(matdiis, b)
            fock = np.zeros((size,size), dtype=data_type)
            for i in range(ndiis):
                fock = fock + c_diis[i]*fockdiis[i]
        
        e,c = scipy.linalg.eigh(fock, ovlp)
        new_dm = mf.make_rdm1(c, mo_occ)
        e_scf = mf.energy_tot(new_dm, h1e=h1e, vhf=mf.get_veff(dm=new_dm))
        diff = np.linalg.norm(new_dm - old_dm)
        if(diff < tol):
            print("Convengence reached after %d iterations!" % icycle)
            conv = True
            break
        
        if(mf.verbose >= 4):
            print("Cycle %d, diff = %.10f, scf_ene = %.14f" % (icycle+1, diff, e_scf))
        if(icycle < diis_start):
            old_dm = damping_factor*new_dm + (1.0-damping_factor)*old_dm
        else:
            old_dm = new_dm.copy()

    if(not conv):
        print("Convergence not reached for subsystem!")
        exit()
    else:
        mf.converged = True
        mf.e_tot = e_scf
        mf.mo_energy = e
        mf.mo_coeff = c
        mf.mo_occ = mo_occ

            