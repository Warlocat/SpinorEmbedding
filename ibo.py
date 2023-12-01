import numpy as np
import cube
from pyscf import gto, dft, x2c
from pyscf.tools import molden
from pyscf.lo.ibo import ibo as ibo_pyscf
from pyscf.lo.iao import iao as iao_pyscf
from scipy.linalg import sqrtm

def mdot(*args):
   """chained matrix product: mdot(A,B,C,..) = A*B*C*...
   No attempt is made to optimize the contraction order."""
   r = args[0]
   for a in args[1:]:
      r = np.dot(r,a)
   return r

def sqrtminv(A):
   return np.linalg.inv(sqrtm(A))

def SymOrth(C,S=1):
   """symmetrically orthogonalize orbitals C with respect to overlap
   matrix S (i.e., such that Cnew^T S Cnew == id)."""
   return np.dot(C, sqrtminv(mdot(C.T.conj(),S,C)))


mol = gto.M(
    atom='''
O      -1.88976346      -1.73052779      -0.00000000
O       0.73083671      -1.86180909      -0.00000000
H      -0.19263533      -2.19581239       0.00000000
C      -1.77509557      -0.50262242       0.00000000
C      -0.49162796       0.18594861      -0.00000000
C       0.66714963      -0.53605177      -0.00000000
H      -0.45329458       1.27411004      -0.00000000
H      -2.68867003       0.12791146       0.00000000
H       1.64898979      -0.05366678      -0.00000000
''',
    basis="3-21g",
    verbose=4
)

mf = dft.RKS(mol)
mf.xc = 'b3lyp'
mf.kernel()
ci = mf.mo_coeff[:,mf.mo_occ>0]
mf_spinor = x2c.dft.RKS(mol)
mf_spinor.xc = 'b3lyp'
mf_spinor.kernel()
ci_spinor = mf_spinor.mo_coeff[:,mf_spinor.mo_occ>0]
print(ci.shape)
print(ci_spinor.shape)

mol_minao = gto.M(
    atom='''
O      -1.88976346      -1.73052779      -0.00000000
O       0.73083671      -1.86180909      -0.00000000
H      -0.19263533      -2.19581239       0.00000000
C      -1.77509557      -0.50262242       0.00000000
C      -0.49162796       0.18594861      -0.00000000
C       0.66714963      -0.53605177      -0.00000000
H      -0.45329458       1.27411004      -0.00000000
H      -2.68867003       0.12791146       0.00000000
H       1.64898979      -0.05366678      -0.00000000
''',
    basis="minao"
)

def get_iao(s1, s2, s12, coeff):
    nbas = s1.shape[0]
    s1inv = np.linalg.inv(s1)
    s2inv = np.linalg.inv(s2)
    
    coeff_tilde = SymOrth(mdot(s1inv,s12,s2inv,s12.T.conj(),coeff), s1)
    
    Oi1 = mdot(coeff, coeff.T.conj(), s1)
    Oti1 = mdot(coeff_tilde, coeff_tilde.T.conj(), s1)
    
    iao = mdot(Oi1, Oti1, s1inv, s12) + mdot(np.eye(nbas)-Oi1, np.eye(nbas)-Oti1, s1inv, s12)
    return SymOrth(iao, s1)

def get_ibo(ovlp, ao_slices_minao, coeff, iao, p=4, maxiter = 200, gradthreshold = 1e-8):
    Ca = mdot(iao.T.conj(), ovlp, coeff)
    Ni = Ca.shape[1]
    NAtom = len(ao_slices_minao)
    assert ao_slices_minao[NAtom-1][3] == Ca.shape[0], \
    "# Basis for ao_slices %d != Ca.shape[0] %d" % (ao_slices_minao[NAtom-1][3], Ca.shape[0])
    conv = False
    for iter in range(maxiter):
        grad = 0.0
        for i in range(Ni):
            for j in range(i):
                Avalue = 0.0
                Bvalue = 0.0
                Ci = 1. * Ca[:,i]
                Cj = 1. * Ca[:,j]
                for aa in range(NAtom):
                    astart = ao_slices_minao[aa][2]
                    aend = ao_slices_minao[aa][3]
                    Cia = Ci[astart:aend]
                    Cja = Cj[astart:aend]
                    Qii = mdot(Cia.T.conj(), Cia)
                    Qjj = mdot(Cja.T.conj(), Cja)
                    Qij = mdot(Cia.T.conj(), Cja)
                    Bvalue += 0.5*p*Qij*(Qii**(p-1) - Qjj**(p-1))
                    Avalue += 0.25*p*(p-1)*Qij**2*(Qii**(p-2) + Qjj**(p-2)) - 0.125*p*(Qii**(p-1) - Qjj**(p-1))*(Qii-Qjj)
                grad += Bvalue**2
                phi = 0.25*np.arctan2(Bvalue,-Avalue)
                cp = np.cos(phi)
                sp = np.sin(phi)
                Ca[:,i] = cp * Ci + sp * Cj
                Ca[:,j] =-sp * Ci + cp * Cj
        grad = grad**0.5
        if(grad < gradthreshold):
            conv = True
            break
    if(conv):
        print("IBO localization converged in %d iterations! Final gradient %.2e" % (iter+1, grad))
    else:
        print("ERROR: IBO localization not converged!")
        exit()
             
    return mdot(iao, SymOrth(Ca))

def get_ibo_spinor(ovlp, ao_slices_minao, coeff, iao, p=4, maxiter = 200, gradthreshold = 1e-8):
    Ca = mdot(iao.T.conj(), ovlp, coeff)
    Ni = Ca.shape[1]
    NAtom = len(ao_slices_minao)
    assert ao_slices_minao[NAtom-1][3] == Ca.shape[0], \
    "# Basis for ao_slices %d != Ca.shape[0] %d" % (ao_slices_minao[NAtom-1][3], Ca.shape[0])
    conv = False
    for iter in range(maxiter):
        grad = 0.0
        for i in range(Ni):
            for j in range(i):
                Ci = 1. * Ca[:,i]
                Cj = 1. * Ca[:,j]

                Avalue = 0.0
                Bvalue = 0.0
                for aa in range(NAtom):
                    astart = ao_slices_minao[aa][2]
                    aend = ao_slices_minao[aa][3]
                    Cia = Ci[astart:aend]
                    Cja = Cj[astart:aend]
                    Qii = (mdot(Cia.T.conj(), Cia)).real
                    Qjj = (mdot(Cja.T.conj(), Cja)).real
                    Qij = mdot(Cia.T.conj(), Cja)
                    ReQij = Qij.real
                    ImQij = Qij.imag
                    
                    Bvalue += (p*Qii**(p-1) - p*Qjj**(p-1))*ImQij
                    Avalue += (p*Qii**(p-1) - p*Qjj**(p-1))*ReQij
                tau = 0.5*np.arctan2(Bvalue,-Avalue)
                c2t = np.cos(2.0*tau)
                s2t = np.sin(2.0*tau)
                
                Avalue = 0.0
                Bvalue = 0.0        
                for aa in range(NAtom):
                    astart = ao_slices_minao[aa][2]
                    aend = ao_slices_minao[aa][3]
                    Cia = Ci[astart:aend]
                    Cja = Cj[astart:aend]
                    Qii = (mdot(Cia.T.conj(), Cia)).real
                    Qjj = (mdot(Cja.T.conj(), Cja)).real
                    Qij = mdot(Cia.T.conj(), Cja)
                    ReExp2itQij = Qij.real*c2t - Qij.imag*s2t
                    Bvalue += 0.5*p*ReExp2itQij*(Qii**(p-1) - Qjj**(p-1))
                    Avalue += 0.25*p*(p-1)*ReExp2itQij**2*(Qii**(p-2) + Qjj**(p-2)) - 0.125*p*(Qii**(p-1) - Qjj**(p-1))*(Qii-Qjj)
                grad += Bvalue**2
                phi = 0.25*np.arctan2(Bvalue,-Avalue)
                cp = np.cos(phi)
                sp = np.sin(phi)
                Ca[:,i] = cp * np.exp(-1.0*complex(0,tau)) * Ci + sp * np.exp(1.0*complex(0,tau)) * Cj
                Ca[:,j] =-sp * np.exp(-1.0*complex(0,tau)) * Ci + cp * np.exp(1.0*complex(0,tau)) * Cj
        grad = grad**0.5
        if(grad < gradthreshold):
            conv = True
            break
    if(conv):
        print("IBO localization converged in %d iterations! Final gradient %.2e" % (iter+1, grad))
    else:
        print("ERROR: IBO localization not converged!")
        exit()
             
    return mdot(iao, SymOrth(Ca))


s1 = mol.intor("int1e_ovlp")
s2 = mol_minao.intor("int1e_ovlp")
s12 = gto.intor_cross('int1e_ovlp', mol, mol_minao)
s1_spinor = mol.intor("int1e_ovlp_spinor")
s2_spinor = mol_minao.intor("int1e_ovlp_spinor")
s12_spinor = gto.intor_cross('int1e_ovlp_spinor', mol, mol_minao)
iao = get_iao(s1, s2, s12, ci)
iao_spinor = get_iao(s1_spinor, s2_spinor, s12_spinor, ci_spinor)


ao_slices = mol_minao.aoslice_by_atom()
ao_slices_spinor = mol_minao.aoslice_2c_by_atom()
ibo = get_ibo(s1, ao_slices, ci, iao)
ibo_spinor = get_ibo_spinor(s1_spinor, ao_slices_spinor, ci_spinor, iao_spinor)
molden.from_mo(mol, "iao.molden", iao)
molden.from_mo(mol, "ibo.molden", ibo)

ibo_ghf = np.vstack(mol.sph2spinor_coeff()).dot(ibo_spinor)
for ii in range(ibo.shape[1]):
    cube.orbital(mol, f'ibo_ghf_{ii}', ibo_ghf[:,2*ii:2*ii+1])

# iaopyscf = iao_pyscf(mol, ci)
# iaopyscf = SymOrth(iaopyscf, s1)
# ibopyscf = ibo_pyscf(mol, ci, iaos=iao)
# molden.from_mo(mol, "iaopyscf.molden", iaopyscf)
# molden.from_mo(mol, "ibopyscf.molden", ibopyscf)