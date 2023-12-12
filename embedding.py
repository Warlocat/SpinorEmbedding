import numpy as np
from numpy import dot as matmul
import ibo
from myhf import myscf_solver
from pyscf import dft, scf
from functools import reduce
    
def pick_active_mulliken(loc_mo, ovlp, active_atoms, ao_slices):
    nocc = loc_mo.shape[1]
    act_mo_indices = []
    for i in range(nocc):
        for a in active_atoms:
            cia = loc_mo[ao_slices[a,2]:ao_slices[a,3],i]
            sa = ovlp[ao_slices[a,2]:ao_slices[a,3],ao_slices[a,2]:ao_slices[a,3]]
            mulliken_q = (matmul(cia.T.conj(), matmul(sa, cia))).real
            if(mulliken_q > 0.4):
                act_mo_indices.append(i)
                break
    act_mo = loc_mo[:,act_mo_indices].copy()
    return act_mo, act_mo_indices

def pick_active(loc_mo, ovlp, active_atoms, ao_slices, method="mulliken"):
    if(method == "mulliken"):
        return pick_active_mulliken(loc_mo, ovlp, active_atoms, ao_slices)
    else:
        print("ERROR: Unknown method in pick_active: %s" % method)
        exit()

def get_embedding_potential(active_atoms, mol, mf, ovlp = None, loc_mo=None):
    if(isinstance(mf, dft.rks.RKS) or isinstance(mf, scf.rhf.RHF)):
        mo_occ = mf.mo_coeff[:,mf.mo_occ>0]
    else:
        print("unrestricted case is not implemented yet")
        exit()
    if(ovlp is None):
        ovlp = mol.intor("int1e_ovlp")
    if(loc_mo is None):
        loc_mo = ibo.get_ibo(mol,mo_occ)
    act_mo, act_indices = pick_active(loc_mo, ovlp, active_atoms, mol.aoslice_by_atom())
    act_occ = mf.mo_occ[act_indices]
        
    den_tot = mf.make_rdm1()
    den_act = mf.make_rdm1(act_mo, act_occ)
    
    den_env = den_tot - den_act
    e_tot = mf.e_tot
    e_act = mf.energy_tot(den_act, h1e=mf.get_hcore(mol), vhf=mf.get_veff(dm=den_act))

    v_correction = mf.get_veff(dm=den_tot) - mf.get_veff(dm=den_act)
    nelec_act = round(np.trace(matmul(ovlp, den_act)))
    nelec_act = int(nelec_act)

    print("Number of active electrons: %d" % nelec_act)
    return v_correction, nelec_act, e_tot, e_act, den_act, den_env

def get_embedding_potential_spinor(active_atoms, mol, mf_spinor, ovlp = None, loc_mo=None):
    mo_occ = mf_spinor.mo_coeff[:,mf_spinor.mo_occ>0]
    if(ovlp is None):
        ovlp = mol.intor("int1e_ovlp_spinor")
    if(loc_mo is None):
        loc_mo = ibo.get_ibo(mol, mo_occ, spinor = True)
    act_mo, act_indices = pick_active(loc_mo, ovlp, active_atoms, mol.aoslice_2c_by_atom())
    act_occ = mf_spinor.mo_occ[act_indices]
        
    den_tot = mf_spinor.make_rdm1()
    den_act = mf_spinor.make_rdm1(act_mo, act_occ)
    
    den_env = den_tot - den_act
    e_tot = mf_spinor.e_tot
    e_act = mf_spinor.energy_tot(den_act, h1e=mf_spinor.get_hcore(mol), vhf=mf_spinor.get_veff(dm=den_act))

    v_correction = mf_spinor.get_veff(dm=den_tot) - mf_spinor.get_veff(dm=den_act)
    nelec_act = round(np.trace(matmul(ovlp, den_act)).real)
    nelec_act = int(nelec_act)

    print("Number of active electrons: %d" % nelec_act)
    return v_correction, nelec_act, e_tot, e_act, den_act, den_env

def spinor_in_scalar(active_atoms, mol, mf_low, mf_high, ovlp = None, huzinaga_factor = 1.0, mu = 1.0e6):
    ovlp_spinor = mol.intor("int1e_ovlp_spinor")
    n2c = ovlp_spinor.shape[0]
    if(ovlp is None):
        ovlp = mol.intor("int1e_ovlp")
    v, nelec_act, e_tot_low, e_act_low, da, db = get_embedding_potential(active_atoms, mol, mf_low, ovlp)
    da = 0.5*da
    db = 0.5*db
    ca, cb = mol.sph2spinor_coeff()
    v_spinor  = reduce(matmul, (ca.conj().T,  v, ca))
    v_spinor += reduce(matmul, (cb.conj().T,  v, cb))
    d_act_low = reduce(matmul, (ca.conj().T, da, ca))
    d_act_low+= reduce(matmul, (cb.conj().T, da, cb))
    d_env_low = reduce(matmul, (ca.conj().T, db, ca))
    d_env_low+= reduce(matmul, (cb.conj().T, db, cb))
    sd_spinor = matmul(ovlp_spinor, d_env_low)

    h1e_spinor = mf_high.get_hcore(mol)
    if(huzinaga_factor > 0.0):
        mu = 0.0
    h1e_embed = h1e_spinor + v_spinor + mu*matmul(sd_spinor, ovlp_spinor)
    mf_high.max_cycle = 100
    mf_high.conv_tol = 1e-7
    mo_occ = np.zeros(n2c)
    mo_occ[:nelec_act] = 1
    myscf_solver(mol, mf_high, ovlp_spinor, h1e_embed, mo_occ, sd_spinor, dm0 = d_act_low, huzinage_factor = huzinaga_factor)

    d_act_high = mf_high.make_rdm1(mf_high.mo_coeff, mf_high.mo_occ)
    e_act_high = mf_high.energy_tot(d_act_high, h1e=h1e_spinor, vhf=mf_high.get_veff(dm=d_act_high))
    e_correction = np.trace(matmul(v_spinor, d_act_high-d_act_low)).real
    e_tot = e_act_high - e_act_low + e_tot_low + e_correction
    print("Total energy: e_tot = e_act_high - e_act_low + e_tot_low + e_correction")
    print("%f = %f - %f + %f + %f" % (e_tot, e_act_high, e_act_low, e_tot_low, e_correction))
    e_correction = np.trace(matmul(v_spinor, -d_act_low)).real
    e_tot = mf_high.e_tot - e_act_low + e_tot_low + e_correction
    print("Total energy: mf_high.e_tot - e_act_low + e_tot_low + e_correction2")
    print("%f = %f - %f + %f + %f" % (e_tot, mf_high.e_tot, e_act_low, e_tot_low, e_correction))
    
def spinor_in_spinor(active_atoms, mol, mf_low, mf_high, huzinaga_factor = 1.0, mu=1.0e6):
    ovlp_spinor = mol.intor("int1e_ovlp_spinor")
    n2c = ovlp_spinor.shape[0]
    v, nelec_act, e_tot_low, e_act_low, da, db = get_embedding_potential_spinor(active_atoms, mol, mf_low, ovlp_spinor)
    sd_spinor = matmul(ovlp_spinor, db)

    h1e_spinor = mf_high.get_hcore(mol)
    if(huzinaga_factor > 0.0):
        mu = 0.0
    h1e_embed = h1e_spinor + v + mu*matmul(sd_spinor, ovlp_spinor)
    mf_high.max_cycle = 100
    mf_high.conv_tol = 1e-7
    mo_occ = np.zeros(n2c)
    mo_occ[:nelec_act] = 1
    myscf_solver(mol, mf_high, ovlp_spinor, h1e_embed, mo_occ, sd_spinor, dm0 = da, huzinage_factor = huzinaga_factor)

    d_act_high = mf_high.make_rdm1(mf_high.mo_coeff, mf_high.mo_occ)
    e_act_high = mf_high.energy_tot(d_act_high, h1e=h1e_spinor, vhf=mf_high.get_veff(dm=d_act_high))
    e_correction = np.trace(matmul(v, d_act_high-da)).real
    e_tot = e_act_high - e_act_low + e_tot_low + e_correction
    print("Total energy: e_tot = e_act_high - e_act_low + e_tot_low + e_correction")
    print("%f = %f - %f + %f + %f" % (e_tot, e_act_high, e_act_low, e_tot_low, e_correction))
    e_correction = np.trace(matmul(v, -da)).real
    e_tot = mf_high.e_tot - e_act_low + e_tot_low + e_correction
    print("Total energy: mf_high.e_tot - e_act_low + e_tot_low + e_correction2")
    print("%f = %f - %f + %f + %f" % (e_tot, mf_high.e_tot, e_act_low, e_tot_low, e_correction))

def scalar_in_scalar(active_atoms, mol, mf_low, mf_high, ovlp = None, huzinaga_factor = 0.5, mu=1.0e6):
    if(ovlp is None):
        ovlp = mol.intor("int1e_ovlp")
    v, nelec_act, e_tot_low, e_act_low, da, db = get_embedding_potential(active_atoms, mol, mf_low)
    sd = matmul(ovlp,db)
    nbas = v.shape[0]

    h1e = mf_high.get_hcore(mol)
    if(huzinaga_factor > 0.0):
        mu = 0.0
    h1e_embed = h1e + v + mu*matmul(sd, ovlp)
    mf_high.max_cycle = 100
    mf_high.conv_tol = 1e-7
    mo_occ = np.zeros(nbas)
    mo_occ[:nelec_act//2] = 2
    myscf_solver(mol, mf_high, ovlp, h1e_embed, mo_occ, sd, dm0 = da, huzinage_factor = huzinaga_factor)

    d_act_high = mf_high.make_rdm1(mf_high.mo_coeff, mf_high.mo_occ)
    e_act_high = mf_high.energy_tot(d_act_high, h1e=h1e, vhf=mf_high.get_veff(dm=d_act_high))
    e_correction = np.trace(matmul(v, d_act_high-da))
    e_tot = e_act_high - e_act_low + e_tot_low + e_correction
    print("Total energy: e_tot = e_act_high - e_act_low + e_tot_low + e_correction")
    print("%f = %f - %f + %f + %f" % (e_tot, e_act_high, e_act_low, e_tot_low, e_correction))

    e_correction = np.trace(matmul(v, -da))
    e_tot = mf_high.e_tot - e_act_low + e_tot_low + e_correction
    print("Total energy: mf_high.e_tot - e_act_low + e_tot_low + e_correction2")
    print("%f = %f - %f + %f + %f" % (e_tot, mf_high.e_tot, e_act_low, e_tot_low, e_correction))