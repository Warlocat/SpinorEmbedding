import numpy as np
import ibo
from pyscf import dft, scf, socutils
    
def pick_active_mulliken(loc_mo, ovlp, active_atoms, ao_slices):
    nocc = loc_mo.shape[1]
    act_mo_indices = []
    for i in range(nocc):
        for a in active_atoms:
            cia = loc_mo[ao_slices[a,2]:ao_slices[a,3],i]
            sa = ovlp[ao_slices[a,2]:ao_slices[a,3],ao_slices[a,2]:ao_slices[a,3]]
            mulliken_q = np.dot(cia.T.conj(), np.dot(sa, cia))
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

def get_embedding_potential(active_atoms, mol, mf, loc_mo=None):
    ovlp = mol.intor("int1e_ovlp")
    if(isinstance(mf, dft.rks.RKS) or isinstance(mf, scf.rhf.RKS)):
        mo_occ = mf.mo_coeff[:,mf.mo_occ>0]
    else:
        print("unrestricted case is not implemented yet")
        exit()
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
    ovlpDenv = np.dot(ovlp, den_env)
    nelec_act = round(np.trace(np.dot(ovlp, den_act)))
    return v_correction, ovlpDenv, nelec_act, e_tot, e_act

def spinor_embedding(active_atoms, mol, mf_low, mf_high, mu=1.0e6):
    v, sd, nelec_act, e_tot_low, e_act_low = get_embedding_potential(active_atoms, mol, mf_low)
    ca, cb = mol.sph2spinor_coeff()
    v_spinor  = ca.conj().T.dot( v).dot(ca)
    v_spinor += cb.conj().T.dot( v).dot(cb)
    sd_spinor = ca.conj().T.dot(sd).dot(ca)
    sd_spinor+= cb.conj().T.dot(sd).dot(cb)
    n2c = v_spinor.shape[0]

    h1e_spinor = mf_high.with_x2c.get_hcore(mol)
    def get_hcore(mol):
        return h1e_spinor + v_spinor + mu*sd_spinor.dot(mol.intor("int1e_ovlp_spinor"))
    mf_high.get_hcore = get_hcore
    def get_occ(mo_energy=None, mo_coeff=None):
        mo_occ = np.zeros(n2c)
        mo_occ[:nelec_act] = 1
        return mo_occ
    mf_high.get_occ = get_occ
    mf_high.kernel()
    d_act_high = mf_high.make_rdm1(mf_high.mo_coeff, mf_high.mo_occ)
    e_act_high = mf_high.energy_tot(d_act_high, h1e=h1e_spinor, vhf=mf_high.get_veff(dm=d_act_high))
    e_tot = e_act_high - e_act_low + e_tot_low
    print("Total energy: e_tot = e_act_high - e_act_low + e_tot_low")
    print("%f = %f - %f + %f" % (e_tot, e_act_high, e_act_low, e_tot_low))
    
def scalar_embedding(active_atoms, mol, mf_low, mf_high, mu=1.0e6):
    v, sd, nelec_act, e_tot_low, e_act_low = get_embedding_potential(active_atoms, mol, mf_low)
    nbas = v.shape[0]

    h1e = mf_high.get_hcore(mol)
    def get_hcore(mol):
        return h1e + v + mu*sd.dot(mol.intor("int1e_ovlp"))
    mf_high.get_hcore = get_hcore
    def get_occ(mo_energy=None, mo_coeff=None):
        mo_occ = np.zeros(nbas)
        mo_occ[:nelec_act//2] = 2
        return mo_occ
    mf_high.get_occ = get_occ
    mf_high.kernel()
    d_act_high = mf_high.make_rdm1(mf_high.mo_coeff, mf_high.mo_occ)
    e_act_high = mf_high.energy_tot(d_act_high, h1e=h1e, vhf=mf_high.get_veff(dm=d_act_high))
    e_tot = e_act_high - e_act_low + e_tot_low
    print("Total energy: e_tot = e_act_high - e_act_low + e_tot_low")
    print("%f = %f - %f + %f" % (e_tot, e_act_high, e_act_low, e_tot_low))