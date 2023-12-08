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

def get_embedding_potential(active_atoms, mol, mf, ovlp = None, loc_mo=None):
    if(isinstance(mf, dft.rks.RKS) or isinstance(mf, scf.rhf.RKS)):
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
    nelec_act = round(np.trace(np.dot(ovlp, den_act)))
    return v_correction, nelec_act, e_tot, e_act, den_act, den_env

def spinor_embedding(active_atoms, mol, mf_low, mf_high, ovlp = None, Huzinaga = True, mu=1.0e6):
    ovlp_spinor = mol.intor("int1e_ovlp_spinor")
    n2c = ovlp_spinor.shape[0]
    if(ovlp is None):
        ovlp = mol.intor("int1e_ovlp")
    v, nelec_act, e_tot_low, e_act_low, da, db = get_embedding_potential(active_atoms, mol, mf_low, ovlp)
    da = 0.5*da
    db = 0.5*db
    ca, cb = mol.sph2spinor_coeff()
    v_spinor  = ca.conj().T.dot( v).dot(ca)
    v_spinor += cb.conj().T.dot( v).dot(cb)
    
    d_act_low = ca.conj().T.dot(da*0.5).dot(ca)
    d_act_low+= cb.conj().T.dot(da*0.5).dot(cb)

    d_env_low = ca.conj().T.dot(db*0.5).dot(ca)
    d_env_low+= cb.conj().T.dot(db*0.5).dot(cb)

    sd_spinor = np.dot(ovlp_spinor, d_env_low)
    

    def get_occ(mo_energy=None, mo_coeff=None):
        mo_occ = np.zeros(n2c)
        mo_occ[:nelec_act] = 1
        return mo_occ
    mf_high.get_occ = get_occ
    h1e_spinor = mf_high.get_hcore(mol)
    if(Huzinaga):
        def get_hcore(mol):
            return h1e_spinor + v_spinor
        mf_high.get_hcore = get_hcore
        def get_fock(h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
            fock = scf.hf.get_fock(mf=mf_high, h1e=h1e, s1e=s1e, vhf=vhf, dm=dm, cycle=cycle, diis=diis,
                diis_start_cycle=diis_start_cycle, level_shift_factor=level_shift_factor, damp_factor=damp_factor)
            sdf = np.dot(sd_spinor, fock)
            return fock - sdf - sdf.T.conj()
        mf_high.get_fock = get_fock
    else:
        def get_hcore(mol):
            return h1e_spinor + v_spinor + mu*sd_spinor.dot(mol.intor("int1e_ovlp_spinor"))
        mf_high.get_hcore = get_hcore
    
    mf_high.kernel(dm0 = d_act_low)
    d_act_high = mf_high.make_rdm1(mf_high.mo_coeff, mf_high.mo_occ)
    e_act_high = mf_high.energy_tot(d_act_high, h1e=h1e_spinor, vhf=mf_high.get_veff(dm=d_act_high))
    e_correction = np.trace(np.dot(v_spinor, d_act_high-d_act_low)).real
    e_tot = e_act_high - e_act_low + e_tot_low + e_correction
    print("Total energy: e_tot = e_act_high - e_act_low + e_tot_low + e_correction")
    print("%f = %f - %f + %f + %f" % (e_tot, e_act_high, e_act_low, e_tot_low, e_correction))
    e_correction = np.trace(np.dot(v_spinor, -d_act_low)).real
    e_tot = mf_high.e_tot - e_act_low + e_tot_low + e_correction
    print("Total energy: mf_high.e_tot - e_act_low + e_tot_low + e_correction2")
    print("%f = %f - %f + %f + %f" % (e_tot, mf_high.e_tot, e_act_low, e_tot_low, e_correction))
    
def scalar_embedding(active_atoms, mol, mf_low, mf_high, ovlp = None, Huzinaga = True, mu=1.0e6):
    if(ovlp is None):
        ovlp = mol.intor("int1e_ovlp")
    v, nelec_act, e_tot_low, e_act_low, da, db = get_embedding_potential(active_atoms, mol, mf_low)
    sd = np.dot(ovlp,db)
    nbas = v.shape[0]

    def get_occ(mo_energy=None, mo_coeff=None):
        mo_occ = np.zeros(nbas)
        mo_occ[:nelec_act//2] = 2
        return mo_occ
    mf_high.get_occ = get_occ
    h1e = mf_high.get_hcore(mol)
    if(Huzinaga):
        def get_hcore(mol):
            return h1e + v
        mf_high.get_hcore = get_hcore
        def get_fock(h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
            fock = scf.hf.get_fock(mf=mf_high, h1e=h1e, s1e=s1e, vhf=vhf, dm=dm, cycle=cycle, diis=diis,
                diis_start_cycle=diis_start_cycle, level_shift_factor=level_shift_factor, damp_factor=damp_factor)
            sdf = 0.5*np.dot(sd, fock)
            return fock - sdf - sdf.T.conj()
        mf_high.get_fock = get_fock
    else:
        def get_hcore(mol):
            return h1e + v + 0.5*mu*sd.dot(mol.intor("int1e_ovlp"))
        mf_high.get_hcore = get_hcore
    
    mf_high.kernel(dm0 = da)
    d_act_high = mf_high.make_rdm1(mf_high.mo_coeff, mf_high.mo_occ)
    e_act_high = mf_high.energy_tot(d_act_high, h1e=h1e, vhf=mf_high.get_veff(dm=d_act_high))
    e_correction = np.trace(np.dot(v, d_act_high-da))
    e_tot = e_act_high - e_act_low + e_tot_low + e_correction
    print("Total energy: e_tot = e_act_high - e_act_low + e_tot_low + e_correction")
    print("%f = %f - %f + %f + %f" % (e_tot, e_act_high, e_act_low, e_tot_low, e_correction))
    e_correction = np.trace(np.dot(v, -da))
    e_tot = mf_high.e_tot - e_act_low + e_tot_low + e_correction
    print("Total energy: mf_high.e_tot - e_act_low + e_tot_low + e_correction2")
    print("%f = %f - %f + %f + %f" % (e_tot, mf_high.e_tot, e_act_low, e_tot_low, e_correction))