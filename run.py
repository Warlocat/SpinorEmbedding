import numpy as np
import ibo, embedding
from pyscf import gto, dft, x2c
from pyscf.tools import molden
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
    basis="cc-pvdz",
    verbose=4
)

mf = dft.RKS(mol, xc='b3lyp')
mf.kernel()
ci = mf.mo_coeff[:,mf.mo_occ>0]


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


s1 = mol.intor("int1e_ovlp")
s2 = mol_minao.intor("int1e_ovlp")
s12 = gto.intor_cross('int1e_ovlp', mol, mol_minao)
iaos = ibo.get_iao(s1, s2, s12, ci)
ao_slices = mol_minao.aoslice_by_atom()
ibos = ibo.get_ibo(s1, ao_slices, ci, iaos)
molden.from_mo(mol, "iao.molden", iaos)
molden.from_mo(mol, "ibo.molden", ibos)

ao_slices = mol.aoslice_by_atom()
active_atoms = [0,1,2]
mo_active, active_orbs = embedding.pick_active(ibos, s1, active_atoms, ao_slices)
molden.from_mo(mol, "mo_act.molden", mo_active)



# mf_spinor = x2c.dft.RKS(mol, xc='b3lyp')
# mf_spinor.kernel()
# ci_spinor = mf_spinor.mo_coeff[:,mf_spinor.mo_occ>0]
# s1_spinor = mol.intor("int1e_ovlp_spinor")
# s2_spinor = mol_minao.intor("int1e_ovlp_spinor")
# s12_spinor = gto.intor_cross('int1e_ovlp_spinor', mol, mol_minao)
# iao_spinor = ibo.get_iao(s1_spinor, s2_spinor, s12_spinor, ci_spinor)
# ao_slices_spinor = mol_minao.aoslice_2c_by_atom()
# ibo_spinor = ibo.get_ibo_spinor(s1_spinor, ao_slices_spinor, ci_spinor, iao_spinor)
# ibo_ghf = np.vstack(mol.sph2spinor_coeff()).dot(ibo_spinor)
# for ii in range(ibo.shape[1]):
#     cube.orbital(mol, f'ibo_ghf_{ii}', ibo_ghf[:,2*ii:2*ii+1])

# iaopyscf = iao_pyscf(mol, ci)
# iaopyscf = SymOrth(iaopyscf, s1)
# ibopyscf = ibo_pyscf(mol, ci, iaos=iao)
# molden.from_mo(mol, "iaopyscf.molden", iaopyscf)
# molden.from_mo(mol, "ibopyscf.molden", ibopyscf)
