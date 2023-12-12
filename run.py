import embedding
from pyscf import gto, dft, scf
from pyscf.x2c import dft as x2cdft
from pyscf.x2c import x2c 
from pyscf.socutils import x2camf_hf
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
    verbose=4,
    symmetry=False
)

mf = dft.RKS(mol, xc='bp86')
mf.kernel()
active_atoms = [0,1,2]

# mf_spinor = x2cdft.UKS(mol)
# mf_spinor.xc = 'b3lyp'
mf_spinor = x2c.UHF(mol)
mf_spinor.with_x2c = x2camf_hf.X2CAMF_UHF(mol, with_gaunt=False, with_breit=False)
embedding.spinor_in_scalar(active_atoms, mol, mf, mf_spinor)

# mf_exact = dft.RKS(mol, xc='bp86')
# embedding.scalar_in_scalar(active_atoms, mol, mf, mf_exact)

# mf_scalar = dft.RKS(mol, xc='b3lyp')
# embedding.scalar_in_scalar(active_atoms, mol, mf, mf_scalar)


# mf = x2cdft.UKS(mol)
# mf.xc = 'bp86'
# mf.with_x2c = x2camf_hf.X2CAMF_UHF(mol, with_gaunt=False, with_breit=False)
# mf.kernel()
# active_atoms = [0,1,2]
# mf_spinor = x2cdft.UKS(mol)
# mf_spinor.xc = 'b3lyp'
# mf_spinor.with_x2c = x2camf_hf.X2CAMF_UHF(mol, with_gaunt=False, with_breit=False)
# embedding.spinor_in_spinor(active_atoms, mol, mf, mf_spinor)


# mf = scf.RHF(mol)
# mf.kernel()
# active_atoms = [0,1,2]

# mf_spinor = x2c.UHF(mol)
# mf_spinor.with_x2c = x2camf_hf.X2CAMF_UHF(mol, with_gaunt=False, with_breit=False)
# embedding.spinor_in_scalar(active_atoms, mol, mf, mf_spinor)