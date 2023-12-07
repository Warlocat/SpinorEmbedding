import numpy as np
import ibo, embedding
from pyscf import gto, dft, x2c
from pyscf.socutils import x2camf_hf
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
active_atoms = [0,1,2]

mf_spinor = x2c.UKS(mol)
mf.xc = 'b3lyp'
mf.with_x2c = x2camf_hf.X2CAMF_UHF(mol, with_gaunt=True, with_breit=True)
embedding.spinor_embedding(active_atoms, mol, mf, mf_spinor)

# embedding.scalar_embedding(active_atoms, mol, mf, mf)
