import numpy as np
    
def pick_active_mulliken(loc_mo, ovlp, active_atoms, ao_slices):
    nocc = loc_mo.shape[1]
    active_orbs = []
    for i in range(nocc):
        for a in active_atoms:
            cia = loc_mo[ao_slices[a,2]:ao_slices[a,3],i]
            sa = ovlp[ao_slices[a,2]:ao_slices[a,3],ao_slices[a,2]:ao_slices[a,3]]
            mulliken_q = np.dot(cia.T.conj(), np.dot(sa, cia))
            if(mulliken_q > 0.4):
                active_orbs.append(i)
                break
    mo_active = loc_mo[:,active_orbs].copy()
    return mo_active, active_orbs

def pick_active(loc_mo, ovlp, active_atoms, ao_slices, method="mulliken"):
    if(method == "mulliken"):
        return pick_active_mulliken(loc_mo, ovlp, active_atoms, ao_slices)
    else:
        print("ERROR: Unknown method in pick_active: %s" % method)
        exit()