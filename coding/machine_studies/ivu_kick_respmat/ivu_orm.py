
import numpy as np

import pyaccel
from pymodels import si



def set_ivu_kick(model, ivu_indices, vkick):
    """Set vertical kick in the selected IVU elements.

    Args:
        model: Accelerator model.
        ivu_indices (list): Indices of the IVU elements.
        vkick (float): Vertical kick to apply in each IVU element [rad].

    """
    for idx in ivu_indices:
        model[idx].pass_method = 'str_mpole_symplectic4_pass'
        model[idx].vkick_polynom = vkick


def calc_ivu_orm(delta_vkick=5e-6, ivu_number=1):
    """Calculate IVU orbit response matrix column.

    Args:
        delta_vkick (float, optional): Finite difference kick [rad].
            Defaults to 5e-6.
        ivu_number (int, optional): IVU number to use. Use 1 for the first
            IVU18 and 2 for the second IVU18. Defaults to 1.

    Returns:
        orm (numpy.ndarray): IVU response column with shape (2*n_bpms,).
            Each element has units of [m/rad] (or equivalently [µm/µrad])
            The first n_bpms entries correspond to the horizontal response,
            and the last n_bpms entries correspond to the vertical response.
        info (dict): Dictionary with auxiliary information.

    """

    model = si.create_accelerator()

    spos = pyaccel.lattice.find_spos(model, indices='open')
    bpm_indices = pyaccel.lattice.find_indices(model, 'fam_name', 'BPM')
    ivu_indices_all = pyaccel.lattice.find_indices(model, 'fam_name', 'IVU18')

    if ivu_number == 1:
        ivu_indices = ivu_indices_all[:2]
    elif ivu_number == 2:
        ivu_indices = ivu_indices_all[2:]
    else:
        raise ValueError("Invalid IVU number. Use 1 or 2.")
    
    set_ivu_kick(model, ivu_indices, +delta_vkick/2)
    cod_pos = pyaccel.tracking.find_orbit4(model, indices='open')  # [m]

    set_ivu_kick(model, ivu_indices, -delta_vkick/2)
    cod_neg = pyaccel.tracking.find_orbit4(model, indices='open')  # [m]

    set_ivu_kick(model, ivu_indices, 0.0)

    n_bpms = len(bpm_indices)
    orm = np.zeros(2*n_bpms)

    orm[:n_bpms] = (cod_pos[0, bpm_indices] - cod_neg[0, bpm_indices]) / delta_vkick  # [m/rad]
    orm[n_bpms:] = (cod_pos[2, bpm_indices] - cod_neg[2, bpm_indices]) / delta_vkick  # [m/rad]

    info = {
        'model': model,
        'spos': spos,  # [m]
        'spos_bpms':spos[bpm_indices],  # [m]
        'bpm_indices': bpm_indices,
        'ivu_indices': ivu_indices,
        'ivu_indices_all': ivu_indices_all,
        'delta_vkick': delta_vkick,  # [rad]
    }

    return orm, info