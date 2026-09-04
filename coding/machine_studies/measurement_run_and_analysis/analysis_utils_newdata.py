
import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mathphys.functions import load

# parent_dir = os.path.dirname(os.getcwd())
# ivu_orm_dir = os.path.join(parent_dir, "ivu_kick_respmat")
# sys.path.append(ivu_orm_dir)

# from ivu_orm import calc_ivu_orm


def calculate_currents(data):
    """Correct and add current information to the measured data.
    Also updates b_sum keys to hold the mean value per acquisition.

    Args:
        data (list): List of measurement dictionaries.

    Returns:
        data (list): List of measurement dictionaries.

    """

    for data_i in data:
        data_meas = data_i["data"]  # list of n_acq dictionaries
        for data_acq in data_meas: 
            b1_sum = np.mean(data_acq['b1_sum'])  # Take the mean over antennas and BPms
            b2_sum = np.mean(data_acq['b2_sum'])
            curr = data_acq['stored_current']

            bt_sum = b1_sum + b2_sum

            # Calculate and update/add current keys
            data_acq['b1_curr'] = b1_sum * curr / bt_sum  # [mA]
            data_acq['b2_curr'] = b2_sum * curr / bt_sum  # [mA]
            data_acq["delta_curr"] = data_acq["b1_curr"] - data_acq["b2_curr"]  # [mA]

            # Update sum keys
            data_acq['b1_sum'] = b1_sum
            data_acq['b2_sum'] = b2_sum
            
    return data


def add_charge_information(data):
    """Add charge information to the measured data.

    Args:
        data (list): List of measurement dictionaries.

    Returns:
        data (list): List of measurement dictionaries.

    """
    light_speed = 299_792_458.0  # [m/s]
    circumference = 518.3898999999917  # [m]
    rev_frequency = light_speed / circumference  # [Hz]

    for data_i in data:
        data_meas = data_i["data"]
        for data_acq in data_meas:     
            b1_charge = 1e6*data_acq['b1_curr'] / rev_frequency  # [nC]
            b2_charge = 1e6*data_acq['b2_curr'] / rev_frequency  # [nC]

            delta_charge = b1_charge - b2_charge  # [nC]

            data_acq["b1_charge"] = b1_charge
            data_acq["b2_charge"] = b2_charge
            data_acq["delta_charge"] = delta_charge
            
        # Add mean delta_charge over acquisitions
        delta_charge_array = np.array([data_acq['delta_charge'] for data_acq in data_meas])
        for data_acq in data_meas:
            data_acq['delta_charge_mean'] = float(np.mean(delta_charge_array))

    return data


def get_acquisition_parity_from_bpm_indices(data_acq):
    """Get acquisition BPM parity from the first BPM index.

    Args:
        data_acq (dict): One acquisition dictionary.

    Returns:
        parity (str): Acquisition parity, 'odd' or 'even'.

    """
    first_bpm_idx = int(data_acq["bpm_indcs"][0])  # []

    if first_bpm_idx % 2 == 0:
        parity = "even"
    else:
        parity = "odd"

    return parity


def join_odd_even_bpm_arrays(odd_array, even_array, return_list=False):
    """Join odd and even BPM arrays into one full BPM array.

    Args:
        odd_array (array-like): Odd BPM array. First dimension is BPM index.
        even_array (array-like): Even BPM array. First dimension is BPM index.
        return_list (bool): Whether to return a Python list instead of a numpy array.

    Returns:
        joined_array (numpy.ndarray or list): Full BPM array.
    """
    odd_array = np.asarray(odd_array)
    even_array = np.asarray(even_array)

    if odd_array.shape != even_array.shape:
        raise ValueError(
            "Odd and even BPM arrays must have the same shape: "
            f"odd={odd_array.shape}, even={even_array.shape}."
        )

    joined_shape = (2 * odd_array.shape[0],) + odd_array.shape[1:]  
    # e.g., (160,) + (500,) = (160, 500)

    joined_array = np.empty(
        joined_shape,
        dtype=np.result_type(odd_array, even_array),
    )

    # Convention:
    # even BPM data -> Python indexes 0, 2, 4, ...
    # odd  BPM data -> Python indexes 1, 3, 5, ...
    joined_array[::2] = even_array
    joined_array[1::2] = odd_array

    if return_list:
        joined_array = joined_array.tolist()

    return joined_array


def join_odd_even_acquisitions(data):
    """Join alternating odd/even BPM acquisitions into full-BPM acquisitions.

    Args:
        data (list): List of measurement dictionaries. Inside each
            configuration, data_i["data"] must be ordered as
            odd, even, odd, even, ...

    Returns:
        data_joined (list): Data with odd/even sub-acquisitions joined.
            Each acquisition has 160 BPMs.

    """
    position_keys = [
        "b1_posx",
        "b1_posy",
        "b2_posx",
        "b2_posy",
    ]

    scalar_mean_keys = [
        "stored_current",
        "rf_frequency",
        "tunex",
        "tuney",
        "ivu18_08_gap",
        "ivu18_14_gap",
        "b1_sum",
        "b2_sum",
        "bt_sum",
        "b1_curr",
        "b2_curr",
        "delta_curr",
        "b1_charge",
        "b2_charge",
        "delta_charge",
        "delta_charge_mean",
    ]

    data_joined = []

    for data_i in data:
        data_meas = data_i["data"]

        if len(data_meas) % 2 != 0:
            raise ValueError(
                "The number of sub-acquisitions must be even "
                "because the expected order is odd, even, odd, even, ..."
            )

        joined_meas = []

        first_parity = get_acquisition_parity_from_bpm_indices(data_meas[0])

        for acq in range(0, len(data_meas), 2):
            first_acq = data_meas[acq]
            second_acq = data_meas[acq + 1]

            if first_parity == "odd":
                odd_acq = first_acq
                even_acq = second_acq
            elif first_parity == "even":
                even_acq = first_acq
                odd_acq = second_acq

            joined_acq = odd_acq.copy()

            for key in position_keys:
                joined_acq[key] = join_odd_even_bpm_arrays(
                    odd_acq[key],
                    even_acq[key],
                )  # [um]

            for key in scalar_mean_keys:
                if key in odd_acq and key in even_acq:
                    joined_acq[key] = 0.5 * (
                        odd_acq[key] + even_acq[key]
                    )

            joined_acq["bpm_indcs"] = join_odd_even_bpm_arrays(
                odd_acq["bpm_indcs"],
                even_acq["bpm_indcs"],
            )

            joined_acq["bpm_names"] = join_odd_even_bpm_arrays(
                odd_acq["bpm_names"],
                even_acq["bpm_names"],
                return_list=True,
            )

            joined_meas.append(joined_acq)

        params = data_i["params"].copy()
        params["num_acquisitions"] = len(joined_meas)
        params["parity"] = "all"
        params["acquisition_mode"] = "odd_even_joined"

        data_i_joined = data_i.copy()
        data_i_joined["params"] = params
        data_i_joined["data"] = joined_meas

        data_joined.append(data_i_joined)

    return data_joined


def make_delta_q_group(delta_charge, step_nC=0.1):
    """Create integer charge group from measured charge difference.

    Args:
        delta_charge (float): Measured charge difference [nC].
        step_nC (float): Charge grouping step [nC].

    Returns:
        delta_q_group (int): Integer charge group.

    """
    delta_q_group = int(np.round(delta_charge / step_nC))  # []
    return delta_q_group


def make_config_table(data, gap_key="ivu18_08_gap", delta_q_group_step_nC=0.1):
    """Create table with the main configuration parameters per measurement.

    Args:
        data (list): List of measurement dictionaries.
        gap_key (str): IVU gap key.
        delta_q_group_step_nC (float): Charge grouping step [nC].

    Returns:
        config_table (pandas.DataFrame): Table with one row per configuration.

    """
    rows = []

    for i, data_i in enumerate(data):
        params = data_i["params"]
        data_meas = data_i["data"]
        first_acq = data_meas[0]

        gap = np.mean([
            data_acq[gap_key]
            for data_acq in data_meas
        ])  # [mm]

        delta_charge = first_acq["delta_charge_mean"]  # [nC]
        y0 = params["y0"]  # [mm]
        parity = params.get("parity", "all")

        n_acq = len(data_meas)  # []
        n_bpms, n_turns = np.shape(first_acq["b1_posy"])  # [], []

        b1_curr = np.mean(np.array([
            data_acq["b1_curr"] for data_acq in data_meas
        ]))  # [mA]

        b2_curr = np.mean(np.array([
            data_acq["b2_curr"] for data_acq in data_meas
        ]))  # [mA]

        delta_curr = np.mean(np.array([
            data_acq["delta_curr"] for data_acq in data_meas
        ]))  # [mA]

        delta_q_group = make_delta_q_group(
            delta_charge,
            step_nC=delta_q_group_step_nC,
        )  # []

        rows.append({
            "cfg": i,
            "gap": gap,  # [mm]
            "y0": y0,  # [mm]
            "parity": parity,
            "n_acq": n_acq,
            "n_bpms": n_bpms,
            "n_turns": n_turns,

            "delta_charge": delta_charge,  # [nC]
            "delta_q_group": delta_q_group,
            "delta_q_group_nC": delta_q_group_step_nC * delta_q_group,

            "b1_curr": b1_curr,  # [mA]
            "b2_curr": b2_curr,  # [mA]
            "delta_curr": delta_curr,  # [mA]
        })

    config_table = pd.DataFrame(rows)

    return config_table


def plot_currents_vs_acquisition(data, config_table, bunch=1, cfgs=[0]):
    """Plot bunch current versus acquisition number.

    Args:
        data (list): List of selected measurement dictionaries.
        config_table (pandas.DataFrame): Table with information about
            the data to be analyzed.
        bunch (int): Number of the selected bunch (1 or 2)
        cfgs (list): Configuration indexes.

    """
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')

    for cfg in cfgs:
        row = config_table.iloc[cfg]
        data_meas = data[cfg]["data"]

        acq_idx = np.arange(len(data_meas))  # []

        b_curr_str = f'b{bunch}_curr'
        b_curr = np.array([
            data_acq[b_curr_str] for data_acq in data_meas
        ])  # [mA]

        label_cfg = (
            rf"gap={row['gap']:.2f} mm, "
            rf"$y_0$={row['y0']:.2f} mm"
        )

        ax.plot(
            acq_idx,
            b_curr,
            ".-",
            label=rf"{label_cfg}",
        )

    ax.set_title(rf"Mean bunch {bunch} current per acquisition")
    ax.set_xlabel("acquisition index []")
    ax.set_ylabel("current [mA]")
    ax.grid(True)
    fig.legend(loc='outside right')
    plt.show()


def stack_orbit_data(data, plane="y"):
    """Stack bunch orbit data for one plane.

    Args:
        data (list): List of measurement dictionaries.
        plane (str): Transverse plane. Use 'x' or 'y'.

    Returns:
        b1_pos (list): Bunch 1 position [µm].
        b2_pos (list): Bunch 2 position [µm].
        delta_pos (list): Bunch-to-bunch difference [µm].

        All 3 are lists of arrays with shape 
            (n_acq, n_bpms, n_turns), each element 
            corresponding to one configuration
    """
    if plane not in ["x", "y"]:
        raise ValueError("plane must be 'x' or 'y'.")

    key1 = f"b1_pos{plane}"
    key2 = f"b2_pos{plane}"

    b1_pos = []
    b2_pos = []

    for data_i in data:

        b1_cfg = np.array([
            data_acq[key1]
            for data_acq in data_i["data"]
        ])  # [µm]

        b2_cfg = np.array([
            data_acq[key2]
            for data_acq in data_i["data"]
        ])  # [µm]

        b1_pos.append(b1_cfg)
        b2_pos.append(b2_cfg)

    delta_pos = [
        b1 - b2
        for b1, b2 in zip(b1_pos, b2_pos)
    ]  # [µm]

    return b1_pos, b2_pos, delta_pos


def normalize_by_charge(delta_pos, config_table):
    """Normalize orbit difference by bunch charge difference.

    Args:
        delta_pos (numpy.ndarray): Bunch-to-bunch orbit difference [µm].
        config_table (pandas.DataFrame): Configuration table.

    Returns:
        delta_pos_norm (numpy.ndarray): Charge-normalized orbit difference [m/C].

    """
    delta_q = config_table["delta_charge"].to_numpy()  # [nC]
    delta_pos_norm = [
        cfg / delta_q[i]
        for i, cfg in enumerate(delta_pos)
    ]
    
    return delta_pos_norm


def get_delta_u(data, config_table, plane="y", normalized=False):
    """Get orbit-difference array for one plane.

    Args:
        data (list): List of measurement dictionaries.
        config_table (pandas.DataFrame): Configuration table.
        plane (str): Transverse plane. Use 'x' or 'y'.
        normalized (bool): Whether Delta u is charge-normalized.

    Returns:
        delta_u (numpy.ndarray): Orbit-difference array with shape
            (n_configs, n_acq, n_bpms, n_turns).

    """
    _, _, delta_u = stack_orbit_data(data, plane=plane)
    
    if normalized:
        delta_u = normalize_by_charge(delta_u, config_table)

    return delta_u


def get_ylabel(plane="y", normalized=False):
    """Get y label for plotting.

    Args:
        plane (str): Transverse plane.
        normalized (bool): Whether Delta u is charge-normalized.

    Returns:
        ylabel (str): Axis label.

    """
    if normalized:
        ylabel = rf"$\Delta {plane}/\Delta q$ [$\mu$m/nC]"
    else:
        ylabel = rf"$\Delta {plane}$ [$\mu$m]"

    return ylabel


def plot_single_profile(data, config_table, spos_bpms, cfg=0, acq=0, turn=0, plane="y", normalized=False):
    """Plot Delta u profile across BPMs for one configuration, acquisition and turn.

    Args:
        data (list): List of measurement dictionaries.
        config_table (pandas.DataFrame): Configuration table.
        spos_bpms (numpy.ndarray): Array with BPM positions.
        cfg (int): Configuration index.
        acq (int): Acquisition index.
        turn (int): Turn index.
        plane (str): Transverse plane.
        normalized (bool): Whether Delta u is charge-normalized.

    """
    delta_u = get_delta_u(data, config_table, plane=plane, normalized=normalized)
    bpm_indices = data[cfg]['data'][acq]['bpm_indcs']
    spos_bpms = spos_bpms[bpm_indices]

    ylabel = get_ylabel(plane=plane, normalized=normalized)

    profile = delta_u[cfg][acq, :, turn]

    row = config_table.iloc[cfg]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(spos_bpms, profile, ".-")

    ax.set_title(
        rf"{normalized*'Charge-norm. '}"
        rf"$\Delta {plane}$ profile, cfg={cfg}, acq={acq}, turn={turn} "
        rf"(gap={row['gap']:.1f} mm, "
        rf"$\Delta q$={row['delta_charge']:.1f} nC, "
        rf"$y_0$={row['y0']:.1f} mm)"
    )
    ax.set_xlabel("spos [m]")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    plt.show()


def plot_turns_in_acquisition(data, config_table, spos_bpms, cfg=0, acq=0, turns=[0], plane="y", normalized=False, legend=False):
    """Plot Delta u profiles across BPMs for all turns inside one acquisition.

    Args:
        data (list): List of measurement dictionaries.
        config_table (pandas.DataFrame): Configuration table.
        spos_bpms (numpy.ndarray): Array with BPM positions.
        cfg (int): Configuration index.
        acq (int): Acquisition index.
        turns (list): Turn indices.
        plane (str): Transverse plane.
        normalized (bool): Whether Delta u is charge-normalized.
        legend (bool): Whether to include legend.

    """
    delta_u = get_delta_u(data, config_table, plane=plane, normalized=normalized)
    bpm_indices = data[cfg]['data'][acq]['bpm_indcs']
    spos_bpms = spos_bpms[bpm_indices]
    
    ylabel = get_ylabel(plane=plane, normalized=normalized)

    row = config_table.iloc[cfg]

    fig, ax = plt.subplots(figsize=(10, 5))

    for turn in turns:
        profile = delta_u[cfg][acq, :, turn]  # [um] or [um/nC]
        ax.plot(
            spos_bpms,
            profile,
            ".-",
            alpha=0.4,
            label=rf"turn {turn}",
        )

    ax.set_title(
        rf"{normalized*'Charge-norm. '}"
        rf"$\Delta {plane}$ $vs$ $spos$ per turn, acq={acq} "
        rf"(gap={row['gap']:.1f} mm, "
        rf"$\Delta q$={row['delta_charge']:.2f} nC, "
        rf"$y_0$={row['y0']:.1f} mm)"
    )
    ax.set_xlabel("spos [m]")
    ax.set_ylabel(ylabel)
    ax.grid(True)

    if legend:
        ax.legend()

    plt.show()


def plot_one_bpm_for_acquisitions(data, config_table, spos_bpms, cfg=0, acqs=[0], bpm=0, plane="y", normalized=False, alpha=0.3):
    """Plot, for one BPM, the Delta u profile across turns for several acquisitions.

    Args:
        data (list): List of measurement dictionaries.
        config_table (pandas.DataFrame): Configuration table.
        spos_bpms (numpy.ndarray): Array with BPM positions.
        cfg (int): Configuration index.
        acqs (list): Acquisition indices.
        bpm (int): BPM index.
        plane (str): Transverse plane.
        normalized (bool): Whether Delta u is charge-normalized.

    """
    delta_u = get_delta_u(data, config_table, plane=plane, normalized=normalized)
    bpm_indices = data[cfg]['data'][0]['bpm_indcs']  # For each cfg, all acqs have the same bpm_indcs
    spos_bpms = spos_bpms[bpm_indices]

    ylabel = get_ylabel(plane=plane, normalized=normalized)

    row = config_table.iloc[cfg]

    fig, ax = plt.subplots(figsize=(10, 5))

    turns = range(delta_u[0].shape[2])

    for i, acq in enumerate(acqs):
        bpm_profile = delta_u[cfg][acq, bpm, :]  # [um] or [um/nC]

        ax.plot(
            turns,
            bpm_profile,
            ".-",
            alpha=alpha,
            color=f'C{i}',
            label=f"acq {acq}"
        )

        mean_value = np.mean(bpm_profile)
        ax.axhline(
            mean_value,
            linestyle='--',
            linewidth=2.0,
            color=f'C{i}',
            label=f"<acq {acq}> = {mean_value:.2f} µm"
        )

    ax.set_title(
        rf"{normalized*'Charge-norm. '}"
        rf"$\Delta {plane}$ BPM {bpm} $vs$ turns "
        rf"(gap={row['gap']:.1f} mm, "
        rf"$\Delta q$={row['delta_charge']:.2f} nC, "
        rf"$y_0$={row['y0']:.1f} mm)"
    )
    ax.set_xlabel("turn idx []")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()
    plt.show()


def plot_acquisition_means(data, config_table, spos_bpms, cfg=0, acqs="all", plane="y", normalized=False, error_bars=False, error_par="std", legend=False):
    """Plot mean Delta u profile for specified acquisitions of one configuration.

    Args:
        data (list): List of measurement dictionaries.
        config_table (pandas.DataFrame): Configuration table.
        spos_bpms (numpy.ndarray): Array with BPM positions.
        cfg (int): Configuration index.
        acqs (list or str): Acquisitions to plot. Use 'all' to plot all.
        plane (str): Transverse plane.
        normalized (bool): Whether Delta u is charge-normalized.
        error_bars (bool): Whether to include error bars.
        error_par (str): Error parameter. Use 'std' or 'sem'.
        legend (bool): Whether to include legend.

    """
    delta_u = get_delta_u(data, config_table, plane=plane, normalized=normalized)
    bpm_indices = data[cfg]['data'][0]['bpm_indcs']  # For each cfg, all acqs have the same bpm_indcs
    spos_bpms = spos_bpms[bpm_indices]

    ylabel = get_ylabel(plane=plane, normalized=normalized)

    delta_u_cfg = delta_u[cfg]  # shape (n_acq, n_bpms, n_turns)

    mean_by_acq = np.mean(delta_u_cfg, axis=2)  # shape (n_acq, n_bpms)
    std_by_acq = np.std(delta_u_cfg, axis=2)  # shape (n_acq, n_bpms)
    sem_by_acq = std_by_acq / np.sqrt(delta_u_cfg.shape[2])  # divide std by sqrt(n_turns)

    if error_par == "std":
        err_by_acq = std_by_acq
        err_label = "std"
    elif error_par == "sem":
        err_by_acq = sem_by_acq
        err_label = "sem"
    else:
        raise ValueError("error_par must be 'std' or 'sem'.")

    row = config_table.iloc[cfg]

    if acqs == "all":
        acqs = range(mean_by_acq.shape[0])

    fig, ax = plt.subplots(figsize=(10, 5))

    for acq in acqs:
        if error_bars:
            ax.errorbar(
                spos_bpms,
                mean_by_acq[acq, :],
                yerr=err_by_acq[acq, :],
                marker=".",
                capsize=3,
                alpha=0.5,
                label=rf"acq {acq}",
            )
        else:
            ax.plot(
                spos_bpms,
                mean_by_acq[acq, :],
                ".-",
                alpha=0.4,
                label=rf"acq {acq}",
            )

    ax.set_title(
        rf"{normalized*'Charge-norm. '}"
        rf"$\langle\Delta {plane}\rangle$ profile per acquisition "
        rf"(gap={row['gap']:.2f} mm, "
        rf"$\Delta q$={row['delta_charge']:.3f} nC, "
        rf"$y_0$={row['y0']:.2f} mm)"
    )
    ax.set_xlabel("spos [m]")
    ax.set_ylabel(ylabel)
    ax.grid(True)

    if legend:
        ax.legend()

    plt.show()


def process_delta_u(delta_u):
    """Calculate mean, std and standard error of Delta u for each configuration.

    Args:
        delta_u (numpy.ndarray): Orbit-difference array with shape
            (n_configs, n_acq, n_bpms, n_turns).

    Returns:
        delta_u_proc (dict): Dictionary with mean, std and sem arrays.
            Arrays have shape (n_configs, n_bpms).

    """
    mean = []
    std = []
    sem = []

    for cfg in delta_u:

        mean_cfg = np.mean(cfg, axis=(0,2))
        std_cfg = np.std(cfg, axis=(0,2))

        n_samples = cfg.shape[0] * cfg.shape[2]

        sem_cfg = std_cfg / np.sqrt(n_samples)

        mean.append(mean_cfg)
        std.append(std_cfg)
        sem.append(sem_cfg)

    delta_u_proc = {
        "mean": mean,
        "std": std,
        "sem": sem,
    }

    return delta_u_proc


def get_processed_delta_u(data, config_table, plane="y", normalized=False):
    """Get chosen processed Delta u data.

    Args:
        data (list): List of measurement dictionaries.
        config_table (pandas.DataFrame): Configuration table.
        plane (str): Transverse plane.
        normalized (bool): Whether to use charge-normalized Delta u.

    Returns:
        processed (dict): Processed data dictionary.

    """
    _, _, delta_u = stack_orbit_data(data, plane=plane)
    
    if normalized:
        delta_u = normalize_by_charge(delta_u, config_table)

    delta_u_proc = process_delta_u(delta_u)

    return delta_u_proc


def plot_config_mean(data, config_table, spos_bpms, cfgs=None, ref_cfg=None, plane="y", normalized=False, error_par="sem", alpha=0.7):
    """Plot mean profile and error bars for selected configurations.

    Args:
        data (list): List of measurement dictionaries.
        config_table (pandas.DataFrame): Configuration table.
        spos_bpms (numpy.ndarray): Array with BPM positions.
        cfgs (list): Configuration indexes. None uses all configurations.
        ref_cfg (int): Configuration used as zero reference. The reference
            configuration must contain data from all BPMs.
        plane (str): Transverse plane.
        normalized (bool): Whether Delta u is charge normalized.
        error_par (str): Error parameter. Use 'std' or 'sem'.
        alpha (float): Plot transparency.
        
    """

    delta_u_proc = get_processed_delta_u(
        data,
        config_table,
        plane=plane,
        normalized=normalized,
    )

    ylabel = get_ylabel(
        plane=plane,
        normalized=normalized,
    )

    if cfgs is None:
        cfgs = range(len(delta_u_proc["mean"]))

    fig, ax = plt.subplots(figsize=(10, 5))

    for cfg in cfgs:

        bpm_indices = data[cfg]["data"][0]["bpm_indcs"]
        spos_bpms_cfg = spos_bpms[bpm_indices]

        mean = delta_u_proc["mean"][cfg]
        err = delta_u_proc[error_par][cfg]

        if ref_cfg is not None:

            mean_ref = delta_u_proc["mean"][ref_cfg][bpm_indices]
            err_ref = delta_u_proc[error_par][ref_cfg][bpm_indices]

            mean = mean - mean_ref

            # Uncertainty of the difference between two independent measurements.
            if cfg == ref_cfg:
                err = np.zeros_like(err)
            else:
                err = np.sqrt(
                    err**2 + err_ref**2
                )

        row = config_table.iloc[cfg]

        ax.errorbar(
            spos_bpms_cfg,
            mean,
            yerr=err,
            marker=".",
            capsize=3,
            alpha=alpha,
            label=(
                rf"$gap$={row['gap']:.1f} mm | "
                rf"$\Delta q$={row['delta_charge']:.1f} nC | "
                rf"$y_0$={row['y0']:.1f} mm"
            ),
        )

    ax.set_title(
        rf"{normalized*'Charge-norm. '}"
        rf"$\langle\Delta {plane}\rangle_{{\rm acqs,turns}}$ profile"
    )

    ax.set_xlabel("spos [m]")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()

    plt.show()


def plot_delta_u_bpm_vs_y0(data, config_table, bpm=0, gap=None, plane="y", normalized=False, error_par="sem"):
    """Plot one BPM value as a function of y0.

    Args:
        data (list): List of measurement dictionaries.
        config_table (pandas.DataFrame): Configuration table.
        bpm (int): BPM index.
        gap (float): IVU gap [mm].
        plane (str): Transverse plane.
        normalized (bool): Whether to plot charge-normalized data.
        error_par (str): Error parameter. Use 'std' or 'sem'.

    """
    if gap is None:
        gap = config_table["gap"].unique()[0]  # [mm]

    delta_u_proc = get_processed_delta_u(
        data,
        config_table,
        plane=plane,
        normalized=normalized,
    )

    ylabel = get_ylabel(
        plane=plane,
        normalized=normalized,
    )

    bpm_indices = [
        data_cfg['data'][0]['bpm_indcs']
        for data_cfg in data
    ]

    bpm_index_list = [
        (np.where(bpm_idx == bpm))
        for bpm_idx in bpm_indices
    ]

    mean = np.array([
        delta_u_proc_mean_cfg[bpm_index[0][0]] for
        delta_u_proc_mean_cfg, bpm_index in 
        zip(delta_u_proc["mean"], bpm_index_list)
    ])  # [um] or [um/nC]

    err = np.array([
            delta_u_proc_err_cfg[bpm_index[0][0]] for
            delta_u_proc_err_cfg, bpm_index in
            zip(delta_u_proc[error_par], bpm_index_list)
        ])  # [um] or [um/nC]

    delta_q_group = config_table["delta_q_group"].to_numpy()  # []
    delta_q_group_nC = config_table["delta_q_group_nC"].to_numpy()  # [nC]
    gap_values = config_table["gap"].to_numpy()  # [mm]

    fig, ax = plt.subplots(figsize=(8, 5))

    groups = np.sort(np.unique(delta_q_group))
    groups_nC = np.sort(np.unique(delta_q_group_nC))

    for group, group_nC in zip(groups, groups_nC):
        mask = (
            np.isclose(gap_values, gap, rtol=0, atol=1e-6)
            & (delta_q_group == group)
        )

        idx = np.where(mask)[0]

        if len(idx) == 0:
            continue

        y0_vals = config_table.iloc[idx]["y0"].to_numpy()  # [mm]

        order = np.argsort(y0_vals)

        idx = idx[order]
        y0_vals = y0_vals[order]

        ax.errorbar(
            y0_vals,
            mean[idx],
            yerr=err[idx],
            marker="o",
            capsize=3,
            linestyle="-",
            label=rf"$\Delta q_{{\rm group}}$ = {group_nC:.1f} nC",
        )

    ax.set_title(
        rf"{normalized*'Charge-norm. '}"
        rf"$\Delta {plane}$ vs $y_0$, BPM {bpm}, "
        rf"gap = {gap:.2f} mm, error = {error_par}"
    )

    ax.set_xlabel(r"$y_0$ [mm]")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()
    plt.show()


# Fit part

def get_ivu_orms(orm, data):
    """Split IVU ORM vector into x and y planes and select 
        elements corresponding to proper BPM indices.

    Args:
        orm (numpy.ndarray): IVU orbit response vector [um/urad].
        data (list): List of measurement dictionaries.

    Returns:
        M_ivu_x_cfgs (list): List of horizontal IVU response columns
            for each data configuration [um/urad].
        M_ivu_y_cfgs (list): List of vertical IVU response columns
            for each data configuration [um/urad].

    """
    n_bpms = len(orm) // 2  # []

    M_ivu_x = orm[:n_bpms]  # [um/urad]
    M_ivu_y = orm[n_bpms:]  # [um/urad]

    M_ivu_x_cfgs = [
        M_ivu_x[data_i["data"][0]["bpm_indcs"]]
        for data_i in data
    ]

    M_ivu_y_cfgs = [
        M_ivu_y[data_i["data"][0]["bpm_indcs"]]
        for data_i in data
    ]

    return M_ivu_x_cfgs, M_ivu_y_cfgs


def filter_delta_u_ivu(delta_u_proc_mean, plane="y"):
    """Filter processed orbit difference to isolate the IVU contribution.

    Args:
        delta_u_proc_mean (numpy.ndarray): Processed orbit difference array
            with shape (n_configs, n_bpms) [um].
        plane (str): Transverse plane.

    Returns:
        delta_u_filt (numpy.ndarray): Filtered orbit difference with shape
            (n_configs, n_bpms) [um].

    """
    delta_u_filt = delta_u_proc_mean.copy()  # [um]

    return delta_u_filt


def fit_theta_u(delta_u_filt, M_ivu_u_cfgs, delta_u_err=None):
    """Project filtered orbit differences onto IVU response vectors.

    Args:
        delta_u_filt (list): Filtered orbit differences.
            One array per configuration [um].
        M_ivu_u_cfgs (list): IVU response vector for each
            data configuration [um/urad].
        delta_u_err (list): Orbit uncertainty per configuration [um].

    Returns:
        theta_fit (dict): Projection fit result.
    """

    theta = []
    theta_err = []
    model = []
    residual = []
    residual_rms = []
    chi2 = []
    reduced_chi2 = []

    for cfg in range(len(delta_u_filt)):

        delta = delta_u_filt[cfg]
        M = M_ivu_u_cfgs[cfg]

        if delta.shape[0] != M.shape[0]:
            raise ValueError(
                f"Configuration {cfg}: incompatible BPM dimensions "
                f"delta={delta.shape}, M={M.shape}"
            )

        if delta_u_err is None:
            weights = np.ones_like(delta)
            err = None

        else:
            err = np.maximum(delta_u_err[cfg], 1e-30)
            weights = 1 / err**2

        numerator = np.sum(
            weights * delta * M
        )

        denominator = np.sum(
            weights * M**2
        )

        theta_cfg = numerator / denominator

        model_cfg = theta_cfg * M
        residual_cfg = delta - model_cfg

        if err is None:
            theta_err_cfg = np.nan
            chi2_cfg = np.nan
            reduced_chi2_cfg = np.nan

        else:
            theta_err_cfg = 1 / np.sqrt(denominator)

            chi2_cfg = np.sum(
                (residual_cfg / err)**2
            )

            dof_cfg = len(delta) - 1

            if dof_cfg > 0:
                reduced_chi2_cfg = chi2_cfg / dof_cfg
            else:
                reduced_chi2_cfg = np.nan

        theta.append(theta_cfg)
        theta_err.append(theta_err_cfg)
        model.append(model_cfg)
        residual.append(residual_cfg)

        residual_rms.append(
            np.sqrt(np.mean(residual_cfg**2))
        )

        chi2.append(chi2_cfg)
        reduced_chi2.append(reduced_chi2_cfg)

    theta_fit = {
        "theta": np.array(theta),
        "theta_err": np.array(theta_err),
        "model": model,
        "residual": residual,
        "residual_rms": np.array(residual_rms),
        "chi2": np.array(chi2),
        "reduced_chi2": np.array(reduced_chi2),
    }

    return theta_fit


def fit_global_bpm_vector(delta_u_proc, M_ivu_u_cfgs, plane="y", error_par="sem"):
    """Fit global BPM vector projection for one plane.

    Args:
        delta_u_proc (dict): Processed Delta u dictionary.
        M_ivu_u_cfgs (list): list of IVU response vector for each 
            data configuration [um/urad].
        plane (str): Transverse plane.
        error_par (str): Error parameter. Use 'std' or 'sem'.

    Returns:
        bpm_fit (dict): Global BPM-vector fit result.

    """
    delta_u_filt = filter_delta_u_ivu(
        delta_u_proc["mean"],
        plane=plane,
    )  # [um]

    delta_u_err = delta_u_proc[error_par]  # [um]

    theta_fit = fit_theta_u(
        delta_u_filt=delta_u_filt,
        M_ivu_u_cfgs=M_ivu_u_cfgs,
        delta_u_err=delta_u_err,
    )

    bpm_fit = {
        "plane": plane,
        "delta_u_filt": delta_u_filt,
        "delta_u_err": delta_u_err,
        "theta_fit": theta_fit,
    }

    return bpm_fit


def fit_global_bpm_vectors(delta_x_proc, delta_y_proc, M_ivu_x_cfgs, M_ivu_y_cfgs, error_par="sem"):
    """Fit global BPM vector projection for x and y planes.

    Args:
        delta_x_proc (dict): Processed Delta x dictionary.
        delta_y_proc (dict): Processed Delta y dictionary.
        M_ivu_x_cfgs (list): List of horizontal IVU response vectors
            for each data configuration [um/urad].
        M_ivu_x_cfgs (list): List of vertical IVU response vectors
            for each data configuration [um/urad].
        error_par (str): Error parameter. Use 'std' or 'sem'.

    Returns:
        bpm_fits (dict): Global BPM-vector fit results for both planes.

    """
    bpm_fits = {
        "x": fit_global_bpm_vector(
            delta_u_proc=delta_x_proc,
            M_ivu_u_cfgs=M_ivu_x_cfgs,
            plane="x",
            error_par=error_par,
        ),
        "y": fit_global_bpm_vector(
            delta_u_proc=delta_y_proc,
            M_ivu_u_cfgs=M_ivu_y_cfgs,
            plane="y",
            error_par=error_par,
        ),
    }

    return bpm_fits


def plot_theta_projection(
    data,
    config_table,
    spos_bpms,
    delta_u_filt,
    delta_u_err,
    theta_fit,
    M_ivu_u_cfgs,
    plane="y",
    cfg=0,
    ref_cfg=None,
    normalized=False,
):
    """Plot measured orbit, theta projection and residual.

    Args:
        data (list): List of measurement dictionaries.
        config_table (pandas.DataFrame): Configuration table.
        spos_bpms (numpy.ndarray): Full BPM longitudinal positions [m].
        delta_u_filt (list): Mean orbit differences per configuration [um].
        delta_u_err (list): Orbit uncertainties per configuration [um].
        theta_fit (dict): Theta projection fit result.
        M_ivu_u_cfgs (list): IVU response vectors per configuration [um/urad].
        plane (str): Transverse plane.
        cfg (int): Configuration index.
        ref_cfg (int): Reference configuration. None plots absolute quantities.
        normalized (bool): Whether to plot charge-normalized quantities.

    """

    row = config_table.iloc[cfg]

    bpm_indices = data[cfg]["data"][0]["bpm_indcs"]
    spos_cfg = spos_bpms[bpm_indices]

    delta_cfg = delta_u_filt[cfg]
    err_cfg = delta_u_err[cfg]

    theta_cfg = theta_fit["theta"][cfg]  # [urad]
    theta_err_cfg = theta_fit["theta_err"][cfg]  # [urad]

    M = M_ivu_u_cfgs[cfg]  # [um/urad]

    q_cfg = row["delta_charge"]  # [nC]

    if ref_cfg is None:

        if normalized:
            delta_plot = delta_cfg / q_cfg
            delta_err_plot = err_cfg / np.abs(q_cfg)

            theta_plot = theta_cfg / q_cfg
            theta_err_plot = theta_err_cfg / np.abs(q_cfg)

            ylabel = r"orbit difference [$\mu$m/nC]"
            theta_unit = r"$\mu$rad/nC"

        else:
            delta_plot = delta_cfg
            delta_err_plot = err_cfg

            theta_plot = theta_cfg
            theta_err_plot = theta_err_cfg

            ylabel = r"orbit difference [$\mu$m]"
            theta_unit = r"$\mu$rad"

        title_ref = ""

    else:

        row_ref = config_table.iloc[ref_cfg]

        # ref_cfg contains all BPMs, so global BPM indices can be used directly.
        delta_ref = delta_u_filt[ref_cfg][bpm_indices]
        err_ref = delta_u_err[ref_cfg][bpm_indices]

        theta_ref = theta_fit["theta"][ref_cfg]  # [urad]
        theta_err_ref = theta_fit["theta_err"][ref_cfg]  # [urad]

        q_ref = row_ref["delta_charge"]  # [nC]

        if normalized:

            delta_plot = (
                delta_cfg / q_cfg
                - delta_ref / q_ref
            )  # [um/nC]

            delta_err_plot = np.sqrt(
                (err_cfg / q_cfg)**2
                + (err_ref / q_ref)**2
            )  # [um/nC]

            theta_plot = (
                theta_cfg / q_cfg
                - theta_ref / q_ref
            )  # [urad/nC]

            theta_err_plot = np.sqrt(
                (theta_err_cfg / q_cfg)**2
                + (theta_err_ref / q_ref)**2
            )  # [urad/nC]

            ylabel = r"reference-subtracted orbit [$\mu$m/nC]"
            theta_unit = r"$\mu$rad/nC"

        else:

            # Scale the reference to the charge of the plotted configuration.
            charge_scale = q_cfg / q_ref

            delta_plot = (
                delta_cfg
                - charge_scale * delta_ref
            )  # [um]

            delta_err_plot = np.sqrt(
                err_cfg**2
                + (charge_scale * err_ref)**2
            )  # [um]

            theta_plot = (
                theta_cfg
                - charge_scale * theta_ref
            )  # [urad]

            theta_err_plot = np.sqrt(
                theta_err_cfg**2
                + (charge_scale * theta_err_ref)**2
            )  # [urad]

            ylabel = r"reference-subtracted orbit [$\mu$m]"
            theta_unit = r"$\mu$rad"

        title_ref = (
        rf"$y_{{0,\rm ref}}$={row_ref['y0']:.2f} mm"
    )

    # Orbit predicted by the fitted theta.
    model_plot = theta_plot * M
    residual_plot = delta_plot - model_plot

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.errorbar(
        spos_cfg,
        delta_plot,
        yerr=delta_err_plot,
        marker=".",
        linestyle="none",
        capsize=2,
        alpha=0.5,
        label="exp. data",
    )

    ax.plot(
        spos_cfg,
        model_plot,
        "-",
        alpha=0.9,
        label=(
            rf"model: $\theta M_{{\rm IVU}}$ |  "
            rf"$\theta={theta_plot:.3f}\pm{theta_err_plot:.3f}$ "
            rf"{theta_unit}"
        ),
    )

    ax.axhline(0, color='black', linestyle='--', linewidth=1.3)

    # ax.plot(
    #     spos_cfg,
    #     residual_plot,
    #     ".-",
    #     alpha=0.3,
    #     label="residual",
    # )

    ax.set_title(
        rf"$\Delta {plane}$ orbit deviation "
        rf"(gap={row['gap']:.1f} mm, "
        rf"$\Delta q$={q_cfg:.2f} nC, "
        rf"$y_0$={row['y0']:.2f} mm, {title_ref})"
    )

    ax.set_xlabel("spos [m]")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()

    plt.show()


def plot_ivu_projection(
    data,
    config_table,
    spos_bpms,
    delta_u_filt,
    delta_u_err,
    M_ivu_u_cfgs,
    kperp_fit,
    cfg,
    ref_cfg,
    plane="y",
    normalized=False,
):
    """Plot reference-subtracted IVU orbit and model from the kperp fit.

    Args:
        data (list): List of measurement dictionaries.
        config_table (pandas.DataFrame): Configuration table.
        spos_bpms (numpy.ndarray): Full BPM longitudinal positions [m].
        delta_u_filt (list): Mean orbit differences per configuration [um].
        delta_u_err (list): Orbit uncertainties per configuration [um].
        M_ivu_u_cfgs (list): IVU response vectors per configuration [um/urad].
        kperp_fit (dict): Kick-factor fit result.
        cfg (int): Configuration index.
        ref_cfg (int): Reference configuration index.
        plane (str): Transverse plane.
        normalized (bool): Whether to plot charge-normalized quantities.

    """

    row = config_table.iloc[cfg]
    row_ref = config_table.iloc[ref_cfg]

    bpm_indices = data[cfg]["data"][0]["bpm_indcs"]
    spos_cfg = spos_bpms[bpm_indices]

    # ref_cfg contains all BPMs.
    delta_cfg = delta_u_filt[cfg]
    delta_ref = delta_u_filt[ref_cfg][bpm_indices]

    err_cfg = delta_u_err[cfg]
    err_ref = delta_u_err[ref_cfg][bpm_indices]

    M = M_ivu_u_cfgs[cfg]  # [um/urad]

    q_cfg = row["delta_charge"]  # [nC]
    q_ref = row_ref["delta_charge"]  # [nC]

    y0_cfg = row["y0"] 
    y0_ref = row_ref["y0"] 
    delta_y0 = y0_cfg - y0_ref  

    # Global slope extracted from theta/dq vs y0.
    a = kperp_fit["a"]  # [urad/(nC mm)]
    a_err = kperp_fit["a_err"]  # [urad/(nC mm)]

    if normalized:

        # Reference-subtracted experimental orbit per unit charge.
        delta_ivu = (
            delta_cfg / q_cfg
            - delta_ref / q_ref
        )  # [um/nC]

        delta_ivu_err = np.sqrt(
            (err_cfg / q_cfg)**2
            + (err_ref / q_ref)**2
        )  # [um/nC]

        # IVU kick predicted by the global slope.
        theta_ivu = a * delta_y0  # [urad/nC]
        theta_ivu_err = np.abs(delta_y0) * a_err  # [urad/nC]

        model_ivu = theta_ivu * M  # [um/nC]

        ylabel = r"Orbit difference [$\mu$m/nC]"
        theta_unit = r"$\mu$rad/nC"

    else:

        # Scale reference orbit to the charge of cfg before subtraction.
        charge_scale = q_cfg / q_ref

        delta_ivu = (
            delta_cfg
            - charge_scale * delta_ref
        )  # [um]

        delta_ivu_err = np.sqrt(
            err_cfg**2
            + (charge_scale * err_ref)**2
        )  # [um]

        # Same slope prediction expressed at the charge of cfg.
        theta_ivu = q_cfg * a * delta_y0  # [urad]
        theta_ivu_err = np.abs(q_cfg * delta_y0) * a_err  # [urad]

        model_ivu = theta_ivu * M  # [um]

        ylabel = r"Orbit difference [$\mu$m]"
        theta_unit = r"$\mu$rad"

    residual_ivu = delta_ivu - model_ivu

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.errorbar(
        spos_cfg,
        delta_ivu,
        yerr=delta_ivu_err,
        marker=".",
        linestyle="none",
        capsize=2,
        alpha=0.6,
        label="reference-subtracted exp. data",
    )

    ax.plot(
        spos_cfg,
        model_ivu,
        "-",
        alpha=0.9,
        label=(
            rf"model: $\theta_{{\rm IVU}}M_{{\rm IVU}}$ | "
            rf"$\theta_{{\rm IVU}}="
            rf"{theta_ivu:.3f}\pm{theta_ivu_err:.3f}$ "
            rf"{theta_unit}"
        ),
    )

    ax.axhline(0, color='black', linestyle='--', linewidth=1.3)

    # ax.plot(
    #     spos_cfg,
    #     residual_ivu,
    #     ".-",
    #     alpha=0.4,
    #     label="residual",
    # )

    ax.set_title(
        rf"IVU orbit distortion "
        rf"(gap={row['gap']:.1f} mm, "
        rf"$y_0$={y0_cfg:.2f} mm, "
        rf"$y_{{0,\rm ref}}$={y0_ref:.2f} mm)"
    )

    ax.set_xlabel("spos [m]")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()

    plt.show()


def plot_theta_values(config_table, theta_fit, gaps="all", plane="y", normalized=False):
    """Plot projected theta values versus y0 with error bars.

    Args:
        config_table (pandas.DataFrame): Configuration table.
        theta_fit (dict): Projection fit result.
        gaps (list or str): IVU gaps [mm]. Use 'all' for all gaps.
        plane (str): Transverse plane.
        normalized (bool): Whether to plot theta/delta_q.

    """
    theta = theta_fit["theta"]          # [urad]
    theta_err = theta_fit["theta_err"]  # [urad]

    y0 = config_table["y0"].to_numpy()  # [mm]
    delta_q_meas = config_table["delta_charge"].to_numpy()  # [nC]
    delta_q_group = config_table["delta_q_group"].to_numpy()  # []
    delta_q_group_nC = config_table["delta_q_group_nC"].to_numpy()  # [nC]
    gap_values = config_table["gap"].to_numpy()  # [mm]

    if gaps == "all":
        gaps = np.sort(np.unique(gap_values))
    elif np.isscalar(gaps):
        gaps = [gaps]

    fig, ax = plt.subplots(figsize=(8, 5))

    if normalized:
        ylabel = r"$\hat{\theta}/\Delta q_{\rm meas}$ [$\mu$rad/nC]"
        title_start = rf"$\hat{{\theta}}^{plane}/\Delta q_{{\rm meas}}$"
    else:
        ylabel = r"$\hat{\theta}$ [$\mu$rad]"
        title_start = rf"$\hat{{\theta}}^{plane}$"

    for gap in gaps:
        mask_gap = np.isclose(gap_values, gap, rtol=0, atol=1e-6)
        groups = np.sort(np.unique(delta_q_group[mask_gap]))

        for group in groups:
            mask = mask_gap & (delta_q_group == group)

            if len(y0[mask]) == 0:
                continue

            group_nC = np.mean(delta_q_group_nC[mask])  # [nC]

            if normalized:
                y_plot = theta[mask] / delta_q_meas[mask]
                y_err = theta_err[mask] / np.abs(delta_q_meas[mask])
            else:
                y_plot = theta[mask]
                y_err = theta_err[mask]

            ax.errorbar(
                y0[mask],
                y_plot,
                yerr=y_err,
                fmt="o",
                capsize=4,
                label=(
                    rf"$gap$ = {gap:.1f} mm | "
                    rf"$\Delta q_{{\rm group}}$ = {group_nC:.1f} nC"
                ),
            )

    ax.set_title(rf"{title_start} vs $y_0$")
    ax.set_xlabel(r"$y_0$ [mm]")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()
    plt.show()


def get_delta_q_groups_from_nC(delta_q_groups_nC, delta_q_group_step_nC=0.1):
    """Convert charge-group values in nC to integer group labels.

    Args:
        delta_q_groups_nC (float or list): Charge groups [nC].
        delta_q_group_step_nC (float): Charge grouping step [nC].

    Returns:
        delta_q_groups (numpy.ndarray): Integer charge groups.

    """
    if delta_q_groups_nC is None:
        return None

    if np.isscalar(delta_q_groups_nC):
        delta_q_groups_nC = [delta_q_groups_nC]

    delta_q_groups = np.array([
        int(np.round(delta_q_nC / delta_q_group_step_nC))
        for delta_q_nC in delta_q_groups_nC
    ])

    return delta_q_groups


def select_configs(config_table, gap, delta_q_groups_nC=None, delta_q_group_step_nC=0.1):
    """Select configurations for a given gap and, optionally, charge groups.

    Args:
        config_table (pandas.DataFrame): Configuration table.
        gap (float): IVU gap [mm].
        delta_q_groups_nC (float or list): Charge groups [nC].
        delta_q_group_step_nC (float): Charge grouping step [nC].

    Returns:
        cfg_idx (numpy.ndarray): Selected configuration positional indexes.

    """
    gap_values = config_table["gap"].to_numpy()  # [mm]

    mask = np.isclose(gap_values, gap, rtol=0, atol=1e-6)

    delta_q_groups = get_delta_q_groups_from_nC(
        delta_q_groups_nC,
        delta_q_group_step_nC=delta_q_group_step_nC,
    )

    if delta_q_groups is not None:
        delta_q_group_values = config_table["delta_q_group"].to_numpy()  # []
        mask = mask & np.isin(delta_q_group_values, delta_q_groups)

    cfg_idx = np.where(mask)[0]  # []

    return cfg_idx


def fit_theta_over_dq_vs_y0(
    y0,
    delta_q_meas,
    delta_q_group,
    theta,
    theta_err,
):
    """Fit theta/delta_q_meas = offset(delta_q_group) + a*y0.

    Args:
        y0 (numpy.ndarray): Bump amplitudes [mm].
        delta_q_meas (numpy.ndarray): Measured bunch charge differences [nC].
        delta_q_group (numpy.ndarray): Integer charge groups.
        theta (numpy.ndarray): Projected kick angles [urad].
        theta_err (numpy.ndarray): Uncertainty of projected kick angles [urad].

    Returns:
        fit (dict): Weighted fit result.

    """

    theta_over_dq = theta / delta_q_meas  # [urad/nC]

    # Propagate theta uncertainty to theta/dq, neglecting dq uncertainty.
    theta_over_dq_err = (
        theta_err / np.abs(delta_q_meas)
    )  # [urad/nC]

    groups = np.sort(np.unique(delta_q_group))
    n_groups = len(groups)

    # Design matrix for:
    # theta/dq = offset(group) + a*y0
    #
    # One column per charge-group offset, plus one last column for y0.
    # Shape: (n_measurements, n_groups + 1)
    X = np.zeros((len(theta_over_dq), n_groups + 1))

    # Select which offset applies to each measurement:
    # rows belonging to group i get a 1 in column i.
    for i, group in enumerate(groups):
        mask = delta_q_group == group
        X[mask, i] = 1.0

    # Last column multiplies the common slope a.
    X[:, -1] = y0  # [mm]

    # Weighted least squares:
    # minimizing sum[(y_i - y_fit_i)^2 / sigma_i^2]
    weights = 1 / theta_over_dq_err**2
    sqrt_weights = np.sqrt(weights)

    # Multiplying each row by sqrt(weight) converts the weighted problem
    # into a standard least-squares problem.
    X_weighted = X * sqrt_weights[:, None]
    y_weighted = theta_over_dq * sqrt_weights

    # Solve X_weighted @ coeffs ~= y_weighted.
    # Shapes:
    #   X_weighted : (n_measurements, n_parameters)
    #   y_weighted : (n_measurements,)
    #   coeffs     : (n_parameters,)
    coeffs, _, _, _ = np.linalg.lstsq(
        X_weighted,
        y_weighted,
        rcond=None,
    )

    # coeffs = [offset_group1, offset_group2, ..., a]
    offsets = coeffs[:-1]  # [urad/nC]
    a = coeffs[-1]  # [urad/(nC mm)]

    # Evaluate the fitted model at the original measurement points.
    theta_over_dq_fit = X @ coeffs
    residual = theta_over_dq - theta_over_dq_fit

    # Number of measurements left after fitting all model parameters.
    dof = len(theta_over_dq) - len(coeffs)

    # Parameter covariance for weighted least squares:
    # cov = (X^T W X)^(-1).
    # pinv is used instead of inv for better numerical robustness.
    cov = np.linalg.pinv(
        X_weighted.T @ X_weighted
    )

    # Diagonal of cov contains parameter variances.
    coeffs_err = np.sqrt(np.diag(cov))

    offsets_err = coeffs_err[:-1]
    a_err = coeffs_err[-1]

    # Chi-square compares residuals with the expected measurement errors.
    chi2 = np.sum(
        (residual / theta_over_dq_err)**2
    )

    if dof > 0:
        reduced_chi2 = chi2 / dof
    else:
        reduced_chi2 = np.nan

    fit = {
        "theta_over_dq": theta_over_dq,
        "theta_over_dq_err": theta_over_dq_err,
        "theta_over_dq_fit": theta_over_dq_fit,
        "residual": residual,
        "groups": groups,
        "offsets": offsets,
        "offsets_err": offsets_err,
        "a": a,
        "a_err": a_err,
        "dof": dof,
        "chi2": chi2,
        "reduced_chi2": reduced_chi2,
    }

    return fit


def fit_kperp_from_theta(
    config_table,
    theta_fit,
    gap,
    beam_voltage=3.0e9,
    delta_q_groups_nC=None,
    delta_q_group_step_nC=0.1,
    plane="y",
):
    """Fit kick factor from projected theta values for one gap.

    Args:
        config_table (pandas.DataFrame): Configuration table.
        theta_fit (dict): Projection fit result.
        gap (float): IVU gap [mm].
        beam_voltage (float): Beam energy divided by charge [V].
        delta_q_groups_nC (float or list): Charge groups to include [nC].
        delta_q_group_step_nC (float): Charge grouping step [nC].
        plane (str): Transverse plane.

    Returns:
        kperp_fit (dict): Kick-factor fit result.

    """
    cfg_idx = select_configs(
        config_table=config_table,
        gap=gap,
        delta_q_groups_nC=delta_q_groups_nC,
        delta_q_group_step_nC=delta_q_group_step_nC,
    )

    if len(cfg_idx) == 0:
        raise ValueError("No configurations selected for this gap/charge group.")

    theta = theta_fit["theta"][cfg_idx]  # [urad]
    theta_err = theta_fit["theta_err"][cfg_idx]  # [µrad]

    y0 = config_table.iloc[cfg_idx]["y0"].to_numpy()  # [mm]
    delta_q_meas = config_table.iloc[cfg_idx]["delta_charge"].to_numpy()  # [nC]
    delta_q_group = config_table.iloc[cfg_idx]["delta_q_group"].to_numpy()  # []

    line_fit = fit_theta_over_dq_vs_y0(
        y0=y0,
        delta_q_meas=delta_q_meas,
        delta_q_group=delta_q_group,
        theta=theta,
        theta_err=theta_err,
    )

    a = line_fit["a"]  # [urad/(nC mm)]
    a_err = line_fit["a_err"]  # [urad/(nC mm)]

    kperp = 1e-6 * beam_voltage * a  # [V/(pC m)]
    kperp_err = 1e-6 * beam_voltage * a_err  # [V/(pC m)]

    kperp_fit = {
        "gap": gap,
        "plane": plane,
        "cfg_idx": cfg_idx,

        "theta": theta,
        "theta_err": theta_err,

        "delta_q_meas": delta_q_meas,
        "delta_q_group": delta_q_group,
        "y0": y0,

        "theta_over_dq": line_fit["theta_over_dq"],
        "theta_over_dq_err": line_fit["theta_over_dq_err"],
        "theta_over_dq_fit": line_fit["theta_over_dq_fit"],

        "groups": line_fit["groups"],
        "offsets": line_fit["offsets"],
        "offsets_err": line_fit["offsets_err"],

        "a": a,
        "a_err": a_err,

        "kperp": kperp,
        "kperp_err": kperp_err,

        "residual": line_fit["residual"],
        "dof": line_fit["dof"],
        "chi2": line_fit["chi2"],
        "reduced_chi2": line_fit["reduced_chi2"],

        "delta_q_group_step_nC": delta_q_group_step_nC,
    }

    return kperp_fit


def plot_theta_and_fit(kperp_fit, normalized=True):
    """Plot theta or theta/delta_q versus y0 with fitted lines.

    Args:
        kperp_fit (dict): Kick-factor fit result.
        normalized (bool): Whether to plot theta/delta_q.

    """
    gap = kperp_fit["gap"]  # [mm]
    plane = kperp_fit["plane"]  # []

    y0 = kperp_fit["y0"]  # [mm]
    theta = kperp_fit["theta"]  # [urad]
    theta_err = kperp_fit["theta_err"]  # [urad]
    delta_q_meas = kperp_fit["delta_q_meas"]  # [nC]
    delta_q_group = kperp_fit["delta_q_group"]  # []

    groups = kperp_fit["groups"]
    offsets = kperp_fit["offsets"]  # [urad/nC]
    a = kperp_fit["a"]  # [urad/(nC mm)]
    step_nC = kperp_fit["delta_q_group_step_nC"]  # [nC]

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, group in enumerate(groups):
        mask = delta_q_group == group
        group_nC = group * step_nC  # [nC]

        if normalized:
            y_plot = theta[mask] / delta_q_meas[mask]  # [urad/nC]
            y_err = (
                theta_err[mask]
                / np.abs(delta_q_meas[mask])
            )

            y0_fit = np.linspace(
                np.min(y0[mask]),
                np.max(y0[mask]),
                100,
            )  # [mm]

            y_fit = offsets[i] + a * y0_fit  # [urad/nC]

            ylabel = r"$\hat{\theta}/\Delta q_{\rm meas}$ [$\mu$rad/nC]"
            title_start = rf"$\hat{{\theta}}^{plane}/\Delta q_{{\rm meas}}$"

        else:
            y_plot = theta[mask]  # [urad]
            y_err = theta_err[mask]  # [µrad]

            order = np.argsort(y0[mask])
            y0_fit = y0[mask][order]  # [mm]

            y_fit = (
                delta_q_meas[mask][order]
                * kperp_fit["theta_over_dq_fit"][mask][order]
            )  # [urad]

            ylabel = r"$\hat{\theta}$ [$\mu$rad]"
            title_start = rf"$\hat{{\theta}}^{plane}$"

        ax.errorbar(
            y0[mask],
            y_plot,
            yerr=y_err,
            fmt="o",
            markersize=5.5,
            capsize=4,
            label=rf"$\Delta q_{{\rm group}}$ = {group_nC:.1f} nC",
        )

        ax.plot(
            y0_fit,
            y_fit,
            "-",
        )

    ax.set_title(
        rf"{title_start} vs $y_0$, "
        rf"gap = {gap:.1f} mm"
    )
    ax.set_xlabel(r"$y_0$ [mm]")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()
    plt.show()


def print_kperp_summary(kperp_fit):
    """Print kick-factor fit summary.

    Args:
        kperp_fit (dict): Kick-factor fit result.

    """
    print("Kick-factor fit")
    print("---------------")
    print(f"plane     : {kperp_fit['plane']}")
    print(f"gap       : {kperp_fit['gap']:.1f} mm")
    print(f"n configs : {len(kperp_fit['cfg_idx'])}")
    print("")
    print(
        f"a         : {kperp_fit['a']:.3e} +/- "
        f"{kperp_fit['a_err']:.3e} [urad/(nC mm)]"
    )
    print(
        f"kperp     : {kperp_fit['kperp']:.3f} +/- "
        f"{kperp_fit['kperp_err']:.3f} [V/(pC m)]"
    )
    print(
        f"chi2/dof  : {kperp_fit['reduced_chi2']:.3f}"
    )


def make_kperp_results_table(kperp_fits, labels=None):
    """Create results table from selected kick-factor fits.

    Args:
        kperp_fits (list): List of kick-factor fit dictionaries.
        labels (list): Names for each fit.

    Returns:
        results_table (pandas.DataFrame): Results summary table.

    """
    if labels is None:
        labels = [f"fit_{i}" for i in range(len(kperp_fits))]

    rows = []

    for label, kperp_fit in zip(labels, kperp_fits):
        step_nC = kperp_fit["delta_q_group_step_nC"]  # [nC]

        groups_nC = step_nC * kperp_fit["groups"]  # [nC]

        groups_label = ", ".join([
            f"{group_nC:.1f}"
            for group_nC in groups_nC
        ])

        rows.append({
            "label": label,
            "gap_mm": kperp_fit["gap"],  # [mm]
            "plane": kperp_fit["plane"],
            "delta_q_groups_nC": groups_label,
            "n_configs": len(kperp_fit["cfg_idx"]),
            "dof": kperp_fit["dof"],

            "a_urad_per_nC_mm": kperp_fit["a"],
            "a_err_urad_per_nC_mm": kperp_fit["a_err"],

            "kperp_v_per_pC_m": kperp_fit["kperp"],
            "kperp_err_v_per_pC_m": kperp_fit["kperp_err"],
        })

    results_table = pd.DataFrame(rows)

    return results_table


def plot_kperp_vs_gap(fit_table, expected_table=None):
    """Plot kick factor as a function of IVU gap.

    Args:
        fit_table (pandas.DataFrame): Fit summary table.
        expected_table (pandas.DataFrame): Expected/simulated values.

    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        fit_table["gap_mm"],
        fit_table["kperp_v_per_pC_m"],
        yerr=fit_table["kperp_err_v_per_pC_m"],
        marker="o",
        capsize=4,
        linestyle="-",
        label="measurement fit",
    )

    if expected_table is not None:
        ax.plot(
            expected_table["gap_mm"],
            expected_table["kperp_v_per_pC_m"],
            "s--",
            label="expected/simulated",
        )

    ax.set_title(r"Transverse kick factor versus IVU gap")
    ax.set_xlabel("gap [mm]")
    ax.set_ylabel(r"$k_\perp$ [V/(pC m)]")
    ax.grid(True)
    ax.legend()
    plt.show()