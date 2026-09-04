
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


def select_data_by_parity(data, parity="odd"):
    """Select measurement dictionaries by BPM parity.

    Args:
        data (list): List of measurement dictionaries.
        parity (str): BPM parity. Use 'odd', 'even' or 'all'.

    Returns:
        data_sel (list): Selected measurement dictionaries.

    """
    if parity == "all":
        data_sel = data
    elif parity in ["odd", "even"]:
        data_sel = [
            data_i for data_i in data
            if data_i["params"]["parity"] == parity
        ]
    else:
        raise ValueError("parity must be 'odd', 'even' or 'all'.")

    return data_sel


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

        gap = first_acq[gap_key]  # [mm]
        delta_charge = first_acq["delta_charge_mean"]  # [nC]
        y0 = params["y0"]  # [mm]

        parity = params["parity"]

        n_acq = params["num_acquisitions"]  # []
        n_bpms, n_turns = np.shape(first_acq["b1_posy"])  # [], []

        # Mean current values across acquisitions
        b1_curr = np.mean(np.array([
            data_acq["b1_curr"] for data_acq in data_meas
        ]))  # [mA]
        
        b2_curr = np.mean(np.array([
            data_acq["b2_curr"] for data_acq in data_meas
        ]))  # [mA]

        delta_curr = np.mean(np.array([
            data_acq["delta_curr"] for data_acq in data_meas
        ]))  # [mA]

        # In values to group similar Delta q's from diff. configurations
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


def plot_currents_vs_acquisition(config_table, bunch=1, cfgs=[0], data=None):
    """Plot bunch current versus acquisition number.

    Args:
        config_table (pandas.DataFrame): Table with information about
            the data to be analyzed.
        bunch (int): Number of the selected bunch (1 or 2)
        cfgs (list): Configuration indexes.
        data (list): List of selected measurement dictionaries.

    """
    if data is None:
        data = data_sel

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
            rf"$y_0$={row['y0']:.2f} mm, {row['parity']}"
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
        b1_pos (numpy.ndarray): Bunch 1 position array with shape
            (n_configs, n_acq, n_bpms, n_turns) [µm].
        b2_pos (numpy.ndarray): Bunch 2 position array with shape
            (n_configs, n_acq, n_bpms, n_turns) [µm].
        delta_pos (numpy.ndarray): Bunch-to-bunch difference with shape
            (n_configs, n_acq, n_bpms, n_turns) [µm].

    """
    if plane not in ["x", "y"]:
        raise ValueError("plane must be 'x' or 'y'.")

    key1 = f"b1_pos{plane}"
    key2 = f"b2_pos{plane}"

    b1_pos = np.array([
        [data_acq[key1] for data_acq in data_i["data"]]
        for data_i in data
    ])  # [µm]

    b2_pos = np.array([
        [data_acq[key2] for data_acq in data_i["data"]]
        for data_i in data
    ])  # [µm]

    delta_pos = b1_pos - b2_pos  # [µm]

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
    delta_pos_norm = delta_pos / delta_q[:, None, None, None]  # [µm/nC]
    
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


def get_spos_by_parity(spos_bpms, parity="odd"):
    """Select BPM positions for a given BPM parity.

    Args:
        spos_bpms (numpy.ndarray): BPM positions array [m].
        parity (str): BPM parity. Use 'odd', 'even' or 'all'.

    Returns:
        spos_bpms (numpy.ndarray): Array with selected BPMs 
            positions [m].

    """
    if parity == "all":
        spos_bpms = spos_bpms
    elif parity == 'odd':
        spos_bpms = spos_bpms[1::2]
    elif parity == 'even':
        spos_bpms = spos_bpms[::2]
    else:
        raise ValueError("parity must be 'odd', 'even' or 'all'.")

    return spos_bpms


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
    ylabel = get_ylabel(plane=plane, normalized=normalized)

    profile = delta_u[cfg, acq, :, turn]

    row = config_table.iloc[cfg]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(spos_bpms, profile, ".-")

    ax.set_title(
        rf"{normalized*'Charge-norm. '}"
        rf"$\Delta {plane}$ profile, cfg={cfg}, acq={acq}, turn={turn} "
        rf"(gap={row['gap']:.1f} mm, "
        rf"$\Delta q$={row['delta_charge']:.1f} nC, "
        rf"$y_0$={row['y0']:.1f} mm, "
        rf"{row['parity']})"
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
    ylabel = get_ylabel(plane=plane, normalized=normalized)

    row = config_table.iloc[cfg]

    fig, ax = plt.subplots(figsize=(10, 5))

    for turn in turns:
        profile = delta_u[cfg, acq, :, turn]  # [um] or [um/nC]
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
        rf"$y_0$={row['y0']:.1f} mm, "
        rf"{row['parity']})"
    )
    ax.set_xlabel("spos [m]")
    ax.set_ylabel(ylabel)
    ax.grid(True)

    if legend:
        ax.legend()

    plt.show()


def plot_one_bpm_for_acquisitions(data, config_table, spos_bpms, cfg=0, acqs=[0], bpm=0, plane="y", normalized=False):
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
    ylabel = get_ylabel(plane=plane, normalized=normalized)

    row = config_table.iloc[cfg]

    fig, ax = plt.subplots(figsize=(10, 5))

    turns = range(delta_u.shape[3])

    for i, acq in enumerate(acqs):
        bpm_profile = delta_u[cfg, acq, bpm, :]  # [um] or [um/nC]

        ax.plot(
            turns,
            bpm_profile,
            ".-",
            alpha=0.3,
            color=f'C{i}',
            label=f"acq {acq}"
        )

        mean_value = np.mean(bpm_profile)
        ax.axhline(
            mean_value,
            linestyle='--',
            linewidth=1.5,
            color=f'C{i}',
            label=f"<acq {acq}> = {mean_value:.2f} µm"
        )

    ax.set_title(
        rf"{normalized*'Charge-norm. '}"
        rf"$\Delta {plane}$ BPM {bpm} $vs$ turns "
        rf"(gap={row['gap']:.1f} mm, "
        rf"$\Delta q$={row['delta_charge']:.2f} nC, "
        rf"$y_0$={row['y0']:.1f} mm, "
        rf"{row['parity']})"
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
        rf"$y_0$={row['y0']:.2f} mm, "
        rf"{row['parity']})"
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
    mean = np.mean(delta_u, axis=(1, 3))  # [um] or [um/nC]
    std = np.std(delta_u, axis=(1, 3))  # [um] or [um/nC]

    n_samples = delta_u.shape[1] * delta_u.shape[3]  # []
    sem = std / np.sqrt(n_samples)  # [um] or [um/nC]

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


def plot_config_mean(data, config_table, spos_bpms, cfgs=[0], plane="y", normalized=False, error_par="sem"):
    """Plot mean profile and error bars for one configuration.

    Args:
        data (list): List of measurement dictionaries.
        config_table (pandas.DataFrame): Configuration table.
        spos_bpms (numpy.ndarray): Array with BPM positions.
        cfgs (list): Configuration indexes.
        plane (str): Transverse plane.
        normalized (bool): Whether Delta u is charge normalized.
        error_par (str): Error parameter. Use 'std' or 'sem'.

    """
    delta_u_proc = get_processed_delta_u(data, config_table, plane=plane, normalized=normalized)

    ylabel = get_ylabel(normalized=normalized)

    fig, ax = plt.subplots(figsize=(10, 5))

    for cfg in cfgs:
        mean = delta_u_proc["mean"][cfg]  # []
        err_key = error_par
        err = delta_u_proc[err_key][cfg]  # []

        row = config_table.iloc[cfg]

        ax.errorbar(
            spos_bpms,
            mean,
            yerr= err,
            marker=".",
            capsize=3,
            alpha=0.7,
            label=rf"$gap$={row['gap']:.1f} mm | $\Delta q$={row['delta_charge']:.1f} nC | $y_0$={row['y0']:.1f} mm"
        )

    ax.set_title(
        rf"{normalized*'Charge-norm. '}"
        rf"$\langle\Delta {plane}$"
        r"$\rangle_{acqs,turns}$ profile"
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

    mean = delta_u_proc["mean"][:, bpm]  # [um] or [um/nC]
    err = delta_u_proc[error_par][:, bpm] # [um] or [um/nC]

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

def get_ivu_orms_by_parity(orm, n_bpms=160, parity="odd"):
    """Select IVU ORM vectors for a given BPM parity.

    Args:
        orm (numpy.ndarray): IVU orbit response vector [um/urad].
        n_bpms (int): Number of BPMs in the full ORM.
        parity (str): BPM parity. Use 'odd', 'even' or 'all'.

    Returns:
        M_ivu_x (numpy.ndarray): Horizontal IVU response [um/urad].
        M_ivu_y (numpy.ndarray): Vertical IVU response [um/urad].

    """
    M_ivu_x_all = orm[:n_bpms]  # [um/urad]
    M_ivu_y_all = orm[n_bpms:]  # [um/urad]

    if parity == "all":
        M_ivu_x = M_ivu_x_all
        M_ivu_y = M_ivu_y_all
    elif parity == "odd":
        M_ivu_x = M_ivu_x_all[1::2]  # [um/urad]
        M_ivu_y = M_ivu_y_all[1::2]  # [um/urad]
    elif parity == "even":
        M_ivu_x = M_ivu_x_all[::2]  # [um/urad]
        M_ivu_y = M_ivu_y_all[::2]  # [um/urad]
    else:
        raise ValueError("parity must be 'odd', 'even' or 'all'.")

    return M_ivu_x, M_ivu_y


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


def fit_theta_u(delta_u_filt, M_ivu_u, delta_u_err=None):
    """Project filtered orbit differences onto the IVU response vector.

    Args:
        delta_u_filt (numpy.ndarray): Filtered orbit difference with shape
            (n_configs, n_bpms) [um].
        M_ivu_u (numpy.ndarray): IVU response vector [um/urad].
        delta_u_err (numpy.ndarray): Orbit-difference uncertainty with shape
            (n_configs, n_bpms) [um].

    Returns:
        theta_fit (dict): Projection fit result.

    """
    if M_ivu_u.shape[0] != delta_u_filt.shape[1]:
        raise ValueError(
            "M_ivu_u and delta_u_filt have incompatible BPM dimensions: "
            f"M_ivu_u has {M_ivu_u.shape[0]} BPMs, "
            f"delta_u_filt has {delta_u_filt.shape[1]} BPMs."
        )

    if delta_u_err is None:
        weights = np.ones_like(delta_u_filt)  # []
    else:
        delta_u_err = np.maximum(delta_u_err, 1e-30)  # [um]
        weights = 1.0 / delta_u_err**2  # [1/um²]

    numerator = np.sum(
        weights * delta_u_filt * M_ivu_u[None, :],
        axis=1,
    )  # [1/urad]

    denominator = np.sum(
        weights * M_ivu_u[None, :]**2,
        axis=1,
    )  # [1/urad²]

    theta = numerator / denominator  # [urad]

    model = theta[:, None] * M_ivu_u[None, :]  # [um]
    residual = delta_u_filt - model  # [um]
    residual_rms = np.sqrt(np.mean(residual**2, axis=1))  # [um]

    theta_fit = {
        "theta": theta,
        "model": model,
        "residual": residual,
        "residual_rms": residual_rms,
    }

    return theta_fit


def fit_global_bpm_vector(delta_u_proc, M_ivu_u, plane="y", error_par="sem"):
    """Fit global BPM vector projection for one plane.

    Args:
        delta_u_proc (dict): Processed Delta u dictionary.
        M_ivu_u (numpy.ndarray): IVU response vector [um/urad].
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
        M_ivu_u=M_ivu_u,
        delta_u_err=delta_u_err,
    )

    bpm_fit = {
        "plane": plane,
        "delta_u_filt": delta_u_filt,
        "delta_u_err": delta_u_err,
        "theta_fit": theta_fit,
    }

    return bpm_fit


def fit_global_bpm_vectors(delta_x_proc, delta_y_proc, M_ivu_x, M_ivu_y, error_par="sem"):
    """Fit global BPM vector projection for x and y planes.

    Args:
        delta_x_proc (dict): Processed Delta x dictionary.
        delta_y_proc (dict): Processed Delta y dictionary.
        M_ivu_x (numpy.ndarray): Horizontal IVU response vector [um/urad].
        M_ivu_y (numpy.ndarray): Vertical IVU response vector [um/urad].
        error_par (str): Error parameter. Use 'std' or 'sem'.

    Returns:
        bpm_fits (dict): Global BPM-vector fit results for both planes.

    """
    bpm_fits = {
        "x": fit_global_bpm_vector(
            delta_u_proc=delta_x_proc,
            M_ivu_u=M_ivu_x,
            plane="x",
            error_par=error_par,
        ),
        "y": fit_global_bpm_vector(
            delta_u_proc=delta_y_proc,
            M_ivu_u=M_ivu_y,
            plane="y",
            error_par=error_par,
        ),
    }

    return bpm_fits


def plot_theta_projection(
    config_table,
    spos_bpms,
    delta_u_filt,
    theta_fit,
    plane="y",
    cfg=0,
):
    """Plot filtered orbit, projected model and residual for one configuration.

    Args:
        config_table (pandas.DataFrame): Configuration table.
        spos_bpms (numpy.ndarray): BPM longitudinal positions [m].
        delta_u_filt (numpy.ndarray): Filtered orbit difference [um].
        theta_fit (dict): Projection fit result.
        plane (str): Transverse plane.
        cfg (int): Configuration index.

    """
    row = config_table.iloc[cfg]

    delta_u_cfg = delta_u_filt[cfg, :]  # [um]
    model_cfg = theta_fit["model"][cfg, :]  # [um]
    residual_cfg = theta_fit["residual"][cfg, :]  # [um]
    theta_cfg = theta_fit["theta"][cfg]  # [urad]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        spos_bpms,
        delta_u_cfg,
        ".",
        alpha=0.5,
        label="filtered data",
    )

    ax.plot(
        spos_bpms,
        model_cfg,
        "-",
        alpha=0.8,
        label=rf"$\hat{{\theta}}M_{{\rm IVU}}$, $\hat{{\theta}}={theta_cfg:.3f}$ urad",
    )

    ax.plot(
        spos_bpms,
        residual_cfg,
        ".-",
        alpha=0.4,
        label="residual",
    )

    ax.set_title(
        rf"$\Delta {plane}$ projection, cfg={cfg} "
        rf"(gap={row['gap']:.1f} mm, "
        rf"$\Delta q$={row['delta_charge']:.2f} nC, "
        rf"$y_0$={row['y0']:.2f} mm, "
        rf"{row['parity']})"
    )
    ax.set_xlabel("spos [m]")
    ax.set_ylabel(r"orbit difference [$\mu$m]")
    ax.grid(True)
    ax.legend()
    plt.show()


def plot_theta_values(config_table, theta_fit, gaps="all", plane="y", normalized=False):
    """Plot projected theta values versus y0.

    Args:
        config_table (pandas.DataFrame): Configuration table.
        theta_fit (dict): Projection fit result.
        gaps (list or str): IVU gaps [mm]. Use 'all' for all gaps.
        plane (str): Transverse plane.
        normalized (bool): Whether to plot theta/delta_q.

    """
    theta = theta_fit["theta"]  # [urad]

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
                y_plot = theta[mask] / delta_q_meas[mask]  # [urad/nC]
            else:
                y_plot = theta[mask]  # [urad]

            ax.plot(
                y0[mask],
                y_plot,
                marker="o",
                linewidth=1,
                markersize=5,
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


def fit_theta_over_dq_vs_y0(y0, delta_q_meas, delta_q_group, theta):
    """Fit theta/delta_q_meas = offset(delta_q_group) + a*y0.

    Args:
        y0 (numpy.ndarray): Bump amplitudes [mm].
        delta_q_meas (numpy.ndarray): Measured bunch charge differences [nC].
        delta_q_group (numpy.ndarray): Integer charge groups.
        theta (numpy.ndarray): Projected kick angles [urad].

    Returns:
        fit (dict): Fit result.

    """
    theta_over_dq = theta / delta_q_meas  # [urad/nC]

    groups = np.sort(np.unique(delta_q_group))
    n_groups = len(groups)

    X = np.zeros((len(theta_over_dq), n_groups + 1))  # []

    for i, group in enumerate(groups):
        mask = delta_q_group == group
        X[mask, i] = 1.0

    X[:, -1] = y0  # [mm]

    coeffs, _, _, _ = np.linalg.lstsq(X, theta_over_dq, rcond=None)

    offsets = coeffs[:-1]  # [urad/nC]
    a = coeffs[-1]  # [urad/(nC mm)]

    theta_over_dq_fit = X @ coeffs  # [urad/nC]
    residual = theta_over_dq - theta_over_dq_fit  # [urad/nC]

    dof = len(theta_over_dq) - len(coeffs)  # []

    if dof > 0:
        residual_var = np.sum(residual**2) / dof  # [(urad/nC)^2]
        cov = residual_var * np.linalg.pinv(X.T @ X)  # []
        coeffs_err = np.sqrt(np.diag(cov))  # []
    else:
        coeffs_err = np.full_like(coeffs, np.nan)  # []

    fit = {
        "theta_over_dq": theta_over_dq,
        "theta_over_dq_fit": theta_over_dq_fit,
        "residual": residual,
        "groups": groups,
        "offsets": offsets,
        "offsets_err": coeffs_err[:-1],
        "a": a,
        "a_err": coeffs_err[-1],
        "dof": dof,
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

    y0 = config_table.iloc[cfg_idx]["y0"].to_numpy()  # [mm]
    delta_q_meas = config_table.iloc[cfg_idx]["delta_charge"].to_numpy()  # [nC]
    delta_q_group = config_table.iloc[cfg_idx]["delta_q_group"].to_numpy()  # []

    line_fit = fit_theta_over_dq_vs_y0(
        y0=y0,
        delta_q_meas=delta_q_meas,
        delta_q_group=delta_q_group,
        theta=theta,
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
        "delta_q_meas": delta_q_meas,
        "delta_q_group": delta_q_group,
        "y0": y0,
        "theta_over_dq": line_fit["theta_over_dq"],
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

            order = np.argsort(y0[mask])
            y0_fit = y0[mask][order]  # [mm]

            y_fit = (
                delta_q_meas[mask][order]
                * kperp_fit["theta_over_dq_fit"][mask][order]
            )  # [urad]

            ylabel = r"$\hat{\theta}$ [$\mu$rad]"
            title_start = rf"$\hat{{\theta}}^{plane}$"

        ax.plot(
            y0[mask],
            y_plot,
            "o",
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
    print(f"dof       : {kperp_fit['dof']}")
    print("")
    print(
        f"a         : {kperp_fit['a']:.3e} +/- "
        f"{kperp_fit['a_err']:.3e} [urad/(nC mm)]"
    )
    print(
        f"kperp     : {kperp_fit['kperp']:.3f} +/- "
        f"{kperp_fit['kperp_err']:.3f} [V/(pC m)]"
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