#!/usr/bin/env python-sirius

import os
import sys
import importlib
import warnings

import numpy as np

import pycolleff.impedances as imp
import pycolleff.materials_params as mat_par

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath('../'))
utils = importlib.import_module('utils')


def main():
    el = imp.Element(name='RW Delta52')

    al_cond = mat_par.al_cond
    al_rel_time = mat_par.al_rel_time
    neg_cond = mat_par.neg_cond
    neg_rel_time = mat_par.neg_rel_time
    neg_thick = 0.65e-6  # thicknes for undulators < than regular chamber.
    ndfe_cond = mat_par.ndfe_cond
    ndfe_mur = mat_par.ndfe_mur

    length = 1.439  # [m] undulator length is 1.2m
    energy = 3e9  # [eV]

    epb = np.array([1, 1, 1, 1, 1])
    mub = np.array([1, 1, 1, 1, ndfe_mur])
    ange = np.array([0, 0, 0, 0, 0])
    angm = np.array([0, 0, 0, 0, 0])
    sigmadc = np.array([0, neg_cond, al_cond, 0, ndfe_cond])  # [S]
    tau = np.array([0, neg_rel_time, al_rel_time, 0, 0], dtype=float)  # [s]

    radius = 7.6e-3/2 + np.array([-neg_thick, 0, 0.9e-3, 1.0e-3])  # [m]

    ang_freq = imp.get_default_reswall_w(radius=radius[0], energy=energy)

    epr, mur = imp.prepare_inputs_epr_mur(
        ang_freq, epb, mub, ange, angm, sigmadc, tau)

    # There is no undulator yet, so the last layer is air:
    epr = np.delete(epr, -1, axis=0)
    mur = np.delete(mur, -1, axis=0)
    radius = np.delete(radius, -1, axis=0)

    # The format of the chamber is an elipse, so the parallel plates model
    # suits better than round:
    Zll, Zdx, Zdy, Zqx, Zqy = imp.multilayer_flat_chamber(
        ang_freq, length, energy, epr, mur, radius, precision=70)

    Zll, ang_freq = imp.get_impedance_for_negative_w(
        Zll, ang_freq, impedance_type='ll')
    el.ang_freq = ang_freq
    el.Zll = Zll
    el.Zdx = imp.get_impedance_for_negative_w(Zdx, impedance_type='t')
    el.Zdy = imp.get_impedance_for_negative_w(Zdy, impedance_type='t')
    el.Zqx = imp.get_impedance_for_negative_w(Zqx, impedance_type='t')
    el.Zqy = imp.get_impedance_for_negative_w(Zqy, impedance_type='t')
    return el


def calc_wakes(el):
    idx = el.ang_freq > 0
    wp = el.ang_freq[idx]
    spos = imp.DEFAULT_SPOS_RW.copy()
    el.pos = spos
    el.Wll = imp.from_impedance_to_wake(spos, wp, el.Zll[idx], plane='long')
    el.Wdx = imp.from_impedance_to_wake(spos, wp, el.Zdx[idx], plane='trans')
    el.Wdy = imp.from_impedance_to_wake(spos, wp, el.Zdy[idx], plane='trans')
    el.Wqx = imp.from_impedance_to_wake(spos, wp, el.Zqx[idx], plane='trans')
    el.Wqy = imp.from_impedance_to_wake(spos, wp, el.Zqy[idx], plane='trans')


if __name__ == '__main__':
    el = main()

    calc_wakes(el)

    props = ['Zll', 'Zdx', 'Zdy', 'Zqx', 'Zqy']
    utils.plot_element(el, props=props)
    props = [p.replace('Z', 'W') for p in props]
    utils.plot_element(el, props=props, logx=False)

    el.save(overwrite=True)
