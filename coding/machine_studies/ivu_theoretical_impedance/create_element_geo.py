#!/usr/bin/env python-sirius

import os
import sys
import importlib

import numpy as _np

import pycolleff.impedances as imp
import pycolleff.process_wakes as ems

sys.path.append(os.path.abspath('../'))
utils = importlib.import_module('utils')


def main(name=None):
    folder = [
        '', 'home', 'facs', 'data', 'em_simulations',
        'transitions_undulators', 'delta52', 'fullchamb_gap1350', 'echo3d',
        'll', 'ss100_bl0500', 'analysis', 'SimulData.pickle']
    wakedir = os.path.sep.join(folder)
    dataL = ems.load_processed_data(wakedir)

    folder[-4] = 'dy'
    wakedir = os.path.sep.join(folder)
    dataY = ems.load_processed_data(wakedir)

    folder[-4] = 'dx'
    folder[-6] = 'fullchamb_gap1350_symm_hplane'
    wakedir = os.path.sep.join(folder)
    dataX = ems.load_processed_data(wakedir)

    wl = dataL.freq*2*_np.pi
    wx = dataX.freq*2*_np.pi
    wy = dataY.freq*2*_np.pi
    w = utils.unique_sorted(wl, wx, wy)

    sl = dataL.s
    sx = dataX.s
    sy = dataY.s
    s = utils.unique_sorted(sl, sx, sy)

    el = imp.Element(name=name)
    el.ang_freq = w
    el.pos = s

    el.Zll = utils.interpolate_impedance(w, wl, dataL.Zll)
    # Detuning Impedance from ECHO3D is not reliable yet
    # el.Zqx = utils.interpolate_impedance(w, wl, dataL.Zqx)
    # el.Zqy = utils.interpolate_impedance(w, wl, dataL.Zqy)
    el.Zdy = utils.interpolate_impedance(w, wy, dataY.Zdy)
    el.Zdx = utils.interpolate_impedance(w, wx, dataX.Zdx)

    el.Wll = utils.interpolate_wake(s, sl, dataL.Wll)
    # Detuning Impedance from ECHO3D is not reliable yet
    # el.Wqx = utils.interpolate_wake(s, sl, dataL.Wqx)
    # el.Wqy = utils.interpolate_wake(s, sl, dataL.Wqy)
    el.Wdx = utils.interpolate_wake(s, sx, dataX.Wdx)
    el.Wdy = utils.interpolate_wake(s, sy, dataY.Wdy)
    return el


if __name__ == '__main__':
    el = main(name='Trans Delta52')
    utils.plot_element(el, props='all', logx=False, logy=False)
    el.save(overwrite=True)
