import numpy as np
import matplotlib.pyplot as mplt

from mathphys.functions import load, save
from apsuite.commisslib.meas_bpms_signals import AcqBPMsSignals


# acqbpm = AcqBPMsSignals()
# acqbpm.params.acq_rate = 'ADCSwap'
# acqbpm.params.signals2acq = 'ABCD'
# acqbpm.params.nrpoints_after = 200000
# print(acqbpm.params)

# folder = '~/shared/screens_iocs/data_by_day/2022-05-31-SI_lifetime_meas/'
folder = './2022-05-31-SI_lifetime_meas/'

files = [
    'lifetime_two_single_bunches_rfgap=0.785MV_test.pickle',
    'lifetime_two_single_bunches_rfgap=0.897MV_test.pickle',
    'lifetime_two_single_bunches_rfgap=1.012MV_test.pickle',
    'lifetime_two_single_bunches_rfgap=1.162MV_test.pickle',
    'lifetime_two_single_bunches_rfgap=1.292MV_test.pickle',
    'lifetime_two_single_bunches_rfgap=1.457MV_test.pickle',
    'lifetime_two_single_bunches_rfgap=1.644MV_test.pickle',
    'lifetime_two_single_bunches_rfgap=1.735MV_test.pickle',
    'lifetime_two_single_bunches_rfgap=1.832MV_test.pickle',
    'lifetime_two_single_bunches_rfgap=1.939MV_test.pickle',
    'lifetime_two_single_bunches_rfgap=1.998MV_test.pickle',
    'lifetime_two_single_bunches_rfgap=2.053MV_test.pickle',
    'lifetime_two_single_bunches_rfgap=2.116MV_test.pickle',
    'lifetime_two_single_bunches_rfgap=2.189MV_test.pickle',
]


ant_raw = []
curr, tstmp = [], []
for fil in files:
    data = load(folder + fil)
    tmp = []
    for dt in data['data']:
        tmp.append([dt['antenna_' + ant] for ant in 'abcd'])
        curr.append(dt['current'])
        tstmp.append(dt['antenna_time'])
    ant_raw.extend(tmp)
ant_raw = np.array(ant_raw)
curr = np.array(curr)
tstmp = np.array(tstmp)
tstmp -= tstmp.min()

fig, ax = mplt.subplots()
ax.plot(ant_raw[:, 0].T)
fig.show()


ant_hil = AcqBPMsSignals.calc_hilbert_transform(ant_raw, axis=2)
ant_amp = np.abs(ant_hil)

fig, ax = mplt.subplots()
ax.plot(ant_amp[:, 0].T)
fig.show()

ant_amax = ant_amp.argmax(axis=2)
amin = ant_amax.min()
ant_amp2 = np.roll(ant_amp, -amin + 16, axis=2)

fig, ax = mplt.subplots()
ax.plot(ant_amp2[:, 0].T)
fig.show()

b1_sigs = ant_amp2.max(axis=2)
b2_sigs = ant_amp2[..., 200:].max(axis=2)

b1_posx, b1_posy = AcqBPMsSignals.calc_positions_from_amplitudes(b1_sigs.T)
b2_posx, b2_posy = AcqBPMsSignals.calc_positions_from_amplitudes(b2_sigs.T)
b1_sum = b1_sigs.sum(axis=1)
b2_sum = b2_sigs.sum(axis=1)
bt_sum = b1_sum + b2_sum

b1_curr = b1_sum * curr / bt_sum
b2_curr = b2_sum * curr / bt_sum

fig, ax = mplt.subplots()
ax.plot(tstmp, b1_curr)
ax.plot(tstmp, b2_curr)
fig.show()

fig, (ax, ay) = mplt.subplots(2, 1, sharex=True)
ax.plot(tstmp, b1_posx)
ax.plot(tstmp, b2_posx)
ay.plot(tstmp, b1_posy)
ay.plot(tstmp, b2_posy)
fig.show()
 