# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# %%
NTWK = np.load('network.data.npy', allow_pickle=True).item()

# %%
### PLOT ###
fig = plt.figure(figsize=(7,5.5))
plt.subplots_adjust()
# afferent stimulation
ax1 = plt.subplot2grid((6,1), (0,0))
ax1.plot(NTWK['t'], NTWK['faff_waveform'], 'k-')
ax1.set_xticks([]);ax1.set_ylabel(r'$\nu_a$ (Hz)')
# populations activity (instant. firing rates)
ax2 = plt.subplot2grid((6,1), (1, 0), rowspan=2)
COLORS = ['tab:green', 'tab:red', 'tab:orange', 'tab:purple']
for i, pop in enumerate(NTWK['POPS']):
    rate = NTWK['rates'][i]
    rate = gaussian_filter1d(rate, int(20./0.1)) # smoothing
    rate[rate<0.01] = 0.01
    ax2.semilogy(NTWK['t'], rate, '-', color=COLORS[i], label=pop)
ax2.legend(frameon=False)
ax2.set_xticks([]);ax2.set_ylabel('pop act. (Hz)')
# sample Vm traces 
ax3 = plt.subplot2grid((6,1), (3, 0), rowspan=3)
N = [3,1,1,1] # number displayed per population
j=0 # index to shift the Vm trace
for i, pop in enumerate(NTWK['POPS']):
    for n in range(N[i]):
        ax3.plot(NTWK['t'], NTWK['VMs'][i][n]-20*j, 
                 '-', color=COLORS[i])
        j+=1
ax3.plot([0.09, 0.09], [-60, -50], 'k-')
ax3.annotate('10mV', (0.1, -70))
ax3.set_yticks([]);ax3.set_ylabel('sample Vm traces')
ax3.set_xlabel('time (ms)')
plt.show()
# %%

from src.network import Model, run_3pop_ntwk_model

# Model['RmS_PvInh'] = 290.
# Model['RmD_PvInh'] = 171.
# Model['Ri_PvInh'] = 3.
# Model['Vtresh_PvInh'] = -53.

# Model['RmS_SstInh'] = 40.
# Model['RmD_SstInh'] = 300.
# Model['Ri_SstInh'] = 50.
# Model['Vtresh_SstInh'] = -63.

NTWK = run_3pop_ntwk_model(Model, REC_POPS,
                            with_Vm=3,
                               verbose=args.verbose)

 
save(NTWK, REC_POPS) 
