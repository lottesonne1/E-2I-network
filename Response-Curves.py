# %% [markdown]
# #Cellular Model

# %%
import numpy as np
from brian2 import * 
from src.cell import single_cell_simulation
from src.stimulation import *
from src.default_params import params
from src.plot import plot_with_stim

# %%

def rate_stim_simulation(exc_rate, params,
                         inh_to_exc_rate_factor=3.,
                         label='Pv',
                         model='single-compartment',
                         tstop=2):


    t = np.arange(int(tstop/params['dt']))*params['dt']

    # Exc -> cell, Nsyn from network parameters
    Nsyn = int(params['p_PyrExc_%sInh' % label]*params['N_PyrExc'])
    _, exc_events = spikes_from_time_varying_rate(t, exc_rate+0*t,
                                                  N=1, Nsyn=Nsyn)


    # Inh -> cell, Nsyn from network parameters
    Nsyn = int(params['p_PvInh_%sInh' % label]*params['N_PvInh'])+\
                int(params['p_SstInh_%sInh' % label]*params['N_SstInh'])
    inh_rate = inh_to_exc_rate_factor*exc_rate
    _, inh_events = spikes_from_time_varying_rate(t, inh_rate+0*t,
                                                  N=1, Nsyn=Nsyn)

    resp = single_cell_simulation(params,
                                  exc_events,
                                  inh_events,
                                  model=model,
                                  tstop=tstop)

    return resp

params['qNMDA'] = 0.
resp = rate_stim_simulation(3., params, tstop=5)
fig, AX = plot_with_stim(resp, figsize=(10,3))

show()


# %%

def build_freq_scan(F_exc, params,
                    inh_to_exc_rate_factor=3.,
                    label='Pv',
                    model='single-compartment',
                    tmin = 1.):
    F_out = []
    for f in F_exc:
        resp = rate_stim_simulation(f, params,
                                    inh_to_exc_rate_factor=inh_to_exc_rate_factor,
                                    label=label,
                                    model=model,
                                    tstop=np.clip(1/f, 5, 100))
        # calculate firing response:
        cond = (resp['spikes']>tmin)
        F_out.append(len(resp['spikes'][cond])/\
                                (resp['tstop']-tmin))
    return np.array(F_out)

# %%
F_exc = np.logspace(np.log10(0.1), np.log10(8), 10)
F_out = build_freq_scan(F_exc, params)

# %%
def plot_freq_scan(F_exc, F_out,
                   color='tab:grey'):
    fig, ax = plt.subplots(1, figsize=(3,2)) 
    ax.plot(F_exc, F_out, 'o-', ms=5, color=color)
    ax.set_xlabel('input freq. $F_{in}$ (Hz)')
    ax.set_ylabel('output freq. $F_{out}$ (Hz)')
    ax.set_xscale('log')
    # plt.yscale('log')
    inset = fig.add_axes([0.35,0.55,0.3,0.3])
    inset.plot(F_exc, F_out, 'o-', ms=4, color=color)
    inset.set_xticks([0,5,10])
    inset.set_yticks([0,20,40])
    inset.set_xlabel('$F_{in}$ (Hz)')
    inset.set_ylabel('$F_{out}$ (Hz)')
plot_freq_scan(F_exc, F_out)

# %%
