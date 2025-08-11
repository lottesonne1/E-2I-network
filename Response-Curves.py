# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

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
                           inh_to_exc_rate_factor=4.,
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


resp = rate_stim_simulation(1.4, params)
fig, AX = plot_with_stim(resp)

show()

