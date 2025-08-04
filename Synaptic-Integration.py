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
# #Synaptic Integration

# %%
import numpy as np
from brian2 import * 
from src.cell import single_cell_simulation
from src.default_params import params
from src.plot import plot_Vm 


# %% [markdown]
# #Excitatory Events

# %%
exc_events = 0.2+np.arange(20)*0.025
print(exc_events)

Vm = single_cell_simulation(params,
                            exc_events,
                            [],
                            tstop=1)


plot_Vm(Vm, params)
show()

# %% [markdown]
# #Inhibitory Events

# %%
inh_events = 0.2+np.arange(20)*0.025
print(inh_events)

Vm = single_cell_simulation(params,
                            [],
                            inh_events,
                            tstop=1)

plot_Vm(Vm, params)
show()

# %% [markdown]
# #Mixed Events

# %%
exc_events = np.cumsum(\
        np.random.exponential(0.1, size=8))
print(exc_events)
inh_events = np.cumsum(\
        np.random.exponential(0.1, size=8))
print(inh_events)

Vm = single_cell_simulation(params,
                            exc_events,
                            inh_events,
                            tstop=1)

plot_Vm(Vm, params)
# adding events on plot
plot(exc_events, -71+0*exc_events, 'go')
plot(inh_events, -71+0*inh_events, 'ro')
xlim([0,1])
xlabel('time (s)')
ylabel('$V_m$ (mV)')
show()

