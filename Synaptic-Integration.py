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
# #Excitatory Synapses

# %%
exc_events = 0.2+np.arange(20)*0.025
print(exc_events)

Vm = single_cell_simulation(params,
                            exc_events,
                            [],
                            tstop=1)


plot_Vm(Vm, params)
show()

# %%
inh_events = 0.2+np.arange(20)*0.025
print(inh_events)

Vm = single_cell_simulation(params,
                            [],
                            inh_events,
                            tstop=1)

plot_Vm(Vm, params)
show()

