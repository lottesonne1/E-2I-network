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
from src.cell import get_neuron_group
from src.default_params import params
from src.plot import plot_Vm 

# %%

# initialize brian2 "network"
network = Network(collect())

cell = get_neuron_group(params)
network.add(cell)
M = StateMonitor(cell, ['V','I0'], record=0)
network.add(M)
cell.V = params['El']*mV 
cell.I0 = 0*pA
network.run(0.1*second)
cell.I0 = 300*pA
network.run(0.2*second)
cell.I0 = 0*pA
network.run(0.1*second)

plot_Vm(M.V[0]/mV, params)
show()

