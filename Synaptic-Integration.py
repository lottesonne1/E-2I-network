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
params['Vthre'] = -53 
exc_events = 0.2+np.arange(20)*0.025 # multiple  equally spaced timed events 
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

# %% [markdown]
# #Simulation 1 excitatory event
# %%
exc_events = [0.1] # single event at 100 ms
Vm_1event = single_cell_simulation(params,
                            exc_events,
                            [],
                            tstop=1)
evoked_1 = Vm_1event - params['El']
plot_Vm(evoked_1, params)
show()

# %% [markdown]
# #Simulation 2 excitatory events
# %%
dt=1e-4 # seconds
exc_events = [0.1, 0.1+dt] #events start at 100 ms 
Vm_2events = single_cell_simulation(params,
                            exc_events,
                            [],
                            tstop=1)

evoked_2 = Vm_2events - params['El']

plot_Vm(evoked_2, params)
show()

# %%
#Loop for 10 excitatory events 
dt=1e-4  
nevoked_list = []
params['Vtresh']=0 

for n in range(0, 10):
    exc_events = [0.1 + i * dt for i in range(n)]
    Vm_nevents = single_cell_simulation(params, 
                                exc_events, 
                                [], 
                                tstop=1)
    evoked_n = Vm_nevents - params['El']
    nevoked_list.append(evoked_n)

for n, evoked_n in enumerate(nevoked_list, start=1):
    expected = n * evoked_1
    plot(evoked_n, label=f'{n} events')
    plot(expected, '--', label=f'{n}×1 event')

xlabel('time (steps)')
ylabel('Evoked $V_m$ (mV)')
legend()
show()

# %%
#evoked_10events vs 10*evoked_1event
plot(10 * evoked_1, '--', label='10 × (1 event)')
plot(nevoked_list[9], label='10 simultaneous events')

xlabel('Time (steps)')
ylabel('Evoked $V_m$ (mV)')
legend()
show()
# %%
#evoked_2events vs 2*evoked_1event
plot(evoked_1, label='1 event')
plot(evoked_2, label='2 events')
plot(2 * evoked_1, '--', label='2 × (1 event)', alpha=0.6)
legend()
xlabel('time')
ylabel(mV)
show()

# %% [markdown]
# #Multiple mixed Events

# %%
exc_events = [0.1]     
inh_events = [0.1]
Vm_ei = single_cell_simulation(params,
                            exc_events,
                            inh_events,
                            tstop=1)

plot_Vm(Vm_ei, params)
plot(exc_events, [-71 for _ in exc_events], 'go')
plot(inh_events, [-71 for _ in inh_events], 'ro')
xlim([0,1])
xlabel('time (s)')
ylabel('$V_m$ (mV)')
show()
print(exc_events)
print(inh_events)

# %%
