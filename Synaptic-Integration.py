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
params['model'] = 'two-compartments'
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
# adding events on plot
plot(exc_events, -71+0*exc_events, 'go')
plot(inh_events, -71+0*inh_events, 'ro')
xlim([0,1])

# %% [markdown]
# #Simulation 1 excitatory event
# %%
exc_events = [0.1] # single event at 100 ms
Vm_1event = single_cell_simulation(params,
                            exc_events,
                            [],
                            tstop=1)
evoked_1 = Vm_1event - params['El']

# %% [markdown]
# #Simulation 2 excitatory events
dt=1e-4 # seconds
exc_events = [0.1, 0.1+dt] #events start at 100 ms 
Vm_2events = single_cell_simulation(params,
                            exc_events,
                            [],
                            tstop=1)

evoked_2 = Vm_2events - params['El']

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

# %% 
fig, AX = plt.subplots(1, 2, figsize=(7,3))
inset = fig.add_axes([0.93, 0.2, 0.04, 0.6])
cmap = mpl.cm.viridis_r
bounds = np.arange(len(nevoked_list)+1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=inset, orientation='vertical',
             ticks=[],
             label="n synapses")
#cb.set_ticks(ticks=np.arange(len(nevoked_list))+1.5,
#             labels=[str(i) for i in 1+np.arange(len(nevoked_list))])
for n, evoked_n in enumerate(nevoked_list, start=1):
    expected = n * evoked_1
    plot_Vm(expected, params, ax=AX[0],
            color=cmap(n/(len(nevoked_list))))
    plot_Vm(evoked_n, params, ax=AX[1],
            color=cmap(n/(len(nevoked_list))))

for ax in AX:
    ax.set_ylim([-1,60])
    ax.set_xlim([0, 0.4])
    ax.set_xlabel('time (s)')
AX[0].set_ylabel('depolarization (mV)')
legend()
show()

# %%
# comparison  
evoked_1_ = np.array(evoked_1)
peak_expected = []
peak_actual = [] 

for n in range(1, 11):
    expected = np.max(n * evoked_1)
    actual = np.max(nevoked_list[n-1])
    
    peak_expected.append(expected)
    peak_actual.append(actual)

# %%
#comparison plot 
plt.figure()
plt.plot(peak_expected, peak_actual, 'o-', label='Actual vs Expected')
plt.plot(peak_expected, peak_expected, 'k--', label='Perfect Linearity')

plt.xlabel('expected depolarization (mV)')
plt.ylabel('modelled depolarization (mV)')
plt.title('SST-INs')
plt.legend()
plt.show()

# %%
#evoked_10events vs 10*evoked_1event
plot(10 * evoked_1, '--', label='10 × (1 event)')
plot(nevoked_list[9], label='10 simultaneous events')
xlabel('Time (steps)')
ylabel('Evoked $V_m$ (mV)')
show()