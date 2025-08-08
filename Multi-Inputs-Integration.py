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
# #Multiinput Integration

# %%
import numpy as np
from brian2 import * 
from src.cell import single_cell_simulation
from src.default_params import params
from src.plot import plot_Vm

# %%
# 1 excitatory event 
exc_events = [0.1]
Vm_dict = single_cell_simulation(params,
                              exc_events,
                             [],
                            #  model ='single-compartment',
                             model ='two-compartments',
                            tstop=1)
evoked_1 = Vm_dict['Vm_soma'] - params['El']
plot_Vm(Vm_dict['Vm_soma'], params) 
plot_Vm(Vm_dict['Vm_dend'], params, color='r') 
print(np.max(evoked_1))
show()

# %%
# Loop for N events 

def simulate_increasing_simultaneous_events(params,
                                            model='two-compartments',
                                            Nmax=10):
    
    # disable spiking by putting threshold very high
    params['Vtresh'] = 100 * mV

    #Loop for Nmax excitatory events 
    nevoked_list = []

    for n in range(Nmax):

        exc_events = [0.1 +\
                       i * params['dt'] for i in range(n)]

        Vm_dict = single_cell_simulation(params, 
                                        exc_events, 
                                        [], 
                                        model=model,  
                                        tstop=1)

        evoked_n = Vm_dict['Vm_soma'] - params['El']
        nevoked_list.append(evoked_n)

    return nevoked_list

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


#%%

# SIMULATION
nevoked_list =\
    simulate_increasing_simultaneous_events(params)

# PLOTTING
fig, AX = plt.subplots(1, 2, figsize=(7,3))
inset = fig.add_axes([0.93, 0.2, 0.04, 0.6])
cmap = mpl.cm.viridis_r
bounds = np.arange(len(n_evoked_list)+1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=inset, orientation='vertical',
             ticks=[],
             label="n synapses")
#cb.set_ticks(ticks=np.arange(len(n_evoked_list))+1.5,
#             labels=[str(i) for i in 1+np.arange(len(nevoked_list))])
for n, evoked_n in enumerate(n_evoked_list, start=1):
    expected = n * evoked_1
    plot_Vm(expected, params, ax=AX[0],
            color=cmap(n/(len(n_evoked_list))))
    plot_Vm(evoked_n, params, ax=AX[1],
            color=cmap(n/(len(n_evoked_list))))

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
non_linearity = [] 

for n in range(1, 10):
    expected = np.max(n * evoked_1)
    actual = np.max(nevoked_list[n])
    percentual_diff = (actual - expected) / expected * 100

    peak_expected.append(expected)
    peak_actual.append(actual)
    non_linearity.append(percentual_diff)


# %%
np.save('data/single-comp-multi-integ-PV.npy',
        dict(params=params,
             peak_expected=peak_expected,
             peak_actual=peak_actual,
             non_linearity=non_linearity
        ))

# %%
#comparison plot 
res = np.load('data/single-comp-multi-integ-PV.npy',
                  allow_pickle=True).item()

plt.figure()
plt.plot(res['peak_expected'], res['peak_actual'], 'o-', label='simulation')
plt.plot(res['peak_expected'], res['peak_expected'], 'k--', label='linearity')

plt.xlabel('expected depolarization (mV)')
plt.ylabel('modelled depolarization (mV)')
plt.title('PV-INs')
plt.legend()
plt.show()


# %%
threshold = 1

def find_nl_kick_level(n_evoked_list):
    pass
    #return depol[iCond]

depol = np.array(res['peak_expected'])+res['params']['El']

iCond = min(np.argwhere(np.abs(non_linearity) > threshold))
plt.figure()
plt.plot(depol, np.abs(non_linearity), 'o') 
plt.plot(depol, threshold+0*depol, 'r:') 
plt.plot([depol[iCond]], [np.abs(non_linearity)[iCond]], 'ro', ms=10) 
plt.xlabel('depolarization level (mV)')
plt.ylabel('abs. non-linearity (%)')
plt.title('PV-INs')
plt.legend()
plt.show()

print(depol[iCond])
    
# %%

def build_multi_inputs_data(params,
                            NMDA_AMPA_ratio=0.,
                            label='PV'):
    
    params['qNMDA'] = NMDA_AMPA_ratio*params['qAMPA']

    # simulate increasing simultaneous events
    n_evoked_list = simulate_increasing_simultaneous_events(params)

    np.save('data/single-comp-multi-integ-%s.npy' % label,
            dict(params=params,
                peak_expected=peak_expected,
                peak_actual=peak_actual,            
            ))
    
# %%
if False:
    from src.default_params import params
    build_multi_inputs_data(params, NMDA_AMPA_ratio=0., label='PV')
    build_multi_inputs_data(params, NMDA_AMPA_ratio=2.7, label='SST')

# %%
def plot_multi_inputs_data(params,
                           label='PV', color='tab:red'):
    res = np.load('data/single-comp-multi-integ-%s.npy' % label,
                  allow_pickle=True).item()

    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))
    ax.plot(res['peak_expected'], res['peak_actual'], 'o-', 
             label='simulation', color=color)
    ax.plot(res['peak_expected'], res['peak_expected'],
             'k:', label='linearity')
    ax.set_xlabel('expected depolarization (mV)')
    ax.set_ylabel('modelled depolarization (mV)')
    ax.set_title(label)
    ax.legend(frameon=False)
    return fig, ax

# %%
plot_multi_inputs_data(params, label='PV', color='tab:red')
#plot_multi_inputs_data(params, label='SST', color='tab:orange')


# %% Parameter scan 

def build_nonlinearity_scan_data(params,
                            RmSs = np.linspace(50, 300, 3),
                            RmDs = np.linspace(100, 500, 5),
                            Ris  = np.linspace(3, 100, 4),
                            NMDA_AMPA_ratio=0.,
                            label='PV'):
    
    params['qNMDA'] = NMDA_AMPA_ratio*params['qAMPA']

    NL_kick_level = np.zeros((len(RmSs),
                            len(RmDs),
                            len(Ris)))
    
    for iRmS, iRmD, iRi in itertools.product(\
        range(len(RmSs)), range(len(RmDs)), range(len(Ris))):
        
        params['RmS'] = RmSs[iRmS]
        params['RmD'] = RmDs[iRmD]
        params['Ri'] = Ris[iRi]

        if False:
            n_evoked_list =\
                simulate_increasing_simultaneous_events(params)
            
            NL_kick_level[RmS, RmD, Ri] =\
                find_nl_kick_level(n_evoked_list)
            
    np.save('data/nonlinearity-params-scan-two-comp-%s.npy' % label,
            dict(params=params,
                peak_expected=peak_expected,
                peak_actual=peak_actual,            
            ))

# %%
build_nonlinearity_scan_data(params, label='PV', NMDA_AMPA_ratio=0.)
build_nonlinearity_scan_data(params, label='SST', NMDA_AMPA_ratio=2.7)
