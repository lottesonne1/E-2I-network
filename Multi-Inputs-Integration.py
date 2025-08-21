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
import itertools

# by default:
label = 'SST'
NMDA_AMPA_ratio = 2.7

# %%
# Loop for N events 

def simulate_increasing_simultaneous_events(params,
                                            model='two-compartments',
                                            NMDA_AMPA_ratio=0.,
                                            Nmax=10):    
    params['Vtresh'] = 100 * mV
    params['NMDA_AMPA_ratio'] = NMDA_AMPA_ratio
    params['qNMDA'] = params['NMDA_AMPA_ratio']*params['qAMPA']

    nevoked_list = []

    for n in range(1, Nmax+1):

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

#%%

# SIMULATION
nevoked_list =\
    simulate_increasing_simultaneous_events(params,
                                            model='two-compartments',
                                            NMDA_AMPA_ratio=NMDA_AMPA_ratio)

# PLOTTING
fig, AX = plt.subplots(1, 2, figsize=(7,3), dpi=200)
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
    expected = n * nevoked_list[0]
    plot_Vm(expected, params, ax=AX[0],
            color=cmap(n/(len(nevoked_list))))
    plot_Vm(evoked_n, params, ax=AX[1],
            color=cmap(n/(len(nevoked_list))))

for ax in AX:
    ax.set_ylim([-1,60])
    ax.set_xlim([0, 0.4])
    ax.set_xlabel('time (s)')
AX[0].set_ylabel('depolarization (mV)')
title = 'Multi-input integration'
legend()
show()

# %%
# comparison peaks 
def compute_peaks(nevoked_list, Nmax=10):

    peak_expected, peak_actual, non_linearity = [], [], []

    for n in range(1, Nmax+1):
        expected = np.max(n * nevoked_list[0])
        actual = np.max(nevoked_list[n-1])
        
        peak_expected.append(expected)
        peak_actual.append(actual)
        non_linearity.append( (actual - expected) / expected * 100 )

    return np.array(peak_expected), np.array(peak_actual), non_linearity

# %%

peak_expected, peak_actual, non_linearity\
      = compute_peaks(nevoked_list, Nmax=10)
np.save('data/single-comp-multi-integ-%s.npy' % label,
        dict(params=params,
             peak_expected=peak_expected,
             peak_actual=peak_actual,
             non_linearity=non_linearity
             ))

# %%
#comparison plot
res = np.load('data/single-comp-multi-integ-%s.npy' % label,
              allow_pickle=True).item()
non_linearity = np.array( (res['peak_actual'] - res['peak_expected']) / res['peak_expected'] * 100 )
plt.figure(dpi=200)
plt.plot(res['peak_expected'], res['peak_actual'], 'o-', label='simulation')
plt.plot(res['peak_expected'], res['peak_expected'], 'k--', label='linearity')
plt.xlabel('expected depolarization (mV)')
plt.ylabel('modelled depolarization (mV)')
plt.title('INs')
plt.legend()
plt.show()


# %%
# find non-linearity kick level
threshold = 25

def find_nl_kick_level(peak_expected, El, non_linearity, 
                       threshold=threshold):
    depol = np.array(peak_expected) + El
    above = np.where(np.abs(non_linearity) > threshold)[0]
    if len(above) == 0:
        return None
    
    iCond = above[0]
    return iCond, peak_expected, non_linearity, depol
depol = np.array(res['peak_expected'])+res['params']['El']
iCond = min( np.argwhere( np.abs(non_linearity) > threshold))

plt.figure(dpi=200)
plt.plot(depol, np.abs(non_linearity), 'o') 
plt.plot(depol, threshold+0*depol, 'r:') 
plt.plot([depol[iCond]], [np.abs(non_linearity)[iCond]], 'ro', ms=10) 
plt.xlabel('depolarization level (mV)')
plt.ylabel('abs. non-linearity (%)')
plt.title('INs')
plt.legend()
plt.show()

print(depol[iCond])

# %%

def build_multi_inputs_data(params,
                            NMDA_AMPA_ratio=0.,
                            label='PV'):
    
    params['qNMDA'] = NMDA_AMPA_ratio*params['qAMPA']
    nevoked_list = simulate_increasing_simultaneous_events(params)
    peak_expected, peak_actual, non_linearity = compute_peaks(nevoked_list, Nmax=10)

    np.save('data/single-comp-multi-integ-%s.npy' % label,
            dict(params=params,
                peak_expected=peak_expected,
                peak_actual=peak_actual,
                non_linearity=non_linearity
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

    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5), dpi=200)
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
from src.default_params import params
plot_multi_inputs_data(params, label='PV', color='tab:red')
plot_multi_inputs_data(params, label='SST', color='tab:orange')


# %% Parameter scan 

def build_nonlinearity_scan_data(params,
                                 RmSs=np.linspace(50, 300, 3),
                                 RmDs=np.linspace(100, 500, 5),
                                 Ris=np.linspace(3, 100, 4),
                                 Nsyn=10,
                                 label='PV',
                                 NMDA_AMPA_ratio=0.,
                                 NL_kick_threshold=25.):
    
    params['qNMDA'] = NMDA_AMPA_ratio*params['qAMPA']

    Peak_Actual = np.zeros((len(RmSs), len(RmDs), len(Ris), Nsyn))
    Peak_Expected = np.zeros((len(RmSs), len(RmDs), len(Ris), Nsyn))

    for iRmS, iRmD, iRi in itertools.product(
        range(len(RmSs)),
        range(len(RmDs)),
        range(len(Ris))
    ):
        params['RmS'] = RmSs[iRmS]
        params['RmD'] = RmDs[iRmD]
        params['Ri'] = Ris[iRi]

        nevoked_list = simulate_increasing_simultaneous_events(
                                        params, 
                                        NMDA_AMPA_ratio=NMDA_AMPA_ratio,
                                        Nmax=Nsyn)

        peak_expected, peak_actual, _ = compute_peaks(nevoked_list)
        Peak_Actual[iRmS, iRmD, iRi, :] = peak_actual
        Peak_Expected[iRmS, iRmD, iRi, :] = peak_expected

    np.save('data/nonlinearity-params-scan-two-comp-%s.npy' % label,
            dict(params=params, 
                 Peak_Actual=Peak_Actual,
                 Peak_Expected=Peak_Expected,
                 RmSs=RmSs, RmDs=RmDs, Ris=Ris))

# %%
if False:
    build_nonlinearity_scan_data(params, label='PV', NMDA_AMPA_ratio=0.)
    build_nonlinearity_scan_data(params, label='SST', NMDA_AMPA_ratio=2.7)

#%%
#basic stats
label = 'PV'
res = np.load('data/nonlinearity-params-scan-two-comp-%s.npy' % label, 
              allow_pickle=True).item()


# %%

fig, AX = plt.subplots(1, len(res['Ris']) ,figsize=(8,2))
plt.subplots_adjust(bottom=0.25, right=.85)

cmap = mpl.cm.autumn

vmin = -65 
vmax = np.nanmax(res['NL_kick_level'])

def rescale(x):
    return (x-vmin)/(vmax-vmin)

def set_2d_scan_axes(ax, res):
    ax.set_xlabel('$R_m^D$ (M$\Omega$)')
    ax.set_ylabel('$R_m^S$ (M$\Omega$)' if i==0 else '')
    ax.set_yticks(range(len(res['RmSs'])))
    ax.set_xticks(range(len(res['RmDs'])))
    ax.set_xticklabels([r for r in res['RmDs']], rotation=60)
    ax.set_yticklabels([r for r in res['RmSs']] if i==0 else [])

for i, ri in enumerate(res['Ris']):
    AX[i].set_title('$R_i$ = %.1f M$\Omega$' % ri )
    AX[i].imshow(rescale(res['NL_kick_level'][:,:,i]), 
                 vmin=0, vmax=1, origin='lower',
                 aspect='auto', cmap=cmap)
    set_2d_scan_axes(AX[i], res)

inset = fig.add_axes([0.93, 0.2, 0.04, 0.6])
bounds = np.linspace(vmin, vmax, 10)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=inset, orientation='vertical',
            #  ticks=[],
             label="depol")


# %%

NL_kick_level[iRmS, iRmD, iRi] = np.max(non_linearity)

kick = find_nl_kick_level(peak_expected, params['El'], non_linearity, threshold)
if kick is None:
    NL_kick_level[iRmS, iRmD, iRi] = np.nan
else:
    iCond, _, _, depol = kick
    NL_kick_level[iRmS, iRmD, iRi] = depol[iCond]
