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

# %%
# Loop for N events 

def simulate_increasing_simultaneous_events(params,
                                            model='two-compartments',
                                            NMDA_AMPA_ratio=0.,
                                            Nmax=10):    
    params['Vtresh'] = 100 * mV
    params['NMDA_AMPA_ratio'] = NMDA_AMPA_ratio
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
                                            NMDA_AMPA_ratio=0.)

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

peak_expected, peak_actual, non_linearity = compute_peaks(nevoked_list, Nmax=10)
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

def find_nl_kick_level(peak_expected, El, non_linearity, threshold=threshold):
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
if True:
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
plot_multi_inputs_data(params, label='PV', color='tab:red')
plot_multi_inputs_data(params, label='SST', color='tab:orange')


# %% Parameter scan 

def build_nonlinearity_scan_data(params,
                                 RmSs=np.linspace(50, 300, 3),
                                 RmDs=np.linspace(100, 500, 5),
                                 Ris=np.linspace(3, 100, 4),
                                 label='PV',
                                 NMDA_AMPA_ratio=0.):
    
    params['qNMDA'] = NMDA_AMPA_ratio*params['qAMPA']

    NL_kick_level = np.zeros((len(RmSs), 
                              len(RmDs), 
                              len(Ris)))

    for iRmS, iRmD, iRi in itertools.product(
        range(len(RmSs)),
        range(len(RmDs)),
        range(len(Ris))
    ):
        params['RmS'] = RmSs[iRmS]
        params['RmD'] = RmDs[iRmD]
        params['Ri'] = Ris[iRi]

        nevoked_list = simulate_increasing_simultaneous_events(
            params, NMDA_AMPA_ratio=NMDA_AMPA_ratio
        )
        peak_expected, peak_actual, non_linearity = compute_peaks(nevoked_list)

        kick = find_nl_kick_level(peak_expected, params['El'], non_linearity, threshold)
        if kick is None:
            NL_kick_level[iRmS, iRmD, iRi] = np.nan
        else:
            iCond, _, _, depol = kick
            NL_kick_level[iRmS, iRmD, iRi] = depol[iCond]

    np.save('data/nonlinearity-params-scan-two-comp-%s.npy' % label,
            dict(params=params, NL_kick_level=NL_kick_level, 
                 peak_expected=peak_expected,
                 peak_actual=peak_actual,
                 RmSs=RmSs, RmDs=RmDs, Ris=Ris))

# %%
build_nonlinearity_scan_data(params, label='PV', NMDA_AMPA_ratio=0.)
build_nonlinearity_scan_data(params, label='SST', NMDA_AMPA_ratio=2.7)

#%%
#basic stats
res = np.load('data/nonlinearity-params-scan-two-comp-PV.npy', allow_pickle=True).item()
NL = res['NL_kick_level']
params = res['params']      # params (note: this holds only the last params values)
RmSs = res['RmSs']
RmDs = res['RmDs']
Ris = res['Ris']
idxs = np.argwhere(np.abs(NL) >= threshold)   # list of (iRmS, iRmD, iRi) indices


print("Shape:", NL.shape)
print('NL mean:', NL.mean())
print('NL std:', NL.std())
print('NL min:', NL.min())
print('NL max:', NL.max())  
print("Fraction above threshold:", np.mean(np.abs(NL) > 25))
print(f'Number of parameter combos >= {threshold}%: {len(idxs)}')
combos = []
for (iRmS, iRmD, iRi) in idxs:
    combo = {
        'RmS': RmSs[iRmS],
        'RmD': RmDs[iRmD],
        'Ri' : Ris[iRi],
        'NL' : NL[iRmS, iRmD, iRi]
    }
    combos.append(combo)

if len(combos) > 0:
    RmS_vals = np.array([c['RmS'] for c in combos])
    RmD_vals = np.array([c['RmD'] for c in combos])
    Ri_vals  = np.array([c['Ri']  for c in combos])

    print('RmS in [%.3g, %.3g]' % (RmS_vals.min(), RmS_vals.max()))
    print('RmD in [%.3g, %.3g]' % (RmD_vals.min(), RmD_vals.max()))
    print('Ri  in [%.3g, %.3g]' % (Ri_vals.min(), Ri_vals.max()))
else:
    print('No combos crossed threshold.')

print(f"Found {len(idxs)} parameter combos with |NL| >= {threshold}%:\n")

for (iRmS, iRmD, iRi) in idxs:
    RmS_val = RmSs[iRmS]
    RmD_val = RmDs[iRmD]
    Ri_val  = Ris[iRi]
    NL_val  = NL[iRmS, iRmD, iRi]

    print(f"  RmS={RmS_val:.1f}, RmD={RmD_val:.1f}, Ri={Ri_val:.1f}  -->  NL={NL_val:.2f}")