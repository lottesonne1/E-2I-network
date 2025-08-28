# %% [markdown]
# #Multiinput Integration

# %%
import numpy as np
from brian2 import * 
from src.cell import single_cell_simulation
from src.default_params import params
from src.plot import plot_Vm
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl


# by default:
label = 'PV'
NMDA_AMPA_ratio = 0

# %%
# Loop for N events 

def simulate_increasing_simultaneous_events(params,
                                            model='two-compartments',
                                            NMDA_AMPA_ratio=0,
                                            Nmax=12):    
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
nevoked_list = simulate_increasing_simultaneous_events(params,
                                                      model='two-compartments',
                                                      NMDA_AMPA_ratio=NMDA_AMPA_ratio,
                                                      Nmax=12)

#%%
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
title = 'Multi-input integration %s' % label
legend()
show()

# %%
# PEAKS + NON LINEARITY 
def compute_peaks(nevoked_list, Nmax=12):

    peak_expected, peak_actual, non_linearity = [], [], []

    for n in range(1, Nmax+1):
        expected = np.max(n * nevoked_list[0])
        actual = np.max(nevoked_list[n-1])
        
        peak_expected.append(expected)
        peak_actual.append(actual)
        non_linearity.append( (actual - expected) / expected * 100 )

    return np.array(peak_expected), np.array(peak_actual), non_linearity

# %%
# SAFE PEAKS 
def run_sim(params, label, NMDA_AMPA_ratio=0):
    nevoked_list =\
        simulate_increasing_simultaneous_events(params,
                                                model='two-compartments',
                                                NMDA_AMPA_ratio=NMDA_AMPA_ratio)

    peak_expected, peak_actual, non_linearity = compute_peaks(nevoked_list,
                                                              Nmax=12)

    np.save('data/two-comp-multi-integ-%s.npy' % label,
            dict(params=params,
                 peak_expected=peak_expected,
                 peak_actual=peak_actual,
                 non_linearity=non_linearity
             ))
run_sim(params, 'PV', NMDA_AMPA_ratio=0.)
run_sim(params, 'SST', NMDA_AMPA_ratio=2.7)

# %%
#PEAKS plot
def plot_peaks(label='PV', color='tab:red'):
    res = np.load('data/two-comp-multi-integ-%s.npy' % label,
                  allow_pickle=True).item()
    plt.figure(dpi=200)
    plt.plot(res['peak_expected'], res['peak_actual'], 'o-', label='simulation', color=color)
    plt.plot(res['peak_expected'], res['peak_expected'], 'k--', label='linearity')
    plt.xlabel('expected depolarization (mV)')
    plt.ylabel('modelled depolarization (mV)')
    plt.title('%s multi input integration' % label)
    plt.legend()
    plt.show()
plot_peaks(label='PV', color='tab:red')
plot_peaks(label='SST', color='tab:orange')
# %%
# non-linearity kick level
threshold = 25
label = 'PV' 

def find_nl_kick_level(peak_expected, non_linearity, 
                       threshold=threshold, label='SST'):
    depol = np.array(peak_expected) + El
    above = np.where(np.abs(non_linearity) > threshold)[0]
    if len(above) == 0:
        return None
    
    iCond = above[0]
    return iCond, peak_expected, non_linearity, depol

res = np.load('data/two-comp-multi-integ-%s.npy' % label,
                  allow_pickle=True).item()
peak_expected = res['peak_expected']
non_linearity = res['non_linearity']
El = res['params']['El']
depol = np.array(peak_expected) + El
iCond = min( np.argwhere( np.abs(non_linearity) > threshold))

plt.figure(dpi=200)
plt.plot(depol, np.abs(non_linearity), 'o') 
plt.plot(depol, threshold+0*depol, 'r:') 
plt.plot([depol[iCond]], [np.abs(non_linearity)[iCond]], 'ro', ms=10) 
plt.xlabel('depolarization level (mV)')
plt.ylabel('abs. non-linearity (%)')
plt.title('%s non-linearity kick level' % label)
plt.legend()
plt.show()

print(depol[iCond])

# %%
# SAVE DATA FUNCTION
def build_multi_inputs_data(params,
                            NMDA_AMPA_ratio=NMDA_AMPA_ratio,
                            label='SST'):
    
    params['qNMDA'] = NMDA_AMPA_ratio*params['qAMPA']
    nevoked_list = simulate_increasing_simultaneous_events(params)
    peak_expected, peak_actual, non_linearity = compute_peaks(nevoked_list, Nmax=12)

    np.save('data/two-comp-multi-integ-%s.npy' % label,
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
                           label=label, color='tab:red'):
    res = np.load('data/two-comp-multi-integ-%s.npy' % label,
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
                                 RmSs=np.linspace(40, 300, 6),
                                 RmDs=np.linspace(100, 600, 10),
                                 Ris=np.linspace(3, 150, 8),
                                 Nsyn=12,
                                 label='PV',
                                 NMDA_AMPA_ratio=0.):
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

#%%
# non-linearity kick level scan
def find_nl_kick_level_scan(label='SST', threshold=25):
    res = np.load('data/nonlinearity-params-scan-two-comp-%s.npy' % label,
                  allow_pickle=True).item()

    Peak_Actual = res['Peak_Actual']
    Peak_Expected = res['Peak_Expected']
    RmSs, RmDs, Ris = res['RmSs'], res['RmDs'], res['Ris']
    El = res['params']['El']
    Nsyn = Peak_Expected.shape[-1]

    depol_at_threshold = np.full((len(RmSs), len(RmDs), len(Ris)), np.nan)

    for iRmS, iRmD, iRi in itertools.product(
        range(len(RmSs)),
        range(len(RmDs)),
        range(len(Ris))
    ):
        peak_expected = Peak_Expected[iRmS, iRmD, iRi, :]
        peak_actual = Peak_Actual[iRmS, iRmD, iRi, :]
        non_linearity = (peak_actual - peak_expected) / peak_expected * 100

        kick = find_nl_kick_level(peak_expected, El, non_linearity, threshold)
        if kick is not None:
            iCond, _, _, depol = kick
            depol_at_threshold[iRmS, iRmD, iRi] = depol[iCond]

    return depol_at_threshold
depol_at_threshold = find_nl_kick_level_scan(label=label, threshold=25)

# %%
#Plot parameter scan 10
fig, AX = plt.subplots(1, len(res['Ris']) ,figsize=(8,2))
plt.subplots_adjust(bottom=0.25, right=.85)

cmap = mpl.cm.autumn
vmin = np.nanmin(depol_at_threshold)
vmax = np.nanmax(depol_at_threshold)

def rescale(x):
    return (x-vmin)/(vmax-vmin)

def set_2d_scan_axes(ax, res):
    ax.set_xlabel('$R_m^D$ (M$\Omega$)')
    ax.set_ylabel('$R_m^S$ (M$\Omega$)' if i==0 else '')
    ax.set_yticks(range(len(res['RmSs'])))
    step = max(1, len(res['RmDs']) // 5)
    ax.set_xticks(range(0, len(res['RmDs']), step))
    ax.set_xticklabels([int(r) for r in res['RmDs'][::step]], rotation=60)
    ax.set_yticklabels([int(r) for r in res['RmSs']] if i==0 else [])

for i, ri in enumerate(res['Ris']):
    AX[i].set_title('$R_i$ = %.1f M$\Omega$' % ri )
    AX[i].imshow(rescale(depol_at_threshold[:,:,i]), 
                 vmin=0, vmax=1, origin='lower',
                 aspect='auto', cmap=cmap)
    set_2d_scan_axes(AX[i], res)

inset = fig.add_axes([0.93, 0.2, 0.04, 0.6])
bounds = np.linspace(vmin, vmax, 10)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                  cax=inset, 
                  orientation='vertical',
                  #  ticks=[],
                  label="Depolarization (mV)")

#%%
#plot parameters for n synapses 
def plot_full_parameter_grid(label='PV', last_n=7):
    res = np.load('data/nonlinearity-params-scan-two-comp-%s.npy' % label, allow_pickle=True).item()
    Nonlinearity = (res["Peak_Actual"] - res["Peak_Expected"]) / res["Peak_Expected"] * 100
    RmSs, RmDs, Ris = res['RmSs'], res['RmDs'], res['Ris']
    Nsyn = Nonlinearity.shape[-1]
    n_indices = np.arange(max(0, Nsyn-last_n), Nsyn)
    cmap = plt.cm.RdBu

    fig, AX = plt.subplots(len(n_indices), len(Ris),
                           figsize=(2.5*len(Ris), 2*len(n_indices)),
                           dpi=200)

    if len(n_indices) == 1:
        AX = np.array([AX])
    AX = np.atleast_2d(AX)

    absMax = np.max(np.abs(Nonlinearity))
    vmin, vmax = -absMax, absMax

    for row_idx, n in enumerate(n_indices):
        for j, ri in enumerate(Ris):
            ax = AX[row_idx, j]
            im = ax.imshow(Nonlinearity[:,:,j,n], origin='lower',
                           aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
            if row_idx == 0:
                ax.set_title(f"$R_i$={ri:.1f} MΩ")
            if j == 0:
                ax.set_ylabel(f"n={n+1}\n$R_m^S$ (MΩ)")
                ax.set_yticks(range(len(RmSs)))
                ax.set_yticklabels([f"{s:.0f}" for s in RmSs])
            else:
                ax.set_yticks([])
            if row_idx == len(n_indices)-1:
                ax.set_xlabel("$R_m^D$ (MΩ)")
                ax.set_xticks(range(len(RmDs)))
                ax.set_xticklabels([f"{d:.0f}" for d in RmDs], rotation=60)
            else:
                ax.set_xticks([])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
                 cax=cbar_ax, orientation='vertical', label="Nonlinearity (%)")

    plt.suptitle('%s: parameter scan' % label, y=0.99)
    plt.tight_layout(rect=[0,0,0.9,0.97])
    plt.show()
    return fig, AX
_ = plot_full_parameter_grid(label='PV');

#%%
if True:
    plot_full_parameter_grid(label='PV')
    plot_full_parameter_grid(label='SST')

#%%
# Input resistance 
from src.cell import get_neuron_group 
def measure_input_resistance_steps(params,
                                   model="single-compartment",
                                   steps_pA=np.linspace(-50, 50, 11), 
                                   step_start=0.2,  
                                   step_dur=0.4,     
                                   settle=0.1      
                                   ):

    defaultclock.dt = params['dt'] * second
    net = Network(collect())
    cell = get_neuron_group(params, model=model)
    net.add(cell)

    if model == "two-compartments":
        mon = StateMonitor(cell, ['Vs', 'V', 'I0'], record=0)
    else:
        mon = StateMonitor(cell, ['V', 'I0'], record=0)
    net.add(mon)

    if model == "two-compartments":
        cell.Vs = params['El'] * mV
        cell.V  = params['El'] * mV
    else:
        cell.V  = params['El'] * mV
    cell.I0 = 0 * pA

    pre_dur   = step_start
    between   = 0.15 
    per_step_total = step_dur + between
    net.run(pre_dur * second)
    for I in steps_pA:
        cell.I0 = I * pA
        net.run(step_dur * second)
        cell.I0 = 0 * pA
        net.run(between * second)
    t = np.asarray(mon.t / second)

    if model == "two-compartments":
        Vs = np.asarray(mon.Vs[0] / mV)
        Vd = np.asarray(mon.V[0]  / mV)
        soma_trace = Vs
    else:
        V  = np.asarray(mon.V[0]  / mV)
        soma_trace = V
        Vd = None

    dV_ss = []
    step_onsets = pre_dur + np.arange(len(steps_pA)) * per_step_total
    for onset in step_onsets:
        start = onset + (step_dur - settle)  
        end   = onset + step_dur
        mask  = (t >= start) & (t <= end)
        Vmean = np.mean(soma_trace[mask])
        dV_ss.append(Vmean - params['El'])

    dV_ss = np.asarray(dV_ss)  
    slope, intercept = np.polyfit(steps_pA, dV_ss, 1)  
    Rin_MOhm = slope  

    return steps_pA, dV_ss, Rin_MOhm, t, soma_trace, Vd
#%%
# Plot input resistance



def plot_input_resistance(params, model="two-compartments", steps, dV, Rin, t, Vs, Vd):
    steps, dV, Rin, t, Vs, Vd = measure_input_resistance_steps(
    params, model=model,
    steps_pA=np.linspace(-50, 50, 11),
    step_start=0.2, step_dur=0.4, settle=0.1
    )
    print(f"Estimated Rin (soma) = {Rin:.1f} MΩ")
    # IV curve
    plt.figure(dpi=200)
    plt.plot(steps, dV, 'o-')
    plt.axhline(0, ls=':', c='k')
    plt.axvline(0, ls=':', c='k')
    plt.xlabel('Injected current (pA)')
    plt.ylabel('Steady-state ΔV_soma (mV)')
    plt.title('I–V curve (Rin = %.1f MΩ)' % Rin)
    plt.tight_layout()
    plt.show()

# Voltage trace (soma ± dendrite)
    plt.figure(dpi=200)
    plt.plot(t, Vs, label='Vs (soma)')
    if Vd is not None:
        plt.plot(t, Vd, label='Vd (dend)')
        plt.axhline(params['El'], ls='--', c='k', lw=0.8, label='El')
    plt.xlabel('Time (s)')
    plt.ylabel('V (mV)')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

# %%
if True:
   plot_input_resistance(steps, dV, Rin, t, Vs, Vd)
   plot_input_resistance(params, model, steps, dV, Rin, t, Vs, Vd, model="single-compartment")