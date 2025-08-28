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
                         inh_to_exc_rate_factor=3.,
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

# %%
for freq in [0.1, 0.5, 1., 3.]:
    params['qNMDA'] = 0.
    resp = rate_stim_simulation(freq, params, tstop=5)
    fig, AX = plot_with_stim(resp, figsize=(10,3))

#show()


# %%

def build_freq_scan(F_exc, params,
                    inh_to_exc_rate_factor=3.,
                    label='Pv',
                    model='single-compartment',
                    tmin = 1.):
    F_out = []
    for f in F_exc:
        resp = rate_stim_simulation(f, params,
                                    inh_to_exc_rate_factor=inh_to_exc_rate_factor,
                                    label=label,
                                    model=model,
                                    tstop=np.clip(1/f, 5, 100))
        # calculate firing response:
        cond = (resp['spikes']>tmin)
        F_out.append(len(resp['spikes'][cond])/\
                                (resp['tstop']-tmin))
    return np.array(F_out)

# %%
F_exc = np.logspace(np.log10(0.1), np.log10(8), 10)
F_out = build_freq_scan(F_exc, params)

# %%
def plot_freq_scan(F_exc, F_out,
                   color='tab:grey'):
    fig, ax = plt.subplots(1, figsize=(3,2)) 
    ax.plot(F_exc, F_out, 'o-', ms=5, color=color)
    ax.set_xlabel('input freq. $F_{in}$ (Hz)')
    ax.set_ylabel('output freq. $F_{out}$ (Hz)')
    ax.set_xscale('log')
    # plt.yscale('log')
    inset = fig.add_axes([0.35,0.55,0.3,0.3])
    inset.plot(F_exc, F_out, 'o-', ms=4, color=color)
    inset.set_xticks([0,5,10])
    inset.set_yticks([0,20,40])
    inset.set_xlabel('$F_{in}$ (Hz)')
    inset.set_ylabel('$F_{out}$ (Hz)')
plot_freq_scan(F_exc, F_out)

# %%
F_exc
# %%
import itertools

F_exc = np.logspace(np.log10(0.3), np.log10(8), 8)
Fout = build_freq_scan(F_exc, params, model='single-compartment')
np.save('data/excitability-params-scan-single-comp.npy', dict(params=params,
                                                              Fout=Fout))

def build_nonlinearity_scan_data(params,
                                 RmSs=np.linspace(40, 300, 2),
                                 RmDs=np.linspace(100, 600, 3),
                                 Ris=np.linspace(3, 150, 1),
                                 Nsyn=12,
                                 label='PV',
                                 NMDA_AMPA_ratio=0.):
    params['qNMDA'] = NMDA_AMPA_ratio*params['qAMPA']

    Fouts = np.zeros((len(RmSs), len(RmDs), len(Ris), len(F_exc)))

    for iRmS, iRmD, iRi in itertools.product(
        range(len(RmSs)),
        range(len(RmDs)),
        range(len(Ris))
    ):
        params['RmS'] = RmSs[iRmS]
        params['RmD'] = RmDs[iRmD]
        params['Ri'] = Ris[iRi]

        Fouts[iRmS, iRmD, iRi, :] = build_freq_scan(F_exc, params,
                                                    model='two-compartments')

    np.save('data/excitability-params-scan-two-comp-%s.npy' % label,
            dict(params=params, 
                 Fouts=Fouts,
                 RmSs=RmSs, RmDs=RmDs, Ris=Ris))
    
# %%
if False:
    build_nonlinearity_scan_data(params, label='PV', NMDA_AMPA_ratio=0.)
    build_nonlinearity_scan_data(params, label='SST', NMDA_AMPA_ratio=2.7)


# %%
#%%
#plot parameters for n synapses 

def compute_square_difference(res, res0):
    SD = np.abs(np.sum([res0['Fout']+1e-12, -res["Fouts"]], axis=-1)).mean(axis=-1)
    return SD/np.mean(res0['Fout'])

def plot_full_parameter_grid(label='PV', last_n=7):

    # load the single compartment data
    res0 = np.load('data/excitability-params-scan-single-comp.npy', allow_pickle=True).item()

    # load the two compartment data
    res = np.load('data/excitability-params-scan-two-comp-%s.npy' % label, allow_pickle=True).item()

    SquareDifference = compute_square_difference(res, res0)

    RmSs, RmDs, Ris = res['RmSs'], res['RmDs'], res['Ris']
    cmap = plt.cm.bone_r

    fig, AX = plt.subplots(1, len(Ris),
                           figsize=(2.5*len(Ris), 2*1),
                           dpi=200)

    absMax = np.max(np.abs(SquareDifference))
    vmin, vmax = 0, absMax

    for j, ri in enumerate(Ris):
        ax = AX[j]
        im = ax.imshow(SquareDifference[:,:,j], origin='lower',
                        aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_title(f"$R_i$={ri:.1f} MΩ")
        if j == 0:
            ax.set_ylabel(f"\n$R_m^S$ (MΩ)")
            ax.set_yticks(range(len(RmSs)))
            ax.set_yticklabels([f"{s:.0f}" for s in RmSs])
        else:
            ax.set_yticks([])
        ax.set_xlabel("$R_m^D$ (MΩ)")
        ax.set_xticks(range(len(RmDs)))
        ax.set_xticklabels([f"{d:.0f}" for d in RmDs], rotation=60)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
                    cax=cbar_ax, orientation='vertical', label="Residual (%)")

    plt.suptitle('%s: parameter scan' % label, y=0.99)
    plt.tight_layout(rect=[0,0,0.9,0.97])
    plt.show()
    return fig, AX
_ = plot_full_parameter_grid(label='PV');

# %%
fig, ax = plt.subplots(1)

res0 = np.load('data/excitability-params-scan-single-comp.npy', allow_pickle=True).item()



# load the two compartment data
label='PV'
res = np.load('data/excitability-params-scan-two-comp-%s.npy' % label, allow_pickle=True).item()

config = (0,2,0)
plt.plot(res0['Fout'], 'bo')
#plt.plot(res['Fouts'][config], 'ro')
print(100*compute_square_difference(res, res0)[config])

for iRmS, iRmD, iRi in itertools.product(
    range(len(res['RmSs'])),
    range(len(res['RmDs'])),
    range(len(res['Ris']))
):
    plt.plot(res['Fouts'][iRmS, iRmD, iRi], 'k-', alpha=0.1)
    
# %%
