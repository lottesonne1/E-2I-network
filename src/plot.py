import matplotlib.pylab as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

def plot_with_stim(resp,
                   tlim=None,
                   peak=0,
                   color='k',
                   figsize=(7,3),
                   AX=None):

    if AX is None:
        fig = plt.figure(figsize=figsize)
        AX = []
        AX.append(plt.subplot2grid((3,1),(0,0), 
                                   rowspan=2,
                                   colspan=1))
        AX.append(plt.subplot2grid((3,1),(2,0), 
                                   rowspan=1,
                                   colspan=1))
    else:
        fig = None

    if tlim is None:
        tlim = [0, len(resp['Vm_soma'])*resp['dt']]

    resp['Vm_soma'][np.array(resp['spikes']/resp['dt'], 
                             dtype=int)] = peak
    AX[0].plot(np.arange(len(resp['Vm_soma']))*resp['dt'], 
               resp['Vm_soma'], color=color)
    AX[0].plot([5e-3,5e-3], [-20,0], 'k-', linewidth=1)
    AX[0].annotate(' 20mV', (1e-3,-10), 
                   rotation=90, va='center', ha='right')
    AX[0].plot([5e-3,105e-3], [-20,-20], 'k-')
    AX[0].annotate('100ms', (55e-3,-21), va='top', ha='center')

    AX[1].plot(resp['inh_events'],
               np.random.randint(0, 20, 
                                 size=len(resp['inh_events'])),
               'o', ms=1, color='r')
    AX[1].annotate('inh ', (0,0), ha='right',
                   xycoords='axes fraction', color='r')
    AX[1].plot(resp['exc_events'],
               np.random.randint(20, 100, 
                                 size=len(resp['exc_events'])),
               'o', ms=1, color='g')
    AX[1].annotate('exc ', (0,1), ha='right', va='top',
                   xycoords='axes fraction', color='g')


    for ax in AX:
        ax.axis('off')
        ax.set_xlim(tlim)

    return fig, AX

def plot_Vm(V, params, 
            ax=None,
            color='k',
            linestyle='-',
            peak=0):

    Vm = V+0.*V
    if ax is None:
        ax = plt.gca()

    reset = params['Vreset']

    ispikes = np.argwhere(\
            (Vm[1:]==params['Vreset']) &
            (Vm[:-1]>params['Vreset']))

    Vm[ispikes] = peak

    return ax.plot(np.arange(len(Vm))*params['dt'], 
                   Vm, 
                   linestyle,
                   color=color)

def plot_ntwk(NTWK,
              log=True,
              colors = ['tab:green', 'tab:red',
                        'tab:orange', 'tab:purple']):

    fig = plt.figure(figsize=(9, 7))
    plt.subplots_adjust()

    grid, AX = (11,1), []
    # afferent stimulation
    AX.append(plt.subplot2grid(grid, (0,0)))
    AX[-1].plot(NTWK['t'], NTWK['faff_waveform'], 'k-')
    AX[-1].set_xticklabels([])
    AX[-1].set_ylabel(r'$\nu_a$ (Hz)')

    # populations activity (instant. firing rates)
    AX.append(plt.subplot2grid(grid, (1, 0), rowspan=2))
    for i, pop in enumerate(NTWK['POPS']):
        rate = NTWK['rates'][i]
        rate = gaussian_filter1d(rate, int(20./0.1)) # smoothing
        rate[rate<0.01] = 0.01
        if log:
            AX[-1].semilogy(NTWK['t'], rate, 
                            '-', color=colors[i], label=pop)
        else:
            AX[-1].plot(NTWK['t'], rate, 
                        '-', color=colors[i], label=pop)
        AX[-1].annotate(i*'\n'+' '+pop, (0,.95), va='top',
                        color=colors[i], 
                        xycoords='axes fraction')
    AX[-1].set_xticklabels([])
    AX[-1].set_ylabel('pop act. (Hz)')

    # raster plot
    AX.append(plt.subplot2grid(grid, (3, 0), rowspan=2))
    n=0
    for i, pop in enumerate(NTWK['POPS']):
        AX[-1].plot(NTWK['raster'][i]['t'],
                    NTWK['raster'][i]['i']+n, 
                    '.', color=colors[i], ms=1)
        n+= NTWK['Model']['N_%s' % pop]
    AX[-1].set_ylabel('neuron ID')

    # sample Vm traces 
    AX.append(plt.subplot2grid(grid, (5, 0), rowspan=6))

    N = [3,1,1,1] # number displayed per population
    j=0 # index to shift the Vm trace
    for i, pop in enumerate(NTWK['POPS']):
        for n in range(N[i]):
            Vm = NTWK['VMs'][i][n]
            ispikes = np.argwhere(\
                  (Vm[1:]==NTWK['Model']['%s_Vreset' % pop]) &
                    (Vm[:-1]>NTWK['Model']['%s_Vreset' % pop]))
            Vm[ispikes] = -10
            AX[-1].plot(NTWK['t'], NTWK['VMs'][i][n]-70*j, 
                    '-', color=colors[i])
            j+=1
    AX[-1].plot([0.09, 0.09], [-60, -40], 'k-')
    AX[-1].annotate(' 20mV', (0.1, -50))
    AX[-1].set_yticks([])
    AX[-1].set_ylabel('sample Vm traces')
    AX[-1].set_xlabel('time (ms)')
    return fig