import matplotlib.pylab as plt
import numpy as np

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
