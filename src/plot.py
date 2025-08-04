import matplotlib.pylab as plt
import numpy as np

def plot_Vm(V, params, 
            ax=None,
            peak=0):

    Vm = V+0.*V
    if ax is None:
        ax = plt.gca()

    reset = params['Vreset']

    ispikes = np.argwhere(\
            (Vm[1:]==params['Vreset']) &
            (Vm[:-1]>params['Vreset']))

    Vm[ispikes] = peak

    return ax.plot(np.arange(len(Vm))*params['dt'], Vm)
