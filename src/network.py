from brian2 import *
from .synapses import get_synapses_eqs
from .cell import get_neuron_group


def single_network_simulation(params):

    defaultclock.dt = params['dt']*second

    network = Network(collect())

    # build neurons


    # synaptic equations
    EXC_SYNAPSES_EQUATIONS, ON_EXC_EVENT,\
            INH_SYNAPSES_EQUATIONS, ON_INH_EVENT =\
                    get_synapses_eqs(params)

    pass

if __name__=='__main__':

    print(2)
