from brian2 import *
from .synapses import get_synapses_eqs


def get_membrane_equation(params):

    eqs = """
    dV/dt = ( %(Gl)f*nS * ( %(El)f*mV -V ) + I ) / ( %(Cm)f * pF ) : volt (unless refractory)
    I = I0 + gE * ( %(Ee)f*mV -V ) + gI * ( %(Ei)f*mV -V ) : amp
    I0 : amp
    gE : siemens
    gI : siemens
    """ % params

    return eqs

def get_neuron_group(params):
    
    eqs = get_membrane_equation(params)

    return NeuronGroup(1, eqs, 
                    threshold='V > %(Vtresh)f*mV' % params, 
                    reset='V=%(Vreset)f * mV ' % params, 
                    refractory=params['Trefrac']*1e-3*second, 
                    method='euler')

def single_cell_simulation(params, 
                           exc_events,
                           inh_events,
                           tstop=1):

    network = Network(collect())

    # create cell
    cell = get_neuron_group(params)
    network.add(cell)

    # get synaptic equations
    EXC_SYNAPSES_EQUATIONS, ON_EXC_EVENT,\
            INH_SYNAPSES_EQUATIONS, ON_INH_EVENT =\
                    get_synapses_eqs(params)


    # create synaptic events
    # - excitatory:
    exc_spikes = SpikeGeneratorGroup(1,
                                     np.zeros(len(exc_events), dtype=int),
                                     np.array(exc_events)*second)
    network.add(exc_spikes)
    # - inhibitory:
    inh_spikes = SpikeGeneratorGroup(1,
                                     np.zeros(len(inh_events), dtype=int),
                                     np.array(inh_events)*second)
    network.add(inh_spikes)

    # create synapses
    # - excitatory:
    exc_synapses = Synapses(exc_spikes, cell,
                            model=EXC_SYNAPSES_EQUATIONS.format(**params),
                            on_pre=ON_EXC_EVENT.format(**params),
                            method='exponential_euler')
    exc_synapses.connect(i=0, j=0)
    network.add(exc_synapses)
    # - inhibitory:
    inh_synapses = Synapses(inh_spikes, cell,
                            model=INH_SYNAPSES_EQUATIONS.format(**params),
                            on_pre=ON_INH_EVENT.format(**params),
                            method='exponential_euler')
    inh_synapses.connect(i=0, j=0)
    network.add(inh_synapses)

    # record membrane potential
    M = StateMonitor(cell, ['V'], record=0)
    network.add(M)

    # initialize and run
    cell.V = params['El']*mV 
    network.run(tstop*second)

    return M.V[0]/mV


if __name__=='__main__':

     params = {
          'Gl':10, # nS, leak conductance ( 1 / membrane resistance)
          'Cm':200, #pF, membrane capacitance
          'El': -70, #mV, leak reversal potential / rest potential
          'Ee': 0, #mV, excitatory reversal potential 
          'Ei': -80, #mV, excitatory reversal potential 
          'Vtresh':-50, #mV, spiking threshold
          'Vreset':-70, #mV, post-spike reset level
          'Trefrac':5, #ms, refractory period
                 }

     # initialize brian2 "network"
     network = Network(collect())

     cell = get_neuron_group(params)
     network.add(cell)
     M = StateMonitor(cell, ['V','I0'], record=0)
     network.add(M)
     cell.V = params['El']*mV 
     cell.I0 = 0*pA
     network.run(0.1*second)
     cell.I0 = 300*pA
     network.run(0.2*second)
     cell.I0 = 0*pA
     network.run(0.1*second)

     from .plot import plot_Vm
     plot_Vm(M.V[0]/mV, params)
     show()
