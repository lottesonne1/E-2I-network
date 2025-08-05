from brian2 import *
from .synapses import get_synapses_eqs

def get_neuron_group(params,
                     model='single-compartement'):
    
    if model=='single-compartement':

        eqs = """
        dV/dt = ( %(Gl)f*nS * ( %(El)f*mV -V ) + I ) / ( %(Cm)f * pF ) : volt (unless refractory)
        I = I0 + gE * ( %(Ee)f*mV -V ) + gI * ( %(Ei)f*mV -V ) : amp
        I0 : amp
        gE : siemens
        gI : siemens
        """ % params

        return NeuronGroup(1, eqs, 
                        threshold='V > %(Vtresh)f*mV' % params, 
                        reset='V=%(Vreset)f * mV ' % params, 
                        refractory=params['Trefrac']*1e-3*second, 
                        method='euler')

    elif model=='two-compartements':

        # soma eq
        RVs = '%(Ri)f*Mohm * %(El)f*mV + %(RmS)f*Mohm * V + %(RmS)f*Mohm * %(Ri)f*Mohm * I0' % params
        Rs = '%(Ri)f*Mohm + %(RmS)f*Mohm' % params
        Ds = ' %(RmS)f*Mohm * %(Ri)f*Mohm * %(CmS)f*pF ' % params
        eqs = "dVs/dt = ( %s - ( %s ) * Vs ) / ( %s ) : volt (unless refractory)\n" % (RVs, Rs, Ds)
        # dendrite eq
        RVd = '%(Ri)f*Mohm * %(El)f*mV + %(RmD)f*Mohm * Vs + %(RmS)f*Mohm * %(Ri)f*Mohm * I' % params
        Rd = '%(Ri)f*Mohm + %(RmD)f*Mohm' % params
        Dd = ' %(RmD)f*Mohm * %(Ri)f*Mohm * %(CmD)f*pF ' % params
        eqs += "dV/dt = ( %s - ( %s ) * V ) / ( %s ) : volt (unless refractory)\n" % (RVd, Rd, Dd)
        eqs += """
        I = gE * ( %(Ee)f*mV - V ) + gI * ( %(Ei)f*mV - V ) : amp
        I0 : amp
        gE : siemens
        gI : siemens
        """ % params
        print(eqs)

        return NeuronGroup(1, eqs, 
                           threshold='Vs > %(Vtresh)f*mV' % params, 
                           reset='Vs=%(Vreset)f * mV ' % params, 
                           refractory=params['Trefrac']*1e-3*second, 
                           method='euler')

def single_cell_simulation(params, 
                           exc_events,
                           inh_events,
                           tstop=1):

    defaultclock.dt = params['dt']*second

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
          'RmS':150, # M Ohm, membrane resistance Soma
          'RmD':300, # M Ohm, membrane resistance Dendrite
          'Ri':100, # M Ohm
          'CmD':300, #pF, membrane capacitance
          'CmS':300, #pF, membrane capacitance
          'El': -70, #mV, leak reversal potential / rest potential
          'Ee': 0, #mV, excitatory reversal potential 
          'Ei': -80, #mV, excitatory reversal potential 
          'Vtresh':-50, #mV, spiking threshold
          'Vreset':-70, #mV, post-spike reset level
          'Trefrac':5, #ms, refractory period
          'dt':1e-4,
                 }

     # initialize brian2 "network"
     network = Network(collect())

     cell = get_neuron_group(params,
                             model='two-compartements')
     network.add(cell)
     M = StateMonitor(cell, ['Vs','V'], record=0)
     network.add(M)
     cell.Vs = params['El']*mV 
     cell.V = params['El']*mV 
     cell.I0 = 0*pA
     network.run(0.1*second)
     cell.I0 = 300*pA
     network.run(0.2*second)
     cell.I0 = 0*pA
     network.run(0.1*second)

     from .plot import plot_Vm
     plot_Vm(M.Vs[0]/mV, params)
     plot_Vm(M.V[0]/mV, params)
     show()
