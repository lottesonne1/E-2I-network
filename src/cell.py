from brian2 import *
from .synapses import get_synapses_eqs

def get_neuron_group(params,
                     model='single-compartment'):
    
    if model=='single-compartment':

        eqs = """
        dV/dt = ( ( 1 / ( %(Rm)f*Mohm ) ) * ( %(El)f*mV -V ) + I ) / ( %(Cm)f * pF ) : volt (unless refractory)
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

    elif model=='two-compartments':

        # soma eq
        As = '%(El)f*mV /  ( %(RmS)f*Mohm * %(CmS)f*pF ) ' % params
        Bs = ' V /  ( %(Ri)f*Mohm * %(CmS)f*pF ) ' % params
        Cs = ' ( %(Ri)f*Mohm + %(RmS)f*Mohm ) / ( %(RmS)f*Mohm * %(Ri)f*Mohm * %(CmS)f*pF )' % params
        eqs = "dVs/dt =  %s + %s - ( %s ) * Vs + I0 / ( %f*pF ) : volt (unless refractory)\n" % (As, Bs, Cs, params['CmS'])
        # dendrite eq
        Ad = '%(El)f*mV /  ( %(RmD)f*Mohm * %(CmD)f*pF ) ' % params
        Bd = ' Vs /  ( %(Ri)f*Mohm * %(CmD)f*pF ) ' % params
        Cd = ' ( %(Ri)f*Mohm + %(RmD)f*Mohm ) / ( %(RmD)f*Mohm * %(Ri)f*Mohm * %(CmD)f*pF )' % params
        eqs += "dV/dt =  %s + %s - ( %s ) * V + I / ( %f * pF ) : volt (unless refractory)\n" % (Ad, Bd, Cd, params['CmD'])
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
                           model='single-compartment',
                           tstop=1):

    defaultclock.dt = params['dt']*second

    network = Network(collect())

    # create cell
    cell = get_neuron_group(params,
                            model=model)
    cell.I0 = 0*pA

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

    spikes = SpikeMonitor(cell)
    network.add(spikes)
    
    sim = {'exc_events':exc_events,
           'inh_events':inh_events,
           'Vm_dend':None, # by default
           'dt':params['dt'],
           'tstop':tstop,
           }

    if model=='single-compartment':

        # record membrane potential
        M = StateMonitor(cell, ['V'], record=0)
        network.add(M)

        # initialize and run
        cell.V = params['El']*mV 
        network.run(tstop*second)

        sim['Vm_soma'] = M.V[0]/mV
        sim['spikes'] = np.array(spikes.t)

    elif model=='two-compartments':

        # record membrane potential
        M = StateMonitor(cell, ['Vs', 'V'], record=0)
        network.add(M)

        # initialize and run
        cell.Vs = params['El']*mV 
        cell.V = params['El']*mV 
        network.run(tstop*second)

        sim['Vm_soma'] = M.Vs[0]/mV
        sim['Vm_dend'] = M.V[0]/mV
        sim['spikes'] = np.array(spikes.t)

    return sim

if __name__=='__main__':

     params = {
          'RmS':150, # M Ohm, membrane resistance Soma
          'RmD':300, # M Ohm, membrane resistance Dendrite
          'Ri':100, # M Ohm
          'CmD':300, #pF, membrane capacitance Dendrite
          'CmS':300, #pF, membrane capacitance Soma
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
                             model='two-compartments')
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
