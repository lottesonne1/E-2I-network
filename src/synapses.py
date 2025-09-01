from brian2 import *

def double_exp_normalization(T1, T2):
    # peak normalization of double exponential
    return T1/(T2-T1)*((T2/T1)**(T2/(T2-T1)))

def get_Glutamatergic_eqs(params):

    params['nNMDA'] = double_exp_normalization(params['tauRiseNMDA'],
                                               params['tauDecayNMDA'])
    EXC_SYNAPSES_EQUATIONS =\
        """dg{name}DecayAMPA/dt = -g{name}DecayAMPA/({tauDecayAMPA}*ms) : 1 (clock-driven)
        dg{name}RiseNMDA/dt = -g{name}RiseNMDA/({tauRiseNMDA}*ms) : 1 (clock-driven)
        dg{name}DecayNMDA/dt = -g{name}DecayNMDA/({tauDecayNMDA}*ms) : 1 (clock-driven)
        g{name}AMPA = ({qAMPA}*nS)*(g{name}DecayAMPA) : siemens
        g{name}NMDA = ({qNMDA}*nS)*{nNMDA}*(g{name}DecayNMDA-g{name}RiseNMDA)/(1+{etaMg}*{cMg}*exp(-V_post/({V0NMDA}*mV))) : siemens
        G{name}_post = g{name}AMPA+g{name}NMDA : siemens (summed)""".format(**params)
    ON_EXC_EVENT = 'g{name}DecayAMPA += 1; g{name}DecayNMDA += 1; g{name}RiseNMDA += 1'

    return EXC_SYNAPSES_EQUATIONS, ON_EXC_EVENT

def get_Gabaergic_eqs(params):

    INH_SYNAPSES_EQUATIONS =\
            """dg{name}DecayGABA/dt = -g{name}DecayGABA/({tauDecayGABA}*ms) : 1 (clock-driven)
               g{name}GABA = ({qGABA}*nS)*(g{name}DecayGABA) : siemens
               G{name}_post = g{name}GABA : siemens (summed)""".format(**params)
    ON_INH_EVENT = 'g{name}DecayGABA += 1'

    return INH_SYNAPSES_EQUATIONS, ON_INH_EVENT 


def get_synapses_eqs(params):

    EXC_SYNAPSES_EQUATIONS, ON_EXC_EVENT = \
        get_Glutamatergic_eqs(params)
    INH_SYNAPSES_EQUATIONS, ON_INH_EVENT = \
        get_Gabaergic_eqs(params)

    return EXC_SYNAPSES_EQUATIONS, ON_EXC_EVENT,\
            INH_SYNAPSES_EQUATIONS, ON_INH_EVENT

def get_syn_onevent_params(source_pop, target_pop, Model):

    if 'Exc' in source_pop:

        P = {'name':source_pop+target_pop,
             'qAMPA': Model['Q_'+source_pop+'_'+target_pop],
             }
        if 'NAR_'+target_pop in Model:
            P['qNMDA'] = Model['NAR_'+target_pop]*\
                    Model['Q_'+source_pop+'_'+target_pop]
            print(target_pop, P['qNMDA'])
        else:
            P['qNMDA'] = 0.

        for k in ['tauDecayAMPA', 
                  'tauRiseNMDA', 'tauDecayNMDA', 
                  'cMg', 'etaMg', 'V0NMDA', 'Mg_NMDA']:
             P[k] = Model[k]

        SYNAPSES_EQUATIONS, ON_EVENT = \
                        get_Glutamatergic_eqs(P)

    elif 'Inh' in source_pop:

        P = {'name':source_pop+target_pop,
             'qGABA': Model['Q_'+source_pop+'_'+target_pop],
             }
        for k in ['tauDecayGABA']:
             P[k] = Model[k]

        SYNAPSES_EQUATIONS, ON_EVENT =\
                        get_Gabaergic_eqs(P)
        
    return SYNAPSES_EQUATIONS, ON_EVENT, P



if __name__=='__main__':

    #######################################
    ##    define synaptic events here    ##
    #######################################
    exc_events = [0.1, 0.2] # in s
    inh_events = [0.5] # in s

    #

    from .cell import get_neuron_group
    from .default_params import params

    # initialize brian2 "network"
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
    M = StateMonitor(cell, ['V','I0'], record=0)
    network.add(M)

    # initialize and run
    cell.V = -70*mV 
    network.run(1*second)

    # plot
    plot(M.V[0]/mV)
    show()
