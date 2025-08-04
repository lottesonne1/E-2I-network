from brian2 import *

def double_exp_normalization(T1, T2):
    # peak normalization of double exponential
    return T1/(T2-T1)*((T2/T1)**(T2/(T2-T1)))


def get_synapses_eqs(params):

    params['nNMDA'] = double_exp_normalization(params['tauRiseNMDA'],params['tauDecayNMDA'])    

    EXC_SYNAPSES_EQUATIONS =\
            """dgDecayAMPA/dt = -gDecayAMPA/({tauDecayAMPA}*ms) : 1 (clock-driven)
               dgRiseNMDA/dt = -gRiseNMDA/({tauRiseNMDA}*ms) : 1 (clock-driven)
               dgDecayNMDA/dt = -gDecayNMDA/({tauDecayNMDA}*ms) : 1 (clock-driven)
               gAMPA = ({qAMPA}*nS)*(gDecayAMPA) : siemens
               gNMDA = ({qNMDA}*nS)*{nNMDA}*(gDecayNMDA-gRiseNMDA)/(1+{etaMg}*{cMg}*exp(-V_post/({V0NMDA}*mV))) : siemens
               gE_post = gAMPA+gNMDA : siemens (summed)""".format(**params)
    ON_EXC_EVENT = 'gDecayAMPA += 1; gDecayNMDA += 1; gRiseNMDA += 1'

    INH_SYNAPSES_EQUATIONS =\
            """dgDecayGABA/dt = -gDecayGABA/({tauDecayGABA}*ms) : 1 (clock-driven)
               gGABA = ({qGABA}*nS)*(gDecayGABA) : siemens
               gI_post = gGABA : siemens (summed)""".format(**params)
    ON_INH_EVENT = 'gDecayGABA+= 1'

    return EXC_SYNAPSES_EQUATIONS, ON_EXC_EVENT,\
            INH_SYNAPSES_EQUATIONS, ON_INH_EVENT



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
