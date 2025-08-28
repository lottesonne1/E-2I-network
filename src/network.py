from brian2 import *
from .synapses import get_Gabaergic_eqs, get_Glutamatergic_eqs
from .cell import get_neuron_group

import itertools

def single_network_simulation(Model,
                              with_Vm=4,
                              verbose=True):


    # synaptic equations

    REC_POPS = ['PyrExc', 'PvInh', 'SstInh']

    print('initializing simulation [...]')
    NTWK = build_populations(Model,
                             REC_POPS,
                             AFFERENT_POPULATIONS=['AffExc'],
                             with_Vm=with_Vm,
                             verbose=verbose)

    build_up_recurrent_connections(NTWK, params,
                                   SEED=Model['SEED'], 
                                   verbose=verbose)

    Model['tstop'] = Model['rise']+3*(3.*Model['rise']+Model['DT'])

    NTWK['t_array'] = np.arange(\
                int(Model['tstop']/Model['dt']))*Model['dt']
    NTWK['faff_waveform'] = waveform(NTWK['t_array'], Model)

    """
    for i, tpop in enumerate(REC_POPS): # both on excitation and inhibition
        construct_feedforward_input(NTWK, tpop, 'AffExc',
                                    NTWK['t_array'],
                                    NTWK['faff_waveform'],
                                    verbose=verbose,
                                    SEED=int(37*SEED+i)%13)

    initialize_to_rest(NTWK)
    
    network_sim = collect_and_run(NTWK,
                                  verbose=verbose)
    
    """
    print('-> done !')

    return NTWK

def collect_and_run(NTWK, verbose=False):
    """
    collecting all the Brian2 objects and running the simulation
    """
    NTWK['dt'], NTWK['tstop'] = NTWK['Model']['dt'], NTWK['Model']['tstop'] 
    brian2.defaultclock.dt = NTWK['dt']*brian2.ms
    net = brian2.Network(brian2.collect())
    OBJECT_LIST = []
    for key in ['POPS',
                'REC_SYNAPSES', 'RASTER',
                'POP_ACT', 'VMS',
                'PRE_SPIKES', 'PRE_SYNAPSES']:
        if key in NTWK.keys():
            net.add(NTWK[key])

    print('running simulation [...]')
    net.run(NTWK['tstop']*brian2.ms)
    return net

def get_syn_and_conn_matrix(Model,
                            POPULATIONS,
                            AFFERENT_POPULATIONS=[],
                            verbose=False):

    SOURCE_POPULATIONS = POPULATIONS+AFFERENT_POPULATIONS
    
    # creating empty arry of objects (future dictionnaries)
    M = np.empty((len(SOURCE_POPULATIONS), len(POPULATIONS)), dtype=object)
    # default initialisation
    for i, j in itertools.product(range(len(SOURCE_POPULATIONS)), 
                                  range(len(POPULATIONS))):

        source_pop, target_pop = SOURCE_POPULATIONS[i], POPULATIONS[j]

        if len(source_pop.split('Exc'))>1:
            Erev, Ts = Model['Ee'], Model['tauDecayAMPA'] 
        elif len(source_pop.split('Inh'))>1:
            Erev, Ts = Model['Ei'], Model['tauDecayGABA'] # common Erev and Tsyn to all inhibitory currents
        else:
            print(' /!\ SOURCE POP COULD NOT BE CLASSIFIED AS Exc or Inh /!\ ')
            print('-----> set to Exc by default')
            Erev, Ts = Model['Ee'], Model['tauDecayAMPA'] 
            Erev, Ts = Model['Ee'], Model['Tse']

        # CONNECTION PROBABILITY AND SYNAPTIC WEIGHTS
        if ('p_'+source_pop+'_'+target_pop in Model.keys()) and ('Q_'+source_pop+'_'+target_pop in Model.keys()):
            pconn, Qsyn = Model['p_'+source_pop+'_'+target_pop], Model['Q_'+source_pop+'_'+target_pop]
        else:
            if verbose:
                print('No connection for:', source_pop,'->', target_pop)
            pconn, Qsyn = 0., 0.
                
        M[i, j] = {'pconn': pconn, 'Q': Qsyn,
                   'Erev': Erev, 'Tsyn': Ts,
                   'name':source_pop+target_pop}

    return M

def built_up_neuron_params(Model,
                           NRN_KEY, N=1):
    """ we construct a dictionary from the """

    params = {'name':NRN_KEY, 'N':N}

    if Model['type_%s' % NRN_KEY]=='single-compartment':
        keys = ['Rm', 'Cm', 'Ee', 'Ei',
                'Trefrac', 'El', 'Vtresh', 'Vreset']
    elif Model['type_%s' % NRN_KEY]=='two-compartments':
        keys = ['RmS', 'CmS', 'RmD', 'CmD', 'Ri', 'Ee', 'Ei',
                'Trefrac', 'El', 'Vtresh', 'Vreset']
    else:
        print()
        print('neuronal type not recognized !! ')
        print('                     ---> BREAK ')
        keys = []

    for k in keys:
        if NRN_KEY+'_'+k in Model:
            params[k] = Model[NRN_KEY+'_'+k]
            print(params[k])
        else:
            # default value
            params[k] = Model[k]

    return params

def build_populations(Model,
                      POPULATIONS,
                      AFFERENT_POPULATIONS=[],
                      with_Vm=2,
                      verbose=False):
    """
    sets up the neuronal populations
    and construct a network object containing everything: NTWK
    """

    ## NEURONS AND CONNECTIVITY MATRIX
    NEURONS = []
    for pop in POPULATIONS:
        NEURONS.append({'name':pop, 'N':Model['N_'+pop]})

    NTWK = {'NEURONS':NEURONS, 'Model':Model,
            'POPULATIONS':np.array(POPULATIONS),
            'M':get_syn_and_conn_matrix(Model,
                                        POPULATIONS,
                                        AFFERENT_POPULATIONS=AFFERENT_POPULATIONS,
                                        verbose=verbose)}
    
    ########################################################################
    ####  Setting up 
    ########################################################################
    
    NTWK['POPS'] = []
    for ii, nrn in enumerate(NEURONS):
        neuron_params = built_up_neuron_params(Model,
                                               nrn['name'], N=nrn['N'])
        NTWK['POPS'].append(get_neuron_group(neuron_params,
                                             N=nrn['N'],
                                             verbose=verbose))
        nrn['params'] = neuron_params

    ########################################################################
    #### Recordings
    ########################################################################
    
    NTWK['POP_ACT'] = []
    for pop in NTWK['POPS']:
        NTWK['POP_ACT'].append(PopulationRateMonitor(pop))

    NTWK['RASTER'] = []
    for pop in NTWK['POPS']:
        NTWK['RASTER'].append(SpikeMonitor(pop))
        
    if with_Vm>0:
        NTWK['VMS'] = []
        for pop in NTWK['POPS']:
            NTWK['VMS'].append(StateMonitor(pop, 'V', record=np.arange(with_Vm)))
            
    NTWK['PRE_SPIKES'], NTWK['PRE_SYNAPSES'] = [], [] # for future afferent inputs
    
    return NTWK

def build_up_recurrent_connections(NTWK, Model,
                                   SEED=1, verbose=False):
    """
    Construct the synapses from the connectivity matrix 
    """
    CONN = np.empty((len(NTWK['POPS']), len(NTWK['POPS'])), dtype=object)
    CONN2 = []

    np.random.seed(SEED)

    if verbose:
        print('------------------------------------------------------')
        print('drawing random connections [...]')
        print('------------------------------------------------------')
        
    for ii, jj in itertools.product(range(len(NTWK['POPS'])), 
                                    range(len(NTWK['POPS']))):

        source_pop, target_pop = \
            NTWK['NEURONS'][ii]['name'], NTWK['NEURONS'][jj]['name']

        if ('p_'+source_pop+'_'+target_pop in Model.keys()) and\
                            (Model['p_'+source_pop+'_'+target_pop]>0):

            pconn = Model['p_'+source_pop+'_'+target_pop]

            if 'Exc' in source_pop:
                params2 = params.copy()
                if 'qNMDAi_%s' % target_pop in params2:
                    params2['qNMDA'] = params2['qNMDAi_%s' % target_pop]
                SYNAPSES_EQUATIONS, ON_EVENT = \
                                get_Glutamatergic_eqs(params2)

            elif 'Inh' in source_pop:
                SYNAPSES_EQUATIONS, ON_EVENT =\
                                get_Gabaergic_eqs(params2)

            CONN[ii,jj] = Synapses(NTWK['POPS'][ii], NTWK['POPS'][jj],
                            model=SYNAPSES_EQUATIONS.format(**params2),
                            on_pre=ON_EVENT.format(**params2),
                            method='exponential_euler')

            # N.B. the following brian2 settings:
            # CONN[ii,jj].connect(p=NTWK['M'][ii,jj]['pconn'], condition='i!=j')
            # does not fix synaptic numbers, so we draw manually the connections
            N_per_cell = int(pconn*NTWK['POPS'][ii].N)
            if ii==jj: # need to take care of no autapse
                i_rdms = np.concatenate([\
                                np.random.choice(
                                    np.delete(np.arange(NTWK['POPS'][ii].N), [iii]), N_per_cell)\
                                          for iii in range(NTWK['POPS'][jj].N)])
            else:
                i_rdms = np.concatenate([\
                                np.random.choice(np.arange(NTWK['POPS'][ii].N), N_per_cell)\
                                          for jjj in range(NTWK['POPS'][jj].N)])

            j_fixed = np.concatenate([np.ones(N_per_cell,dtype=int)*jjj for jjj in range(NTWK['POPS'][jj].N)])

            CONN[ii,jj].connect(i=i_rdms, j=j_fixed) 
            CONN2.append(CONN[ii,jj])

        else:
            print(source_pop, target_pop, 'not connected')

    NTWK['REC_SYNAPSES'] = CONN2 

def initialize_to_rest(NTWK):
    """
    Vm to resting potential and conductances to 0
    """
    for ii in range(len(NTWK['POPS'])):
        NTWK['POPS'][ii].V = NTWK['NEURONS'][ii]['params']['El']*mV
        for jj in range(len(NTWK['POPS'])):
            if NTWK['M'][jj,ii]['pconn']>0: # if connection
                exec("NTWK['POPS'][ii].G"+NTWK['M'][jj,ii]['name']+" = 0.*nS")

if __name__=='__main__':

    from .default_params import params
    single_network_simulation(params)
