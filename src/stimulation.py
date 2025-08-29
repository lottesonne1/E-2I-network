import numpy as np
from brian2 import *
from .synapses import get_syn_onevent_params

def deal_with_multiple_spikes_per_bin(indices, times, t, verbose=False, debug=False):
    """
    Brian2 constraint:
    spikes have to be shifted to insure no overlapping presynaptic spikes !
    """
    dt = t[1]-t[0]
    
    if verbose:
        print('Insuring only 1 presynaptic spikes per dt [...]')

    indices2, times2 = np.empty(0, dtype=int), np.empty(0) 
    for nn in np.array(np.unique(indices), dtype=int):
        if debug:
            print('neuron ', nn)
        binned_spikes = np.histogram(times[nn==indices], bins=t)[0]
        new_binned_spikes = 0.*binned_spikes
        range_of_spk_num = np.arange(1, np.max(binned_spikes)+1)
        for spk_num in range_of_spk_num[::-1]:
            if debug:
                print(spk_num, binned_spikes)
            # let's find the empty ones
            iempty = np.argwhere(new_binned_spikes==0).flatten()
            # let's find the times corresponding to this high spike number:
            for jj in np.argwhere(binned_spikes==spk_num).flatten():
                new_binned_spikes[iempty[np.argmin((iempty-jj)**2)]] += 1
                binned_spikes[jj] -= 1
            if debug:
                print(spk_num, binned_spikes, new_binned_spikes)

        times2 = np.concatenate([times2, np.array(t[:-1][new_binned_spikes==1]+dt/2.)])
        indices2 = np.concatenate([indices2, nn*np.ones(len(t[:-1][new_binned_spikes==1]), dtype=int)])

    return indices2, times2

def spikes_from_time_varying_rate(time_array, rate_array,
                                  N=10,
                                  Nsyn=1,
                                  SEED=1):
    """
    GENERATES a POISSON PROCESS TO FEED POST_SYNAPTIC CELLS

    N is the number of different processes generated
    Nsyn is the number of presynaptic cell per connection

    /!\ time_array in seconds !!
    /!\ rate_array in Hz !!
    """
    np.random.seed(SEED) # setting the seed !
    
    ## time_array in ms !!
    # so multplying rate array
    DT = (time_array[1]-time_array[0])

    # indices and spike times for the post-synaptic cell:
    indices, times = [], []

    # trivial way to generate inhomogeneous poisson events
    for it in range(len(time_array)):
        rdm_num = np.random.random(N)
        for ii in np.arange(N)[rdm_num<DT*Nsyn*rate_array[it]]:
            indices.append(ii) # all the indices
            times.append(time_array[it]) # all the same time !
                
    return np.array(indices), np.array(times)

def construct_feedforward_input(NTWK, Model,
                                target_pop, afferent_pop,\
                                t, rate_array,\
                                verbose=False,
                                SEED=1):
    """
    This generates an input asynchronous from post synaptic neurons to post-synaptic neurons

    POPS and AFFERENCE_ARRAY should be 1D arrrays as their is only one 
    source population

    'pop_for_conductance' is the string identifying the source conductance
    that will be incremented by the afferent input !!

    """

    if ('p_'+afferent_pop+'_'+target_pop in Model) and\
            (Model['p_'+afferent_pop+'_'+target_pop]>0):
        # if non-zero projection [...]

        Nsyn = Model['p_'+afferent_pop+'_'+target_pop]*Model['N_'+afferent_pop]
        
        #finding the target pop in the brian2 objects
        ipop = np.flatnonzero(NTWK['POPULATIONS']==target_pop)[0]
        
        if verbose:
            print('drawing Poisson process for afferent input [...]')
            
        indices, times = spikes_from_time_varying_rate(t, rate_array,\
                                                       NTWK['POPS'][ipop].N,
                                                       Nsyn,
                                                       SEED=(SEED+2)**2%100)

        # insuring no more than one prespike per bin
        indices, times = deal_with_multiple_spikes_per_bin(indices, times, t, verbose=verbose)

        # incorporating into Brian2 objects
        spikes = SpikeGeneratorGroup(NTWK['POPS'][ipop].N, indices, times*second)
        # sorted = True, see "deal_with_multiple_spikes_per_bin"

        SYNAPSES_EQUATIONS, ON_EVENT, P = get_syn_onevent_params(afferent_pop, 
                                                                 target_pop, 
                                                                 Model)

        synapse = Synapses(spikes, NTWK['POPS'][ipop],
                        model=SYNAPSES_EQUATIONS.format(**P),
                        on_pre=ON_EVENT.format(**P),
                        method='exponential_euler')
        synapse.connect('i==j')

        NTWK['PRE_SPIKES'].append(spikes)
        NTWK['PRE_SYNAPSES'].append(synapse)
        
    else:
        spikes, synapse = None, None
        indices, times, pre_indices, pre_times = [], [], [], []
        if verbose:
            print('Nsyn = 0 for', afferent_pop+'_'+target_pop)
    
    # afferent array
    NTWK['Rate_%s_%s' % (afferent_pop, target_pop)] = rate_array
    
    # storing quantities:
    if 'iRASTER_PRE' in NTWK.keys():
        NTWK['iRASTER_PRE'].append(indices)
        NTWK['tRASTER_PRE'].append(times)

    else: # we create the key
        NTWK['iRASTER_PRE'] = [indices]
        NTWK['tRASTER_PRE'] = [times]

if __name__=='__main__':

    from .cell import *
    from .default_params import params
    from .plot import plot_with_stim

    tstop = 2
    params['dt'] = 5e-5
    t = np.arange(int(tstop/params['dt']))*params['dt']

    Nsyn = int(params['p_PyrExc_PvInh']*params['N_PyrExc'])

    rate = 4 # Hz

    _, exc_events = spikes_from_time_varying_rate(t, rate+0*t,
                                                  N=1,
            Nsyn = int(params['p_PyrExc_PvInh']*params['N_PyrExc']))

    _, inh_events = spikes_from_time_varying_rate(t, 4*rate+0*t,
                                                  N=1,
            Nsyn = int(params['p_PvInh_PvInh']*params['N_PvInh'])+\
                int(params['p_SstInh_PvInh']*params['N_SstInh']))

    resp = single_cell_simulation(params, 
                                exc_events,
                                inh_events,
                                model='single-compartment',
                                tstop=2)

    fig, AX = plot_with_stim(resp)
    show()
