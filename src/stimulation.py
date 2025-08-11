import numpy as np

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
