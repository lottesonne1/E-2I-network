import numpy as np

def spikes_from_time_varying_rate(time_array, rate_array,
                                  N=10,
                                  Nsyn=1,
                                  SEED=1):
    """
    GENERATES a POISSON PROCESS TO FEED POST_SYNAPTIC CELLS

    N is the number of different processes generated
    Nsyn is the number of presynaptic cell per connection

    /!\ time_array in ms !!
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
        for ii in np.arange(N)[rdm_num<DT*Nsyn*rate_array[it]*1e-3]: # /!\ ms to seconds !! /!\
            indices.append(ii) # all the indices
            times.append(time_array[it]) # all the same time !
                
    return np.array(indices), np.array(times)


