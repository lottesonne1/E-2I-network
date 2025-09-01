# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
NTWK = np.load('new.data.npy', allow_pickle=True).item()

from src.plot import plot_ntwk


plot_ntwk(NTWK, log=True) 
# %%

from src.network import Model, run_3pop_ntwk_model, save

Model['PvInh_RmS'] = 290.
Model['PvInh_RmD'] = 171.
Model['PvInh_Ri'] = 3.
Model['PvInh_Vtresh'] = -53.

Model['SstInh_RmS'] = 40.
Model['SstInh_RmD'] = 300.
Model['SstInh_Ri'] = 50.
Model['SstInh_Vtresh'] = -63.

REC_POPS =  ['PyrExc', 'PvInh', 'SstInh', 'DsInh']
NTWK = run_3pop_ntwk_model(Model, REC_POPS,
                            with_Vm=3,
                               verbose=True)

 
save(NTWK, REC_POPS,
     filename='new.data.npy') 

# %%
