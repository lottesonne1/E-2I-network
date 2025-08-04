params = {
    # ----------------------------- #
    # ---- Cellular Parameters ---- #
    # ----------------------------- #
    'Gl':10, # [nS], leak conductance ( 1 / membrane resistance)
    'Cm':200, # [pF], membrane capacitance
    'El': -70, # [mV], leak reversal potential / rest potential
    'Ee': 0, # [mV], excitatory reversal potential 
    'Ei': -80, # [mV], excitatory reversal potential 
    'Vtresh':-50, # [mV], spiking threshold
    'Vreset':-60, # [mV], post-spike reset level
    'Trefrac':5, # [ms], refractory period
    # ----------------------------- #
    # ---- Synaptic Parameters ---- #
    # ----------------------------- #
    'qAMPA':4,# [nS] # 
    'qNMDA':4*2.7,# [nS] # NMDA-AMPA ratio=2.7
    'qGABA':10,# [nS] # 
    'tauDecayAMPA':5,# [ms]
    'tauDecayGABA':5,# [ms]
    'tauRiseNMDA': 3,# [ms]
    'tauDecayNMDA': 70,# [ms]
    # -- MG-BLOCK PARAMS  -- #
    'cMg': 1., # mM
    'etaMg': 0.33, # 1/mM
    'V0NMDA':1./0.08,# [mV]
    'Mg_NMDA':1.,# mM
    # ----------------------------- #
    # ----  Network Parameters ---- #
    # ----------------------------- #

    #
    'dt':1e-4, # second
         }
