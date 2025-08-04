params = {
    # ----------------------------- #
    # ---- Cellular Parameters ---- #
    # ----------------------------- #
    'Gl':10, # nS, leak conductance ( 1 / membrane resistance)
    'Cm':200, #pF, membrane capacitance
    'El': -70, #mV, leak reversal potential / rest potential
    'Ee': 0, #mV, excitatory reversal potential 
    'Ei': -80, #mV, excitatory reversal potential 
    'Vtresh':-50, #mV, spiking threshold
    'Trefrac':5, #ms, refractory period
    # ----------------------------- #
    # ---- Synaptic Parameters ---- #
    # ----------------------------- #
    'qAMPA':2,# [nS] # Destexhe et al., 1998: "0.35 to 1.0 nS"
    'qNMDA':2*2.7,# [nS] # NMDA-AMPA ratio=2.7
    'qGABA':8,# [nS] # Destexhe et al., 1998: "0.35 to 1.0 nS"
    'tauDecayAMPA':5,# [ms], Destexhe et al. 1998: "the decay time constant is about 5 ms (e.g., Hestrin, 1993)"
    'tauDecayGABA':5,# [ms] Destexhe et al. 1998
    'tauRiseNMDA': 3,# [ms], Farinella et al., 2014
    'tauDecayNMDA': 70,# [ms], FITTED --- Destexhe et al.:. 25-125ms, Farinella et al., 2014: 70ms
    ##########################
    # -- MG-BLOCK PARAMS  -- #
    ##########################
    'cMg': 1., # mM
    'etaMg': 0.33, # 1/mM
    'V0NMDA':1./0.08,# [mV]
    'Mg_NMDA':1.,# mM
         }
