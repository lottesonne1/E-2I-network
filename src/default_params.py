params = {
    # ----------------------------- #
    # ---- Cellular Parameters ---- #
    # ----------------------------- #
    #  === -> single-compartment 
    'Rm':100, # [Mohm], membrane resistance
    'Cm':200, # [pF], membrane capacitance
    'El': -70, # [mV], leak reversal potential / ++ potential
    'Ee': 0, # [mV], excitatory reversal potential 
    'Ei': -80, # [mV], excitatory reversal potential 
    'Vtresh':-53, # [mV], spiking threshold
    'Vreset':-70, # [mV], post-spike reset level
    'Trefrac':5, # [ms], refractory period
    #  === -> two-compartments
    'RmS':200, # [Mohm], membrane resistance Soma
    'RmD':200, # M Ohm, membrane resistance Dendrite
    'Ri':3, # M Ohm, intra-compartment resistance
    'CmD':100, #pF, membrane capacitance Dendrite
    'CmS':100, #pF, membrane capacitance Soma
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
    # network populations
    "REC_POPS":["PyrExc", "PvInh", "SstInh", "DsInh"],
    "AFF_POPS":["AffExc"],
    # numbers of neurons in populations:
    "N_PyrExc":4000, "N_PvInh":500, "N_SstInh":500, "N_DsInh":500, "N_AffExc":100,
    # connectivity parameters:
    "p_PyrExc_PyrExc":0.05, "p_PyrExc_PvInh":0.05, "p_PyrExc_SstInh":0.05, 
    "p_PvInh_PyrExc":0.05, "p_PvInh_PvInh":0.05, "p_PvInh_SstInh":0.05, 
    "p_SstInh_PyrExc":0.05, "p_SstInh_PvInh":0.05, "p_SstInh_SstInh":0.05, 
    "p_AffExc_PyrExc":0.1, "p_AffExc_PvInh":0.1, "p_AffExc_SstInh":0.1, 
    # additional params
    "qAMPA_AffExc":4.0,
    "PyrExc_Vthre":-50.0, "DsInh_Vthre":-50.0,
    #
    'dt':1e-4, # second
}



{
    " -------------------------------------------------------------------- ": " ",
    " == Afferent Stimulation Waveform properties for Simulation        == ": " ",
    " -------------------------------------------------------------------- ": " ",
    "AffExc_IncreasingStep_onset": 1000,
    "AffExc_IncreasingStep_baseline": 0.0,
    "AffExc_IncreasingStep_length": 1000.0,
    "AffExc_IncreasingStep_size": 4.0,
    "AffExc_IncreasingStep_smoothing": 100
}
