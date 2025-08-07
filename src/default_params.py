params = {
    # ----------------------------- #
    # ---- Cellular Parameters ---- #
    # ----------------------------- #
    #  === -> single-compartment 
    'Rm':100, # [Mohm], membrane resistance
    'Cm':200, # [pF], membrane capacitance
    'El': -70, # [mV], leak reversal potential / rest potential
    'Ee': 0, # [mV], excitatory reversal potential 
    'Ei': -80, # [mV], excitatory reversal potential 
    'Vtresh':-50, # [mV], spiking threshold
    'Vreset':-60, # [mV], post-spike reset level
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

    #
    'dt':1e-4, # second
         }



{
    " -------------------------------------------------------------------- ": " ",
    " ---- ######### Initialisation by default parameters ############ --- ": " ",
    "   UNIT SYSTEM is : ms, mV, pF, nS, pA, Hz  ": " ",
    "                        (arbitrary and unconsistent, so see code)": " ",
    " -------------------------------------------------------------------- ": " ",
    " ": " ",
    " - numbers of neurons in population": " ",
    "REC_POPS":["RecExc", "RecInh", "DsInh"],
    "AFF_POPS":["AffExc"],
    "N_RecExc":4000, "N_RecInh":1000, "N_DsInh":500, "N_AffExc":100,
    " ": " ",
    " - synaptic weights": " ",
    "Q_RecExc_RecExc":2.0, "Q_RecExc_RecInh":2.0, 
    "Q_RecInh_RecExc":10.0, "Q_RecInh_RecInh":10.0, 
    "Q_AffExc_RecExc":4.0, "Q_AffExc_RecInh":4.0,
    " ": " ",
    " - synaptic time constants": " ",
    "Tsyn_Exc":5.0, "Tsyn_Inh":5.0,
    " ": " ",
    " - synaptic reversal potentials": " ",
    "Erev_Exc":0.0, "Erev_Inh": -80.0,
    " ": " ",
    " - connectivity parameters": " ",
    "p_RecExc_RecExc":0.05, "p_RecExc_RecInh":0.05, 
    "p_RecInh_RecExc":0.05, "p_RecInh_RecInh":0.05, 
    "p_AffExc_RecExc":0.1, "p_AffExc_RecInh":0.1, 
    " ": " ",
    " - afferent stimulation": " ",
    "F_AffExc":20.0, "F_DsInh":0.0,
    " ": " ",
    " - recurrent activity (for single cell simulation only)": " ",
    "F_RecExc":1.0, "F_RecInh":1.0,
    " ": " ",
    " - simulation parameters": " ",
    "dt":0.1, "tstop": 6000.0, "SEED":3, " low by default, see later": " ",
    " ": " ",
    " -------------------------------------------------------------------- ": " ",
    " == cellular properties (based on AdExp), population by population == ": " ",
    " -------------------------------------------------------------------- ": " ",
    " --> Excitatory population (RecExc, recurrent excitation)": " ",
    "RecExc_Gl":10.0, "RecExc_Cm":200.0,"RecExc_Trefrac":5.0,
    "RecExc_El":-70.0, "RecExc_Vthre":-50.0, "RecExc_Vreset":-70.0, "RecExc_deltaV":0.0,
    "RecExc_a":0.0, "RecExc_b": 0.0, "RecExc_tauw":1e9,
    " --> Inhibitory population (RecInh, recurrent inhibition)": " ",
    "RecInh_Gl":10.0, "RecInh_Cm":200.0,"RecInh_Trefrac":5.0,
    "RecInh_El":-70.0, "RecInh_Vthre":-53.0, "RecInh_Vreset":-70.0, "RecInh_deltaV":0.0,
    "RecInh_a":0.0, "RecInh_b": 0.0, "RecInh_tauw":1e9,
    " --> Disinhibitory population (DsInh, disinhibition)": " ",
    "DsInh_Gl":10.0, "DsInh_Cm":200.0,"DsInh_Trefrac":5.0,
    "DsInh_El":-70.0, "DsInh_Vthre":-50.0, "DsInh_Vreset":-70.0, "DsInh_deltaV":0.0,
    "DsInh_a":0.0, "DsInh_b": 0.0, "DsInh_tauw":1e9,
    " -------------------------------------------------------------------- ": " ",
    " == Afferent Stimulation Waveform properties for Simulation        == ": " ",
    " -------------------------------------------------------------------- ": " ",
    "AffExc_IncreasingStep_onset": 1000,
    "AffExc_IncreasingStep_baseline": 0.0,
    "AffExc_IncreasingStep_length": 1000.0,
    "AffExc_IncreasingStep_size": 4.0,
    "AffExc_IncreasingStep_smoothing": 100
}
