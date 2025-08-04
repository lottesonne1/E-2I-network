from brian2 import *

def get_membrane_equation(params):

    eqs = """
    dV/dt = ( %(Gl)f*nS * ( %(El)f*mV -V ) + I ) / ( %(Cm)f * pF ) : volt (unless refractory)
    I = I0 + gE * ( %(Ee)f*mV -V ) + gI * ( %(Ei)f*mV -V ) : amp
    I0 : amp
    gE : siemens
    gI : siemens
    """ % params

    return eqs

def get_neuron_group(params):
    
    eqs = get_membrane_equation(params)

    return NeuronGroup(1, eqs, 
                    threshold='V > %(Vtresh)f*mV' % params, 
                    reset='V=%(El)f * mV ' % params, 
                    refractory=params['Trefrac']*1e-3*second, 
                    method='euler')

if __name__=='__main__':

     params = {
          'Gl':10, # nS, leak conductance ( 1 / membrane resistance)
          'Cm':200, #pF, membrane capacitance
          'El': -70, #mV, leak reversal potential / rest potential
          'Ee': 0, #mV, excitatory reversal potential 
          'Ei': -80, #mV, excitatory reversal potential 
          'Vtresh':-50, #mV, spiking threshold
          'Trefrac':5, #ms, refractory period
                 }

     # initialize brian2 "network"
     network = Network(collect())

     cell = get_neuron_group(params)
     network.add(cell)
     M = StateMonitor(cell, ['V','I0'], record=0)
     network.add(M)
     cell.V = -70*mV 
     cell.I0 = 0*pA
     network.run(0.1*second)
     cell.I0 = 300*pA
     network.run(0.2*second)
     cell.I0 = 0*pA
     network.run(0.1*second)
     plot(M.V[0]/mV)
     show()
