# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %%
from brian2 import * 
from src.cell import get_neuron_group 

# %matplotlib inline

# %% [markdown]
# #Fisrt try single neuron, LIF blueprint

# %%
start_scope()

params = {
    'Gl':10, # nS, leak conductance ( 1 / membrane resistance)
    'Cm':200, #pF, membrane capacitance
    'El': -70, #mV, leak reversal potential / rest potential
    'Ee': 0, #mV, excitatory reversal potential 
    'Ei': -80, #mV, excitatory reversal potential 
    'Vtresh':-50, #mV, spiking threshold
    'Trefrac':5, #ms, refractory period
            }

G = get_neuron_group(params)
M = StateMonitor(G, ['V','I0'], record=0)

G.V = -70*mV 
G.I0 = 0*pA
run(0.1*second)
G.I0 = 300*pA
run(0.2*second)
G.I0 = 0*pA
run(0.1*second)
plot(M.V[0]/mV)

# %%
#Plotting 
for v in Vs:
    plot(M.t/ms, v)
xlabel('Time (ms)')
ylabel('v');

# %% [markdown]
# #PV + SST model single cell 

# %%
start_scope()

#Parameters 
N_PV = 1 
N_SST = 1 
tau = 10*ms #membrane time constant
vr = -70*mV #reset potential after spike 
vt0 = -50*mV #baseline threshold 
delta_vt0 = 5*mV #adaptation after spike 
tau_t = 100*ms #threshold recovery time constant 
sigma = 0.5*(vt0 - vr) #noise amplitude 
v_drive = 2*(vt0 - vr) #constant drive, bias current
duration = 100*ms

#Equations 
#basic: dv/dt=(1-v)/tau : 1 (unless refractory)
#LIF: dv/dt = (v_drive + vr - v)/tau + sigma*xi*tau**-0.5 : volt
                                 #dvt/dt = (vt0 - vt)/tau_t : volt
#simpler LIF: dv/dt = ((El - v) + I)/tau : volt
            #I : volt
#HH: 
eqs = ''' 
dv/dt = (v_drive + vr - v)/tau + sigma*xi*tau**-0.5 : volt
dvt/dt = (vt0 - vt)/tau_t : volt
'''
reset = '''
v = vr
vt += delta_vt0
'''

#Monitoring PV 
PV = NeuronGroup(N_PV, eqs,  threshold='v > vt', reset=reset, refractory=5*ms, method='euler')
M_PV = StateMonitor(PV, 'v', record=0)
S_PV = SpikeMonitor(PV)
spike_counts_PV = S_PV.count 
PV.v = 'rand() * (vt0 - vr) + vr' #randomized initial MP
PV.vt = vt0 #baseline threshold 

#Monitoring SST 
SST = NeuronGroup(N_SST, eqs,  threshold='v > vt', reset=reset, refractory=5*ms, method='euler')
M_SST = StateMonitor(SST, 'v', record=0)
S_SST = SpikeMonitor(SST)
spike_counts_SST = S_SST.count 
SST.v = 'rand() * (vt0 - vr) + vr' #randomized initial MP
SST.vt = vt0 #baseline threshold 
run(duration)

#Plotting 

print(f'PV spike count: {S_PV.count[0]}')
print(f'SST spike count: {S_SST.count[0]}')

figure(figsize=(10, 4))

subplot(1, 2, 1)
plot(M_PV.t/ms, M_PV.v[0]/mV)
title('PV Neuron')
xlabel('Time (ms)')
ylabel('v (mV)')

subplot(1, 2, 2)
plot(M_SST.t/ms, M_SST.v[0]/mV)
title('SST Neuron')
xlabel('Time (ms)')
ylabel('v (mV)')

tight_layout()
show();

# %% [markdown]
# #Simpler LIF 

# %%
start_scope()

#Parameters 
N_PV = 1 
N_SST = 1 
tau = 10*ms #membrane time constant
vr = -70*mV #reset potential after spike 
v_reset = -65*mV
duration = 100*ms
threshold = 'v > 0*mV'


eqs_PV = ''' 
dv/dt = (-v + I)/tau : volt
I : volt
'''

#Monitoring PV 
PV = NeuronGroup(N_PV, eqs_PV,  threshold=threshold, reset='v = v_reset', refractory=5*ms, method='exact')
M_PV = StateMonitor(PV, 'v', record=0)
S_PV = SpikeMonitor(PV)
PV.v = vr #initial MP
PV.I = 15*mV #constant input 
spike_counts_PV = S_PV.count 


eqs_SST = '''
dv/dt = (-v + I - w)/tau : volt
dw/dt = -w / (200*ms) : volt
I : volt
'''
reset_sst = '''
v = v_reset
w += 2*mV  # adaptation increment
'''

#Monitoring SST 
SST = NeuronGroup(N_SST, eqs_SST,  threshold=threshold, reset='v = reset_sst', refractory=5*ms, method='euler')
M_SST = StateMonitor(SST, 'v', record=0)
S_SST = SpikeMonitor(SST)
SST.v = vr #initial MP
SST.w = 0*mV 
SST.I = 15*mV 
spike_counts_SST = S_SST.count 



run(duration)

#Plotting 

print(f'PV spike count: {S_PV.count[0]}')
print(f'SST spike count: {S_SST.count[0]}')

figure(figsize=(10, 4))

subplot(1, 2, 1)
plot(M_PV.t/ms, M_PV.v[0]/mV)
title('PV Neuron')
xlabel('Time (ms)')
ylabel('v (mV)')

subplot(1, 2, 2)
plot(M_SST.t/ms, M_SST.v[0]/mV)
title('SST Neuron')
xlabel('Time (ms)')
ylabel('v (mV)')

tight_layout()
show();
