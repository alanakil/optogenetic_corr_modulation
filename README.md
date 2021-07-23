## Optogenetically-induced correlation modulation in V1 

Here, we simulated a balanced neural network in order to replicate 
results from experiments in visual cortex V1 of macaque monkeys.

In experiments, subsets of pyramidal cells were stimulated repeatedly 
optogenetically. This led to an initial increase in noise correlations, 
followed by a dynamic decrease of correlations. Firing rates remained constant across trials.

We simulated the same experiment and found that inhibitory plasticity, which
is induced by optogenetic input to control firing rates, is the mechanism
behind the dynamic reversal of noise correlations.

To do a single run of the experiment, the file "ArianaSims_EE_EI_IE_II_STDP.m"
 can be used. It allows for plasticity on any synapses and the actual neural 
network used is the same as in Akil et al. (https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008958).

To show the robustness of our results, several simulations were run using 
different seeds and different scaling factors for synapses. 
To do that, we used bash scripts (script_run_*.sh) which call the function myscript.m.
This allowed us to swiftly run simulations in parallel using different 
parameters in compute servers.
