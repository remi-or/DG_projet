CHOSEN_SAMPLES = ['cat6', 'cat7', 'michael13', 'centaur0', 'horse5', 'dog7', 'victoria12', 'cat2']
CHOSEN_SAMPLES_ID = [7, 8, 50, 11, 42, 31, 68, 2]

## Code to infer samples' id
# import os
# samples = [file.split('.')[0]
#            for file in os.listdir('deep_eikonal/data/TOSCA/raw/')
#            if file.endswith('.png')]
# samples = {x : i for i, x in enumerate(samples)}
# CHOSEN_SAMPLES = [samples[x] for x in CHOSEN_SAMPLES]