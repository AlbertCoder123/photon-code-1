import numpy as np
from scipy import stats, optimize
import pandas as pd
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

from functions import Photon_Spectrum
from functions import rayleigh
from functions import compton
from functions import photoelectric
from functions import pair_production
from functions import cross_sections

E_initial = 6
E_threshold = 0.01
num_interactions = 0

energies = []
interaction_type = []
#hi
#jon was here
photon_energy = E_initial 

N = 10

for i in range(N):
    photon_energy = Photon_Spectrum(i) 
    E_threshold = 0.01

    while photon_energy >= E_threshold:
        lower_energies = [x for x in cross_sections.keys() if x <= photon_energy]
        closest_low_energy = max(lower_energies)

        upper_energies = [ x for x in cross_sections.keys() if x >= photon_energy]
        closest_high_energy = min(upper_energies)
    
        cross_section_values_low = cross_sections[closest_low_energy]
        cross_section_values_high = cross_sections[closest_high_energy]

        if closest_high_energy - closest_low_energy != 0:
            slope = []
            new_cross_section = []
            for i in range(len(cross_section_values_high)):
                slope.append((cross_section_values_high[i] - cross_section_values_low[i]) / (closest_high_energy - closest_low_energy))
                new_cross_section.append(cross_section_values_low[i] + slope[i] * (photon_energy - closest_low_energy))
    
            interaction = np.random.choice(['rayleigh', 'compton', 'photoelectric', 'pair+trip'], p = new_cross_section/np.sum(new_cross_section))
        else:
            interaction = np.random.choice(['rayleigh', 'compton', 'photoelectric', 'pair+trip'], p = cross_section_values_low/np.sum(cross_section_values_low))
        #interaction_type.append(interaction)
        #energies.append(photon_energy)

        if interaction == 'rayleigh':
            photon_energy = rayleigh(photon_energy)
        elif interaction == 'compton':
            photon_energy = compton(photon_energy)
        elif interaction == 'photoelectric':
            photon_energy = photoelectric(photon_energy)
        else:
            photon_energy = pair_production(photon_energy)
            break
        #num_interactions += 1
    
    