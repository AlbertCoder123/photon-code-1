import numpy as np
from scipy import stats, optimize
import pandas as pd
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

def photoelectric(E):
    K_BE = 0.5431
    E_new = E - K_BE
    E = E_new
    return E

def rayleigh(E):
    E = E
    return E

def pair_production(E):
    rest_E = 1.022
    E_new = E - rest_E
    E = E_new
    return E

def compton(E):
    
    theta_values = [] 
    photon_energy_values = []
    fraction_of_energy_transferred = []
    T_values = []
    
    rest_mass = 0.511
    alpha = E/rest_mass
    r_e = 2.8179403262*(10**(-15))
    def comp_cross(x):
       cross_value = (np.pi)*(r_e**2)*np.sin(x)*(1 + (np.cos(x))**2)/(1+alpha*(1-np.cos(x))**2)*(1 + (alpha**2)*(1-(np.cos(x))**2)/(1+(np.cos(x))**2)/(1+alpha*(1-np.cos(x))))
       return cross_value
    theta = 5
    while theta > np.pi:
        R_1 = np.random.uniform(0, np.pi)
        max_x = minimize_scalar(lambda x: -comp_cross(x))
        R_2 = np.random.uniform(0, -max_x.fun)
        if comp_cross(R_1) > R_2:
            theta = R_1
            break
        else:
            theta = 5
    T = E*alpha*(1-np.cos(theta))/(1+alpha*(1-np.cos(theta)))
    theta_values.append(theta)
    photon_energy_values.append(E)
    fraction_of_energy_transferred.append(T/E)
    E_new = E - T
    E = E_new
    return E

cross_sections = { 
    0.0010: [1.37, 0.0132, 4080, 0],
    0.0015: [1.27, 0.0267, 1370, 0],
    0.0020: [1.15, 0.0418, 616, 0],
    0.003: [0.909, 0.0707, 192, 0],
    0.004: [0.708, 0.0943, 82.0, 0],
    0.005: [0.558, 0.112, 41.9, 0],
    0.006: [0.449, 0.126, 24.1, 0],
    0.008: [0.31, 0.144, 9.92, 0],
    0.01: [0.231, 0.155, 4.94, 0],
    0.015: [0.133, 0.17, 1.37, 0],
    0.02: [0.0886, 0.177, 0.544, 0],
    0.03: [0.0469, 0.183, 0.146, 0],
    0.04: [0.0287, 0.183, 0.0568, 0],
    0.05: [0.0194, 0.18, 0.0272, 0],
    0.06: [0.0139, 0.177, 0.0149, 0],
    0.08: [0.00816, 0.17, 0.00577, 0],
    0.1: [0.00535, 0.163, 0.00276, 0],
    0.15: [0.00244, 0.147, 0.000731, 0],
    0.2: [0.00139, 0.135, 0.000289, 0],
    0.3: [0.000622, 0.118, 0.0000816, 0],
    0.4: [0.000351, 0.106, 0.0000349, 0],
    0.5: [0.000225, 0.0966, 0.0000188, 0],
    0.6: [0.000156, 0.0894, 0.0000117, 0],
    0.8: [0.0000879, 0.0786, 0.00000592, 0],
    1: [0.0000563, 0.0707, 0.00000368, 0],
    1.25: [0.000036, 0.0632, 0.00000233, 0.0000178],
    1.5: [0.000025, 0.0574, 0.00000169, 0.0000982],
    2: [0.0000141, 0.049, 0.00000106, 0.000391],
    3: [0.00000626, 0.0385, 0.000000594, 0.00113],
    4: [0.00000352, 0.0322, 0.000000408, 0.00187],
    5 :[0.00000225, 0.0278, 0.000000309, 0.00254],
    6: [0.00000156, 0.0245, 0.000000248, 0.00316]
}

def Photon_Spectrum(n):
    if (n==0):
        return 0
    elif (n>0) and (n<=2480):
        return 0.25 # Photon Energy in MeV
    elif (n>2480) and (n<=15000): 
        return 0.5 # Photon Energy in MeV 
    elif (n>15000) and (n<=27290): 
        return 0.75 # Photon Energy in MeV 
    elif (n>27290) and (n<=37590): 
        return 1 # Photon Energy in MeV 
    elif (n>37590) and (n<=46310): 
        return 1.25 # Photon Energy in MeV 
    elif (n>46310) and (n<=53760): 
        return 1.5 # Photon Energy in MeV 
    elif (n>53760) and (n<=60140): 
        return 1.75 # Photon Energy in MeV 
    elif (n>60140) and (n<=65680): 
        return 2 # Photon Energy in MeV 
    elif (n>65680) and (n<=70460): 
        return 2.25 # Photon Energy in MeV 
    elif (n>70460) and (n<=74630): 
        return 2.5 # Photon Energy in MeV 
    elif (n>74630) and (n<=78290): 
        return 2.75 # Photon Energy in MeV 
    elif (n>78290) and (n<=81510): 
        return 3 # Photon Energy in MeV 
    elif (n>81510) and (n<=84330): 
        return 3.25 # Photon Energy in MeV 
    elif (n>84330) and (n<=86860): 
        return 3.5 # Photon Energy in MeV 
    elif (n>86860) and (n<=89090): 
        return 3.75 # Photon Energy in MeV 
    elif (n>89090) and (n<=91060): 
        return 4 # Photon Energy in MeV 
    elif (n>91060) and (n<=92790): 
        return 4.25 # Photon Energy in MeV 
    elif (n>92790) and (n<=94330): 
        return 4.5 # Photon Energy in MeV 
    elif (n>94330) and (n<=95670): 
        return 4.75 # Photon Energy in MeV 
    elif (n>95670) and (n<=96840): 
        return 5 # Photon Energy in MeV 
    elif (n>96840) and (n<=97850): 
        return 5.25 # Photon Energy in MeV 
    elif (n>97850) and (n<=98710): 
        return 5.5 # Photon Energy in MeV 
    elif (n>98710) and (n<=99420): 
        return 5.75 # Photon Energy in MeV 
    elif (n>99420) and (n<=100000): 
        return 6 # Photon Energy in MeV end


