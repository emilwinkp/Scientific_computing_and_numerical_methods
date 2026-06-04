## BOBINA DE HELMHOLTZ Y ANTIHELMHOLTZ 

### AUTOR: Emil Winkler Partida 

import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np 
import subprocess, os

_DIR = os.path.dirname(os.path.abspath(__file__))
def _fig(name): return os.path.join(_DIR, name)
from scipy.special import ellipk, ellipe  # K(m), E(m)  (parametro m = k^2)

# Parametros de normalizacion

a = 1.0 # radio
PRE = 1.0 # = mu0 I /(2 pi)

def elliptic_m1(rho, z, d):
    """ Parametro m de las funciones elipticas en funcion de rho y z"""
    return 4.0 * a * rho / ((a + rho)**2 + (z + d/2)**2)

def elliptic_m2(rho, z, d):
    return 4.0 * a * rho / ((a + rho)**2 + (z - d/2)**2)

def B_field(rho, z, d):
    pass

def Bz_axis(z, d):
    """ Campo magnetico a lo largo del eje z"""
    return PRE * a**2 * (1/((z - d/2)**2 + a**2)**(3/2)+ 1/((z + d/2)**2 + a**2)**(3/2))


if __name__ == "__main__":
    # zt = np.array([0.0, 0.5, 1.0, 2.0])
    z = np.linspace(-4, 4, 800)
    Bz_eje = Bz_axis(z, a)
    fig, ax = plt.subplots(figsize = (7.2,4.6))
    ax.plot(z, Bz_eje, label = r'$B_z$')