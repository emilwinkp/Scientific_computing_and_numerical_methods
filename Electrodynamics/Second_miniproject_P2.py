## BOBINA DE HELMHOLTZ Y ANTIHELMHOLTZ 

### AUTOR: Emil Winkler Partida 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

def Bz_axis(z, d):
    """ Campo magnetico a lo largo del eje z"""
    return PRE * a**2 * (1/((z - d/2)**2 + a**2)**(3/2)+ 1/((z + d/2)**2 + a**2)**(3/2))

### PRIMERA PARTE - BOBINA DE HELMHOLTZ
def B_field(rho, z, d):
    rho = np.asarray(rho, dtype=float)
    z = np.asarray(z, dtype=float)
    Dp2_1 = (a + rho)**2 + (z + d/2)**2
    Dm2_1 = (a - rho)**2 + (z + d/2)**2
    Dp2_2 = (a + rho)**2 + (z - d/2)**2
    Dm2_2 = (a - rho)**2 + (z - d/2)**2
    m1 = 4.0 * a * rho /Dp2_1
    m2 = 4.0 * a * rho /Dp2_2
    K1 = ellipk(m1)
    E1 = ellipe(m1)
    K2 = ellipk(m2)
    E2 = ellipe(m2)

    Bz_1 = PRE * (1.0/np.sqrt(Dp2_1)) * (K1 + (a**2 - rho**2 - z**2)/Dm2_1 * E1)
    Bz_2 = PRE * (1.0/np.sqrt(Dp2_2)) * (K2 + (a**2 - rho**2 - z**2)/Dm2_2 * E2)
    Bz = Bz_1 + Bz_2

    with np.errstate(divide='ignore', invalid='ignore'):
        Brho_1 = PRE * ((z + d/2)/(rho*np.sqrt(Dp2_1))) * (-K1 + (a**2 + rho**2 + (z + d/2)**2)/Dm2_1 * E1)
        Brho_2 = PRE * ((z - d/2)/(rho*np.sqrt(Dp2_2))) * (-K2 + (a**2 + rho**2 + (z - d/2)**2)/Dm2_2 * E2)
    Brho_1 = np.where(rho < 1e-9, 0.0, Brho_1)
    Brho_2 = np.where(rho < 1e-9, 0.0, Brho_2)
    Brho = Brho_1 + Brho_2
    return Brho, Bz

def A_phi(rho, z, d):
    rho = np.asarray(rho, dtype=float)
    z = np.asarray(z, dtype=float)
    m1 = elliptic_m1(rho,z,d)
    m2 = elliptic_m2(rho,z,d)
    K1 = ellipk(m1)
    E1 = ellipe(m1)
    K2 = ellipk(m2)
    E2 = ellipe(m2)
    with  np.errstate(divide = 'ignore', invalid='ignore'):
        A1 = 2*PRE * np.sqrt(a/(rho*m1)) * ((1 - m1/2)*K1 - E1)
        A2 = 2*PRE * np.sqrt(a/(rho*m2)) * ((1 - m2/2)*K2 - E2)
    A1 = np.where(rho < 1e-9, 0.0, A1)
    A2 = np.where(rho < 1e-9, 0.0, A2)
    A = A1 + A2
    return A

if __name__ == "__main__":
    d = a  # condicion Helmholtz: d = a

    # ── Fig 1: Bz sobre el eje z ──────────────────────────────────────────────
    z_line = np.linspace(-4, 4, 800)
    fig1, ax1 = plt.subplots(figsize=(7.2, 4.6))
    ax1.plot(z_line, Bz_axis(z_line, d), color='steelblue', label=r'$B_z(\rho=0)$')
    for zc in [-d/2, d/2]:
        ax1.axvline(zc, ls='--', color='gray', lw=0.9, label=f'espira z={zc:+.2f}')
    ax1.set_xlabel(r'$z/a$')
    ax1.set_ylabel(r'$B_z$ (u.n.)')
    ax1.set_title(r'$B_z$ en el eje — Bobina de Helmholtz ($d = a$)')
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(_fig('Bz_eje.pdf'), dpi=150)

    # ── Malla 2-D compartida para Fig 2 y Fig 3 ───────────────────────────────
    rho_1d = np.linspace(1e-3, 3.0, 80)
    z_1d   = np.linspace(-3.0, 3.0, 80)
    RHO, Z = np.meshgrid(rho_1d, z_1d)

    # ── Fig 2: Campo vectorial B(rho, z) ──────────────────────────────────────
    Brho, Bz = B_field(RHO, Z, d)
    Bmag = np.hypot(Brho, Bz)

    fig2, ax2 = plt.subplots(figsize=(7, 5.5))
    strm = ax2.streamplot(
        rho_1d, z_1d, Brho, Bz,
        color=np.log1p(Bmag), cmap='inferno',
        linewidth=1.2, density=1.4, arrowsize=1.2,
    )
    fig2.colorbar(strm.lines, ax=ax2, label=r'$\ln(1 + |\mathbf{B}|)$')
    for zc in [-d/2, d/2]:
        ax2.plot(a, zc, 'o', color='cyan', ms=8, label='espira' if zc < 0 else '')
    ax2.set_xlabel(r'$\rho/a$')
    ax2.set_ylabel(r'$z/a$')
    ax2.set_title(r'Campo $\mathbf{B}(\rho, z)$ — Bobina de Helmholtz')
    ax2.legend(loc='upper right')
    fig2.tight_layout()
    fig2.savefig(_fig('B_campo.pdf'), dpi=150)

    # ── Fig 3: Potencial vectorial A_phi(rho, z) ──────────────────────────────
    Aphi = A_phi(RHO, Z, d)
    flux = RHO * Aphi  # curvas de nivel de rho*A_phi = lineas de campo B

    fig3, ax3 = plt.subplots(figsize=(7, 5.5))
    cf = ax3.contourf(rho_1d, z_1d, flux, levels=40, cmap='RdBu_r')
    ax3.contour(rho_1d, z_1d, flux, levels=20, colors='k', linewidths=0.4, alpha=0.5)
    fig3.colorbar(cf, ax=ax3, label=r'$\rho\,A_\phi$ (u.n.)')
    for zc in [-d/2, d/2]:
        ax3.plot(a, zc, 'o', color='black', ms=8, label='espira' if zc < 0 else '')
    ax3.set_xlabel(r'$\rho/a$')
    ax3.set_ylabel(r'$z/a$')
    ax3.set_title(r'Potencial vectorial $A_\phi(\rho, z)$ — Bobina de Helmholtz')
    ax3.legend(loc='upper right')
    fig3.tight_layout()
    fig3.savefig(_fig('A_phi.pdf'), dpi=150)

    plt.show()