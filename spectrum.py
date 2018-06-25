import numpy as np
import matplotlib.pyplot as plt
import time
import mode_analysis_code
import ion_trapping
import datetime
import coldatoms
print('cooling time test starts at:',datetime.datetime.now())
print('\n')

line_width = 246.0 / 72.0
default_width = 0.95 * line_width
golden_ratio = 1.61803398875
default_height = default_width / golden_ratio

num_ions = 127
# frot = 180.0e3
frot = 1e3*195
v_wall = 1.0

coulomb_force = coldatoms.CoulombForce()
class TrapPotential(object):

    def __init__(self, kz, delta, omega, phi_0):
        self.kz = kz
        self.kx = -(0.5 + delta) * kz
        self.ky = -(0.5 - delta) * kz
        self.phi_0 = phi_0
        self.phi = phi_0
        self.omega = omega
        self.trap_potential = coldatoms.HarmonicTrapPotential(self.kx, self.ky, self.kz)

    def reset_phase(self):
        self.phi = self.phi_0

    def force(self, dt, ensemble, f):
        self.phi += self.omega * 0.5 * dt
        self.trap_potential.phi = self.phi
        self.trap_potential.force(dt, ensemble, f)
        self.phi += self.omega * 0.5 * dt

mode_analysis = mode_analysis_code.ModeAnalysis(N=num_ions,
                                                Vtrap=(0.0, -1750.0, -1970.0),
                                                Vwall=v_wall,
                                                frot=1.0e-3 * frot)

mode_analysis.run()
m_Be = mode_analysis.m_Be
q_Be = mode_analysis.q
XR=1

trap_potential = TrapPotential(2.0 * mode_analysis.Coeff[2], XR*mode_analysis.Cw, mode_analysis.wrot, np.pi / 2.0)
forces = [coulomb_force, trap_potential]

def evolve_ensemble(dt, t_max, ensemble, Bz, forces):
    num_steps = int(t_max / dt)
    coldatoms.bend_kick(dt, Bz, ensemble, forces, num_steps=num_steps)
    coldatoms.bend_kick(t_max - dt * num_steps, Bz, ensemble, forces)

def create_ensemble_thermal(uE, omega_z, mass, charge, T):
    num_ions = int(uE.size / 2)
    x = uE[:num_ions]
    y = uE[num_ions:]
    r = np.sqrt(x**2 + y**2)
    r_hat = np.transpose(np.array([x / r, y / r]))
    phi_hat = np.transpose(np.array([-y / r, x / r]))
    v = np.zeros([num_ions, 2], dtype=np.float64)
    for i in range(num_ions):
        v[i, 0] = omega_z * r[i] * phi_hat[i, 0]
        v[i, 1] = omega_z * r[i] * phi_hat[i, 1]

    kB = 1.38064852e-23
    mu = np.sqrt(kB*T/mass)
    ensemble = coldatoms.Ensemble(num_ions)
    for i in range(num_ions):
        ensemble.x[i, 0] = x[i]
        ensemble.x[i, 1] = y[i]
        ensemble.x[i, 2] = 0.0
        ensemble.v[i, 0] = v[i, 0]+np.random.normal(0,mu,size=None)
        ensemble.v[i, 1] = v[i, 1]+np.random.normal(0,mu,size=None)
        ensemble.v[i, 2] = 0.0+np.random.normal(0,mu,size=None)

    ensemble.ensemble_properties['mass'] = mass
    ensemble.ensemble_properties['charge'] = charge
    return ensemble

T_thermal = 0.005
initial_state = create_ensemble_thermal(mode_analysis.uE,2.0*np.pi*frot,m_Be,q_Be,T_thermal)
wavelength = 313.0e-9
k = 2.0 * np.pi / wavelength
gamma = 2.0 * np.pi * 18.0e6
hbar = 1.0e-34
off = 5e-6
OFF = np.array([0.0,off,0.0])
wy=5e-6
ywidth = wy*np.sqrt(0.5)
z_S0 = 0.005

det_factor = 0
det = -gamma/2*(1+det_factor)
x_S0 = 1
kx = -k
K = np.array([kx, 0.0, 0.0])

class UniformBeam(object):
    """A laser beam with a uniform intensity profile."""

    def __init__(self, S0):
        """Construct a Gaussian laser beam from position, direction, and width.

        S0 -- Peak intensity (in units of the saturation intensity).
        k -- Propagation direction of the beam (need not be normalized).
        """
        self.S0 = S0

    def intensities(self, x):
        return self.S0 * np.ones_like(x[:, 2])
class GaussianBeam(object):
    """A laser beam with a Gaussian intensity profile."""
    def __init__(self, S0, x0, k, wy):
        """Construct a Gaussian laser beam from position, direction, and width.
        S0 -- Peak intensity (in units of the saturation intensity).
        x0 -- A location on the center of the beam.
        k -- Propagation direction of the beam (need not be normalized).
        wy -- 1/e width of the beam."""
        self.S0 = S0
        self.x0 = np.copy(x0)
        self.k_hat = k / np.linalg.norm(k)
        self.wy = wy

    def intensities(self, x):
        xp = x - self.x0
        xperp = xp - np.outer(xp.dot(self.k_hat[:, np.newaxis]), self.k_hat)
        return self.S0 * np.exp(-np.linalg.norm(xperp, axis=1)**2/self.wy**2)
class DopplerDetuning(object):
    def __init__(self, Delta0, K):
        self.Delta0 = Delta0
        self.K = np.copy(K)

    def detunings(self, x, v):
        return self.Delta0 - np.inner(self.K, v)

z_cooling_uni = [
    coldatoms.RadiationPressure(gamma, hbar * np.array([0.0, 0.0, k]),
                                UniformBeam(S0=z_S0),
                                DopplerDetuning(-0.5 * gamma, np.array([0.0, 0.0, k]))),
    coldatoms.RadiationPressure(gamma, hbar * np.array([0.0, 0.0, -k]),
                                UniformBeam(S0=z_S0),
                                DopplerDetuning(-0.5 * gamma, np.array([0.0, 0.0, -k]))),
                            ]
x_cooling_gauss = [coldatoms.RadiationPressure(gamma, hbar*K, GaussianBeam(x_S0,OFF,K,ywidth), DopplerDetuning(det, K) )]

dt = 1.0e-9
sampling_period = 1e-6
num_samples = 5000

# plt.clf()
# plt.axis("equal")
# plt.xlabel('x/um',fontsize=20)
# plt.ylabel('y/um',fontsize=20)
# plt.plot(initial_state.x[:,0]*1e6, initial_state.x[:,1]*1e6,'o',ms=3)
# plt.show()

evolution_time = sampling_period*num_samples*1e3
finite_temperature_ensemble = initial_state.copy()
trap_potential.reset_phase()
trajectories = [finite_temperature_ensemble.x.copy()]
velocities = [finite_temperature_ensemble.v.copy()]
print(XR,'Rwall,',frot/1000,'KHz Rfreq,offset=',off*1e6,'um, Wy=',wy*1e6,'um')
# print('Intensity: SX =',x_S0,'and SZ =',z_S0)
# print('Initial temperature is:',T_thermal*1000,'mK')
print('Laser cooling starts at:',datetime.datetime.now())
print('with',sampling_period*num_samples*1e3,'ms, and dt=',dt*1e9,'ns.')
for i in range(num_samples):
    if i%1000==0:
        print(i)
    evolve_ensemble(dt, sampling_period,
                    finite_temperature_ensemble, mode_analysis.B, forces+x_cooling_gauss+z_cooling_uni)
    trajectories.append(finite_temperature_ensemble.x.copy())
    velocities.append(finite_temperature_ensemble.v.copy())
trajectories = np.array(trajectories)
velocities = np.array(velocities)
print('Laser cooling ends at  :',datetime.datetime.now())

dt = 1.0e-9
# 0.25 micro seconds gives a Nyquist frequency of 2MHz.
sampling_period = 2.5e-7
# Integrating for a total of sampling_period * num_samples = 5.0e-3 s gives a
# frequency resolution of 200Hz.
num_samples = 40000
free_evolution_ensemble=finite_temperature_ensemble.copy()
trap_potential.reset_phase()
trajectories = [free_evolution_ensemble.x.copy()]

for i in range(num_samples):
    if i%4000==0:
        print(i)
    evolve_ensemble(dt, sampling_period,
                    free_evolution_ensemble, mode_analysis.B, forces)
    trajectories.append(free_evolution_ensemble.x.copy())
trajectories = np.array(trajectories)

nu_nyquist = 0.5 / sampling_period
nu_axis = np.linspace(0.0, nu_nyquist, trajectories.shape[0] // 2)
psd =  np.sum(np.abs(np.fft.fft(trajectories[:,:,2],axis=0) / trajectories.shape[0])**2, axis=1)
psd = psd[0 : psd.size//2 : 1] + psd[2*(psd.size//2) : psd.size//2 : -1]
np.save('nu_axis.npy', nu_axis)
np.save('psd.npy', psd)
nu_axis = np.load('nu_axis.npy')
psd = np.load('psd.npy')

# Make a plot of the whole spectrum.
fig = plt.figure()
spl = fig.add_subplot(111)
for e in mode_analysis.axialEvalsE:
    plt.semilogy(np.array([e, e]) / (2.0 * np.pi * 1.0e6),
                 np.array([1.0e-18, 1.0e-13]),
                 color='gray', linewidth=0.5,
                 zorder=-3)
spl.fill_between(nu_axis / 1.0e6, 1.0e-21, psd, zorder=-2)
spl.set_yscale("log")
plt.semilogy(nu_axis / 1.0e6, psd,
             linewidth=0.75, color='blue', zorder=-1)
plt.xlabel(r'$\nu / \rm{MHz}$')
plt.ylabel(r'PSD($z$)')
plt.xlim([1.0, 1.65])
plt.ylim([1.0e-21, 1.0e-13])
plt.gcf().set_size_inches([default_width, default_height])
plt.subplots_adjust(left=0.2, right=0.97, top=0.95, bottom=0.2)
plt.savefig('fig_axial_spectrum.pdf')


# And a close up of the interval [1.4 MHz, 1.5 MHz].
fig = plt.figure()
spl = fig.add_subplot(111)
for e in mode_analysis.axialEvalsE:
    plt.semilogy(np.array([e, e]) / (2.0 * np.pi * 1.0e6),
                 np.array([1.0e-13, 1.0e-5]),
                 color='gray', linewidth=0.5,
                 zorder=-3)
spl.fill_between(nu_axis / 1.0e6, 1.0e-20, psd, zorder=-2)
spl.set_yscale("log")
plt.semilogy(nu_axis / 1.0e6, psd,
             linewidth=0.75, color='blue', zorder=-1)
plt.xlabel(r'$\nu / \rm{MHz}$')
plt.ylabel(r'PSD($z$)')
plt.xlim([1.4, 1.5])
plt.ylim([1.0e-21, 1.0e-13])
plt.gcf().set_size_inches([default_width, default_height])
plt.subplots_adjust(left=0.2, right=0.97, top=0.95, bottom=0.2)
plt.savefig('fig_axial_spectrum_detail.pdf')

#
#
#
#
#
#
#
#
#
#
