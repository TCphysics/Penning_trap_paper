import numpy as np
import matplotlib.pyplot as plt
import time
import mode_analysis_code
import ion_trapping
import datetime
import coldatoms
print('cooling time test starts at:',datetime.datetime.now())
print('\n')

np.random.seed(2155)
num_ions_0 = 127
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

mode_analysis = mode_analysis_code.ModeAnalysis(N=num_ions_0,
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
# initial_state = coldatoms.json_to_ensemble(open('800ions_180.0kHz.txt').read())
# initial_state = ion_trapping.create_ensemble(mode_analysis.uE,2.0 * np.pi * frot,m_Be,q_Be)

def create_ensemble_thermal_ensemble(my_ensemble, omega_z, mass, charge, T):
    num_ions = my_ensemble.x.shaoe[0]
    # x = uE[:num_ions]
    # y = uE[num_ions:]
    # r = np.sqrt(x**2 + y**2)
    # r_hat = np.transpose(np.array([x / r, y / r]))
    # phi_hat = np.transpose(np.array([-y / r, x / r]))
    # v = np.zeros([num_ions, 2], dtype=np.float64)
    # for i in range(num_ions):
    #     v[i, 0] = omega_z * r[i] * phi_hat[i, 0]
    #     v[i, 1] = omega_z * r[i] * phi_hat[i, 1]

    kB = 1.38064852e-23
    mu = np.sqrt(kB*T/mass)
    ensemble = coldatoms.Ensemble(num_ions)
    for i in range(num_ions):
        ensemble.x[i, 0] = my_ensemble.x[i,0]
        ensemble.x[i, 1] = my_ensemble.x[i,1]
        ensemble.x[i, 2] = 0
        ensemble.v[i, 0] = my_ensemble.v[i, 0]+np.random.normal(0,mu,size=None)
        ensemble.v[i, 1] = my_ensemble.v[i, 1]+np.random.normal(0,mu,size=None)
        ensemble.v[i, 2] = 0.0+np.random.normal(0,mu,size=None)

    ensemble.ensemble_properties['mass'] = mass
    ensemble.ensemble_properties['charge'] = charge
    return ensemble

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

T_thermal = 0.015
initial_state = create_ensemble_thermal(mode_analysis.uE,2.0*np.pi*frot,m_Be,q_Be,T_thermal)
wavelength = 313.0e-9
k = 2.0 * np.pi / wavelength
gamma = 2.0 * np.pi * 18.0e6
hbar = 1.0e-34
off = 40e-6
OFF = np.array([0.0,off,0.0])
wy=25e-6
ywidth = wy*np.sqrt(0.5)
z_S0 = 0.005

det_factor = 0
det = -gamma/2*(1+det_factor)
x_S0 = 1
kx = -k
K = np.array([kx, 0.0, 0.0])

print(mode_analysis.wr/(2*np.pi*frot))
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
num_samples = 4000

evolution_time = sampling_period*num_samples*1e3
finite_temperature_ensemble = initial_state.copy()
trap_potential.reset_phase()
trajectories = [finite_temperature_ensemble.x.copy()]
velocities = [finite_temperature_ensemble.v.copy()]
print(XR,'Rwall,',frot/1000,'KHz Rfreq,offset=',off*1e6,'um, Wy=',wy*1e6,'um')
print('Intensity: SX =',x_S0,'and SZ =',z_S0)
# print('Initial temperature is:',T_thermal*1000,'mK')
print('detunings is:',det_factor)
print('Laser cooling starts at:',datetime.datetime.now())
print('with',sampling_period*num_samples*1e3,'ms, and dt=',dt*1e9,'ns.')
for i in range(num_samples):
    if i%500==0:
        print(i)
    evolve_ensemble(dt, sampling_period,
                    finite_temperature_ensemble, mode_analysis.B, forces+x_cooling_gauss+z_cooling_uni)
    trajectories.append(finite_temperature_ensemble.x.copy())
    velocities.append(finite_temperature_ensemble.v.copy())
trajectories = np.array(trajectories)
velocities = np.array(velocities)
print('Laser cooling ends at  :',datetime.datetime.now())
print('with',sampling_period*num_samples*1e3,'ms, and dt=',dt*1e9,'ns.')
print(XR,'Rwall,',frot/1000,'KHz Rfreq,offset=',off*1e6,'um, Wy=',wy*1e6,'um')
print('Intensity: SX =',x_S0,'and SZ =',z_S0)
# print('Initial temperature is:',T_thermal*1000,'mK')
print('detunings is:',det_factor)
#
# plt.clf()
# plt.plot(finite_temperature_ensemble.x[:,0], finite_temperature_ensemble.x[:,2],'go')
# plt.savefig('after_simulation_side_view.pdf')

# plt.clf()
# plt.plot(initial_state.x[:,0], initial_state.x[:,1],'bo')
# plt.plot(finite_temperature_ensemble.x[:,0], finite_temperature_ensemble.x[:,1],'ro')
# plt.savefig('after_simulation.pdf')
# plt.show()

#import from mode_analysis_code
omega=mode_analysis.wrot
phi0=np.pi/2.0

def Velocity_in_Rotating_Frame(velocities,trajectories,sampling_period):
    V = np.zeros(velocities.shape)
    phi= phi0
    Steps = trajectories.shape[0]
    for step in range(Steps):
        x=trajectories[step,:,0]
        y=trajectories[step,:,1]
        vx=velocities[step,:,0]
        vy=velocities[step,:,1]
        s=np.sin(phi)
        c=np.cos(phi)
        V[step,:,0]=c*vx+s*vy-omega*(s*x-c*y)
        V[step,:,1]=c*vy-s*vx-omega*(s*y+c*x)
        phi+=sampling_period*omega
    V[:,:,2]=velocities[:,:,2]
    return V
def Trajectories_in_Rotating_Frame(trajectories,sampling_period):
    X = np.zeros(trajectories.shape)
    phi= phi0
    Steps = trajectories.shape[0]
    for step in range(Steps):
        x=trajectories[step,:,0]
        y=trajectories[step,:,1]
        s=np.sin(phi)
        c=np.cos(phi)
        X[step,:,0]=c*x+s*y
        X[step,:,1]=-s*x+c*y
        phi+=sampling_period*omega
    X[:,:,2]=trajectories[:,:,2]
    return X

V_R = Velocity_in_Rotating_Frame(velocities,trajectories,sampling_period)
X_R = Trajectories_in_Rotating_Frame(trajectories,sampling_period)

kinetic_energy = 0.5*m_Be*np.sum(np.sum(np.square(V_R),axis=2),axis=1)
Ekx = 0.5*m_Be*np.sum(np.square(V_R[:,:,0]),axis=1)
Eky = 0.5*m_Be*np.sum(np.square(V_R[:,:,1]),axis=1)
Ekz = 0.5*m_Be*np.sum(np.square(V_R[:,:,2]),axis=1)
Ek_in_plane = Ekx + Eky

K_Boltzmann = 1.38064852e-23

temperature = kinetic_energy/num_ions/(1.5*K_Boltzmann)
temperature_x = Ekx/num_ions/(0.5*K_Boltzmann)
temperature_y = Eky/num_ions/(0.5*K_Boltzmann)
temperature_z = Ekz/num_ions/(0.5*K_Boltzmann)
temperature_plane = Ek_in_plane/num_ions/K_Boltzmann
Nt=temperature_z.shape[0]
time = np.arange(0,(Nt-1)*1e-6-1,Nt)

Ave_Tz=np.average(temperature_z[Nt-800:Nt-1])
Ave_Tp=np.average(temperature_plane[Nt-800:Nt-1])
print('Tz approaches:',Ave_Tz)
print('Tp approaches:',Ave_Tp)
# np.save('Tz.npy',temperature_z)
# np.save('Tp.npy',temperature_plane)
# Tz=np.load('Tz.npy')
# Tp=np.load('Tp.npy')
title = 'T with offset='+str(off*1e6)+',ywidth='+str(wy*1e6)
plt.clf()
plt.title(title,fontsize=10)
plt.xlabel('time/micro seconds',fontsize=10)
plt.ylabel('Temperature/mK',fontsize=10)
plt.plot(temperature_z*1e3,'b',ms=1)
plt.plot(temperature_plane*1e3,'g',ms=1)
plt.show()


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
