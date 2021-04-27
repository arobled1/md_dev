#===============================================================================
# Path integral molecular dynamics for a 1D quantum harmonic oscillator.
# This code uses velocity verlet with a langevin thermostat to integrate the
# equations of motion for P beads on a ring with harmonic nearest neighbors that
# are subject to an external potential that is also harmonic.
# Code requires Python 3.
#===============================================================================
# Author: Alan Robledo
# Date: 04/26/21
#===============================================================================
# Notes:
#  The virial energy estimator should be close to the value of energy obtained
#  when you plug in omega and kBT into the equation for the energy of a
#  harmonic oscillator in the canonical ensemble.
#===============================================================================
import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import jit

@jit(nopython=True)
# Update velocity at a half step dt/2
def upd_velocity(veloc, force, deltat, mass, num_beads):
    for p in range(num_beads):
        veloc[p] += (force[p] * deltat)/(2 * mass[p])
    return veloc

@jit(nopython=True)
# Update position at a half step dt/2
def upd_position(posit, veloc, deltat, num_beads):
    for p in range(num_beads):
        posit[p] += 0.5 * veloc[p] * deltat
    return posit

@jit(nopython=True)
# Use langevin thermostat to add randomness to your velocities
def rand_kick(veloc, friction, boltz, mass, deltat, num_beads):
    for p in range(num_beads):
        gaussian = np.random.normal(0,1)
        sqt1 = np.sqrt(1 - np.exp(-2*friction*deltat) )
        sqt2 = np.sqrt(boltz) / np.sqrt(mass[p])
        random = gaussian * sqt1 * sqt2
        veloc[p] = veloc[p] * np.exp(- friction * deltat) + random
    return veloc

@jit(nopython=True)
# Compute force from the harmonic springs in ring
def get_harmonic_force(force, xi, mass, frequency, num_beads):
    for p in range(num_beads):
        force[p] = - mass[p] * (frequency**2) * xi[p]
    return force

@jit(nopython=True)
# Compute force from the external potential (harmonic potential U = 1/2mw^2x^2)
def get_ext_force(force, xi, mass, frequency, num_beads):
    for p in range(num_beads):
        force[p] = - mass * (frequency**2) * xi[p]
    return force

@jit(nopython=True)
# Transform primitive coordinates to staged coordinates
def stage_coords(xi, num_beads):
    up_u = np.zeros(num_beads)
    # For the first bead
    up_u[0] = xi[0]
    # For the in between beads
    for k in range(1,num_beads-1):
        up_u[k] = xi[k] - ( k*xi[k+1] + xi[0] ) / (k + 1)
    # For the last bead
    up_u[num_beads-1] = xi[num_beads-1] - ( ( (num_beads-2)*xi[0] ) + xi[0] )/(num_beads - 1)
    return up_u

@jit(nopython=True)
# Transform staged coordinates to primitive coordinates
def inverse_stage_coords(ui, num_beads):
    up_x = np.zeros(num_beads)
    # For the first bead
    up_x[0] = ui[0]
    # For the P bead
    up_x[num_beads-1] = ui[num_beads-1] + ((num_beads-1)/(num_beads))*up_x[0] + (1/(num_beads))*ui[0]
    # For the in between beads (loop goes backwards from bead P to bead 2)
    for k in range(num_beads - 2, 0,-1):
        up_x[k] = ui[k] + (k/(k+1))*up_x[k+1] + (1/(k+1))*ui[0]
    return up_x

@jit(nopython=True)
# Compute external potential in staged coordinates using external potential in primitive coordinates
def inverse_force_transformation(force_x):
    new_ext_u = np.zeros(len(force_x))
    tmp_sum = 0
    # Force acting on the u1 bead
    for l in range(len(force_x)):
        tmp_sum += force_x[l]
    new_ext_u[0] = tmp_sum
    # Force acting on every other bead
    for k in range(1, len(force_x)):
        new_ext_u[k] = force_x[k] + ((k-1)/k)*(new_ext_u[k-1])
    return new_ext_u

@jit(nopython=True)
# Computing the virial estimator
def get_virial_est(xi, mass, frequency, num_beads):
    sum_v = 0
    for p in range(num_beads):
        sum_v += (mass*frequency**2)*xi[p]**2
    # Scale the estimator by the inverse of the number of beads
    sum_v /= pbeads
    return sum_v

#=============================================================================
# This block is for MD prep. Setting up parameters and initial values.
n_steps = 60000                       # Number of time steps
pbeads = 400                          # Number of beads
kbt = 3/15.8                          # Temperature (KbT)
w = 3                                 # Set Frequency
m = 0.01                                 # Set Mass
tmin = 0                              # Starting time
dt = 0.01                             # Delta t
times = np.array([tmin + i * dt for i in range(n_steps)])
gamma = np.sqrt(pbeads)*kbt           # Friction

# Initialize bead positions
primitives_x = np.zeros(pbeads)
# Sample initial positions
for rando in range(pbeads):
    primitives_x[rando] = np.random.normal(0,np.sqrt(kbt/(m*w**2)) )

# Initialize bead velocities
v = np.zeros(pbeads)
# Sample initial velocities from boltzmann distribution
for rando in range(pbeads):
    v[rando] = np.random.normal(0, np.sqrt(kbt/m))

# Masses for the harmonic coupling
m_k = np.zeros(pbeads)
# Masses for the velocities
m_prime_k = np.zeros(pbeads)
# Mass for the first bead
m_k[0] = 0
m_prime_k[0] = m
# Mass for the rest of the beads
for i in range(1, pbeads):
    m_k[i] = (i+1) * m / i
    m_prime_k[i] = (i+1) * m / i

u = np.zeros(pbeads)
# Compute initial stage coords (x --> u)
u = stage_coords(primitives_x, pbeads)
# Initialize harmonic forces
f_u = np.zeros(pbeads)
# Compute initial harmonic forces
f_u = get_harmonic_force(f_u, u, m_k, np.sqrt(pbeads)*kbt, pbeads)
# Inititialize external forces in x
f_x = np.zeros(pbeads)
# Compute initial external forces in x
f_x = get_ext_force(f_x, primitives_x, m, w, pbeads)
# Compute initial forces in staged coords
f_u += (1/pbeads)*inverse_force_transformation(f_x)

virial = np.zeros(n_steps-1)
#=============================================================================
# MD code starts here !!
for i in range(1,n_steps):
    # Update velocity
    v = upd_velocity(v, f_u, dt, m_prime_k, pbeads)
    # Update position
    u = upd_position(u, v, dt, pbeads)
    # Add random velocity
    v = rand_kick(v, gamma, kbt, m_prime_k, dt, pbeads)
    # Update position with random velocity
    u = upd_position(u, v, dt, pbeads)
    # Update harmonic force with random position
    f_u = get_harmonic_force(f_u, u, m_k, np.sqrt(pbeads)*kbt, pbeads)
    # Update positions in x (converting u --> x)
    primitives_x = inverse_stage_coords(u, pbeads)
    # Compute external force with updated positions in x
    f_x = get_ext_force(f_x, primitives_x, m, w, pbeads)
    # Update forces in stage coordinates
    f_u += (1/pbeads)*inverse_force_transformation(f_x)
    # Update velocity based on random force
    v = upd_velocity(v, f_u, dt, m_prime_k, pbeads)
    # Computing the virial energy estimator
    virial[i-1] = get_virial_est(primitives_x, m, w, pbeads)
# MD code ends here
#=============================================================================
# This block is for analysis.

# Computing cumulative average of the virial estimator
cume = np.zeros(n_steps-1)
cume[0] = virial[0]
for i in range(1,len(virial)):
   cume[i] = (i)/(i+1)*cume[i-1] + virial[i]/(i+1)
steps = np.arange(1,n_steps)

# Writing values to file
filename = open("energies.txt", "a+")
for l in range(len(virial)):
   filename.write(str(virial[l]))
   filename.write('              ')
   filename.write(str(cume[l]))
   filename.write('              ')
   filename.write(str(steps[l]) + "\n")
filename.close()

# Plotting the virial estimator
plt.xlim(min(steps)-100, max(steps))
plt.ylim(0,7)
plt.axhline(y=1.5, linewidth=1.5, color='r', label=r'$\epsilon_{vir} = 1.5$')
plt.plot(steps, virial, '-', color='black', label=r'$\epsilon_{vir}$', alpha=0.4)
plt.plot(steps, cume, '-', label='cumulative average', color='blue')
plt.legend(loc='upper left')
plt.xlabel('# of Steps')
plt.ylabel(r'$\epsilon_{vir} \quad / \quad \hbar \omega$')
plt.savefig('virial.pdf')
plt.clf()
#=============================================================================
