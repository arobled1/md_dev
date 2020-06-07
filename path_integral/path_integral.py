#===============================================================================
# Path integral molecular dynamics for a 1D harmonic oscillator.
# This code uses velocity verlet to integrate the equations of motion for P
# beads on a ring with harmonic nearest neighbors that are subject to an
# external potential that is also harmonic.
# Each bead is coupled to a langevin thermostat.
#===============================================================================
# Author: Alan Robledo
# Date: 11/15/19
#===============================================================================
# Notes:
#  The virial energy estimator should be close to the value of energy obtained
#  when you plug in omega and KBT into the equation for the energy of a
#  harmonic oscillator derived from the derivative of the canonical partition
#  function.
#===============================================================================
import numpy as np
import matplotlib.pyplot as plt

# Update velocity at a half step dt/2
def upd_velocity(veloc, force, deltat, mass):
    return veloc + (force * deltat)/(2 * mass)

# Update position at a half step dt/2
def upd_position(posit, veloc, deltat):
    return posit + 0.5 * veloc * deltat

# Use langevin thermostat to add randomness to your velocities
def rand_kick(old_v, friction, boltz, mass, deltat):
    gaussian = np.random.normal(0,1)
    sqt1 = np.sqrt(1 - np.exp(-2*friction*deltat) )
    sqt2 = np.sqrt(boltz) * np.sqrt(mass)
    random = gaussian * sqt1 * sqt2
    new_velocity = old_v * np.exp(- friction * deltat) + random
    return new_velocity

# Compute force from the harmonic potential U = 1/2mw^2x^2
def get_harmonic_force(xi, mass, frequency):
    return - mass * (frequency**2) * xi

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

# Compute external potential in staged coordinates using external potential in primitive coordinates
def inverse_force_transformation(force_x):
    new_ext_u = np.zeros(len(force_x))
    sum = 0
    # Force acting on the u1 bead
    for l in range(len(force_x)):
        sum += force_x[l]
    new_ext_u[0] = sum
    # Force acting on every other bead
    for k in range(1, len(force_x)):
        new_ext_u[k] = force_x[k] + ((k-1)/k)*(new_ext_u[k-1])
    return new_ext_u

#=============================================================================
# This block is for MD prep. Setting up parameters and initial values.

n_steps = 3000000                     # Number of time steps
pbeads = 400                          # Number of beads
kbt = 0.00189873                      # Temperature (KbT)
w = 0.03                              # Set Frequency
m = 1                                 # Set Mass
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
# Sample initial velocities
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
    m_k[i] = (i+1) * m / k
    m_prime_k[i] = (i+1) * m / k

u = np.zeros(pbeads)
# Compute initial stage coords
u[:] = stage_coords(primitives_x[:], pbeads)
# Initialize harmonic forces
f_u = np.zeros(pbeads)
# Compute initial harmonic forces
for p in range(pbeads):
    f_u[p] = get_harmonic_force(u[p], m_k[p], np.sqrt(pbeads)*kbt)
# Inititialize external forces in x
f_x = np.zeros(pbeads)
# Compute initial external forces in x
for p in range(pbeads):
    f_x[p] = get_harmonic_force(primitives_x[p], m, w)
# Compute initial forces in staged coords (eqn 12.6.12)
f_u[:] = f_u[:] + (1/pbeads)*inverse_force_transformation(f_x[:])

virial = []
#=============================================================================
# MD code starts here !!
for i in range(1,n_steps):
    # Update velocity
    for p in range(pbeads):
        v[p] = upd_velocity(v[p], f_u[p], dt, m_prime_k[p])
    # Update position
    for p in range(pbeads):
        u[p] = upd_position(u[p], v[p], dt)
    # Add random velocity
    for p in range(pbeads):
        v[p] = rand_kick(v[p], gamma, kbt, m_prime_k[p], dt)
    # Update position with random velocity
    for p in range(pbeads):
        u[p] = upd_position(u[p], v[p], dt)
    # Update harmonic force with random position
    for p in range(pbeads):
        f_u[p] = get_harmonic_force(u[p], m_k[p], np.sqrt(pbeads)*kbt)
    # Update positions in x
    x[:] = inverse_stage_coords(u[:], pbeads)
    # Compute external force in updated x's
    for p in range(pbeads):
        f_x[p] = get_harmonic_force(x[p], m, w)
    # Update forces in stage coordinates
    f_u[:] = f_u[:] + (1/pbeads)*inverse_force_transformation(f_x[:])
    # Update velocity based on random force
    for p in range(pbeads):
        v[p] = upd_velocity(v[p], f_u[p], dt, m_prime_k[p])
    # Computing the virial energy estimator
    sumv = 0
    for o in range(pbeads):
        sumv += (m*w**2)*primitives_x[o]**2
    # Scale the estimator by the inverse of the number of beads
    virial.append(sumv / pbeads)
# MD code ends here
#=============================================================================
# This block is for analysis.

# Compute cumulative averages of the virial estimator
cume = np.zeros(len(virial))
cume[0] = virial[0]
for i in range(1,len(virial)):
    cume[i] = (i)/(i+1)*cume[i-1] + virial[i]/(i+1)
steps = np.arange(n_steps)

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
plt.ylim(-0.005,0.07)
plt.axhline(y=0.015, linewidth=2, color='r', label=r'$\epsilon_{vir} = 0.015$')
plt.plot(steps, virial, '-', color='black', label=r'$\epsilon_{vir}$', alpha=0.4)
plt.plot(steps, cume, '-', label='cumulative average', color='blue')
plt.legend(loc='upper left')
plt.xlabel('# of Steps')
plt.ylabel(r'$\epsilon_{vir} \quad / \quad \hbar \omega$')
plt.savefig('virial.pdf')
plt.clf()
#=============================================================================
