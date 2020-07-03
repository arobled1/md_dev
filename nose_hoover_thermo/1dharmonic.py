import numpy as np
import matplotlib.pyplot as plt

# Update position
def upd_position(posit, veloc, force, deltat, mass):
    return posit + veloc * deltat + (force/(2*mass) * deltat**2)

# Update velocity
def upd_velocity(veloc, force, deltat, mass):
    return veloc + (force/(2 * mass)) * deltat

def get_force(position, mass, frequency): # F = - grad(U) = - grad(.5 k x^2) = - kx = - m w^2 x
    return - mass * (frequency**2) * position

def get_chain_forces(harmonic_veloc, harmonic_mass, chain_velocities, temp, queue):
    upd_forces = np.zeros(len(chain_velocities))
    # For the first thermostat
    upd_forces[0] = ( ((harmonic_mass*harmonic_veloc)**2)/harmonic_mass - temp) - chain_velocities[1]*chain_velocities[0]*queue
    # For thermostats 2 - M-1
    for j in range(1,len(chain_velocities) - 1):
        upd_forces[j] = ( ((queue*chain_velocities[j-1])**2)/queue - temp) - chain_velocities[j+1] * chain_velocities[j] * queue
    # For thermostat M
    upd_forces[len(chain_velocities) - 1] = ((queue*chain_velocities[len(chain_velocities) - 2])**2)/queue - temp
    return upd_forces

def bin_centers(bin_edges):
    return (bin_edges[1:]+bin_edges[:-1])/2.

def get_harmonic_density_v(veloci, boltz_temp, mass, omega):
    normalization = (m / (2*np.pi * boltz_temp))**0.5
    exp_constant = mass / (2*boltz_temp)
    return normalization * np.exp(-exp_constant * veloci**2 )

tmin = 0                 # Starting time
dt = 0.01                # Time step
n_steps = 2500000        # Number of time steps

times = np.array([tmin + i * dt for i in range(n_steps)])
m = 1                    # Set mass
x = 0                    # Initial position
v = 1                    # Initial velocity
w = 1                    # Angular Frequency for spring
big_q = 1                # Nose hoover parameter
kbt = 1                  # Temperature parameter

chain_q = np.zeros(4)    # intialize chain positions
chain_q[0] = 0
chain_q[1] = 0
chain_q[2] = 0
chain_q[3] = 0
chain_v = np.zeros(4)    # intialize chain velocities
chain_v[0] = 1/big_q
chain_v[1] = -1/big_q
chain_v[2] = 1/big_q
chain_v[3] = -1/big_q
chain_f = np.zeros(4)    # intialize chain forces
chain_f[:] = get_chain_forces(v, m, chain_v, kbt, big_q)
f = get_force(x, m, w)   # Compute inital force

positions = [x]
velocities = [v]
# =============================================================================
# MD Code starts here !!! Integrator is velocity verlet.
for i in range(1,n_steps):
    # Update oscillator position
    x = upd_position(x, v, f, dt, m)
    # Update thermostat positions
    for i in range(len(chain_q)):
        chain_q[i] = upd_position(chain_q[i], chain_v[i], chain_f[i], dt, big_q)
    # Update oscillator velocity at half step
    v = upd_velocity(v, f, dt, m)
    # Update thermostat velocities at half step
    for i in range(len(chain_q)):
        chain_v[i] = upd_velocity(chain_v[i], chain_f[i], dt, big_q)
    # Update oscillator force
    f = get_force(x, m, w) - chain_v[0]*m*v
    # Update thermostat forces
    chain_f[:] = get_chain_forces(v, m, chain_v, kbt, big_q)
    # Update oscillator velocity at half step
    v = upd_velocity(v, f, dt, m)
    # Update thermostat velocities at half step
    for i in range(len(chain_q)):
        chain_v[i] = upd_velocity(chain_v[i], chain_f[i], dt, big_q)
    # Keep position value
    positions.append(x)
    # Keep velocity value
    velocities.append(v)
# MD code ends here !
#============================================================================
# This block is for computing a histogram of the velocities to check that the
#   maxwell boltzmann distribution is produced. The bin centers are plotted
#   as points to show that they line up with the boltzmann distribution.
check = np.arange(-6, 6, 0.1)
vel_hist, vel_bin_edges = np.histogram(velocities[-len(velocities)//2:],bins=20,density=True)
ideal_prediction_v = get_harmonic_density_v(bin_centers(vel_bin_edges), kbt, m, w)
p = plt.plot(bin_centers(vel_bin_edges), vel_hist,marker='o',label='P(v)',linestyle='')
plt.plot(check, get_harmonic_density_v(check, kbt, m, w),linestyle='-',label='Ideal P(v)')
plt.legend(loc='upper left')
plt.xlim(-6, 6)
plt.ylim(-0.01,0.5)
plt.xlabel('v')
plt.ylabel('P(v)')
plt.savefig('harmonic_vel_dens.pdf')
plt.clf()
