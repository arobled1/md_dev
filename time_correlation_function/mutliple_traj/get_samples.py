import numpy as np
import matplotlib.pyplot as plt

def upd_velocity(veloc, force, deltat, mass):
    return veloc + (force * deltat)/(2 * mass)

def upd_position(posit, veloc, deltat):
    return posit + 0.5 * veloc * deltat

def rand_kick(old_v, friction, boltz, mass, deltat):
    gaussian = np.random.normal(0,1)
    sqt1 = np.sqrt(1 - np.exp(-2*friction*deltat) )
    sqt2 = np.sqrt(boltz) * np.sqrt(mass)
    random = gaussian * sqt1 * sqt2
    new_velocity = old_v * np.exp(- friction * deltat) + random
    return new_velocity

def get_force(posit, mass, frequency):
    return - mass * frequency**2 * posit

def bin_centers(bin_edges):
    return (bin_edges[1:]+bin_edges[:-1])/2.

def get_harmonic_density(veloci, inv_temp, mass, omega):
    normalization = (m / (2*np.pi * inv_temp))**0.5
    exp_constant = mass / (2*inv_temp)
    return normalization * np.exp(-exp_constant * veloci**2 )

#============================================================================
# This block is for setting up paramters.
tmin = 0                # Starting time
dt = 0.01               # Delta t
n_steps = 500000        # Number of time steps
kbt = 10                # Temperature
gamma = 10              # Friction

times = np.array([tmin + i * dt for i in range(n_steps)])
x = np.zeros((len(times)))      # Initialize Positions
v = np.zeros((len(times)))      # Initialize Velocities
f = np.zeros((len(times)))      # Initialize Forces
m = 1                           # Set mass
w = np.sqrt(8)                  # Angular frequency
x[0] = .1                       # Initial position
v[0] = .5                       # Initial velocity
f[0] = get_force(x[0], m, w)    # Compute inital force

#============================================================================
# MD starts here!
for i in range(1,n_steps):
    # Update velocity
    v[i] = upd_velocity(v[i-1], f[i-1], dt, m)
    # Update position
    x[i] = upd_position(x[i-1], v[i], dt)
    # Add random velocity
    v[i] = rand_kick(v[i], gamma, kbt, m, dt)
    # Update position based on random velocity
    x[i] = upd_position(x[i], v[i], dt)
    # Update force based on random position
    f[i] = get_force(x[i], m, w)
    # Update velocity based on random force
    v[i] = upd_velocity(v[i], f[i], dt, m)
# MD ends here!

#============================================================================
# This block is for computing a histogram of the velocities to check that the
#   maxwell boltzmann distribution is produced. The bin centers are plotted
#   as points to show that they line up with the boltzmann distribution.
check = np.arange(-15, 15, 0.1)
dist_hist, dist_bin_edges = np.histogram(v[-len(v)//2:],bins=20,density=True)
ideal_prediction_x = get_harmonic_density(bin_centers(dist_bin_edges), kbt, m, w)
p = plt.plot(bin_centers(dist_bin_edges), dist_hist,marker='o',label='P(v)',linestyle='')
plt.plot(check, get_harmonic_density(check, kbt, m, w),linestyle='-',label='Ideal P(v)')
plt.legend(loc='upper left')
plt.xlim(-15, 15)
plt.ylim(-0.01,.14)
plt.xlabel('v')
plt.ylabel('P(v)')
plt.savefig('harmonic_vel_dens.pdf')
plt.clf()

#============================================================================
# This block is for saving every 500th sample and removing the initial values.
save_frequency = 500
samples_x = x[::save_frequency]
samples_x = np.delete(samples_x,0)
samples_v = v[::save_frequency]
samples_v = np.delete(samples_v,0)

#============================================================================
# This block is for writing positions and velocities to file for correlation
#   function calculation.
filename = open("sampled_pos.txt", "a+")
for l in range(len(samples_x)):
    filename.write(str(samples_x[l]))
    filename.write('              ')
    filename.write(str(samples_v[l]) + "\n")
filename.close()
