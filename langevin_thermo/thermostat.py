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

tmin = 0            # Starting time
dt = 0.001          # Delta t
n_steps = 2000000   # Number of time steps
kbt = 1
gamma = kbt

times = np.array([tmin + i * dt for i in range(n_steps)])
x = np.zeros((len(times)))      # Initialize Positions
v = np.zeros((len(times)))      # Initialize Velocities
f = np.zeros((len(times)))      # Initialize Forces
m = 1                           # Set mass
w = 1                           # Angular Frequency for spring
x[0] = 0                        # Initial position
v[0] = 0.1                      # Initial velocity
f[0] = get_force(x[0], m, w)    # Compute inital force

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

dens_mat = []
sigma = kbt/(m*w**2)
dens_x = np.arange(-5,5,0.1)
for j in range(len(dens_x)):
    # dens_mat.append( ((m*w)/(2*np.pi*np.sinh((kbt)**(-1) * w)))**(1/2) * np.exp(-((m*w)/(2*np.sinh((kbt)**(-1) * w))) * ( 2*dens_x[j]**2 * np.cosh((kbt)**(-1) * w) - 2*dens_x[j]**2 )))
    dens_mat.append( (1/np.sqrt(2*np.pi*sigma))*np.exp(-0.5*(sigma)**(-1) * dens_x[j]**2))

_ = plt.hist(x, bins=20, normed=True)
# Keep all lines below for plots of histograms
plt.xlim(-5, 5)
plt.ylim(-0.2,0.6)
plt.plot(dens_x, dens_mat, color='black')
plt.xlabel('x')
plt.ylabel('P(x)')
plt.savefig("histogram.pdf")
