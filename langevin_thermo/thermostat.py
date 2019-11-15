import numpy as np
import matplotlib.pyplot as plt

def upd_velocity(veloc, force, deltat, mass):
    return veloc + (force * deltat)/(2 * mass)

def upd_position(posit, veloc, deltat):
    return posit + 0.5 * veloc * deltat

def rand_kick(old_v, friction, boltz, mass, deltat):
    gaussian = np.random.normal(0,1)
    sqt1 = np.sqrt(1 - np.exp(-2*friction*deltat) )
    sqt2 = boltz * np.sqrt(mass)
    random = gaussian * sqt1 * sqt2
    new_velocity = old_v * np.exp(- friction * deltat) + random
    return new_velocity

def get_force(posit, mass, frequency): # F = - grad(U) = - grad(.5 k x^2) = - kx = - m w^2 x
    return - mass * frequency**2 * posit

tmin = 0           # Starting time
dt = 0.01          # Delta t
n_steps = 1500000  # Number of time steps
gamma = 4
kbt = 1

times = np.array([tmin + i * dt for i in range(n_steps)])
x = np.zeros((len(times)))      # Initialize Positions
v = np.zeros((len(times)))      # Initialize Velocities
f = np.zeros((len(times)))      # Initialize Forces
x[0] = .1                       # Initial position
v[0] = .5                       # Initial velocity
m = 1                           # Set mass
w = 1                           # Angular Frequency for spring
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

 # Keep all lines below for plots of histograms
_ = plt.hist(x, bins=100, density=True)
plt.xlabel('x')
plt.ylabel('P(x)')
plt.savefig('histx.pdf')
plt.clf()

_ = plt.hist(v, bins=100, density=True)
plt.xlabel('v')
plt.ylabel('P(v)')
plt.savefig('histv.pdf')
plt.clf()
