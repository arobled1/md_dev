import numpy as np
import matplotlib.pyplot as plt

def get_force(position): # F = - grad(U) = - grad(.5 k x^2) = - kx = - m w^2 x
    return - m * w**2 * position

def upd_velocity(veloc, force, deltat, mass):
    return veloc + (force * deltat)/(2 * mass)

#============================================================================
# This block is for setting up paramters
tmin = 0                # Starting time
dt = 0.01               # Time step
n_steps = 50000         # Number of time steps

times = np.array([tmin + i * dt for i in range(n_steps)])
x = np.zeros((len(times)))      # Initialize Positions
v = np.zeros((len(times)))      # Initialize Velocities
f = np.zeros((len(times)))      # Initialize Forces
x[0] = 1                # Initial position
v[0] = 1                # Initial velocity
m = 3                   # Set mass
w = 2                   # Angular Frequency for spring
f[0] = get_force(x[0])  # Compute inital force

#============================================================================
# This block is for running MD
for i in range(1,n_steps):
    # Update position
    x[i] = x[i - 1] + v[i - 1] * dt + ((f[i - 1])/(2 * m) * dt**2)
    # Compute velocity at half step
    v_half = upd_velocity(v[i - 1], f[i - 1], dt, m)
    # Update force
    f[i] = get_force(x[i])
    # Update velocity
    v[i] = upd_velocity(v_half, f[i], dt, m)

#============================================================================
# This block is for computing velocity autocorrelation
K = 500    # User specified number of points in each segment  k << M
auto_times = [tmin + i * dt for i in range(K)]
x_auto = []
for n in range(K):
    sum = 0
    for m in range((n_steps-n)):
        sum += x[m]*x[m+n]
    sum = (1/(n_steps-n)) * sum
    x_auto.append(sum)
#============================================================================
# This block is for computing the ideal correlation function
total_E = 0.5 * m * v[0]**2 + 0.5 * m * (w**2) * x[0]**2
constant = total_E / (m*w**2)
ideal = [constant * np.cos(w * j) for j in auto_times]
#============================================================================

# This block is for plotting the correlation function
plt.plot(auto_times, x_auto, '-', color = "blue")
plt.xlabel("t")
plt.ylabel(r'$C_{xx}(t)$')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.tight_layout()
plt.savefig("1d_position_auto.pdf")
plt.clf()

# This block is for plotting the ideal correlation function
plt.plot(auto_times, ideal, '-', color = "red")
plt.xlabel("t")
plt.ylabel(r'$C_{xx}(t)$')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.tight_layout()
plt.savefig("ideal.pdf")
plt.clf()
#============================================================================
