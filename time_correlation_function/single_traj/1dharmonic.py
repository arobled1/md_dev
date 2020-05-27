import numpy as np
import matplotlib.pyplot as plt

def get_force(position): # F = - grad(U) = - grad(.5 k x^2) = - kx = - m w^2 x
    return - m * w**2 * position

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
v[0] = 0                # Initial velocity
m = 1                   # Set mass
w = 2                   # Angular Frequency for spring
f[0] = get_force(x[0])  # Compute inital force

#============================================================================
# This block is for running MD
for i in range(1,n_steps):
    # Update position
    x[i] = x[i - 1] + v[i - 1] * dt + ((f[i - 1])/(2 * m) * dt**2)
    # Compute velocity at half step
    v_half = v[i - 1] + (f[i - 1]/(2 * m)) * dt
    # Update force
    f[i] = get_force(x[i])
    # Update velocity
    v[i] = v_half + (f[i]/(2 * m)) * dt

#============================================================================
# This block is for computing velocity autocorrelation
K = 500    # User specified number of points in each segment  k << M
auto_times = [tmin + i * dt for i in range(K)]

v_auto = []
for n in range(K):
    sum = 0
    for m in range((n_steps-n)):
        sum += v[m]*v[m+n]
    sum = (1/(n_steps-n)) * sum
    v_auto.append(sum)

plt.plot(auto_times,v_auto, '-', color = "blue")
plt.xlabel("t")
plt.ylabel('Velocity Autocorrelation')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.tight_layout()
plt.savefig("autocorrelation.pdf")
plt.clf()
#============================================================================
