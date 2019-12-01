import numpy as np
import matplotlib.pyplot as plt

def get_force(position): # F = - grad(U) = - grad(.5 k x^2) = - kx = - m w^2 x
    return - m * w**2 * position

tmin = 0        # Starting time
dt = 0.01       # Time step
n_steps = 500  # Number of time steps

times = np.array([tmin + i * dt for i in range(n_steps)])
x = np.zeros((len(times)))      # Initialize Positions
v = np.zeros((len(times)))      # Initialize Velocities
f = np.zeros((len(times)))      # Initialize Forces
x[0] = 0                # Initial position
v[0] = 1                # Initial velocity
m = 1                   # Set mass
w = 2                   # Angular Frequency for spring
f[0] = get_force(x[0])  # Compute inital force

for i in range(1,n_steps):
    # Update position
    x[i] = x[i - 1] + v[i - 1] * dt + ((f[i - 1])/(2 * m) * dt**2)
    # Compute velocity at half step
    v_half = v[i - 1] + (f[i - 1]/(2 * m)) * dt
    # Update force
    f[i] = get_force(x[i])
    # Update velocity
    v[i] = v_half + (f[i]/(2 * m)) * dt

plt.xlim(0, max(times))
plt.ylim(-max(x) - .1,max(x) + .1)
plt.plot(times, x, 'o', color='blue')
plt.xlabel("t")
plt.ylabel("x(t)")
plt.savefig("positions.pdf")
