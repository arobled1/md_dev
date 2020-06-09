import numpy as np
import matplotlib.pyplot as plt

def get_force(position, mass, frequency, scaling_factor): # F = - grad(U) = - grad(.5 k x^2) = - kx = - m w^2 x
    return - mass * (frequency**2) * scaling_factor * position

tmin = 0                    # Starting time
a = 0.99                    # scaling to create 2 a fast force and a slow force
n = 5                       # Arbitrary number of small time steps
large_dt = 0.01             # Time step for slow forces
small_dt = large_dt/n       # Time step for fast forces
n_steps = 500               # Number of time steps

times = np.array([tmin + i * large_dt for i in range(n_steps)])
x = 0                       # Initial position
v = 1                       # Initial velocity
m = 1                       # Set mass
w = 2                       # Angular Frequency for spring
positions = [x]
velocities = [v]

for i in range(1,n_steps):
    v = v + (0.5/m) * large_dt * get_force(x, m, w, 1 - a)
    for j in range(n):
        v = v + (0.5/m) * small_dt * get_force(x, m, w, a)
        x = x + small_dt * v
        v = v + (0.5/m) * small_dt * get_force(x, m, w, a)
    v = v + (0.5/m) * large_dt * get_force(x, m, w, 1 - a)
    positions.append(x)
    velocities.append(v)


plt.xlim(0, max(times))
plt.ylim(-max(positions) - .1,max(positions) + .1)
plt.plot(times, positions, '-', color='blue')
plt.xlabel("t")
plt.ylabel("x(t)")
plt.savefig("positions.pdf")
