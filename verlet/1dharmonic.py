import numpy as np
import matplotlib.pyplot as plt

def get_force(position): # F = - grad(U) = - grad(.5 k x^2) = - kx = - m w^2 x
    return - m * w**2 * position

tmin = 0        # Starting time
dt = 0.01       # Time step
n_steps = 3000  # Number of time steps

times = [tmin + i * dt for i in range(n_steps)]
x = []      # Positions
v = []      # Velocities
f = []      # Forces
x.append(0) # Initial position
v.append(1) # Initial velocity
m = 1       # Mass
w = 1       # Angular Frequency for spring
f.append(get_force(x[0])) # Compute inital force

for i in range(n_steps):
    x.append( x[i] + v[i] * dt + ((f[i])/(2 * m) * dt**2) )
    v_half = v[i] + (f[i]/(2 * m)) * dt
    if i < n_steps:
        f.append(get_force(x[i + 1]))
    v.append( v_half + (f[i + 1]/(2 * m)) * dt )

x.pop()
plt.xlim(0, max(times))
plt.ylim(-max(x) - .1,max(x) + .1)
plt.plot(times, x, 'o', color='blue')
plt.xlabel("t")
plt.ylabel("x(t)")
plt.savefig("positions.pdf")
