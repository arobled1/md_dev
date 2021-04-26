import numpy as np
import matplotlib.pyplot as plt

def get_force(position, mass, frequency): # F = - grad(U) = - grad(.5 k x^2) = - kx = - m w^2 x
    return - mass * (frequency**2) * position

def upd_pos(old_pos, vel, time):
    return old_pos + vel*time

def upd_vel(old_vel, force, mass, time):
    return old_vel + (force * time)/(2 * mass)

tmin = 0        # Starting time
dt = 0.01       # Time step
n_steps = 500   # Number of time steps

times = np.array([tmin + i * dt for i in range(n_steps)])
x = 0                   # Initial position
v = 1                   # Initial velocity
m = 1                   # Set mass
w = 2                   # Angular Frequency for spring
f = get_force(x, m, w)  # Compute inital force
positions = [x]
velocities = [v]
for i in range(1,n_steps):
    # Compute velocity at half step
    v = upd_vel(v, f, m, dt)
    # Update position
    x = upd_pos(x, v, dt)
    # Update force
    f = get_force(x, m, w)
    # Update velocity
    v = upd_vel(v, f, m, dt)
    # Keep position value
    positions.append(x)
    # Keep velocity value
    velocities.append(v)

plt.xlim(0, max(times))
plt.ylim(-max(positions) - .1,max(positions) + .1)
plt.plot(times, positions, '-', color='blue')
plt.xlabel("t")
plt.ylabel("x(t)")
plt.savefig("positions.pdf")
