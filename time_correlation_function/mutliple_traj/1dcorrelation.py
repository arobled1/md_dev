import numpy as np
import matplotlib.pyplot as plt

def get_force(position, mas, frequency): # F = - grad(U) = - grad(.5 k x^2) = - kx = - m w^2 x
    return - mas * frequency**2 * position

def upd_velocity(veloc, force, deltat, mass):
    return veloc + (force * deltat)/(2 * mass)

def get_trajectory(total_steps, deltat, init_time, init_pos, init_vel, mass, omega):
    times = np.array([init_time + i * deltat for i in range(total_steps)])
    x = np.zeros((len(times)))                   # Initialize Positions
    v = np.zeros((len(times)))                   # Initialize Velocities
    f = np.zeros((len(times)))                   # Initialize Forces
    x[0] = init_pos                              # Initial position
    v[0] = init_vel                              # Initial velocity
    f[0] = get_force(x[0], mass, omega)          # Compute inital force

    for i in range(1,total_steps):
        # Update position
        x[i] = x[i - 1] + v[i - 1] * deltat + ((f[i - 1])/(2 * mass) * deltat**2)
        # Compute velocity at half step
        v_half = upd_velocity(v[i - 1], f[i - 1], deltat, mass)
        # Update force
        f[i] = get_force(x[i], mass, omega)
        # Update velocity
        v[i] = upd_velocity(v_half, f[i], deltat, mass)
    return x

#============================================================================
# This block is for setting up paramters
tmin = 0                # Starting time
dt = 0.01               # Time step
n_steps = 5000          # Number of time steps
m = 1                   # Set mass
w = np.sqrt(8)          # Angular Frequency for spring
kbt = 10

xs = [float(i.split()[0]) for i in open('sampled_pos.txt').readlines()]
vs = [float(i.split()[1]) for i in open('sampled_pos.txt').readlines()]
trajectories = []

for n in range(len(xs)):
    trajectories.append(get_trajectory(n_steps, dt, tmin, xs[n], vs[n], m, w))

#============================================================================
# This block is for computing position autocorrelation
K = len(xs)
M = len(trajectories[0])
auto_times = [tmin + i * dt for i in range(M)]
x_auto = []
for n in range(M):
    sum = 0
    for l in range(K):
        sum += trajectories[l][0] * trajectories[l][n]
    sum = sum / K
    x_auto.append(sum)

#============================================================================
# This block is for computing the ideal correlation function
constant = kbt / (m*(w**2))
ideal = [constant * np.cos(w * j) for j in auto_times]
#============================================================================

# This block is for plotting the correlation function
plt.plot(auto_times[:800], x_auto[:800], '-', color = "blue")
plt.xlabel("t",fontsize=14)
plt.ylabel(r'$C_{xx}(t)$',fontsize=14)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.tight_layout()
plt.savefig("1d_position_auto.pdf")
plt.clf()

# This block is for plotting the ideal correlation function
plt.plot(auto_times[:800], ideal[:800], '-', color = "red")
plt.xlabel("t",fontsize=14)
plt.ylabel(r'$C_{xx}(t)$',fontsize=14)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.tight_layout()
plt.savefig("ideal.pdf")
plt.clf()
#============================================================================
