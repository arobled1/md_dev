import numpy as np
import matplotlib.pyplot as plt

# Update velocity at a half step dt/2
def upd_velocity(veloc, force, deltat, mass):
    return veloc + (force * deltat)/(2 * mass)

# Update position at a half step dt/2
def upd_position(posit, veloc, deltat):
    return posit + 0.5 * veloc * deltat

# Use langevin thermostat to add randomness to your velocities
def rand_kick(old_v, friction, boltz, mass, deltat):
    gaussian = np.random.normal(0,1)
    sqt1 = np.sqrt(1 - np.exp(-2*friction*deltat) )
    sqt2 = np.sqrt(boltz) / np.sqrt(mass)
    random = gaussian * sqt1 * sqt2
    new_velocity = old_v * np.exp(- friction * deltat) + random
    return new_velocity

# Compute force from the harmonic potential U = 1/2mw^2x^2
def get_harmonic_force(xi, mass, frequency):
    return - mass * (frequency**2) * xi

# Compute force from the harmonic potential U = 1/2mw^2x^2
def get_harmonic_potential(xi, mass, frequency):
    return 0.5 * mass * (frequency**2) * xi**2

# Transform primitive coordinates to staged coordinates
def stage_coords(xi, num_beads):
    up_u = np.zeros(num_beads+1)
    # For the first bead
    up_u[0] = 0.5 * (xi[0] + xi[num_beads])
    # For the in between beads
    for s in range(1,num_beads):
        up_u[s] = xi[s] - (1/(s+1)) * (s * xi[s+1] + xi[0])
    # For the last bead
    up_u[num_beads] = xi[0] - xi[num_beads]
    return up_u

# Transform staged coordinates to primitive coordinates
def inverse_stage_coords(ui, num_beads):
    up_x = np.zeros(num_beads+1)
    # For the first bead
    up_x[0] = ui[0] + 0.5 * ui[num_beads]
    # For the in between beads
    for s in range(1,num_beads):
        sum1 = 0
        for t in range(s,num_beads):
            sum1 += ((s)/(t)) * ui[t]
        up_x[s] = ui[0] + (( (num_beads/2) - s) / num_beads) * ui[num_beads] + sum1
    # For the last bead
    up_x[num_beads] = ui[0] - 0.5 * ui[num_beads]
    return up_x

# Compute external potential in staged coordinates using external potential in primitive coordinates
def inverse_force_transformation(force_x, num_beads):
    new_ext_u = np.zeros(num_beads+1)
    # Force acting on the u1 bead
    sum1 = 0
    for t in range(num_beads+1):
        sum1 += force_x[t]
    new_ext_u[0] = sum1
    # Force acting on middle beads
    for s in range(1,num_beads):
        new_ext_u[s] = force_x[s] + ((s-1)/s) * new_ext_u[s-1]
    # Force on last bead
    sum2 = 0
    for t in range(1,num_beads):
        sum2 += ((0.5 * num_beads - t) / num_beads) * force_x[t]
    new_ext_u[num_beads] = 0.5 * (force_x[0] - force_x[num_beads]) + sum2
    return new_ext_u

def bin_centers(bin_edges):
    return (bin_edges[1:]+bin_edges[:-1])/2.

def get_harmonic_density(pos, inv_temp, mass, omega):
    normalization = np.sqrt((mass*omega) / (4*np.pi*np.tanh(inv_temp*omega/2)))
    exp_constant = np.pi * normalization**2
    return normalization * np.exp(-exp_constant * pos**2 )

#=============================================================================
# This block is for MD prep. Setting up parameters and initial values.
n_steps = 4000000                     # Number of time steps
pbeads = 32                           # Number of beads
kbt = 1/10                            # Temperature (KbT)
w = 1                                 # Set Frequency
m = 1                                 # Set Mass
tmin = 0                              # Starting time
dt = 0.01                             # Delta t
gamma = np.sqrt(pbeads)*kbt           # Friction

# Initialize bead positions
primitives_x = np.zeros(pbeads+1)
# Sample initial positions
for rando in range(pbeads+1):
    primitives_x[rando] = np.random.normal(0,np.sqrt(kbt/(m*w**2)) )

# Initialize bead velocities
v = np.zeros(pbeads+1)
# Sample initial velocities
for rando in range(pbeads+1):
    v[rando] = np.random.normal(0, np.sqrt(kbt/m))

# Masses for the harmonic coupling
m_k = np.zeros(pbeads+1)
# Masses for the velocities
m_prime_k = np.zeros(pbeads+1)
# Mass for the first bead
m_k[0] = 0
m_prime_k[0] = m
# Mass for the middle beads
for i in range(1, pbeads):
    m_k[i] = (i+1) * m / i
    m_prime_k[i] = (i+1) * m / i
# Mass for the last bead
m_k[pbeads] = m / pbeads
m_prime_k[pbeads] = m / pbeads

u = np.zeros(pbeads+1)
# Compute initial stage coords
u[:] = stage_coords(primitives_x[:], pbeads)
# Initialize harmonic forces
f_u = np.zeros(pbeads+1)
# Compute initial harmonic forces for beads
for p in range(1,pbeads+1):
    f_u[p] = get_harmonic_force(u[p], m_k[p], np.sqrt(pbeads)*kbt)
# Inititialize external forces in x
f_x = np.zeros(pbeads+1)
# Compute initial external forces in x
f_x[0] = get_harmonic_force(primitives_x[0], m, w) / (2*pbeads)
for p in range(1,pbeads):
    f_x[p] = get_harmonic_force(primitives_x[p], m, w) / pbeads
f_x[pbeads] = get_harmonic_force(primitives_x[pbeads], m, w) / (2*pbeads)
# Compute initial forces in staged coords
f_u[:] += inverse_force_transformation(f_x[:], pbeads)
#=============================================================================
# MD code starts here !!
lastu_bead = []
for i in range(1,n_steps+1):
    # Update velocity
    for p in range(pbeads+1):
        v[p] = upd_velocity(v[p], f_u[p], dt, m_prime_k[p])
    # Update position
    for p in range(pbeads+1):
        u[p] = upd_position(u[p], v[p], dt)
    # Add random velocity
    for p in range(pbeads+1):
        v[p] = rand_kick(v[p], gamma, kbt, m_prime_k[p], dt)
    # Update position with random velocity
    for p in range(pbeads+1):
        u[p] = upd_position(u[p], v[p], dt)
    # Reset array
    f_u = np.zeros(pbeads+1)
    # Update harmonic force with random position
    for p in range(1,pbeads+1):
        f_u[p] = get_harmonic_force(u[p], m_k[p], np.sqrt(pbeads)*kbt)
    # Update positions in x
    primitives_x[:] = inverse_stage_coords(u[:], pbeads)
    lastu_bead.append(u[pbeads])
    # Compute external force with updated primitives
    f_x[0] = get_harmonic_force(primitives_x[0], m, w) / (2*pbeads)
    for p in range(1,pbeads):
        f_x[p] = get_harmonic_force(primitives_x[p], m, w) / pbeads
    f_x[pbeads] = get_harmonic_force(primitives_x[pbeads], m, w) / (2*pbeads)
    # Compute initial forces in staged coords
    f_u[:] += inverse_force_transformation(f_x[:], pbeads)
    # Update velocity based on random force
    for p in range(pbeads+1):
        v[p] = upd_velocity(v[p], f_u[p], dt, m_prime_k[p])
# MD code ends here
#=============================================================================
# Writing values to file
steps = np.arange(1,n_steps+1)
filename = open("positions.txt", "a+")
for l in range(n_steps):
    filename.write(str(lastu_bead[l]))
    filename.write('              ')
    filename.write(str(steps[l]) + "\n")
filename.close()

# Make a histogram of the open path end-to-end distance
dist_hist, dist_bin_edges = np.histogram(lastu_bead,bins=50,density=True)
ideal_x = np.arange(-6,6,.1)
ideal_prediction_x = get_harmonic_density(ideal_x, 1/kbt, m, w)
plt.plot(ideal_x, ideal_prediction_x,linestyle='-',label='Exact', color='blue')
p = plt.plot(bin_centers(dist_bin_edges), dist_hist, linestyle='--',label=r'$N(u_{P+1})$', color='red')
plt.legend(loc='upper right')
plt.xlim(-6, 6)
plt.ylim(0,0.35)
plt.xlabel(r'$u_{P+1}$')
plt.ylabel(r'$N(u_{P+1})$')
plt.title('Estimator for harmonic oscillator density matrix')
plt.savefig('h_density.pdf')
plt.clf()
