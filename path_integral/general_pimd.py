import numpy as np
import matplotlib.pyplot as plt
import copy

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
    sqt2 = np.sqrt(boltz) * np.sqrt(mass)
    random = gaussian * sqt1 * sqt2
    new_velocity = old_v * np.exp(- friction * deltat) + random
    return new_velocity

# Compute force from the harmonic potential U = 1/2mw^2x^2
def get_harmonic_force(xi, mass, frequency):
    return - mass * (frequency**2) * xi

# Transform primitive coordinates to staged coordinates
def stage_coords(xi, num_beads, num_seg_beads, num_segments):
    up_u = copy.deepcopy(xi)
    # For the segment beads
    for s in range(num_segments):
        for k in range(1,num_seg_beads):
            up_u[(s*num_seg_beads + k)] = xi[(s*num_seg_beads + k)] - ( (k)*xi[(s*num_seg_beads + k + 1) % (num_beads)] + xi[(s*num_seg_beads)] ) / (k+1)
    return up_u

# Transform staged coordinates to primitive coordinates
def inverse_stage_coords(ui, num_beads, num_seg_beads, num_segments):
    up_x = copy.deepcopy(ui)
    # For the in between beads (loop goes backwards from bead P to bead 2)
    for s in range(num_segments):
        for k in range(num_seg_beads-1, 0, -1):
            up_x[(s*num_seg_beads + k)] = ui[(s*num_seg_beads + k)] + (k/(k+1))*up_x[(s*num_seg_beads + k + 1) % (num_beads)] + (1/(k+1))*ui[(s*num_seg_beads)]
    return up_x

# Compute external potential in staged coordinates using external potential in primitive coordinates
def inverse_force_transformation(force_x, num_beads, num_seg_beads, num_segments):
    new_ext_u = np.zeros(len(force_x))
    # Force acting on the segment beads
    for s in range(num_segments):
        for k in range(1,j):
            # Did change sum labels and constants, not subscripts for potentials
            new_ext_u[s*num_seg_beads + k] = force_x[s*num_seg_beads + k] + ((k-1)/(k)) * new_ext_u[s*num_seg_beads + k - 1]
    # Force acting on the end point beads
    for s in range(num_segments-1,-1,-1):
        sum = 0
        # Did not change sum labels nor constants, instead changed subscripts for potentials
        for l in range(1,num_seg_beads+1):
            sum += force_x[s*num_seg_beads + l - 1]
        new_ext_u[s*num_seg_beads] = sum - ((j-1)/j) * ( new_ext_u[((s+1)*num_seg_beads -1) % num_beads] - new_ext_u[s*num_seg_beads - 1] )
    return new_ext_u

#=============================================================================
# This block is for MD prep. Setting up parameters and initial values.
#===================================================
# This tiny block is for new parameters
j = 2                                # Number of beads in chain segment
capital_n = 5
#===================================================
n_steps = 11                     # Number of time steps
pbeads = 10                          # Number of beads
kbt = 0.00189873                      # Temperature (KbT)
w = 0.03                              # Set Frequency
m = 1                                 # Set Mass
tmin = 0                              # Starting time
dt = 0.01                             # Delta t
times = np.array([tmin + i * dt for i in range(n_steps)])
gamma = np.sqrt(pbeads)*kbt           # Friction

# Initialize bead positions
primitives_x = np.zeros(pbeads)
# Sample initial positions
for rando in range(pbeads):
    primitives_x[rando] = np.random.normal(0,np.sqrt(kbt/(m*w**2)) )

# # Initialize bead velocities
v = np.zeros(pbeads)
# Sample initial velocities
for rando in range(pbeads):
    v[rando] = np.random.normal(0, np.sqrt(kbt/m))

# Initialize masses for the segment beads (There are j - 1 beads in each segment)
m_k = np.zeros(j-1)
# Define the segment bead masses
for i in range(2, j+1):
    m_k[i-2] = (m * i) / (i - 1)

u = np.zeros(pbeads)
# Compute initial stage coords
u[:] = stage_coords(primitives_x[:], pbeads, j, capital_n)
# Initialize harmonic forces
f_u = np.zeros(pbeads)
# Compute initial harmonic forces for end point beads
for s in range(capital_n):
    f_u[s*j] = get_harmonic_force(2*u[s*j] - u[(s+1)*j % pbeads] - u[(s-1)*j], m, kbt*np.sqrt(pbeads/j))
# Compute initial harmonic forces for staging beads
for s in range(capital_n):
    for k in range(1,j):
        f_u[s*j + k] = get_harmonic_force(u[s*j + k], m_k[k-1], np.sqrt(pbeads)*kbt)
# Inititialize external forces in x
f_x = np.zeros(pbeads)
# Compute initial external forces in x
for p in range(pbeads):
    f_x[p] = get_harmonic_force(primitives_x[p], m, w)
# Compute initial forces in staged coords (eqn 12.6.12)
f_u[:] = f_u[:] + (1/pbeads)*inverse_force_transformation(f_x[:], pbeads, j, capital_n)

virial = []
#=============================================================================
# MD code starts here !!
for i in range(1,n_steps):
    # Update end point bead velocities
    for s in range(capital_n):
        v[s*j] = upd_velocity(v[s*j], f_u[s*j], dt, m)
    # Update segment bead velocities
    for s in range(capital_n):
        for k in range(1,j):
            v[s*j + k] = upd_velocity(v[s*j + k], f_u[s*j + k], dt, m_k[k-1])
    # Update end point bead positions
    for s in range(capital_n):
        u[s*j] = upd_position(u[s*j], v[s*j], dt)
    # Update segment bead positions
    for s in range(capital_n):
        for k in range(1,j):
            u[s*j + k] = upd_position(u[s*j + k], v[s*j + k], dt)
    # Add random velocity to end point beads
    for s in range(capital_n):
        v[s*j] = rand_kick(v[s*j], gamma, kbt, m, dt)
    # Add random velocity to segment beads
    for s in range(capital_n):
        for k in range(1,j):
            v[s*j + k] = rand_kick(v[s*j + k], gamma, kbt, m_k[k-1], dt)
    # Update position of end point beads with random velocity
    for s in range(capital_n):
        u[s*j] = upd_position(u[s*j], v[s*j], dt)
    # Update position of end point beads with random velocity
    for s in range(capital_n):
        for k in range(1,j):
            u[s*j + k] = upd_position(u[s*j + k], v[s*j + k], dt)
    # Update harmonic force with random position for end point beads
    for s in range(capital_n):
        f_u[s*j] = get_harmonic_force(2*u[s*j] - u[(s+1)*j % pbeads] - u[(s-1)*j], m, kbt*np.sqrt(pbeads/j))
    # Update harmonic force with random position for segment beads
    for s in range(capital_n):
        for k in range(1,j):
            f_u[s*j + k] = get_harmonic_force(u[s*j + k], m_k[k-1], np.sqrt(pbeads)*kbt)
    # Update positions in x
    primitives_x[:] = inverse_stage_coords(u[:], pbeads, j, capital_n)
    # Compute external force with updated primitives
    for p in range(pbeads):
        f_x[p] = get_harmonic_force(primitives_x[p], m, w)
    # Update forces in stage coordinates
    f_u[:] = f_u[:] + (1/pbeads)*inverse_force_transformation(f_x[:], pbeads, j, capital_n)
    # Update end point bead velocities with new force
    for s in range(capital_n):
        v[s*j] = upd_velocity(v[s*j], f_u[s*j], dt, m)
    # Update segment bead velocities with new force
    for s in range(capital_n):
        for k in range(1,j):
            v[s*j + k] = upd_velocity(v[s*j + k], f_u[s*j + k], dt, m_k[k-1])
    # Computing the virial energy estimator
    sumv = 0
    for o in range(pbeads):
        sumv += (m*w**2)*primitives_x[o]**2
    # Scale the estimator by the inverse of the number of beads
    virial.append(sumv / pbeads)

# MD code ends here
#=============================================================================
# # This block is for analysis.

# Computing cumulative average of the virial estimator
cume = np.zeros(len(virial))
cume[0] = virial[0]
for i in range(1,len(virial)):
    cume[i] = (i)/(i+1)*cume[i-1] + virial[i]/(i+1)
steps = np.arange(1,n_steps)

# Writing values to file
filename = open("energies.txt", "a+")
for l in range(len(virial)):
    filename.write(str(virial[l]))
    filename.write('              ')
    filename.write(str(cume[l]))
    filename.write('              ')
    filename.write(str(steps[l]) + "\n")
filename.close()

# Plotting the virial estimator
plt.xlim(min(steps)-100, max(steps))
plt.ylim(-0.005,0.07)
plt.axhline(y=0.015, linewidth=2, color='r', label=r'$\epsilon_{vir} = 0.015$')
plt.plot(steps, virial, '-', color='black', label=r'$\epsilon_{vir}$', alpha=0.4)
plt.plot(steps, cume, '-', label='cumulative average', color='blue')
plt.legend(loc='upper left')
plt.xlabel('# of Steps')
plt.ylabel(r'$\epsilon_{vir} \quad / \quad \hbar \omega$')
plt.savefig('virial.pdf')
plt.clf()
# #=============================================================================
