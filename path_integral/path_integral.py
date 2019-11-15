import numpy as np
import matplotlib.pyplot as plt

def upd_velocity(veloc, force, deltat, mass):
    return veloc + (force * deltat)/(2 * mass)

def upd_position(posit, veloc, deltat):
    return posit + 0.5 * veloc * deltat

def rand_kick(old_v, friction, boltz, mass, deltat):
    gaussian = np.random.normal(0,1)
    sqt1 = np.sqrt(1 - np.exp(-2*friction*deltat) )
    sqt2 = np.sqrt(boltz) * np.sqrt(mass)
    random = gaussian * sqt1 * sqt2
    new_velocity = old_v * np.exp(- friction * deltat) + random
    return new_velocity

def get_harmonic_force(xi, mass, frequency):
    return - mass * frequency**2 * xi

def stage_coords(xi, num_beads):
    up_u = np.zeros(num_beads)
    # For the first bead
    up_u[0] = xi[0]
    # For the in between beads
    for k in range(1,num_beads-1):
        up_u[k] = xi[k] - ( k*xi[k+1] + xi[0] ) / (k + 1)
    # For the last bead
    up_u[num_beads-1] = xi[num_beads-1] - ( ( (num_beads-2)*xi[0] ) + xi[0] )/(num_beads - 1)
    return up_u

def inverse_stage_coords(ui, xi, num_beads):
    up_x = np.zeros(num_beads)
    # For the first bead
    up_x[0] = ui[0]
    # For the P bead
    up_x[num_beads-1] = ui[num_beads-1] + ((num_beads-1)/(num_beads))*up_x[0] + (1/(num_beads))*ui[0]
    # For the in between beads (loop goes backwards from bead P to bead 2)
    for k in range(num_beads - 2, 0,-1):
        up_x[k] = ui[k] + (k/(k+1))*up_x[k+1] + (1/(k+1))*ui[0]
    return up_x

def inverse_force_transformation(force_x):
    new_ext_u = np.zeros(len(force_x))
    sum = 0
    # Force acting on the u1 bead
    for l in range(len(force_x)):
        sum += force_x[l]
    new_ext_u[0] = sum
    # Force acting on every other bead
    for k in range(1, len(force_x)):
        new_ext_u[k] = force_x[k] + ((k-1)/k)*(new_ext_u[k-1])
    return new_ext_u

n_steps = 20                           # Number of time steps
tmin = 0                              # Starting time
dt = 0.01                             # Delta t
times = np.array([tmin + i * dt for i in range(n_steps)])
gamma = 0                           # Friction
kbt = 1                               # Temperature (KbT)
w = 2                                 # Set Frequency
m = 2                                 # Set Mass
pbeads = 3                            # Number of beads
x = np.zeros((pbeads, len(times)))    # Initialize bead positions
x[:,0] = 1                            # Iinitial positions
v = np.zeros((pbeads, len(times)))    # Initialize bead velocities
v[:,0] = 0.5                          # Initial velocities

# Masses for the harmonic coupling
new_m_force = np.zeros(pbeads)
# Masses for the velocities
new_m_vel = np.zeros(pbeads)
    # Mass for the first bead
new_m_force[0] = 0
new_m_vel[0] = m
    # Mass for the rest of the beads
for k in range(1, pbeads):
    new_m_force[k] = (k+1) * m / k
    new_m_vel[k] = (k+1) * m / k

# Initialize staged coords
u = np.zeros((pbeads, len(times)))
# Compute initial stage coords
u[:,0] = stage_coords(x[:,0], pbeads)
# Initialize harmonic forces
f_u = np.zeros((pbeads, len(times)))
# Compute initial harmonic forces
for p in range(pbeads):
    f_u[p,0] = get_harmonic_force(u[p, 0], new_m_force[p], np.sqrt(pbeads)*kbt)
# Inititialize external forces in x
f_x = np.zeros((pbeads, len(times)))
# Compute initial external forces in x
for p in range(pbeads):
    f_x[p,0] = get_harmonic_force(x[p,0], m, w)
# Compute initial forces in staged coords (eqn 12.6.12)
f_u[:,0] = f_u[:,0] + (1/pbeads)*inverse_force_transformation(f_x[:,0])

# MD code starts here!!!!!
for i in range(1,n_steps):
    # Update velocity
    for p in range(pbeads):
        v[p,i] = upd_velocity(v[p, i-1], f_u[p, i-1], dt, new_m_vel[p])
    # Update position
    for p in range(pbeads):
        u[p,i] = upd_position(u[p, i-1], v[p, i], dt)
    # Add random velocity
    for p in range(pbeads):
        v[p,i] = rand_kick(v[p, i], gamma, kbt, new_m_vel[p], dt)
    # Update position with random velocity
    for p in range(pbeads):
        u[p,i] = upd_position(u[p, i], v[p, i], dt)
    # Update harmonic force with random position
    for p in range(pbeads):
        f_u[p,i] = get_harmonic_force(u[p, i], new_m_force[p], np.sqrt(pbeads)*kbt)
    # Update positions in x
    x[:,i] = inverse_stage_coords(u[:,i], x[:,i], pbeads)
    # Compute external force in updated x's
    for p in range(pbeads):
        f_x[p,i] = get_harmonic_force(x[p,i], m, w)
    # Update forces in stage coordinates
    f_u[:,i] = f_u[:,i] + (1/pbeads)*inverse_force_transformation(f_x[:,i])
    # Update velocity based on random force
    for p in range(pbeads):
        v[p,i] = upd_velocity(v[p, i], f_u[p, i], dt, new_m_vel[p])

# Compute the virial energy estimator
virial = []
for q in range(n_steps):
    sumv = 0
    for o in range(pbeads):
        sumv += 0.5*(m*w**2)*x[o,q]*x[o,q] + 0.5*x[o,q]*(-f_x[o,q])
    virial.append(sumv)
virial = np.asarray(virial)
virial = (1/pbeads)*virial

# Compute cumulative average of the virial estimator
cume = np.zeros(len(virial))
cume[0] = virial[0]
for i in range(1,len(virial)):
    cume[i] = (i)/(i+1)*cume[i-1] + virial[i]/(i+1)
steps = np.zeros(n_steps)
for g in range(n_steps):
    steps[g] = g

filename = open("energies.txt", "a+")
for l in range(len(virial)):
    filename.write(str(virial[l]))
    filename.write('              ')
    filename.write(str(cume[l]))
    filename.write('              ')
    filename.write(str(steps[l]) + "\n")
filename.close()

# plt.xlim(min(steps)-100, max(steps))
# plt.ylim(min(energy[:60000])-2 ,max(energy[:60000])+2)
# plt.plot(steps, energy[:60000], '-', color='black', alpha=0.4)
# plt.plot(steps, cume[:60000], '-', color='blue')
# plt.xlabel('# of Steps')
# plt.ylabel('Energy')
# plt.savefig('sunday_virial.pdf')
# plt.clf()

# # Plot positions of beads
# plt.xlim(0, times[3000])
# plt.ylim(min(x[1,:3000]) - 1,max(x[1,:3000]) + 1)
# plt.plot(times[:3000], x[0,:3000], '-', color='blue', label='Atom 1')
# plt.plot(times[:3000], x[1,:3000], '-', color='green', label='Atom 2')
# plt.plot(times[:3000], x[2,:3000], '-', color='red', label='Atom 3')
# plt.xlabel('time (dimensionless units)')
# plt.ylabel('x (dimensionless units)')
# plt.legend()
# plt.savefig('posit.pdf')
# plt.clf()

# # Check conservation of energy with gamma = 0 and plot to make sure it is
# # a straight line
# energy = []
# for j in range(n_steps):
#     sum = 0
#     for p in range(pbeads):
#         sum = sum + 0.5*new_m[p]*v[p,j]*v[p,j] + (0.5*new_m[p]*w**2)*u[p,j]*u[p,j]
#     energy.append(sum)
# plt.xlim(0, times[500])
# plt.ylim(min(energy[:500]) - .01,max(energy[:5000]) + .01)
# plt.plot(times[:500], energy[:500], '.', color='blue')
# plt.xlabel('time (dimensionless units)')
# plt.ylabel('energy (dimensionless units)')
# plt.title("$\gamma = 0$")
# plt.savefig('gam0energy.pdf')
# plt.clf()

# # Histograms for checking x and p
# _ = plt.hist(u[0,:], bins=100, normed=True)
# plt.xlabel('X')
# plt.ylabel('P(X)')
# plt.savefig('xhist0.pdf')
# plt.clf()
# _ = plt.hist(u[pbeads-2,:], bins=100, normed=True)
# plt.xlabel('X')
# plt.ylabel('P(X)')
# plt.savefig('xhistpm1.pdf')
# plt.clf()
# _ = plt.hist(u[pbeads -1,:], bins=100, normed=True)
# plt.xlabel('X')
# plt.ylabel('P(X)')
# plt.savefig('xhistp.pdf')
# plt.clf()