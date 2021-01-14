import numpy as np
import matplotlib.pyplot as plt
import copy

def bubble_sort(eig_energies, eig_vectors):
    new_list = copy.copy(eig_energies)
    new_mat = copy.deepcopy(eig_vectors)
    num_pairs = len(new_list) - 1
    for j in range(num_pairs):
        for i in range(num_pairs - j):
            if new_list[i] > new_list[i+1]:
                new_list[i], new_list[i+1] = new_list[i+1], new_list[i]
                new_mat[:,[i, i+1]] = new_mat[:,[i+1,i]]
    return new_list, new_mat

size = 13
mass = 1
w = 1
kbt = 1/8
#============================================================================
# Creating momentum matrix
momentum_mat = np.zeros((size,size),dtype=np.complex_)
for n in range(1,size):
    momentum_mat[n-1][n] = 0 - np.sqrt(n) * 1j
    momentum_mat[n][n-1] = 0 + np.sqrt(n) * 1j
momentum_mat = momentum_mat * np.sqrt(mass*w*0.5)
#============================================================================
# Creating position matrix
position_matrix = np.zeros((size,size))
for n in range(1,size):
    position_matrix[n-1][n] = 0 + np.sqrt(n)
    position_matrix[n][n-1] = 0 + np.sqrt(n)
position_matrix = position_matrix * np.sqrt(1/(2*mass*w))
#============================================================================
# Creating kinetic energy matrix
ke_matrix = np.zeros((size,size),dtype=np.complex_)
for param1 in range(size):
    for param2 in range(size):
        tmp = 0
        for param3 in range(size):
            tmp += momentum_mat[param1][param3] * momentum_mat[param3][param2]
        ke_matrix[param1][param2] = tmp
ke_matrix = np.real(ke_matrix)
#============================================================================
# Creating harmonic potential matrix
pe_matrix_harmonic = np.zeros((size,size))
for param1 in range(size):
    for param2 in range(size):
        tmp = 0
        for param3 in range(size):
            tmp += position_matrix[param1][param3] * position_matrix[param3][param2]
        pe_matrix_harmonic[param1][param2] = tmp
# #============================================================================
# Creating cubed potential matrix
pe_mat_ex_cubed = np.zeros((size,size))
for param1 in range(size):
    for param2 in range(size):
        tmp = 0
        for param3 in range(size):
            tmp += pe_matrix_harmonic[param1][param3] * position_matrix[param3][param2]
        pe_mat_ex_cubed[param1][param2] = tmp
# #============================================================================
# Creating quartic potential matrix
pe_matrix_quartic = np.zeros((size,size))
for param1 in range(size):
    for param2 in range(size):
        tmp = 0
        for param3 in range(size):
            tmp += pe_matrix_harmonic[param1][param3] * pe_matrix_harmonic[param3][param2]
        pe_matrix_quartic[param1][param2] = tmp
#============================================================================
# Multiply matrices by appropriate constant
ke_matrix = ke_matrix/(2*mass)
pe_matrix_harmonic = 0.5*mass*(w**2)*pe_matrix_harmonic
pe_mat_ex_cubed = 0.1*pe_mat_ex_cubed
pe_matrix_quartic = 0.25*pe_matrix_quartic
#============================================================================
# Creating Hamiltonian matrix (uncomment line below depending on potential being used)
# Line below is for harmonic oscillator
# hamiltonian = ke_matrix + pe_matrix_harmonic
# Line below is for mildly anharmonic oscillator
# hamiltonian = ke_matrix + pe_matrix_harmonic + pe_mat_ex_cubed + pe_matrix_quartic
# Line below is for quartic oscillator
hamiltonian = ke_matrix + pe_matrix_quartic
# Compute eigenvalues and eigenvectors
eigenvals, eigvecs = np.linalg.eig(hamiltonian)
# Sort eigenvalues and eigenvectors in order from smallest to largest eigenvalue and corresponding eigenvector
sort_eigval, sort_eigvec = bubble_sort(eigenvals, eigvecs)
#============================================================================
# Compute times for kubo transformed position autocorrelation 
correlation_times = [i*0.0005 for i in range(40000)]
print(correlation_times)

tmp_pos_in_eigen_basis = np.zeros((size,size))
eigvecs_transpose = np.transpose(sort_eigvec)
for param1 in range(size):
    for param2 in range(size):
        tmp = 0
        for param3 in range(size):
            tmp += eigvecs_transpose[param1][param3] * position_matrix[param3][param2]
        tmp_pos_in_eigen_basis[param1][param2] = tmp
pos_in_eigen_basis = np.zeros((size,size))
for param1 in range(size):
    for param2 in range(size):
        tmp = 0
        for param3 in range(size):
            tmp += tmp_pos_in_eigen_basis[param1][param3] * sort_eigvec[param3][param2]
        pos_in_eigen_basis[param1][param2] = tmp

# Compute partition function
correlation_func = []
partition = 0
for n in range(len(sort_eigval)):
    partition += np.exp(-sort_eigval[n]/kbt)

# Compute kubo transformed position autocorrelation
for tau in correlation_times:
    sum1 = 0
    for n in range(size):
        tmp = 0
        for m in range(size):
            if n == m:
                n_m_element = pos_in_eigen_basis[n][m] * pos_in_eigen_basis[m][n]
                n_m_element = np.exp(1j * (sort_eigval[m] - sort_eigval[n]) * tau) * n_m_element
                n_m_element = n_m_element * np.exp(- sort_eigval[n] / kbt)
                n_m_element = n_m_element / kbt
            else:
                n_m_element = pos_in_eigen_basis[n][m] * pos_in_eigen_basis[m][n]
                n_m_element = np.exp(1j * (sort_eigval[m] - sort_eigval[n]) * tau) * n_m_element
                n_m_element = n_m_element * np.exp(- sort_eigval[n] / kbt)
                n_m_element = n_m_element / (sort_eigval[m] - sort_eigval[n])
                n_m_element = n_m_element * (1 - np.exp(- (sort_eigval[m] - sort_eigval[n]) / kbt ) )
            tmp += n_m_element
        sum1 += tmp
    sum1 = sum1 * (kbt/partition)
    correlation_func.append(sum1)

# Writing values to file
filename = open("corr_check.txt", "a+")
for l in range(len(correlation_times)):
    filename.write(str(correlation_times[l]))
    filename.write('              ')
    filename.write(str(np.real(correlation_func[l])) + "\n")
filename.close()
