module rpmd_functions
implicit none
!==============================================================================!
! Global Variables
integer :: pbeads
integer :: num_samples
integer :: n_steps
double precision :: dt
double precision :: beta_p
double precision :: m
!==============================================================================!
contains
! Update position at a half step dt/2
subroutine upd_position(posit, veloc, tmp_accel)
    integer :: p
    double precision :: posit(pbeads), veloc(pbeads), tmp_accel(pbeads)
    do p = 1, pbeads
        posit(p) = posit(p) + veloc(p) * dt + 0.5 * tmp_accel(p) * dt**2
    enddo
end subroutine upd_position

! Update velocity at a half step dt/2
subroutine upd_velocity(veloc, tmp_accel)
    integer :: p
    double precision :: veloc(pbeads), tmp_accel(pbeads)
    do p = 1, pbeads
        veloc(p) = veloc(p) + 0.5 * tmp_accel(p) * dt
    enddo
end subroutine upd_velocity

! Compute force from the harmonic potential U = 1/2mw^2x^2
subroutine get_harmonic_force(pos, mass, frequency, force)
    double precision :: pos, mass, frequency, force
    force = - mass * (frequency**2) * pos
end subroutine get_harmonic_force

! Compute force from the harmonic potential U = 1/2mw^2x^2
subroutine get_spring_force(ui, mass_k, frequency, harm_force)
    integer :: p
    double precision :: ui(pbeads), mass_k(pbeads), harm_force(pbeads)
    double precision :: frequency
    ! Compute harmonic forces for staging beads
    do p = 1, pbeads
        call get_harmonic_force(ui(p), mass_k(p), frequency, harm_force(p))
    enddo
end subroutine get_spring_force

subroutine get_quarticforce(xi, ext_force)
    integer :: p
    double precision :: xi(pbeads), ext_force(pbeads)
    do p = 1, pbeads
        ext_force(p) = - xi(p)**3
    enddo
end subroutine get_quarticforce

! Transform primitive coordinates (x) to staged coordinates (u')
subroutine x_to_u_transform(xi, up_u)
    integer :: k
    double precision :: xi(pbeads), up_u(pbeads)
    double precision, allocatable :: tmp_u(:)
    allocate(tmp_u(pbeads))
    tmp_u = 0.0d0
    ! For the first bead
    tmp_u(1) = xi(1)
    ! For the in between beads
    do k = 2, pbeads - 1
        tmp_u(k) = xi(k) - ((k - 1.0d0)*xi(k+1) + xi(1) ) / k
    enddo
    ! For the last bead
    tmp_u(pbeads) = xi(pbeads) - ( (pbeads - 1.0d0)*xi(1) + xi(1)) / pbeads
    up_u = tmp_u
end subroutine x_to_u_transform

! Transform staged coordinates (u') to primitive coordinates (x)
subroutine u_to_x_transform(ui, up_x)
    integer :: k
    double precision :: ui(pbeads), up_x(pbeads)
    double precision, allocatable :: tmp_x(:)
    allocate(tmp_x(pbeads))
    tmp_x = 0.0d0
    ! For the first bead
    tmp_x(1) = ui(1)
    ! For the P bead
    tmp_x(pbeads) = ui(pbeads) + ((pbeads-1.0d0)/pbeads)*tmp_x(1) + (1.0d0/pbeads)*ui(1)
    ! For the in between beads (loop goes backwards from bead P to bead 2)
    do k = pbeads-1, 2, -1
        tmp_x(k) = ui(k) + ((k-1.0d0)/k)*tmp_x(k+1) + (1.0d0/k)*ui(1)
    enddo
    up_x = tmp_x
end subroutine u_to_x_transform

! Create the transformation matrix for x --> u
subroutine get_transformation_matrices(tmp_tmat, inverse_tmp_tmat)
    !==========================================================================!
    ! Variables and arrays for lapack
    integer :: cap_n, lda, lwork, info
    integer, allocatable :: ipiv(:)
    double precision, allocatable :: work(:)
    !==========================================================================!
    integer :: index1, index2, index3, ess, kay
    double precision :: tmp_tmat(pbeads,pbeads), inverse_tmp_tmat(pbeads,pbeads)
    cap_n = pbeads
    lda = pbeads
    lwork = pbeads
    allocate(ipiv(pbeads))
    allocate(work(max(1,lwork)))
    ! Populate the first column of the transformation matrix
    do index1 = 2, pbeads - 1
        tmp_tmat(index1,1) = -1.0d0 / index1
    enddo
    tmp_tmat(pbeads,1) = -1.0d0
    ! Populate the diagonal of the transformation matrix
    do index1 = 1, pbeads
        tmp_tmat(index1, index1) = 1.0d0
    enddo
    ! Populate the upper diagonal of the transformation matrix
    do index1 = 3, pbeads
        tmp_tmat(index1 - 1, index1) = - (index1 - 2.0d0) / (index1 - 1.0d0)
    enddo
    ! Saving original matrix
    inverse_tmp_tmat = tmp_tmat
    ! Computing LU factorization for computing the inverse
    call dgetrf(pbeads, pbeads, inverse_tmp_tmat, pbeads, ipiv, info)
    ! Computing the inverse matrix using the LU factorization
    call dgetri(cap_n, inverse_tmp_tmat, lda, ipiv, work, lwork, info)
end subroutine get_transformation_matrices

subroutine get_mass_tensors(transf_mat, inv_transf_mat, tmp_mass_tens, inverse_tmp_mass_mat)
    integer :: index1, index2, index3
    double precision :: transf_mat(pbeads,pbeads), inverse_tmp_mass_mat(pbeads,pbeads)
    double precision :: inv_transf_mat(pbeads,pbeads), tmp_mass_tens(pbeads,pbeads)
    double precision, allocatable :: t_transf_mat(:,:), t_inv_transf_mat(:,:)
    allocate(t_transf_mat(pbeads,pbeads))
    allocate(t_inv_transf_mat(pbeads,pbeads))
    ! Transpose the inverse transformation matrix for matrix multiplication
    t_transf_mat = transpose(transf_mat)
    inverse_tmp_mass_mat = matmul(transf_mat, t_transf_mat)
    do index1 = 1, pbeads
        do index2 = 1, pbeads
            inverse_tmp_mass_mat(index1, index2) = inverse_tmp_mass_mat(index1, index2) / m
        enddo
    enddo
    t_inv_transf_mat = transpose(inv_transf_mat)
    tmp_mass_tens = matmul(t_inv_transf_mat, inv_transf_mat)
    do index1 = 1, pbeads
        do index2 = 1, pbeads
            tmp_mass_tens(index1, index2) = tmp_mass_tens(index1, index2) * m
        enddo
    enddo
    ! Transposed arrays will not be used anymore during the program.
    deallocate(t_transf_mat)
    deallocate(t_inv_transf_mat)
end subroutine get_mass_tensors

subroutine get_staging_velocities(prims_v, transform_matrix, temp1)
    double precision :: transform_matrix(pbeads,pbeads), prims_v(pbeads,num_samples)
    double precision :: temp1(pbeads,num_samples)
    temp1 = matmul(transform_matrix, prims_v)
end subroutine get_staging_velocities

subroutine inverse_force_transformation(force_x, force_u)
    integer :: ell, k
    double precision :: sum1
    double precision :: force_x(pbeads), force_u(pbeads)
    double precision, allocatable :: new_ext_u(:)
    allocate(new_ext_u(pbeads))
    new_ext_u = 0.0d0
    sum1 = 0.0d0
    ! Force acting on the u1 bead
    do ell = 1, pbeads
        sum1 = sum1 + force_x(ell)
    enddo
    new_ext_u(1) = sum1
    ! Force acting on every other bead
    do k = 2, pbeads
        new_ext_u(k) = force_x(k) + ((k - 2.0d0)/(k - 1.0d0)) * new_ext_u(k-1)
    enddo
    force_u = force_u + new_ext_u
end subroutine inverse_force_transformation

! Sample velocities from boltzmann distribution using box-muller transformation
subroutine sample_velocities(samps)
    integer :: p
    double precision :: u1, u2
    double precision :: samps(pbeads)
    ! Sample initial velocities for each RPMD run
    do p = 1, pbeads
        call random_number(u1)
        call random_number(u2)
        samps(p) = sqrt(1.0 / (beta_p * m)) * sqrt(-2.0d0 * log(u1))*cos(8.0d0*atan(1.0d0)*u2)
    enddo
end subroutine
end module rpmd_functions
program parallel_stage_rpmd
use omp_lib
use rpmd_functions
implicit none
integer :: i, k, p, rpmd_index, traj_var, rando_seed
integer :: threads
double precision :: temp, u1, u2, partial_corre_sum
double precision, allocatable :: staging_u(:,:), primitives_x(:,:), prim_momenta(:,:), f_x(:,:), f_u(:,:)
double precision, allocatable :: primitives_v(:,:), stage_vels(:,:)
double precision, allocatable :: initial_centroid_pos(:), correlation_values(:), correlation_times(:)
double precision, allocatable :: m_k(:), m_s(:), transform_mat(:,:), inverse_transform_mat(:,:)
double precision, allocatable :: mass_tensor(:,:), inv_mass_tensor(:,:), save_centroids(:)
double precision, allocatable :: accel1(:), accel2(:)
! =============================================================================
! Reading in data
open(29, file="input", action="readwrite")
read(29,*) pbeads
read(29,*) num_samples
read(29,*) n_steps
read(29,*) dt
read(29,*) temp
read(29,*) m
close(29)
! Redefining temperature for RPMD
beta_p = temp / pbeads

! Allocate and initialize primitive bead positions
allocate(primitives_x(pbeads,num_samples))
primitives_x = 0.0d0
! Reading in position samples (originally in staging variables so no need to transform)
open(28, file='positions.txt')
read(28, *) ((primitives_x(i,k), i = 1, pbeads), k = 1, num_samples)
close(28)

! =============================================================================
! Allocate and initialize staging bead positions
allocate(staging_u(pbeads,num_samples))
staging_u = 0.0d0
! Allocate and initialize masses for the staging beads
allocate(m_k(pbeads))
m_k = 0.0d0
! Allocate forces
allocate(f_u(pbeads,num_samples))
allocate(f_x(pbeads,num_samples))
f_u = 0.0d0
f_x = 0.0d0
! Allocate and initialize transformation matrices
allocate(transform_mat(pbeads, pbeads))
allocate(inverse_transform_mat(pbeads, pbeads))
transform_mat = 0.0d0
inverse_transform_mat = 0.0d0
! Allocate and initialize mass tensors
allocate(mass_tensor(pbeads,pbeads))
allocate(inv_mass_tensor(pbeads,pbeads))
mass_tensor = 0.0d0
inv_mass_tensor = 0.0d0
! Allocate bead primitive and staging velocities
allocate(primitives_v(pbeads,num_samples))
allocate(stage_vels(pbeads,num_samples))
primitives_v = 0.0d0
stage_vels = 0.0d0

! Getting initial velocities from maxwell-boltzmann distribution
call random_seed(rando_seed)
do i = 1, num_samples
    call sample_velocities(primitives_v(:,i))
enddo

! Generate transformation matrix and its inverse
call get_transformation_matrices(transform_mat, inverse_transform_mat)

! Generate mass matrix and its inverse from the transformation matrix
call get_mass_tensors(transform_mat, inverse_transform_mat, mass_tensor, inv_mass_tensor)

call get_staging_velocities(primitives_v, transform_mat, stage_vels)

! Allocating the centroid positions to be used for computing Kxx
allocate(initial_centroid_pos(num_samples))
! Initialize centroids
initial_centroid_pos = 0.0d0
! Allocation kubo transform values
allocate(correlation_values(n_steps+1))
! Initializing correlation values array.
correlation_values = 0.0d0

allocate(save_centroids(num_samples))
save_centroids = 0.0d0

! Compute centroid positions from initial configurations
do i = 1, num_samples
    initial_centroid_pos(i) = sum(primitives_x(:,i)) / pbeads
enddo

! Computing Kxx(t = 0)
do i = 1, num_samples
    correlation_values(1) = correlation_values(1) + (initial_centroid_pos(i)**2 ) / num_samples
enddo
! =============================================================================
! Start of parallel region
!$omp parallel private(i, k, p, rpmd_index, traj_var) &
!$omp& private(u1, u2, partial_corre_sum, accel1, accel2)

!$omp do
! Define the bead masses
do i = 2, pbeads
    m_k(i) = (i * m)/ (i - 1.0d0)
enddo
!$omp end do

!$omp do
! Convert primitive coords (x) to staging coords (u)
do i = 1, num_samples
    call x_to_u_transform(primitives_x(:,i), staging_u(:,i))
enddo
!$omp end do

!$omp do
! Compute initial forces
do i = 1, num_samples
    ! Compute initial forces from harmonic springs
    call get_spring_force(staging_u(:,i), m_k, (1.0d0/beta_p), f_u(:,i))
    ! Inititialize external forces in x
    call get_quarticforce(primitives_x(:, i), f_x(:, i))
    ! Compute initial forces in staged coords using transformation
    call inverse_force_transformation(f_x(:,i), f_u(:,i))
enddo
!$omp end do

! Now moving onto MD
! Need the outermost loop to run sequentially while the inner loop can be run in
! parallel.
do rpmd_index = 2, n_steps
    partial_corre_sum = 0.0d0
    !$omp do
    do traj_var = 1, num_samples
        ! Update acceleration for position update
        accel1 = matmul(inv_mass_tensor, f_u(:,traj_var))
        ! Update position in u
        call upd_position(staging_u(:, traj_var), stage_vels(:, traj_var), accel1)
        ! Update positions in x
        call u_to_x_transform(staging_u(:,traj_var), primitives_x(:,traj_var))
        ! Update velocity at half step
        call upd_velocity(stage_vels(:, traj_var), accel1)
        ! Update harmonic forces with updated u
        call get_spring_force(staging_u(:,traj_var), m_k, (1.0d0/beta_p), f_u(:,traj_var))
        ! Computing external forces with updated x
        call get_quarticforce(primitives_x(:, traj_var), f_x(:, traj_var))
        ! Compute initial forces in staged coords using transformation
        call inverse_force_transformation(f_x(:,traj_var), f_u(:,traj_var))
        ! Update acceleration for position update
        accel2 = matmul(inv_mass_tensor, f_u(:,traj_var))
        ! Update velocity at step
        call upd_velocity(stage_vels(:, traj_var), accel2)
        save_centroids(traj_var) = sum(primitives_x(:,traj_var)) / pbeads
    enddo
    !$omp end do
    ! Have any one of the threads compute compute the sum below. This is done
    ! to prevent a race condition when computing each value of Kxx.
    !$omp single
    do i = 1, num_samples
        correlation_values(rpmd_index) = correlation_values(rpmd_index) + initial_centroid_pos(i)*save_centroids(i) / num_samples
    enddo
    !$omp end single
enddo
!$omp end parallel
! =============================================================================
! Writing correlation times and values to file
open(30,file='stage_correlation.txt', action="readwrite")
allocate(correlation_times(n_steps))
do i =  1, n_steps
    correlation_times(i) = (i - 1.0d0) * dt
enddo

do i = 1, n_steps
    write(30,*) correlation_times(i), correlation_values(i)
enddo
close(30)
end program parallel_stage_rpmd
