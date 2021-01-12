module rpmd_mod
implicit none
contains
! Update position at a half step dt/2
subroutine upd_position(posit, veloc, mass, force, deltat, num_beads)
    integer :: p, num_beads
    double precision :: posit(num_beads), veloc(num_beads), force(num_beads)
    double precision :: mass, deltat
    do p = 1, num_beads
        posit(p) = posit(p) + veloc(p) * deltat + (force(p)/(2*mass) * deltat**2)
    enddo
end subroutine upd_position

! Update velocity at a half step dt/2
subroutine upd_velocity(veloc, force, deltat, mass, num_beads)
    integer :: p, num_beads
    double precision :: veloc(num_beads), force(num_beads)
    double precision :: deltat, mass
    do p = 1, num_beads
        veloc(p) = veloc(p) + (force(p) * deltat)/(2 * mass)
    enddo
end subroutine upd_velocity

! Compute force from the harmonic potential U = 1/2mw^2x^2
subroutine get_spring_force(xi, mass, frequency, harm_force, num_beads)
    integer :: p, num_beads
    double precision :: xi(num_beads), harm_force(num_beads)
    double precision :: mass, frequency
    harm_force(1) = - mass * (frequency**2) * (2*xi(1) - xi(num_beads) - xi(2))
    do p = 2, num_beads - 1
        harm_force(p) = - mass * (frequency**2) * (2*xi(p) - xi(p-1) - xi(p+1))
    enddo
    harm_force(num_beads) = - mass * (frequency**2) * (2*xi(num_beads) - xi(num_beads-1) - xi(1))
end subroutine get_spring_force

! Compute the external force and add it to the spring force
subroutine get_mild_anharm_force(xi, harmonic_force, num_beads)
    integer :: p, num_beads
    double precision :: mild_force
    double precision :: xi(num_beads), harmonic_force(num_beads)
    do p = 1, num_beads
        mild_force = - xi(p) - 0.3d0*xi(p)**2 - 0.04d0*xi(p)**3
        harmonic_force(p) = harmonic_force(p) + mild_force
    enddo
end subroutine get_mild_anharm_force

! Sample velocities from boltzmann distribution using box-muller transformation
subroutine sample_velocities(samps, rpmd_temp, mass, num_beads)
    integer :: p, num_beads
    double precision :: u1, u2, rpmd_temp, mass
    double precision :: samps(num_beads)
    ! Sample initial velocities for each RPMD run
    do p = 1, num_beads
        call random_number(u1)
        call random_number(u2)
        samps(p) = sqrt(1.0 / (rpmd_temp * mass)) * sqrt(-2.0d0 * log(u1))*cos(8.0d0*atan(1.0d0)*u2)
    enddo
end subroutine
end module rpmd_mod
program ring_polymer_dynamics
use rpmd_mod
implicit none
!==============================================================================!
! Setting up arrays and loop variables
integer :: i, j, rando_seed, rpmd_index, traj_var ! Loop variables
integer :: pbeads                   ! Number of beads
integer :: num_samples              ! Number of initial configurations
integer :: n_steps                  ! Number of time steps
double precision :: dt              ! Delta t
double precision :: beta_p          ! Temperature (Beta)
double precision :: m               ! Set Mass
double precision :: temp, partial_corre_sum, initial_corre_value
double precision, allocatable :: primitives_x(:,:), f_x(:,:)
double precision, allocatable :: primitives_v(:,:), initial_centroid_pos(:)
double precision, allocatable :: correlation_times(:), correlation_values(:)
!==============================================================================!
! Reading in parameters
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

! Allocating positions from PIMD
allocate(primitives_x(pbeads,num_samples))
! Reading in samples
open(28, file='positions.txt', action="readwrite")
read(28, *) ((primitives_x(i,j), i = 1, pbeads), j = 1, num_samples)
close(28)
!==============================================================================!
! Allocating the centroid positions to be used for computing Kxx
allocate(initial_centroid_pos(num_samples))
initial_centroid_pos = 0.0d0
! Compute centroid positions from initial configurations (X(t = 0))
do i = 1, num_samples
    initial_centroid_pos(i) = sum(primitives_x(:,i)) / pbeads
enddo

! Allocating kubo transform values
allocate(correlation_values(n_steps+1))
correlation_values = 0.0d0
! Computing Kxx(t = 0)
do i = 1, num_samples
    correlation_values(1) = correlation_values(1) + (initial_centroid_pos(i)**2 ) / num_samples
enddo

! Allocating bead velocities
allocate(primitives_v(pbeads,num_samples))
primitives_v = 0.0d0
! Getting initial velocities from maxwell-boltzmann distribution
call random_seed(rando_seed)
do i = 1, num_samples
    call sample_velocities(primitives_v(:,i), beta_p, m, pbeads)
enddo

! Allocating external forces in x
allocate(f_x(pbeads,num_samples))
f_x = 0.0d0
! Compute initial forces
do i = 1, num_samples
    ! Compute initial forces from harmonic springs
    call get_spring_force(primitives_x(:,i), m, 1.0d0/beta_p, f_x(:,i), pbeads)
    ! Compute initial external forces
    call get_mild_anharm_force(primitives_x(:,i), f_x(:,i), pbeads)
enddo
!==============================================================================!
! Running RPMD
do rpmd_index = 2, n_steps+1
    partial_corre_sum = 0.0d0
    do traj_var = 1, num_samples
        ! Update position
        call upd_position(primitives_x(:,traj_var), primitives_v(:,traj_var), m, f_x(:,traj_var), dt, pbeads)
        ! Update velocity at half step
        call upd_velocity(primitives_v(:,traj_var), f_x(:,traj_var), dt, m, pbeads)
        ! Update harmonic forces in x
        call get_spring_force(primitives_x(:,traj_var), m, 1.0d0/beta_p, f_x(:,traj_var), pbeads)
        ! Update ext forces in x
        call get_mild_anharm_force(primitives_x(:,traj_var), f_x(:,traj_var), pbeads)
        ! Update velocity at step
        call upd_velocity(primitives_v(:,traj_var), f_x(:,traj_var), dt, m, pbeads)
        ! Compute X(n deltat)
        partial_corre_sum = partial_corre_sum + initial_centroid_pos(traj_var) * ( sum(primitives_x(:,traj_var)) / pbeads)
    enddo
    ! Compute Kxx(n deltat)
    correlation_values(rpmd_index) = correlation_values(rpmd_index) + partial_corre_sum / num_samples
enddo
!==============================================================================!
! Writing correlation times and values to file
open(32,file='prim_correlation_parallel.txt', action="readwrite")
allocate(correlation_times(n_steps+1))
do i =  1, n_steps+1
    correlation_times(i) = (i - 1) * dt
enddo

do i = 1, n_steps+1
    write(32,*) correlation_times(i), correlation_values(i)
enddo
close(32)

end program ring_polymer_dynamics
