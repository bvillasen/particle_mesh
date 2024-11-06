#include <stdio.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include "grid.h"
#include "particles.h"

#define BLOCK_SIZE 256

// #define NO_ATOMICS

//Get the CIC index from the particle position ( device function )
__device__ __inline__ void get_indices_CIC( real_t pos_x, real_t pos_y, real_t pos_z, 
                                            real_t dx, real_t dy, real_t dz, 
                                            int &indx_x, int &indx_y, int &indx_z ){
  indx_x = floor( ( pos_x - 0.5*dx ) / dx );
  indx_y = floor( ( pos_y - 0.5*dy ) / dy );
  indx_z = floor( ( pos_z - 0.5*dz ) / dz );
}


// #define INDEX1D(indx_x, indx_y, indx_z, nx, ny) (indx_x + indx_y*nx + indx_z*nx*ny)
__device__ __inline__ int get_index_1D( int indx_x, int indx_y, int indx_z, 
                                        int nx, int ny, int nz ){
if ( indx_x < 0 ) indx_x += nx;
if ( indx_y < 0 ) indx_y += ny;
if ( indx_z < 0 ) indx_z += nz;
if ( indx_x >= nx ) indx_x -= nx;
if ( indx_y >= ny ) indx_y -= ny;
if ( indx_z >= nz ) indx_z -= nz;
return indx_x + nx*indx_y + nx*ny*indx_z;

}

//HIP Kernel to compute the CIC density from the particles positions
__global__ void compute_density_CIC_kernel( int n_particles, real_t L, int nx, int ny, int nz,
                                            real_t *mass, real_t *pos_x, real_t *pos_y, real_t *pos_z,
                                            real_t *density ){

  int tid = blockIdx.x * blockDim.x + threadIdx.x ;
  if ( tid >= n_particles ) return;

  real_t dx, dy, dz, dVol;
  dx = L / nx;
  dy = L / ny;
  dz = L / nz;
  dVol = dx * dy * dz;

  real_t p_x, p_y, p_z, p_dens; // Particle position and mass
  p_dens = mass[tid]/dVol;
  p_x = pos_x[tid];
  p_y = pos_y[tid];
  p_z = pos_z[tid];  

  int indx_x, indx_y, indx_z;
  get_indices_CIC( p_x, p_y, p_z, dx, dy, dz, indx_x, indx_y, indx_z );
  real_t cell_center_x, cell_center_y, cell_center_z;
  cell_center_x = indx_x*dx + 0.5*dx;
  cell_center_y = indx_y*dy + 0.5*dy;
  cell_center_z = indx_z*dz + 0.5*dz;


  real_t delta_x, delta_y, delta_z, cic_factor;
  delta_x = 1 - ( p_x - cell_center_x ) / dx;
  delta_y = 1 - ( p_y - cell_center_y ) / dy;
  delta_z = 1 - ( p_z - cell_center_z ) / dz;


  int indx;  
  // indx = INDEX1D( indx_x, indx_y, indx_z, nx, ny );
  indx = get_index_1D( indx_x, indx_y, indx_z, nx, ny, nz );
  cic_factor = delta_x * delta_y * delta_z;
  #ifdef NO_ATOMICS
  density[indx] +=  p_dens*cic_factor;
  #else
  atomicAdd( &density[indx], p_dens*cic_factor );
  #endif

  // indx = INDEX1D( indx_x+1, indx_y, indx_z, nx, ny );
  indx = get_index_1D( indx_x+1, indx_y, indx_z, nx, ny, nz );
  cic_factor = (1-delta_x) * delta_y * delta_z;
  #ifdef NO_ATOMICS
  density[indx] +=  p_dens*cic_factor;
  #else
  atomicAdd( &density[indx], p_dens*cic_factor );
  #endif

  // indx = INDEX1D( indx_x, indx_y+1, indx_z, nx, ny );
  indx = get_index_1D( indx_x, indx_y+1, indx_z, nx, ny, nz );
  cic_factor = delta_x * (1-delta_y) * delta_z;
  #ifdef NO_ATOMICS
  density[indx] +=  p_dens*cic_factor;
  #else
  atomicAdd( &density[indx], p_dens*cic_factor );
  #endif  

  // indx = INDEX1D( indx_x, indx_y, (indx_z+1), nx, ny );
  indx = get_index_1D( indx_x, indx_y, indx_z+1, nx, ny, nz );
  cic_factor = delta_x * delta_y * (1-delta_z);
  #ifdef NO_ATOMICS
  density[indx] +=  p_dens*cic_factor;
  #else
  atomicAdd( &density[indx], p_dens*cic_factor );
  #endif  

  // indx = INDEX1D( indx_x+1, indx_y+1, indx_z, nx, ny );
  indx = get_index_1D( indx_x+1, indx_y+1, indx_z, nx, ny, nz );
  cic_factor = (1-delta_x) * (1-delta_y) * delta_z;
  #ifdef NO_ATOMICS
  density[indx] +=  p_dens*cic_factor;
  #else
  atomicAdd( &density[indx], p_dens*cic_factor );
  #endif  

  // indx = INDEX1D( indx_x+1, indx_y, indx_z+1, nx, ny );
  indx = get_index_1D( indx_x+1, indx_y, indx_z+1, nx, ny, nz );
  cic_factor = (1-delta_x) * delta_y * (1-delta_z);
  #ifdef NO_ATOMICS
  density[indx] +=  p_dens*cic_factor;
  #else
  atomicAdd( &density[indx], p_dens*cic_factor );
  #endif  

  // indx = INDEX1D( indx_x, indx_y+1, indx_z+1, nx, ny );
  indx = get_index_1D( indx_x, indx_y+1, indx_z+1, nx, ny, nz );
  cic_factor = delta_x * (1-delta_y) * (1-delta_z);
  #ifdef NO_ATOMICS
  density[indx] +=  p_dens*cic_factor;
  #else
  atomicAdd( &density[indx], p_dens*cic_factor );
  #endif  

  // indx = INDEX1D( indx_x+1, indx_y+1, indx_z+1, nx, ny );
  indx = get_index_1D( indx_x+1, indx_y+1, indx_z+1, nx, ny, nz );
  cic_factor = (1-delta_x) * (1-delta_y) * (1-delta_z);
  #ifdef NO_ATOMICS
  density[indx] +=  p_dens*cic_factor;
  #else
  atomicAdd( &density[indx], p_dens*cic_factor );
  #endif
}

__global__ void set_zero_density_kernel( int n_grid, real_t *grid_density ){
  int tid = blockIdx.x * blockDim.x + threadIdx.x ;
  if ( tid < n_grid )  grid_density[tid] = 0;
}

void Grid3D::compute_density_CIC_gpu( Particles &particles){

  int n_grid = nx*ny*nz;
  int n_blocks = ( n_grid - 1 )/BLOCK_SIZE + 1;
  set_zero_density_kernel<<<n_blocks, BLOCK_SIZE>>>( n_grid, density );
  
  n_blocks = ( particles.n_particles - 1 )/BLOCK_SIZE + 1;
  compute_density_CIC_kernel<<<n_blocks, BLOCK_SIZE>>>( particles.n_particles, box_length, nx, ny, nz,
                                                        particles.mass, particles.pos_x, particles.pos_y, particles.pos_z,
                                                        density );

}  




//HIP Kernel to compute the CIC acceleration from the particles positions and the grid gravitational field
__global__ void compute_acceleration_CIC_kernel( int n_particles, real_t L, int nx, int ny, int nz,
                                                 real_t *pos_x, real_t *pos_y, real_t *pos_z, 
                                                 real_t *accel_x, real_t *accel_y, real_t *accel_z,  
                                                 real_t *grav_x, real_t *grav_y, real_t *grav_z ){
  int tid = blockIdx.x * blockDim.x + threadIdx.x ;
  if ( tid >= n_particles ) return;

  real_t dx, dy, dz;
  dx = L / nx;
  dy = L / ny;
  dz = L / nz;

  real_t p_x, p_y, p_z; // Particle position 
  p_x = pos_x[tid];
  p_y = pos_y[tid];
  p_z = pos_z[tid];  

  int indx_x, indx_y, indx_z;
  get_indices_CIC( p_x, p_y, p_z, dx, dy, dz, indx_x, indx_y, indx_z );
  real_t cell_center_x, cell_center_y, cell_center_z;
  cell_center_x = indx_x*dx + 0.5*dx;
  cell_center_y = indx_y*dy + 0.5*dy;
  cell_center_z = indx_z*dz + 0.5*dz;

  real_t delta_x, delta_y, delta_z, cic_factor;
  delta_x = 1 - ( p_x - cell_center_x ) / dx;
  delta_y = 1 - ( p_y - cell_center_y ) / dy;
  delta_z = 1 - ( p_z - cell_center_z ) / dz;

  real_t p_accel_x, p_accel_y, p_accel_z;
  p_accel_x = 0;
  p_accel_y = 0;
  p_accel_z = 0;

  int indx;  
  indx = get_index_1D( indx_x, indx_y, indx_z, nx, ny, nz );
  cic_factor = delta_x * delta_y * delta_z;
  p_accel_x += grav_x[indx] * cic_factor;
  p_accel_y += grav_y[indx] * cic_factor;
  p_accel_z += grav_z[indx] * cic_factor;

  indx = get_index_1D( indx_x+1, indx_y, indx_z, nx, ny, nz );
  cic_factor = (1-delta_x) * delta_y * delta_z;
  p_accel_x += grav_x[indx] * cic_factor;
  p_accel_y += grav_y[indx] * cic_factor;
  p_accel_z += grav_z[indx] * cic_factor;

  indx = get_index_1D( indx_x, indx_y+1, indx_z, nx, ny, nz );
  cic_factor = delta_x * (1-delta_y) * delta_z;
  p_accel_x += grav_x[indx] * cic_factor;
  p_accel_y += grav_y[indx] * cic_factor;
  p_accel_z += grav_z[indx] * cic_factor;
  
  indx = get_index_1D( indx_x, indx_y, indx_z+1, nx, ny, nz );
  cic_factor = delta_x * delta_y * (1-delta_z);
  p_accel_x += grav_x[indx] * cic_factor;
  p_accel_y += grav_y[indx] * cic_factor;
  p_accel_z += grav_z[indx] * cic_factor;

  indx = get_index_1D( indx_x+1, indx_y+1, indx_z, nx, ny, nz );
  cic_factor = (1-delta_x) * (1-delta_y) * delta_z;
  p_accel_x += grav_x[indx] * cic_factor;
  p_accel_y += grav_y[indx] * cic_factor;
  p_accel_z += grav_z[indx] * cic_factor;
  
  indx = get_index_1D( indx_x+1, indx_y, indx_z+1, nx, ny, nz );
  cic_factor = (1-delta_x) * delta_y * (1-delta_z);
  p_accel_x += grav_x[indx] * cic_factor;
  p_accel_y += grav_y[indx] * cic_factor;
  p_accel_z += grav_z[indx] * cic_factor;

  indx = get_index_1D( indx_x, indx_y+1, indx_z+1, nx, ny, nz );
  cic_factor = delta_x * (1-delta_y) * (1-delta_z);
  p_accel_x += grav_x[indx] * cic_factor;
  p_accel_y += grav_y[indx] * cic_factor;
  p_accel_z += grav_z[indx] * cic_factor;
  
  indx = get_index_1D( indx_x+1, indx_y+1, indx_z+1, nx, ny, nz );
  cic_factor = (1-delta_x) * (1-delta_y) * (1-delta_z);
  p_accel_x += grav_x[indx] * cic_factor;
  p_accel_y += grav_y[indx] * cic_factor;
  p_accel_z += grav_z[indx] * cic_factor;

  // Write the particle acceleration to the Particle data arrays
  accel_x[tid] = p_accel_x;
  accel_y[tid] = p_accel_y;
  accel_z[tid] = p_accel_z;
}



void Grid3D::compute_particles_acceleration_CIC_gpu( Particles &device_particles ){

  int n_particles = device_particles.n_particles;
  
  int n_blocks = ( n_particles - 1 )/BLOCK_SIZE + 1;
  compute_acceleration_CIC_kernel<<<n_blocks, BLOCK_SIZE>>>( n_particles, box_length, nx, ny, nz,
                                                             device_particles.pos_x, 
                                                             device_particles.pos_y, 
                                                             device_particles.pos_z,
                                                             device_particles.accel_x, 
                                                             device_particles.accel_y, 
                                                             device_particles.accel_z, 
                                                             gravity_x, gravity_y, gravity_z ); 
                                                             
}
