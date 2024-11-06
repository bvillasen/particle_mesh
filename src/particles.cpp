#include <stdio.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include "particles.h"

#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort

template <typename T>
std::vector<int> get_sorted_indices(const std::vector<T> &v) {

  std::vector<int> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);
  stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
  return idx;
}


#define BLOCK_SIZE 256

void Particles::allocate_memory_host(){
  std::cout << "Allocating particle data in host" << std::endl;
  size_t array_size = n_particles * sizeof(real_t);
  mass =  (real_t *) malloc( array_size );
  pos_x = (real_t *) malloc( array_size );
  pos_y = (real_t *) malloc( array_size );
  pos_z = (real_t *) malloc( array_size );
  vel_x = (real_t *) malloc( array_size );
  vel_y = (real_t *) malloc( array_size );
  vel_z = (real_t *) malloc( array_size );
  accel_x = (real_t *) malloc( array_size );
  accel_y = (real_t *) malloc( array_size );
  accel_z = (real_t *) malloc( array_size );
  delta_time = (real_t *) malloc( array_size );

}

void Particles::allocate_memory_device(){
  std::cout << "Allocating particle data in device" << std::endl;
  size_t array_size = n_particles*sizeof(real_t);
  hipMalloc( (void **)&mass, array_size );
  hipMalloc( (void **)&pos_x, array_size );
  hipMalloc( (void **)&pos_y, array_size );
  hipMalloc( (void **)&pos_z, array_size );
  hipMalloc( (void **)&vel_x, array_size );
  hipMalloc( (void **)&vel_y, array_size );
  hipMalloc( (void **)&vel_z, array_size );
  hipMalloc( (void **)&accel_x, array_size );
  hipMalloc( (void **)&accel_y, array_size );
  hipMalloc( (void **)&accel_z, array_size );
  hipMalloc( (void **)&delta_time, array_size );

  int reduction_size = (n_particles-1)/BLOCK_SIZE + 1;  
  hipMalloc( (void **)&temp_reduction, reduction_size*sizeof(real_t) );
  
}

void Particles::allocate_memory(){
  if (type == 0) allocate_memory_host();
  if (type == 1) allocate_memory_device();
}

void Particles::initialize_random_uniform_sphere( real_t total_mass, real_t radius, real_t center ){
  std::cout << "Initializing random uniform sphere" << std::endl;
  real_t particle_mass = total_mass / n_particles;
  real_t x, y, z;
  int indx = 0;
  
  // Seed the random number generator with the current time
  // std::srand(std::time(0)); 
  printf( "random: %f \n", rand() / (double)RAND_MAX );
  
  while( indx < n_particles ){
    x = 2 * radius * (std::rand() / (double)RAND_MAX) - radius; // Random number in range [-R, R]
    y = 2 * radius * (std::rand() / (double)RAND_MAX) - radius; // Random number in range [-R, R]
    z = 2 * radius * (std::rand() / (double)RAND_MAX) - radius; // Random number in range [-R, R]
    if ( x*x + y*y + z*z  < radius*radius ){
      mass[indx] = particle_mass;
      pos_x[indx] = x + center;
      pos_y[indx] = y + center;
      pos_z[indx] = z + center;
      vel_x[indx] = vel_y[indx] = vel_z[indx] = 0;
      indx += 1;
    }
  }
}

template <typename T>
void sort_array( T *array, T *temp, std::vector<int> &sorted_indices ){
  int n = sorted_indices.size();
  for ( int i=0; i<n; i++){
    int index = sorted_indices[i];
    temp[i] = array[index];
  }
  for ( int i=0; i<n; i++){
    array[i] = temp[i];
  }
}

void Particles::sort_positions( int nx, int ny, int nz,
                                real_t Lx, real_t Ly, real_t Lz ){
  std::cout << "Sorting particles positions" << std::endl;                                
  real_t dx = Lx / nx;
  real_t dy = Ly / ny;
  real_t dz = Lz / nz;
  std::vector<int>global_indices( n_particles );
  for ( int i=0; i<n_particles; i++ ){
    int indx_x = floor( pos_x[i] / dx );
    int indx_y = floor( pos_y[i] / dy );
    int indx_z = floor( pos_z[i] / dz );
    global_indices[i] = indx_x + indx_y*nx + indx_z*nx*ny;  
  }

  std::vector<int> sorted_indices = get_sorted_indices( global_indices );
  real_t *temp = (real_t *)malloc(n_particles*sizeof(real_t));
  sort_array( pos_x, temp, sorted_indices );
  sort_array( pos_y, temp, sorted_indices );
  sort_array( pos_z, temp, sorted_indices );
  free( temp );
}

void Particles::copy_host_to_device( Particles &host_particles ){
  std::cout << "Copying particle data to device" << std::endl;
  if ( n_particles != host_particles.n_particles ) return;
  size_t array_size = n_particles * sizeof(real_t);
  hipMemcpy( mass,  host_particles.mass, array_size, hipMemcpyHostToDevice );  
  hipMemcpy( pos_x, host_particles.pos_x, array_size, hipMemcpyHostToDevice );
  hipMemcpy( pos_y, host_particles.pos_y, array_size, hipMemcpyHostToDevice );
  hipMemcpy( pos_z, host_particles.pos_z, array_size, hipMemcpyHostToDevice );
  hipMemcpy( vel_x, host_particles.vel_x, array_size, hipMemcpyHostToDevice );
  hipMemcpy( vel_y, host_particles.vel_y, array_size, hipMemcpyHostToDevice );
  hipMemcpy( vel_z, host_particles.vel_z, array_size, hipMemcpyHostToDevice );
}


__global__ void update_positions_kernel(int n_particles,  real_t delta_time, 
                                  real_t *pos_x, real_t *pos_y, real_t *pos_z,
                                  real_t *vel_x, real_t *vel_y, real_t *vel_z ){

  int tid = blockIdx.x * blockDim.x + threadIdx.x ;
  if ( tid >= n_particles ) return;

  // Advance the particle positions using the velocities
  pos_x[tid] += delta_time * vel_x[tid];
  pos_y[tid] += delta_time * vel_y[tid];
  pos_z[tid] += delta_time * vel_z[tid];
  
} 


void Particles::update_positions( real_t delta_time ){
  int n_blocks = ( n_particles - 1 )/BLOCK_SIZE + 1;
  update_positions_kernel<<<n_blocks, BLOCK_SIZE>>>(n_particles, delta_time,
                                                    pos_x, pos_y, pos_z,
                                                    vel_x, vel_y, vel_z );
}
  

__global__ void update_velocities_kernel(int n_particles,  real_t delta_time, 
                                  real_t *vel_x, real_t *vel_y, real_t *vel_z,  
                                  real_t *accel_x, real_t *accel_y, real_t *accel_z ){

  int tid = blockIdx.x * blockDim.x + threadIdx.x ;
  if ( tid >= n_particles ) return;

  // Advance the particle velocities using the accelerations
  vel_x[tid] += delta_time * accel_x[tid];
  vel_y[tid] += delta_time * accel_y[tid];
  vel_z[tid] += delta_time * accel_z[tid];
  
} 


void Particles::update_velocities( real_t delta_time ){
  int n_blocks = ( n_particles - 1 )/BLOCK_SIZE + 1;
  update_positions_kernel<<<n_blocks, BLOCK_SIZE>>>(n_particles, delta_time,
                                                    vel_x, vel_y, vel_z,
                                                    accel_x, accel_y, accel_z );
}
  


__global__ void get_delta_time_kernel(int n_particles, real_t dx,  real_t dy,  real_t dz,  
                                      real_t *vel_x, real_t *vel_y, real_t *vel_z,
                                      real_t *accel_x, real_t *accel_y, real_t *accel_z,     
                                      real_t *particles_dt ){

  int tid = blockIdx.x * blockDim.x + threadIdx.x ;
  if ( tid >= n_particles ) return;

  real_t dt = 1e100;

  real_t v, a;
  // X direction
  v = fabs( vel_x[tid] );
  a = fabs( accel_x[tid] );
  dt = fmin( dt, dx/v );          // Velocity limiter
  dt = fmin( dt, sqrt(2*dx/a) );  // Acceleration limiter

  // Y direction
  v = fabs( vel_y[tid] );
  a = fabs( accel_y[tid] );
  dt = fmin( dt, dy/v );          // Velocity limiter
  dt = fmin( dt, sqrt(2*dy/a) );  // Acceleration limiter
   
  // Z direction
  v = fabs( vel_z[tid]);
  a = fabs( accel_z[tid]);
  dt = fmin( dt, dz/v );          // Velocity limiter
  dt = fmin( dt, sqrt(2*dz/a) );  // Acceleration limiter

  // Write dt calculated for each particle
  particles_dt[tid] = dt;
}  

__global__ void minimum_reduction_kernel( int size, real_t *input, real_t *block_reduction ){

  int tid = blockIdx.x * blockDim.x + threadIdx.x ;
  if ( tid >= size ) return;

  __shared__ real_t sh_block_val[BLOCK_SIZE];
  int id = threadIdx.x;

  // Load to shared memory
  sh_block_val[id] = input[tid];
  __syncthreads();

  // Reduce the threads block on shared memory
  for ( int s=blockDim.x/2; s>0; s>>=1 ){
    if ( id < s ) sh_block_val[id] = fmin( sh_block_val[id], sh_block_val[id + s]);
    __syncthreads();
  }

  // write result from this block to gto the block reduction array
  if (id == 0) block_reduction[blockIdx.x] = sh_block_val[0];
} 

real_t Particles::compute_delta_time( real_t cfl, real_t dx, real_t dy, real_t dz ){
  int n_blocks = ( n_particles - 1 )/BLOCK_SIZE + 1;
  get_delta_time_kernel<<<n_blocks, BLOCK_SIZE>>>(n_particles, dx, dy, dz,
                                                  vel_x, vel_y, vel_z,
                                                  accel_x, accel_y, accel_z,
                                                  delta_time );  

  // hipMemcpy( host_delta_time, delta_time, n_particles*sizeof(real_t), hipMemcpyDeviceToHost );
  // hipDeviceSynchronize();

  // real_t host_dt = 1e100;
  // for (int i=0; i<n_particles; i++ ) host_dt = fmin(host_dt, host_delta_time[i]);


  // Perform a reduction to find the minimum delta_time
  int size = n_particles;
  real_t *reduction_input = delta_time;
  real_t *reduction_output = temp_reduction;

  while (size > 0 ){
    n_blocks = (size - 1) / BLOCK_SIZE + 1;
    minimum_reduction_kernel<<<n_blocks, BLOCK_SIZE>>>( size, reduction_input, reduction_output );
    
    size /= BLOCK_SIZE;
    real_t *temp = reduction_input;
    reduction_input = reduction_output;
    reduction_output = temp;
  }

  real_t dt;
  hipMemcpy( &dt, reduction_input, sizeof(real_t), hipMemcpyDeviceToHost );
  hipDeviceSynchronize();
  // printf( "delta_time. host: %f  device: %f  \n", host_dt, dt);
  return cfl * dt;
}