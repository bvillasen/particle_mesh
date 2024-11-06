#include <stdio.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include "grid.h"

#define BLOCK_SIZE 256

void Grid3D::allocate_memory_host(){
  std::cout << "Allocating grid data in host" << std::endl;
  size_t array_size = nx*ny*nz * sizeof(real_t);
  density   = (real_t *) malloc( array_size );
  potential = (real_t *) malloc( array_size );
  gravity_x = (real_t *) malloc( array_size );
  gravity_y = (real_t *) malloc( array_size );
  gravity_z = (real_t *) malloc( array_size );
}


void Grid3D::allocate_memory_device(){
  std::cout << "Allocating grid data in device" << std::endl;
  size_t array_size = nx*ny*nz * sizeof(real_t);
  hipMalloc( (void **)&density,   array_size );
  hipMalloc( (void **)&potential, array_size );
  hipMalloc( (void **)&gravity_x, array_size );
  hipMalloc( (void **)&gravity_y, array_size );
  hipMalloc( (void **)&gravity_z, array_size );
}

void Grid3D::allocate_memory(){
  if (type == 0) allocate_memory_host();
  if (type == 1) allocate_memory_device();
}


void Grid3D::copy_host_to_device( Grid3D &host_grid ){
  std::cout << "Copying grid data to device" << std::endl;
  size_t array_size = nx*ny*nz * sizeof(real_t);
  hipMemcpy( density,   host_grid.density,   array_size, hipMemcpyHostToDevice );
  hipMemcpy( potential, host_grid.potential, array_size, hipMemcpyHostToDevice );  
}


void Grid3D::copy_device_to_host( Grid3D &device_grid ){
  std::cout << "Copying grid data to host" << std::endl;
  size_t array_size = nx*ny*nz * sizeof(real_t);
  hipMemcpy( density,   device_grid.density,   array_size, hipMemcpyDeviceToHost );
  hipMemcpy( potential, device_grid.potential, array_size, hipMemcpyDeviceToHost );
  hipMemcpy( gravity_x, device_grid.gravity_x, array_size, hipMemcpyDeviceToHost );
  hipMemcpy( gravity_y, device_grid.gravity_y, array_size, hipMemcpyDeviceToHost );
  hipMemcpy( gravity_z, device_grid.gravity_z, array_size, hipMemcpyDeviceToHost );
  hipDeviceSynchronize();  
}



void Grid3D::initialize_FFT(){
  std::cout << "Creating HipFFT plans for 3D FFT " << std::endl;;
  hipfftPlan3d( &fft_plan_forward,  nx, ny,  nz, HIPFFT_Z2Z);
  hipfftPlan3d( &fft_plan_backward, nx, ny,  nz, HIPFFT_Z2Z);
  
  std::cout << "Allocating data for 3D FFT in device" << std::endl;;
  size_t array_size = nx * ny * nz * sizeof(Complex_hipfft);
  hipMalloc( (void **)&fft_input,     array_size );
  hipMalloc( (void **)&fft_transform, array_size );
  hipMalloc( (void **)&fft_output,    array_size );
}

__global__ void copy_fft_input_kernel( int n_grid, real_t Grav_constant, 
                                      real_t *density, Complex_hipfft *fft_input ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if ( tid >= n_grid ) return;
  fft_input[tid].x = 4 * M_PI * Grav_constant * density[tid];
  fft_input[tid].y = 0;
}

__global__ void copy_fft_output_kernel( int n_grid, Complex_hipfft *fft_output, real_t *output ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if ( tid > n_grid ) return;
  output[tid] = fft_output[tid].x / n_grid;
}

__global__ void apply_Greens_function_kernel( int nx, int ny, int nz, real_t L, Complex_hipfft *fft_transform ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if ( tid >= nx*ny*nz ) return;
  
  // Get 3D indices form the global 1D thread index
  int indx_x, indx_y, indx_z;
  indx_z = tid / (nx*ny);
  indx_y = (tid - indx_z*nx*ny) / nx;
  indx_x = tid - indx_z*nx*ny - indx_y*nx;

  int i, j, k;
  i = indx_x < nx/2 ? indx_x : indx_x - nx;
  j = indx_y < ny/2 ? indx_y : indx_y - ny;
  k = indx_z < nz/2 ? indx_z : indx_z - nz;
  
  real_t kx, ky, kz;
  kx = 2 * M_PI * i / L;
  ky = 2 * M_PI * j / L;
  kz = 2 * M_PI * k / L;   

  real_t k2 = kx*kx + ky*ky + kz*kz;
  if ( tid == 0 ) k2 = 1.0;
  // if (indx_x == 1 && indx_y == 1 && indx_z == 1 ) printf( "Thread. k2: %f \n", k2 ); 

  fft_transform[tid].x *= -1/k2;
  fft_transform[tid].y *= -1/k2;
}

void Grid3D::compute_gravitational_potential( real_t Grav_constant ){
  int n_grid = nx * ny * nz;                                      
  int n_blocks = ( n_grid - 1 )/BLOCK_SIZE + 1;
  copy_fft_input_kernel<<<n_blocks, BLOCK_SIZE>>>( n_grid, Grav_constant, density, fft_input );

  // Apply forward FFT
  hipfftExecZ2Z( fft_plan_forward, fft_input, fft_transform, HIPFFT_FORWARD );

  // Apply Green's function
  apply_Greens_function_kernel<<<n_blocks, BLOCK_SIZE>>>( nx, ny, nz, box_length, fft_transform );

  // // Apply inverse FFT
  hipfftExecZ2Z( fft_plan_backward, fft_transform, fft_output, HIPFFT_BACKWARD );

  copy_fft_output_kernel<<<n_blocks, BLOCK_SIZE>>>( n_grid, fft_output, potential );
}


__global__ void copy_potential_to_fft_input( int n_grid, real_t *potential, Complex_hipfft *fft_input ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if ( tid >= n_grid ) return;
  fft_input[tid].x = potential[tid];
  fft_input[tid].y = 0;
}


__global__ void apply_gradient_kernel( int direction, int nx, int ny, int nz, real_t L, Complex_hipfft *fft_transform ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if ( tid >= nx*ny*nz ) return;
  
  // Get 3D indices form the global 1D thread index
  int indx_x, indx_y, indx_z;
  indx_z = tid / (nx*ny);
  indx_y = (tid - indx_z*nx*ny) / nx;
  indx_x = tid - indx_z*nx*ny - indx_y*nx;

  int i, j, k;
  i = indx_x < nx/2 ? indx_x : indx_x - nx;
  j = indx_y < ny/2 ? indx_y : indx_y - ny;
  k = indx_z < nz/2 ? indx_z : indx_z - nz;

  real_t kx, ky, kz;
  kx = 2 * M_PI * i / L;
  ky = 2 * M_PI * j / L;
  kz = 2 * M_PI * k / L;   

  real_t gradient_factor;
  if (direction == 0) gradient_factor = kx;
  if (direction == 1) gradient_factor = ky;
  if (direction == 2) gradient_factor = kz;  

  real_t re, im;
  re = fft_transform[tid].x;
  im = fft_transform[tid].y;   
  fft_transform[tid].x = im * gradient_factor;
  fft_transform[tid].y = -re * gradient_factor;
}

void Grid3D::compute_gravitational_field(){

  int n_grid = nx * ny * nz;                                      
  int n_blocks = ( n_grid - 1 )/BLOCK_SIZE + 1;
  copy_potential_to_fft_input<<<n_blocks, BLOCK_SIZE>>>( n_grid, potential, fft_input );                       
  
  // Derivative in the x-direction
  hipfftExecZ2Z( fft_plan_forward, fft_input, fft_transform, HIPFFT_FORWARD );
  apply_gradient_kernel<<<n_blocks, BLOCK_SIZE>>>( 0, nx, ny, nz, box_length, fft_transform );
  hipfftExecZ2Z( fft_plan_backward, fft_transform, fft_output, HIPFFT_BACKWARD );
  copy_fft_output_kernel<<<n_blocks, BLOCK_SIZE>>>( n_grid, fft_output, gravity_x );
  
  // Derivative in the y-direction
  hipfftExecZ2Z( fft_plan_forward, fft_input, fft_transform, HIPFFT_FORWARD );
  apply_gradient_kernel<<<n_blocks, BLOCK_SIZE>>>( 1, nx, ny, nz, box_length, fft_transform );
  hipfftExecZ2Z( fft_plan_backward, fft_transform, fft_output, HIPFFT_BACKWARD );
  copy_fft_output_kernel<<<n_blocks, BLOCK_SIZE>>>( n_grid, fft_output, gravity_y );
  
  // Derivative in the z-direction
  hipfftExecZ2Z( fft_plan_forward, fft_input, fft_transform, HIPFFT_FORWARD );
  apply_gradient_kernel<<<n_blocks, BLOCK_SIZE>>>( 2, nx, ny, nz, box_length, fft_transform );
  hipfftExecZ2Z( fft_plan_backward, fft_transform, fft_output, HIPFFT_BACKWARD );
  copy_fft_output_kernel<<<n_blocks, BLOCK_SIZE>>>( n_grid, fft_output, gravity_z );

}
