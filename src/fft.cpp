#include <stdio.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include "fft.h"
#include "grid.h"



void FFT3D::initialize(){
  std::cout << "Creating HipFFT plans for 3D FFT " << std::endl;;
  hipfftPlan3d( &fft_plan_forward,  nx, ny,  nz, HIPFFT_Z2Z);
  hipfftPlan3d( &fft_plan_backward, nx, ny,  nz, HIPFFT_Z2Z);
  
  std::cout << "Allocating data for 3D FFT in device" << std::endl;;
  size_t array_size = nx * ny * nz * sizeof(Complex_hipfft);
  hipMalloc( (void **)&input,     array_size );
  hipMalloc( (void **)&transform, array_size );
  hipMalloc( (void **)&output,    array_size );
}

// __global__ void copy_fft_input_kernel( int n_grid, real_t Grav_constant, 
//                                       real_t *density, Complex_hipfft *fft_input ){
//   int tid = threadIdx.x + blockIdx.x * blockDim.x;
//   if ( tid > n_grid ) return;
//   fft_input[tid].x = 4 * M_PI * Grav_constant * density[tid];
//   fft_input[tid].y = 0;
// }

// __global__ void copy_fft_output_kernel( int n_grid, Complex_hipfft *fft_output, real_t *potential ){
//   int tid = threadIdx.x + blockIdx.x * blockDim.x;
//   if ( tid > n_grid ) return;
//   potential[tid] =fft_output[tid].x;
// }


// __global__ void apply_Greens_function_kernel( int nx, int ny, int nz, real_t L, Complex_hipfft *fft_transform ){
//   int tid = threadIdx.x + blockIdx.x * blockDim.x;
//   if ( tid > nx*ny*nz ) return;
  
//   // Get 3D indices form the global 1D thread index
//   int indx_x, indx_y, indx_z;
//   indx_z = tid / (nx*ny);
//   indx_y = (tid - indx_z*nx*ny) / nx;
//   indx_x = tid - indx_z*nx*ny - indx_y*nx;

//   int i, j, k;
//   i = indx_x < nx/2 ? indx_x : indx_x - nx;
//   j = indx_y < ny/2 ? indx_y : indx_y - ny;
//   k = indx_z < nz/2 ? indx_z : indx_z - nz;

//   real_t kx, ky, kz;
//   kx = 2 * M_PI * i / L;
//   ky = 2 * M_PI * j / L;
//   kz = 2 * M_PI * k / L;   

//   real_t G = -1. / ( kx*kx + ky*ky + kz*kz );
//   if ( tid == 0 ) G = 1.0;

//   fft_transform[tid].x *= G;
//   fft_transform[tid].y *= G;

// }


// #define BLOCK_SIZE 256

// void FFT3D::compute_gravitational_potential( real_t Grav_constant, Grid3D &grid ){
//   int n_grid = nx * ny * nz;                                      
//   int n_blocks = ( n_grid - 1 )/BLOCK_SIZE + 1;
//   copy_fft_input_kernel<<<n_blocks, BLOCK_SIZE>>>( n_grid, Grav_constant, grid.density, input );

//     // Apply forward FFT
//   hipfftExecZ2Z( fft_plan_forward, input, transform, HIPFFT_FORWARD );

//   // Apply Green's function
//   apply_Greens_function_kernel<<<n_blocks, BLOCK_SIZE>>>( nx, ny, nz, grid.box_length, transform );

//   // Apply inverse FFT
//   hipfftExecZ2Z( fft_plan_backward, transform, output, HIPFFT_BACKWARD );

//   copy_fft_output_kernel<<<n_blocks, BLOCK_SIZE>>>( n_grid, output, grid.potential );
// }


// __global__ void apply_gradient_kernel( int direction, int nx, int ny, int nz, real_t L, Complex_hipfft *fft_transform ){
//   int tid = threadIdx.x + blockIdx.x * blockDim.x;
//   if ( tid > nx*ny*nz ) return;
  
//   // Get 3D indices form the global 1D thread index
//   int indx_x, indx_y, indx_z;
//   indx_z = tid / (nx*ny);
//   indx_y = (tid - indx_z*nx*ny) / nx;
//   indx_x = tid - indx_z*nx*ny - indx_y*nx;

//   int i, j, k;
//   i = indx_x < nx/2 ? indx_x : indx_x - nx;
//   j = indx_y < ny/2 ? indx_y : indx_y - ny;
//   k = indx_z < nz/2 ? indx_z : indx_z - nz;

//   real_t kx, ky, kz;
//   kx = 2 * M_PI * i / L;
//   ky = 2 * M_PI * j / L;
//   kz = 2 * M_PI * k / L;   

//   real_t gradient_factor;
//   if (direction == 0) gradient_factor = kx;
//   if (direction == 1) gradient_factor = ky;
//   if (direction == 2) gradient_factor = kz;  

//   real_t re, im;
//   re = fft_transform[tid].x;
//   im = fft_transform[tid].y;   
//   fft_transform[tid].x = -im * gradient_factor;
//   fft_transform[tid].y = re * gradient_factor;

// }

// void Grid3D::compute_gravitational_field( Grid3D &grid ){


// }