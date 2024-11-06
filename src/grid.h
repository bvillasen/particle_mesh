#ifndef GRID_H
#define GRID_H

#include "global.h"
#include "particles.h"

class Grid3D {
  public:
  int type;  // [0 for host_particles and 1 for device_particles]
  int nx, ny, nz;
  real_t box_length; 
  real_t *density; 

  real_t *potential;
  real_t *gravity_x; 
  real_t *gravity_y; 
  real_t *gravity_z;

  Grid3D( int _type, real_t _box_length, int _nx, int _ny, int _nz ){ 
    type=_type; box_length=_box_length, nx=_nx; ny=_ny; nz=_nz;
  };

  void allocate_memory_host();
  void allocate_memory_device();
  void allocate_memory();

  void copy_host_to_device( Grid3D &host_grid );
  void copy_device_to_host( Grid3D &device_grid );

  void compute_density_CIC_gpu( Particles &particles);
  void compute_particles_acceleration_CIC_gpu( Particles &device_particles );
  
  // For the FFTs
  Complex_hipfft *fft_input, *fft_transform, *fft_output;  
  hipfftHandle fft_plan_forward;
  hipfftHandle fft_plan_backward;
  void initialize_FFT();

  // Gravitational potential and forces
  void compute_gravitational_potential( real_t Grav_constant );
  void compute_gravitational_field();

};






#endif // ifndef GRID_H