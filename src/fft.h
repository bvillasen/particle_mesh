#ifndef FFT_H
#define FFT_H

#include "global.h"
#include "grid.h"




class FFT3D {
  public:
  int type;  // [0 for host_particles and 1 for device_particles]
  int nx, ny, nz;
  Complex_hipfft *input, *transform, *output;  
  hipfftHandle fft_plan_forward;
  hipfftHandle fft_plan_backward;


  FFT3D( int _type, int _nx, int _ny, int _nz ){ 
    type=_type; nx=_nx; ny=_ny; nz=_nz;
  };

  void initialize();
  void compute_gravitational_potential( real_t Grav_constant, Grid3D &grid );
  void compute_gravitational_field( Grid3D &grid );

};






#endif // ifndef FFT_H