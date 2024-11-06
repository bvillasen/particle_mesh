#ifndef PARTICLES_H
#define PARTICLES_H

#include "global.h"

class Particles {
  public:
  int type;  // [0 for host_particles and 1 for device_particles]
  int n_particles;
  real_t *mass;
  real_t *pos_x, *pos_y, *pos_z;       //position
  real_t *vel_x, *vel_y, *vel_z;       //velocity
  real_t *accel_x, *accel_y, *accel_z; //acceleration
  real_t *delta_time;
  real_t *temp_reduction;

  Particles( int _type, int _n_particles){ 
    type=_type; n_particles=_n_particles;
  };

  void allocate_memory_host();
  void allocate_memory_device();
  void allocate_memory();

  void initialize_random_uniform_sphere( real_t total_mass, real_t radius, real_t center );
  void sort_positions( int nx, int ny, int nz, real_t Lx, real_t Ly, real_t Lz );
  void copy_host_to_device( Particles &host_particles );

  void update_positions( real_t delta_time );
  void update_velocities( real_t delta_time );

  real_t compute_delta_time( real_t cfl, real_t dx, real_t dy, real_t dz );

};






#endif // ifndef PARTICLES_H