#include <stdio.h>
#include <iostream>
#include <chrono>
#include "hip/hip_runtime.h"

#include "global.h"
#include "particles.h"
#include "grid.h"
#include "tools.h"
#include "io.h"


char* get_parameter(const std::string & option, char ** begin, char ** end){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end)
  {
    return *itr;
  }
  return 0;
}

bool parameter_exists(const std::string& option, char** begin, char** end ){
  return std::find(begin, end, option) != end;
}


int main(int argc, char * argv[]){

  
  std::cout << "Particle-In-Cell Simulation." << std::endl;
  int nx = 128;
  int ny = 128;
  int nz = 128;
  bool sort_positions = false;

  if (parameter_exists("-nx", argv, argv+argc)) nx = std::stoi(get_parameter("-nx", argv, argv+argc));
  if (parameter_exists("-ny", argv, argv+argc)) ny = std::stoi(get_parameter("-ny", argv, argv+argc));
  if (parameter_exists("-nz", argv, argv+argc)) nz = std::stoi(get_parameter("-nz", argv, argv+argc));
  if (parameter_exists("-sort", argv, argv+argc)) sort_positions = true;

  bool run_density_iterations = true;

  int n_grid = nx * ny * nz;
  int n_particles = n_grid;
  std::cout << "nx: " << nx << "  ny: " << ny << "  nz: " << nz << "  n_grid: " << n_grid << std::endl;  

  real_t box_length = 1.0;
  real_t dx = box_length / nx;
  real_t dy = box_length / ny;
  real_t dz = box_length / nz;
  real_t sphere_density = 1.0;
  real_t sphere_radius = 0.25;
  real_t sphere_volume = 4.0/3.0*M_PI*sphere_radius*sphere_radius*sphere_radius;
  real_t sphere_mass = sphere_density*sphere_volume;
  real_t sphere_center = 0.5;
  real_t Gravitational_constant = 1.0;

  char output_dir[MAXLEN] = "snapshots";
  
  real_t time, delta_time, simulation_time, snapshot_delta_time;
  simulation_time = 0.6;
  snapshot_delta_time = 0.01;
  time = 0;
  real_t cfl = 0.3;

  Particles host_particles( 0, n_particles ); 
  host_particles.allocate_memory();
  host_particles.initialize_random_uniform_sphere( sphere_mass, sphere_radius, sphere_center );
  if (sort_positions) host_particles.sort_positions( nx, ny, nz, box_length, box_length, box_length );
  
  Particles device_particles( 1, n_particles ); 
  device_particles.allocate_memory();
  device_particles.copy_host_to_device( host_particles );

  Grid3D host_grid( 0, box_length, nx, ny, nz );
  host_grid.allocate_memory();

  Grid3D device_grid( 1, box_length, nx, ny, nz );
  device_grid.allocate_memory();
  device_grid.initialize_FFT();

  if (run_density_iterations){

    int n_iter = 100;
    std::cout << "Computing density. n_iter: " << n_iter << std::endl;
    hipDeviceSynchronize();
    auto timer_start = std::chrono::high_resolution_clock::now();
    for ( int iter=0; iter<n_iter; iter++){
      device_grid.compute_density_CIC_gpu( device_particles );
    }
    hipDeviceSynchronize();
    auto timer_stop = std::chrono::high_resolution_clock::now();
    auto duration = duration_cast<std::chrono::milliseconds>(timer_stop - timer_start);
    std::cout << "Elapsed time: " << duration.count() << " millisecs" << std::endl;
    std::cout << "Time per iteration: " << ( (float) duration.count())/n_iter << " millisecs" << std::endl;

    // int snap_id = write_density( output_dir, 0, time, host_grid, device_grid );
    std::cout << "Finished!" << std::endl;  
    return 0;
  }

  // Deposit particles mass onto the grid
  device_grid.compute_density_CIC_gpu( device_particles );
  
  // Compute the gravitational potential
  device_grid.compute_gravitational_potential( Gravitational_constant );

  // Compute the gravitational field
  device_grid.compute_gravitational_field();

  // Compute the particles acceleration based on the grid gravitational field
  device_grid.compute_particles_acceleration_CIC_gpu( device_particles );

  int snap_id = 0;
  snap_id = write_density( output_dir, snap_id, time, host_grid, device_grid );

  int step = 0;
  bool output_step = false;
  while( time < simulation_time ){

    delta_time = device_particles.compute_delta_time( cfl, dx, dy, dz );
    if (time + delta_time > snap_id*snapshot_delta_time ){
      delta_time = snap_id*snapshot_delta_time - time;
      output_step = true;
    }  
    if (time + delta_time > simulation_time ) delta_time = simulation_time - time;
    printf( "Step: %d  time: %e  delta_time: %e  \n", step, time, delta_time );

    
    device_particles.update_velocities( delta_time/2 );

    device_particles.update_positions( delta_time );

    device_grid.compute_density_CIC_gpu( device_particles );
    
    device_grid.compute_gravitational_potential( Gravitational_constant );

    device_grid.compute_gravitational_field();

    device_grid.compute_particles_acceleration_CIC_gpu( device_particles );

    device_particles.update_velocities( delta_time/2 );  

    time += delta_time;
    step += 1;

    if ( output_step ){
      snap_id = write_density( output_dir, snap_id, time, host_grid, device_grid );
      output_step = false;
    }

  }

  printf( "Step: %d  time: %e  \n", step, time);
  std::cout << "Finished!" << std::endl;  
  return 0;
}