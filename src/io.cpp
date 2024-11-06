#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <hdf5.h>
#include "io.h"


void write_hdf5_header( hid_t file_id, real_t time, int nx, int ny, int nz  ){
  hid_t     attribute_id, dataspace_id;
  herr_t    status;
  hsize_t   attr_dims;

  attr_dims = 1;
  dataspace_id = H5Screate_simple(1, &attr_dims, NULL);
  
  // Create a group attribute
  attribute_id = H5Acreate(file_id, "time", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &time);
  status = H5Aclose(attribute_id);

  attribute_id = H5Acreate(file_id, "nx", H5T_STD_I64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_ULONG, &nx);
  status = H5Aclose(attribute_id);

  attribute_id = H5Acreate(file_id, "ny", H5T_STD_I64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_ULONG, &ny);
  status = H5Aclose(attribute_id);

  attribute_id = H5Acreate(file_id, "nz", H5T_STD_I64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_ULONG, &nz);
  status = H5Aclose(attribute_id);

  status = H5Sclose(dataspace_id);
}

void write_hdf5_field( hid_t file_id, real_t *data_field, int nx, int ny, int nz, char *field_name ){

  herr_t    status;
  hid_t     dataset_id, dataspace_id;
  hsize_t   dims3d[3];
  dims3d[0] = nx;
  dims3d[1] = ny;
  dims3d[2] = nz;
  dataspace_id = H5Screate_simple(3, dims3d, NULL);

  char name[25];
  sprintf(name, "/");
  strcat(name, field_name);

  dataset_id = H5Dcreate(file_id, name, H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_field);
  status = H5Dclose(dataset_id);

  status = H5Sclose(dataspace_id);
}

int write_density( char *output_dir, int snap_id, real_t time, Grid3D &host_grid, Grid3D &device_grid ){

  char file_name[MAXLEN];
  char timestep[20];

  strcpy(file_name, output_dir);
  sprintf(timestep, "/snapshot_%d.h5", snap_id);
  strcat(file_name,timestep);

  
  printf( "Writing file: %s  time: %e\n", file_name, time );

  hid_t   file_id;
  herr_t  status;

  int n_grid = host_grid.nx * host_grid.ny * host_grid.nz; 
  hipMemcpy( host_grid.density, device_grid.density, n_grid*sizeof(real_t), hipMemcpyDeviceToHost );
  hipDeviceSynchronize();

  // Create a new file collectively
  file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  write_hdf5_header( file_id, time, host_grid.nx, host_grid.ny, host_grid.nz );
  write_hdf5_field( file_id, host_grid.density, host_grid.nx, host_grid.ny, host_grid.nz, "density" );
  status = H5Fclose(file_id);

  return snap_id + 1;
}