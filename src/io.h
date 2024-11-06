#ifndef IO_H
#define IO_H

#include "global.h"
#include "grid.h"

#define MAXLEN 100 

int write_density( char *output_dir, int snap_id, real_t time, Grid3D &host_grid, Grid3D &device_grid );

#endif
