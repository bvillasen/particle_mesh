#ifndef TOOLS_H
#define TOOLS_H


void print_statistics_grid( int n, real_t *values  ){

  real_t v_sum, v_min, v_max, v;
  v_sum = 0;
  v_min = 1e100;
  v_max = -1e100; 
  for (int i=0; i<n; i++){
    v = fabs(values[i]);
    v_sum += v;
    if ( v < v_min) v_min = v;
    if ( v > v_max) v_max = v;
  }
  real_t v_mean = v_sum / n;
  printf( "min: %f  max: %f  mean: %f \n", v_min, v_max, v_mean );

}

#endif // #ifndef TOOLS_H