#ifndef GLOBAL_H
#define GLOBAL_H

#include <hipfft.h>

#define PRECISION 2

#if PRECISION == 1
#define real_t float
typedef hipfftReal Real_hipfft;
typedef hipfftComplex Complex_hipfft;
#endif

#if PRECISION == 2
#define real_t double
typedef hipfftDoubleReal Real_hipfft;
typedef hipfftDoubleComplex Complex_hipfft;
#endif


#endif