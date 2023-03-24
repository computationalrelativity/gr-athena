#ifndef SphericalHarmonicDecomp_h
#define SphericalHarmonicDecomp_h
#undef USE_LEGENDRE

#include "cctk.h"

extern "C"
{
void SphericalHarmonicDecomp_Read(
     const char *name,
     const int iteration,
     CCTK_REAL *p_time,
     CCTK_REAL *p_Rin,
     CCTK_REAL *p_Rout,
     CCTK_INT *p_lmax,
     CCTK_INT *p_nmax,
     CCTK_REAL **p_re,
     CCTK_REAL **p_im);
}

#ifdef USE_LEGENDRE
# ERROR: Do not activate this option
#endif
#endif
