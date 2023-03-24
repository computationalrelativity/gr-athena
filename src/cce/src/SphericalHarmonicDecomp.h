#ifndef SphericalHarmonicDecomp_h
#define SphericalHarmonicDecomp_h
#undef USE_LEGENDRE

extern "C"
{
void SphericalHarmonicDecomp_Read(
     const char *name,
     const int iteration,
     double *p_time,
     double *p_Rin,
     double *p_Rout,
     int *p_lmax,
     int *p_nmax,
     double **p_re,
     double **p_im);
}

#ifdef USE_LEGENDRE
# ERROR: Do not activate this option
#endif
#endif
