// C/C++ headers

// Athena++ classes headers
#include "../athena.hpp"
#include "reconstruction.hpp"
#include "reconstruction_utils.hpp"

// ----------------------------------------------------------------------------
namespace {

/*
// BAM conventions (e.g.):
// zl(n,i) = rec1d_p_weno5(zimt,zimo,zi,zipo,zipt);
// zr(n,i) = rec1d_m_weno5(zimt,zimo,zi,zipo,zipt);
//
// or- flip the arguments and write one function
//
// zl(n,i) = rec1d_p_weno5(zimt,zimo,zi,zipo,zipt);
// zr(n,i) = rec1d_p_weno5(zipt,zipo,zi,zimo,zimt);
*/

static const Real alpha = 0.7; // CENO3 coef

#pragma omp declare simd
Real rec1d_p_ceno3(const Real uimt,
                   const Real uimo,
                   const Real ui,
                   const Real uipo,
                   const Real uipt);


}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------

void Reconstruction::ReconstructCeno3X1(AthenaArray<Real> &z,
                                        AthenaArray<Real> &zl_,
                                        AthenaArray<Real> &zr_,
                                        const int n_tar,
                                        const int n_src,
                                        const int k,
                                        const int j,
                                        const int il, const int iu)
{
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    const Real zimt = z(n_src,k,j,i-2);
    const Real zimo = z(n_src,k,j,i-1);
    const Real zi   = z(n_src,k,j,i);
    const Real zipo = z(n_src,k,j,i+1);
    const Real zipt = z(n_src,k,j,i+2);

    zl_(n_tar,i+1) = rec1d_p_ceno3(zimt,zimo,zi,zipo,zipt);
    zr_(n_tar,i  ) = rec1d_p_ceno3(zipt,zipo,zi,zimo,zimt);
  }
}

void Reconstruction::ReconstructCeno3X2(AthenaArray<Real> &z,
                                        AthenaArray<Real> &zl_,
                                        AthenaArray<Real> &zr_,
                                        const int n_tar,
                                        const int n_src,
                                        const int k,
                                        const int j,
                                        const int il, const int iu)
{
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    const Real zimt = z(n_src,k,j-2,i);
    const Real zimo = z(n_src,k,j-1,i);
    const Real zi   = z(n_src,k,j,  i);
    const Real zipo = z(n_src,k,j+1,i);
    const Real zipt = z(n_src,k,j+2,i);

    zl_(n_tar,i) = rec1d_p_ceno3(zimt,zimo,zi,zipo,zipt);
    zr_(n_tar,i) = rec1d_p_ceno3(zipt,zipo,zi,zimo,zimt);
  }
}

void Reconstruction::ReconstructCeno3X3(AthenaArray<Real> &z,
                                        AthenaArray<Real> &zl_,
                                        AthenaArray<Real> &zr_,
                                        const int n_tar,
                                        const int n_src,
                                        const int k,
                                        const int j,
                                        const int il, const int iu)
{
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    const Real zimt = z(n_src,k-2,j,i);
    const Real zimo = z(n_src,k-1,j,i);
    const Real zi   = z(n_src,k,  j,i);
    const Real zipo = z(n_src,k+1,j,i);
    const Real zipt = z(n_src,k+2,j,i);

    zl_(n_tar,i) = rec1d_p_ceno3(zimt,zimo,zi,zipo,zipt);
    zr_(n_tar,i) = rec1d_p_ceno3(zipt,zipo,zi,zimo,zimt);
  }
}

// impl -----------------------------------------------------------------------
namespace {

Real ceno3lim( Real d[3] )
{
  Real o3term = 0.0;
  Real absd[3];
  int kmin;

  if ( ((d[0]>=0.) && (d[1]>=0.) && (d[2]>=0.)) ||
        ((d[0]<0.) && (d[1]<0.) && (d[2]<0.))  )
  {
    absd[0] = std::abs( d[0] );
    absd[1] = std::abs( alpha*d[1] );
    absd[2] = std::abs( d[2] );

    kmin = 0;
    if( absd[1] < absd[kmin] ) kmin = 1;
    if( absd[2] < absd[kmin] ) kmin = 2;

    o3term = d[kmin];
  }

  return( o3term );
}

#pragma omp declare simd
Real rec1d_p_ceno3(const Real uimt,
                   const Real uimo,
                   const Real ui,
                   const Real uipo,
                   const Real uipt)
{
  /*
  // Computes u[i + 1/2]
  Real uipt = u[i+2];
  Real uipo = u[i+1];
  Real ui   = u[i];
  Real uimo = u[i-1];
  Real uimt = u[i-2];
  */
  using namespace reconstruction::utils;

  static const Real oocc2 = 1.0 / 2.0;
  static const Real oocc8 = 1.0 / 8.0;

  static const Real cc2  = 2.0;
  static const Real cc3  = 3.0;
  static const Real cc6  = 6.0;
  static const Real cc10 = 10.0;
  static const Real cc15 = 10.0;

  const Real slope = oocc2 * MC2( ( ui - uimo ), ( uipo - ui ) );

  Real tmpL;
  Real tmpd[3];    // these are d^k_i with k = -1,0,1

  tmpL    = ui + slope;
  tmpd[0] = ( cc3 * uimt - cc10 * uimo + cc15 * ui   ) * oocc8 - tmpL;
  tmpd[1] = ( -     uimo + cc6 * ui    + cc3  * uipo ) * oocc8 - tmpL;
  tmpd[2] = ( cc3 * ui   + cc6 * uipo -         uipt ) * oocc8 - tmpL;

  return tmpL + ceno3lim(tmpd);
}

}
// ----------------------------------------------------------------------------

//
// :D
//
