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

#pragma omp declare simd
Real rec1d_p_weno5z(const Real uimt,
                    const Real uimo,
                    const Real ui,
                    const Real uipo,
                    const Real uipt);

const Real optimw[3]  = {1./10., 3./5., 3./10.};// WENO5 optimal weights
const Real EPSL       = 1e-40; //1e-6

}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------

void Reconstruction::ReconstructWeno5ZX1(AthenaArray<Real> &z,
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

    zl_(n_tar,i+1) = rec1d_p_weno5z(zimt,zimo,zi,zipo,zipt);
    zr_(n_tar,i  ) = rec1d_p_weno5z(zipt,zipo,zi,zimo,zimt);
  }
}

void Reconstruction::ReconstructWeno5ZX2(AthenaArray<Real> &z,
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

    zl_(n_tar,i) = rec1d_p_weno5z(zimt,zimo,zi,zipo,zipt);
    zr_(n_tar,i) = rec1d_p_weno5z(zipt,zipo,zi,zimo,zimt);
  }
}

void Reconstruction::ReconstructWeno5ZX3(AthenaArray<Real> &z,
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

    zl_(n_tar,i) = rec1d_p_weno5z(zimt,zimo,zi,zipo,zipt);
    zr_(n_tar,i) = rec1d_p_weno5z(zipt,zipo,zi,zimo,zimt);
  }
}

// impl -----------------------------------------------------------------------
namespace {

#pragma omp declare simd
Real rec1d_p_weno5z(const Real uimt,
                    const Real uimo,
                    const Real ui,
                    const Real uipo,
                    const Real uipt)
{
  /*
  // Computes u[i + 1/2]
  Real uimt = u [i-2];
  Real uimo = u [i-1];
  Real ui   = u [i];
  Real uipo = u [i+1];
  Real uipt = u [i+2];
  */

  const Real othreeotwo = 3.0 / 2.0;

  const Real cc2  = 2.0;
  const Real cc3  = 3.0;
  const Real cc4  = 4.0;
  const Real cc5  = 5.0;
  const Real cc7  = 7.0;
  const Real cc11 = 11.0;

  const Real oocc4 = 1.0 / 4.0;
  const Real oocc6 = 1.0 / 6.0;

  Real up;

  Real uk[3], a[3], b[3], w[3], dsa;
  int j;

  // smoothness coefs, Jiag & Shu '96
  b[0] = othreeotwo * SQR(( uimt-cc2*uimo+ui   )) +
         oocc4 * SQR((uimt-cc4*uimo+cc3*ui));
  b[1] = othreeotwo * SQR(( uimo-cc2*ui  +uipo )) +
         oocc4 * SQR((uimo-uipo ) );
  b[2] = othreeotwo * SQR(( ui  -cc2*uipo+uipt )) +
         oocc4 * SQR((cc3*ui-cc4*uipo+uipt));

  for( j = 0 ; j<3; j++)
  {
    a[j] = optimw[j]*( 1 + std::abs(b[0]-b[2])/( EPSL + b[j]) );
  }

  dsa = 1.0/( a[0] + a[1] + a[2] );

  for( j = 0 ; j<3; j++)
  {
    w[j] = a[j] * dsa;
  }

  uk[0] = oocc6*(   cc2*uimt - cc7*uimo + cc11*ui   );
  uk[1] = oocc6*( -     uimo + cc5*ui   + cc2 *uipo );
  uk[2] = oocc6*(   cc2*ui   + cc5*uipo -        uipt );

  up = w[0]*uk[0] + w[1]*uk[1] + w[2]*uk[2];

  return up;
}

}
// ----------------------------------------------------------------------------

//
// :D
//
