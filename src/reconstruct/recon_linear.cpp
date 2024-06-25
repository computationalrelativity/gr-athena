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

// monotonized central
#pragma omp declare simd
Real rec1d_p_lin_mc2(const Real uimo, const Real ui, const Real uipo);
// van-Leer
#pragma omp declare simd
Real rec1d_p_lin_vl(const Real uimo, const Real ui, const Real uipo);

}
// ----------------------------------------------------------------------------

void Reconstruction::ReconstructLinearVLX1(AthenaArray<Real> &z,
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
    const Real zimo = z(n_src,k,j,i-1);
    const Real zi   = z(n_src,k,j,i);
    const Real zipo = z(n_src,k,j,i+1);

    zl_(n_tar,i+1) = rec1d_p_lin_vl(zimo,zi,zipo);
    zr_(n_tar,i  ) = rec1d_p_lin_vl(zipo,zi,zimo);
  }
}

void Reconstruction::ReconstructLinearVLX2(AthenaArray<Real> &z,
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
    const Real zimo = z(n_src,k,j-1,i);
    const Real zi   = z(n_src,k,j,  i);
    const Real zipo = z(n_src,k,j+1,i);

    zl_(n_tar,i) = rec1d_p_lin_vl(zimo,zi,zipo);
    zr_(n_tar,i) = rec1d_p_lin_vl(zipo,zi,zimo);
  }
}

void Reconstruction::ReconstructLinearVLX3(AthenaArray<Real> &z,
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
    const Real zimo = z(n_src,k-1,j,i);
    const Real zi   = z(n_src,k,  j,i);
    const Real zipo = z(n_src,k+1,j,i);

    zl_(n_tar,i) = rec1d_p_lin_vl(zimo,zi,zipo);
    zr_(n_tar,i) = rec1d_p_lin_vl(zipo,zi,zimo);
  }
}

// ----------------------------------------------------------------------------

void Reconstruction::ReconstructLinearMC2X1(AthenaArray<Real> &z,
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
    const Real zimo = z(n_src,k,j,i-1);
    const Real zi   = z(n_src,k,j,i);
    const Real zipo = z(n_src,k,j,i+1);

    zl_(n_tar,i+1) = rec1d_p_lin_mc2(zimo,zi,zipo);
    zr_(n_tar,i  ) = rec1d_p_lin_mc2(zipo,zi,zimo);
  }
}

void Reconstruction::ReconstructLinearMC2X2(AthenaArray<Real> &z,
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
    const Real zimo = z(n_src,k,j-1,i);
    const Real zi   = z(n_src,k,j,  i);
    const Real zipo = z(n_src,k,j+1,i);

    zl_(n_tar,i) = rec1d_p_lin_mc2(zimo,zi,zipo);
    zr_(n_tar,i) = rec1d_p_lin_mc2(zipo,zi,zimo);
  }
}

void Reconstruction::ReconstructLinearMC2X3(AthenaArray<Real> &z,
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
    const Real zimo = z(n_src,k-1,j,i);
    const Real zi   = z(n_src,k,  j,i);
    const Real zipo = z(n_src,k+1,j,i);

    zl_(n_tar,i) = rec1d_p_lin_mc2(zimo,zi,zipo);
    zr_(n_tar,i) = rec1d_p_lin_mc2(zipo,zi,zimo);
  }
}

// impl -----------------------------------------------------------------------
namespace {

#pragma omp declare simd
Real rec1d_p_lin_mc2(const Real uimo, const Real ui, const Real uipo)
{
  using namespace reconstruction::utils;

  // MC limiter, possibly too aggressive
  Real up;

  Real slope = 0.5 * MC2( ( ui - uimo ), ( uipo - ui ) );
  up = ui + slope;
  return up;
}

#pragma omp declare simd
Real rec1d_p_lin_vl(const Real uimo, const Real ui, const Real uipo)
{
  // L/R slopes  (note _p & opposite convention to PLM!)
  const Real dul = uipo - ui;
  const Real dur = ui   - uimo;
  const Real uc = ui;

  // van Leer slope limiter
  const Real du2 = dul * dur;
  const Real dum = (du2 <= 0.0) ? 0.0 : 2.0 * du2 / (dul + dur);

  const Real ul = uc - 0.5 * dum;
  const Real ur = uc + 0.5 * dum;
  return ur;
}

}
// ----------------------------------------------------------------------------

//
// :D
//
