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
Real rec1d_p_donate(const Real uimo, const Real ui, const Real uipo);

}
// ----------------------------------------------------------------------------

void Reconstruction::ReconstructDonateX1(AthenaArray<Real> &z,
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

    zl_(n_tar,i+1) = rec1d_p_donate(zimo,zi,zipo);
    zr_(n_tar,i  ) = rec1d_p_donate(zipo,zi,zimo);
  }
}

void Reconstruction::ReconstructDonateX2(AthenaArray<Real> &z,
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

    zl_(n_tar,i) = rec1d_p_donate(zimo,zi,zipo);
    zr_(n_tar,i) = rec1d_p_donate(zipo,zi,zimo);
  }
}

void Reconstruction::ReconstructDonateX3(AthenaArray<Real> &z,
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

    zl_(n_tar,i) = rec1d_p_donate(zimo,zi,zipo);
    zr_(n_tar,i) = rec1d_p_donate(zipo,zi,zimo);
  }
}

// impl -----------------------------------------------------------------------
namespace {

#pragma omp declare simd
Real rec1d_p_donate(const Real uimo, const Real ui, const Real uipo)
{
  // center gets put left/right
  return ui;
}

}
// ----------------------------------------------------------------------------

//
// :D
//
