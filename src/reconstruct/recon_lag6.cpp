// C/C++ headers

// Athena++ classes headers
#include "../athena.hpp"
#include "reconstruction.hpp"

// ============================================================================
// LAG6 -- 6th-order Lagrange interpolation (unlimited, pointwise)
//
// 6-point stencil {i-2, i-1, i, i+1, i+2, i+3} symmetric about x_{i+1/2}.
// Coefficients (palindromic, denominator 256):
//   { 3, -25, 150, 150, -25, 3 }
//
// Since the stencil is symmetric about the evaluation point, the left and
// right reconstructed states are identical:  uL = uR = u_{i+1/2}.
//
// Requires NGHOST >= 4.
// ============================================================================

// ----------------------------------------------------------------------------

void Reconstruction::ReconstructLag6X1(AthenaArray<Real>& z,
                                       AthenaArray<Real>& zl_,
                                       AthenaArray<Real>& zr_,
                                       const int n_tar,
                                       const int n_src,
                                       const int k,
                                       const int j,
                                       const int il,
                                       const int iu)
{
  static constexpr Real oo256 = 1.0 / 256.0;

#pragma omp simd simdlen(SIMD_WIDTH)
  for (int i = il; i <= iu; ++i)
  {
    const Real zim2 = z(n_src, k, j, i - 2);
    const Real zim1 = z(n_src, k, j, i - 1);
    const Real zi   = z(n_src, k, j, i);
    const Real zip1 = z(n_src, k, j, i + 1);
    const Real zip2 = z(n_src, k, j, i + 2);
    const Real zip3 = z(n_src, k, j, i + 3);

    const Real uface =
      (3.0 * (zim2 + zip3) - 25.0 * (zim1 + zip2) + 150.0 * (zi + zip1)) *
      oo256;

    zl_(n_tar, i + 1) = uface;
    zr_(n_tar, i)     = uface;
  }
}

void Reconstruction::ReconstructLag6X2(AthenaArray<Real>& z,
                                       AthenaArray<Real>& zl_,
                                       AthenaArray<Real>& zr_,
                                       const int n_tar,
                                       const int n_src,
                                       const int k,
                                       const int j,
                                       const int il,
                                       const int iu)
{
  static constexpr Real oo256 = 1.0 / 256.0;

#pragma omp simd simdlen(SIMD_WIDTH)
  for (int i = il; i <= iu; ++i)
  {
    const Real zim2 = z(n_src, k, j - 2, i);
    const Real zim1 = z(n_src, k, j - 1, i);
    const Real zi   = z(n_src, k, j, i);
    const Real zip1 = z(n_src, k, j + 1, i);
    const Real zip2 = z(n_src, k, j + 2, i);
    const Real zip3 = z(n_src, k, j + 3, i);

    const Real uface =
      (3.0 * (zim2 + zip3) - 25.0 * (zim1 + zip2) + 150.0 * (zi + zip1)) *
      oo256;

    zl_(n_tar, i) = uface;
    zr_(n_tar, i) = uface;
  }
}

void Reconstruction::ReconstructLag6X3(AthenaArray<Real>& z,
                                       AthenaArray<Real>& zl_,
                                       AthenaArray<Real>& zr_,
                                       const int n_tar,
                                       const int n_src,
                                       const int k,
                                       const int j,
                                       const int il,
                                       const int iu)
{
  static constexpr Real oo256 = 1.0 / 256.0;

#pragma omp simd simdlen(SIMD_WIDTH)
  for (int i = il; i <= iu; ++i)
  {
    const Real zim2 = z(n_src, k - 2, j, i);
    const Real zim1 = z(n_src, k - 1, j, i);
    const Real zi   = z(n_src, k, j, i);
    const Real zip1 = z(n_src, k + 1, j, i);
    const Real zip2 = z(n_src, k + 2, j, i);
    const Real zip3 = z(n_src, k + 3, j, i);

    const Real uface =
      (3.0 * (zim2 + zip3) - 25.0 * (zim1 + zip2) + 150.0 * (zi + zip1)) *
      oo256;

    zl_(n_tar, i) = uface;
    zr_(n_tar, i) = uface;
  }
}

// ----------------------------------------------------------------------------

//
// :D
//
