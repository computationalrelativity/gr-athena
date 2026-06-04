// C/C++ headers

// Athena++ classes headers
#include "../athena.hpp"
#include "reconstruction.hpp"

// ============================================================================
// LAG6 -- 6th-order Lagrange interpolation (unlimited, pointwise)
//
// 6-point stencil symmetric about the evaluation face.  Coefficients
// (palindromic, denominator 256):  { 3, -25, 150, 150, -25, 3 }
//
// uL = value at face i+1/2 (right face of cell i) from {i-2..i+3}
// uR = value at face i-1/2 (left  face of cell i) from {i-3..i+2}
//
// For X2/X3, the swap-buffer pattern in calculate_fluxes.cpp uses
// zl(i) from iteration j-1 as L at face j-1/2, and zr(i) from the
// current iteration j as R at face j-1/2.  Both must produce the
// respective cell's face value at j-1/2.
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
    const Real zim3 = z(n_src, k, j, i - 3);
    const Real zim2 = z(n_src, k, j, i - 2);
    const Real zim1 = z(n_src, k, j, i - 1);
    const Real zi   = z(n_src, k, j, i);
    const Real zip1 = z(n_src, k, j, i + 1);
    const Real zip2 = z(n_src, k, j, i + 2);
    const Real zip3 = z(n_src, k, j, i + 3);

    const Real uL =
      (3.0 * (zim2 + zip3) - 25.0 * (zim1 + zip2) + 150.0 * (zi + zip1)) *
      oo256;

    const Real uR =
      (3.0 * (zim3 + zip2) - 25.0 * (zim2 + zip1) + 150.0 * (zim1 + zi)) *
      oo256;

    zl_(n_tar, i + 1) = uL;
    zr_(n_tar, i)     = uR;
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
    const Real zim3 = z(n_src, k, j - 3, i);
    const Real zim2 = z(n_src, k, j - 2, i);
    const Real zim1 = z(n_src, k, j - 1, i);
    const Real zi   = z(n_src, k, j, i);
    const Real zip1 = z(n_src, k, j + 1, i);
    const Real zip2 = z(n_src, k, j + 2, i);
    const Real zip3 = z(n_src, k, j + 3, i);

    const Real uL =
      (3.0 * (zim2 + zip3) - 25.0 * (zim1 + zip2) + 150.0 * (zi + zip1)) *
      oo256;

    const Real uR =
      (3.0 * (zim3 + zip2) - 25.0 * (zim2 + zip1) + 150.0 * (zim1 + zi)) *
      oo256;

    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
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
    const Real zim3 = z(n_src, k - 3, j, i);
    const Real zim2 = z(n_src, k - 2, j, i);
    const Real zim1 = z(n_src, k - 1, j, i);
    const Real zi   = z(n_src, k, j, i);
    const Real zip1 = z(n_src, k + 1, j, i);
    const Real zip2 = z(n_src, k + 2, j, i);
    const Real zip3 = z(n_src, k + 3, j, i);

    const Real uL =
      (3.0 * (zim2 + zip3) - 25.0 * (zim1 + zip2) + 150.0 * (zi + zip1)) *
      oo256;

    const Real uR =
      (3.0 * (zim3 + zip2) - 25.0 * (zim2 + zip1) + 150.0 * (zim1 + zi)) *
      oo256;

    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
  }
}

// ----------------------------------------------------------------------------

//
// :D
//
