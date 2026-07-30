// C/C++ headers
#include <cmath>

// Athena++ classes headers
#include "../athena.hpp"
#include "recon_koren.hpp"
#include "reconstruction.hpp"

namespace reconstruction::koren {

void ReconstructLR(const Real a,
                   const Real b,
                   const Real c,
                   Real& uL,
                   Real& uR)
{
  constexpr Real EPSL = 1e-40;

  const Real dl = c - b;
  const Real dr = b - a;
  if (dl * dr <= 0.0)
  {
    uL = b;
    uR = b;
    return;
  }
  const Real r_fwd   = dl / (dr + EPSL);
  const Real r_bwd   = dr / (dl + EPSL);
  const Real phi_fwd = std::fmax(
    Real(0.0),
    std::fmin(
      Real(2.0) * r_fwd,
      std::fmin((Real(1.0) + Real(2.0) * r_fwd) / Real(3.0), Real(2.0))));
  const Real phi_bwd = std::fmax(
    Real(0.0),
    std::fmin(
      Real(2.0) * r_bwd,
      std::fmin((Real(1.0) + Real(2.0) * r_bwd) / Real(3.0), Real(2.0))));
  const Real slope = std::copysign(
    std::fmin(phi_fwd * std::fabs(dr), phi_bwd * std::fabs(dl)), dl);
  uL = b + 0.5 * slope;
  uR = b - 0.5 * slope;
}

}  // namespace reconstruction::koren

void Reconstruction::ReconstructKorenX1(AthenaArray<Real>& z,
                                        AthenaArray<Real>& zl_,
                                        AthenaArray<Real>& zr_,
                                        const int n_tar,
                                        const int n_src,
                                        const int k,
                                        const int j,
                                        const int il,
                                        const int iu)
{
  #pragma omp simd simdlen(SIMD_WIDTH)
  for (int i = il; i <= iu; ++i)
  {
    const Real zimo = z(n_src, k, j, i - 1);
    const Real zi   = z(n_src, k, j, i);
    const Real zipo = z(n_src, k, j, i + 1);

    Real uL, uR;
    reconstruction::koren::ReconstructLR(zimo, zi, zipo, uL, uR);
    zl_(n_tar, i + 1) = uL;
    zr_(n_tar, i)     = uR;
  }
}

void Reconstruction::ReconstructKorenX2(AthenaArray<Real>& z,
                                        AthenaArray<Real>& zl_,
                                        AthenaArray<Real>& zr_,
                                        const int n_tar,
                                        const int n_src,
                                        const int k,
                                        const int j,
                                        const int il,
                                        const int iu)
{
  #pragma omp simd simdlen(SIMD_WIDTH)
  for (int i = il; i <= iu; ++i)
  {
    const Real zimo = z(n_src, k, j - 1, i);
    const Real zi   = z(n_src, k, j, i);
    const Real zipo = z(n_src, k, j + 1, i);

    Real uL, uR;
    reconstruction::koren::ReconstructLR(zimo, zi, zipo, uL, uR);
    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
  }
}

void Reconstruction::ReconstructKorenX3(AthenaArray<Real>& z,
                                        AthenaArray<Real>& zl_,
                                        AthenaArray<Real>& zr_,
                                        const int n_tar,
                                        const int n_src,
                                        const int k,
                                        const int j,
                                        const int il,
                                        const int iu)
{
  #pragma omp simd simdlen(SIMD_WIDTH)
  for (int i = il; i <= iu; ++i)
  {
    const Real zimo = z(n_src, k - 1, j, i);
    const Real zi   = z(n_src, k, j, i);
    const Real zipo = z(n_src, k + 1, j, i);

    Real uL, uR;
    reconstruction::koren::ReconstructLR(zimo, zi, zipo, uL, uR);
    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
  }
}
