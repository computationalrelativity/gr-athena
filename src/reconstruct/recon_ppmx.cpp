// C/C++ headers
#include <cmath>

// Athena++ classes headers
#include "../athena.hpp"
#include "reconstruction.hpp"

// ----------------------------------------------------------------------------
namespace
{

// PPMX (Colella-Sekora extremum-preserving) parabolic reconstruction.
// Returns ql_ip1 (interface i+1/2 from cell i) and qr_i (interface i-1/2
// from cell i).  Works for reconstruction in any dimension by passing the
// appropriate 5-point stencil.
//
// References:
//   (CS)  Colella & Sekora, JCP, 227, 7069 (2008)
//   (PH)  Peterson & Hammett, SIAM J. Sci. Com, 35, B576 (2013)
// ---------------------------------------------------------------------------

#pragma omp declare simd
inline void PPMX(const Real q_im2,
                 const Real q_im1,
                 const Real q_i,
                 const Real q_ip1,
                 const Real q_ip2,
                 Real& ql_ip1,
                 Real& qr_i)
{
  // --- initial 4th-order interface interpolation (CS eqn 16, PH 3.26/3.27) ---
  Real qlv = (Real(7.0) * (q_i + q_im1) - (q_im2 + q_ip1)) / Real(12.0);
  Real qrv = (Real(7.0) * (q_i + q_ip1) - (q_im1 + q_ip2)) / Real(12.0);

  // --- CS limiters at left face (i-1/2) ---
  Real d2qc = Real(3.0) * ((q_im1 + q_i) - Real(2.0) * qlv);
  Real d2ql = (q_im2 + q_i) - Real(2.0) * q_im1;
  Real d2qr = (q_im1 + q_ip1) - Real(2.0) * q_i;

  Real d2qlim = Real(0.0);
  Real lim_slope = std::fmin(std::fabs(d2ql), std::fabs(d2qr));
  if (d2qc > Real(0.0) && d2ql > Real(0.0) && d2qr > Real(0.0)) {
    d2qlim = SIGN(d2qc) * std::fmin(Real(1.25) * lim_slope, std::fabs(d2qc));
  }
  if (d2qc < Real(0.0) && d2ql < Real(0.0) && d2qr < Real(0.0)) {
    d2qlim = SIGN(d2qc) * std::fmin(Real(1.25) * lim_slope, std::fabs(d2qc));
  }
  if (((q_im1 - qlv) * (q_i - qlv)) > Real(0.0)) {
    qlv = Real(0.5) * (q_i + q_im1) - d2qlim / Real(6.0);
  }

  // --- CS limiters at right face (i+1/2) ---
  d2qc = Real(3.0) * ((q_i + q_ip1) - Real(2.0) * qrv);
  d2ql = d2qr;
  d2qr = (q_i + q_ip2) - Real(2.0) * q_ip1;

  d2qlim = Real(0.0);
  lim_slope = std::fmin(std::fabs(d2ql), std::fabs(d2qr));
  if (d2qc > Real(0.0) && d2ql > Real(0.0) && d2qr > Real(0.0)) {
    d2qlim = SIGN(d2qc) * std::fmin(Real(1.25) * lim_slope, std::fabs(d2qc));
  }
  if (d2qc < Real(0.0) && d2ql < Real(0.0) && d2qr < Real(0.0)) {
    d2qlim = SIGN(d2qc) * std::fmin(Real(1.25) * lim_slope, std::fabs(d2qc));
  }
  if (((q_i - qrv) * (q_ip1 - qrv)) > Real(0.0)) {
    qrv = Real(0.5) * (q_i + q_ip1) - d2qlim / Real(6.0);
  }

  // --- extrema detection and handling ---
  Real qa = (qrv - q_i) * (q_i - qlv);
  Real qb = (q_im1 - q_i) * (q_i - q_ip1);
  if (qa <= Real(0.0) || qb <= Real(0.0))
  {
    Real d2q  = Real(6.0) * (qlv + qrv - Real(2.0) * q_i);
    Real d2qc2 = (q_im1 + q_ip1) - Real(2.0) * q_i;
    Real d2ql2 = (q_im2 + q_i) - Real(2.0) * q_im1;
    Real d2qr2 = (q_i + q_ip2) - Real(2.0) * q_ip1;

    d2qlim = Real(0.0);
    lim_slope = std::fmin(std::fmin(std::fabs(d2ql2), std::fabs(d2qr2)),
                          std::fabs(d2qc2));
    if (d2qc2 > Real(0.0) && d2ql2 > Real(0.0) && d2qr2 > Real(0.0) &&
        d2q > Real(0.0)) {
      d2qlim = SIGN(d2q) * std::fmin(Real(1.25) * lim_slope, std::fabs(d2q));
    }
    if (d2qc2 < Real(0.0) && d2ql2 < Real(0.0) && d2qr2 < Real(0.0) &&
        d2q < Real(0.0)) {
      d2qlim = SIGN(d2q) * std::fmin(Real(1.25) * lim_slope, std::fabs(d2q));
    }

    Real rho = Real(0.0);
    if (std::fabs(d2q) >
        Real(1.0e-12) *
          std::fmax(std::fabs(q_im1), std::fmax(std::fabs(q_i),
                    std::fabs(q_ip1)))) {
      rho = d2qlim / d2q;
    }
    qlv = q_i + (qlv - q_i) * rho;
    qrv = q_i + (qrv - q_i) * rho;
  }
  else
  {
    Real qc = qrv - q_i;
    Real qd = qlv - q_i;
    if (std::fabs(qc) >= Real(2.0) * std::fabs(qd)) {
      qrv = q_i - Real(2.0) * qd;
    }
    if (std::fabs(qd) >= Real(2.0) * std::fabs(qc)) {
      qlv = q_i - Real(2.0) * qc;
    }
  }

  ql_ip1 = qrv;
  qr_i   = qlv;
}

}  // namespace
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// X1-direction
// ----------------------------------------------------------------------------

void Reconstruction::ReconstructPPMXX1(AthenaArray<Real>& z,
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
    const Real zimt = z(n_src, k, j, i - 2);
    const Real zimo = z(n_src, k, j, i - 1);
    const Real zi   = z(n_src, k, j, i);
    const Real zipo = z(n_src, k, j, i + 1);
    const Real zipt = z(n_src, k, j, i + 2);

    Real uL, uR;
    PPMX(zimt, zimo, zi, zipo, zipt, uL, uR);
    zl_(n_tar, i + 1) = uL;
    zr_(n_tar, i)     = uR;
  }
}

// ----------------------------------------------------------------------------
// X2-direction
// ----------------------------------------------------------------------------

void Reconstruction::ReconstructPPMXX2(AthenaArray<Real>& z,
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
    const Real zimt = z(n_src, k, j - 2, i);
    const Real zimo = z(n_src, k, j - 1, i);
    const Real zi   = z(n_src, k, j, i);
    const Real zipo = z(n_src, k, j + 1, i);
    const Real zipt = z(n_src, k, j + 2, i);

    Real uL, uR;
    PPMX(zimt, zimo, zi, zipo, zipt, uL, uR);
    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
  }
}

// ----------------------------------------------------------------------------
// X3-direction
// ----------------------------------------------------------------------------

void Reconstruction::ReconstructPPMXX3(AthenaArray<Real>& z,
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
    const Real zimt = z(n_src, k - 2, j, i);
    const Real zimo = z(n_src, k - 1, j, i);
    const Real zi   = z(n_src, k, j, i);
    const Real zipo = z(n_src, k + 1, j, i);
    const Real zipt = z(n_src, k + 2, j, i);

    Real uL, uR;
    PPMX(zimt, zimo, zi, zipo, zipt, uL, uR);
    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
  }
}
