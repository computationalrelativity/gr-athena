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
inline void rec1d_p_JS_smoothness(
  Real & b_0,
  Real & b_1,
  Real & b_2,
  const Real uimt,
  const Real uimo,
  const Real ui,
  const Real uipo,
  const Real uipt);

#pragma omp declare simd
inline void rec1d_p_weno5stencils(
  Real & u_0,
  Real & u_1,
  Real & u_2,
  const Real uimt,
  const Real uimo,
  const Real ui,
  const Real uipo,
  const Real uipt);

#pragma omp declare simd
Real rec1d_p_weno5(const Real uimt,
                   const Real uimo,
                   const Real ui,
                   const Real uipo,
                   const Real uipt);

#pragma omp declare simd
Real rec1d_p_weno5z(const Real uimt,
                    const Real uimo,
                    const Real ui,
                    const Real uipo,
                    const Real uipt);

// WENO5 optimal weights
static const Real optimw[3]  = {1./10., 3./5., 3./10.};
static const Real EPSL       = 1e-40; //1e-6

// See:
// A novel and robust scale-invariant WENO scheme for hyperbolic conservation
// laws; 2022, Don et. al.
#pragma omp declare simd
Real rec1d_p_weno5d_si(const Real uimt,
                       const Real uimo,
                       const Real ui,
                       const Real uipo,
                       const Real uipt);

// const double W5D_SI_EPSL = 1e-40;
static const double W5D_SI_EPSL = 1e-12;
static const double W5D_SI_p    = 2;
static const double W5D_SI_s    = 1;
static const double W5D_SI_mu_0 = 1e-40;

}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------

void Reconstruction::ReconstructWeno5X1(AthenaArray<Real> &z,
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

    zl_(n_tar,i+1) = rec1d_p_weno5(zimt,zimo,zi,zipo,zipt);
    zr_(n_tar,i  ) = rec1d_p_weno5(zipt,zipo,zi,zimo,zimt);
  }
}

void Reconstruction::ReconstructWeno5X2(AthenaArray<Real> &z,
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

    zl_(n_tar,i) = rec1d_p_weno5(zimt,zimo,zi,zipo,zipt);
    zr_(n_tar,i) = rec1d_p_weno5(zipt,zipo,zi,zimo,zimt);
  }
}

void Reconstruction::ReconstructWeno5X3(AthenaArray<Real> &z,
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

    zl_(n_tar,i) = rec1d_p_weno5(zimt,zimo,zi,zipo,zipt);
    zr_(n_tar,i) = rec1d_p_weno5(zipt,zipo,zi,zimo,zimt);
  }
}

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

// ----------------------------------------------------------------------------

void Reconstruction::ReconstructWeno5dsiX1(AthenaArray<Real> &z,
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

    zl_(n_tar,i+1) = rec1d_p_weno5d_si(zimt,zimo,zi,zipo,zipt);
    zr_(n_tar,i  ) = rec1d_p_weno5d_si(zipt,zipo,zi,zimo,zimt);
  }
}

void Reconstruction::ReconstructWeno5dsiX2(AthenaArray<Real> &z,
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

    zl_(n_tar,i) = rec1d_p_weno5d_si(zimt,zimo,zi,zipo,zipt);
    zr_(n_tar,i) = rec1d_p_weno5d_si(zipt,zipo,zi,zimo,zimt);
  }
}

void Reconstruction::ReconstructWeno5dsiX3(AthenaArray<Real> &z,
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

    zl_(n_tar,i) = rec1d_p_weno5d_si(zimt,zimo,zi,zipo,zipt);
    zr_(n_tar,i) = rec1d_p_weno5d_si(zipt,zipo,zi,zimo,zimt);
  }
}

// impl -----------------------------------------------------------------------
namespace {

#pragma omp declare simd
inline void rec1d_p_JS_smoothness(
  Real & b_0,
  Real & b_1,
  Real & b_2,
  const Real uimt,
  const Real uimo,
  const Real ui,
  const Real uipo,
  const Real uipt)
{
  // smoothness coefs, Jiag & Shu '96
  static const Real othreeotwo = 3.0 / 2.0;

  static const Real cc2  = 2.0;
  static const Real cc3  = 3.0;
  static const Real cc4  = 4.0;

  static const Real oocc4 = 1.0 / 4.0;

  b_0 = othreeotwo * SQR(( uimt-cc2*uimo+ui   )) +
         oocc4 * SQR((uimt-cc4*uimo+cc3*ui));
  b_1 = othreeotwo * SQR(( uimo-cc2*ui  +uipo )) +
         oocc4 * SQR((uimo-uipo ) );
  b_2 = othreeotwo * SQR(( ui  -cc2*uipo+uipt )) +
         oocc4 * SQR((cc3*ui-cc4*uipo+uipt));
}

#pragma omp declare simd
inline void rec1d_p_weno5stencils(
  Real & u_0,
  Real & u_1,
  Real & u_2,
  const Real uimt,
  const Real uimo,
  const Real ui,
  const Real uipo,
  const Real uipt)
{
  static const Real cc2  = 2.0;
  static const Real cc5  = 5.0;
  static const Real cc7  = 7.0;
  static const Real cc11 = 11.0;

  static const Real oocc6 = 1.0 / 6.0;

  u_0 = oocc6*(   cc2*uimt - cc7*uimo + cc11*ui   );
  u_1 = oocc6*( -     uimo + cc5*ui   + cc2 *uipo );
  u_2 = oocc6*(   cc2*ui   + cc5*uipo -      uipt );
}

#pragma omp declare simd
Real rec1d_p_weno5(const Real uimt,
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
  Real uk[3], b[3];
  rec1d_p_JS_smoothness(b[0], b[1], b[2], uimt, uimo, ui, uipo, uipt);

  const Real a_0 = optimw[0] / SQR((EPSL + b[0]));
  const Real a_1 = optimw[1] / SQR((EPSL + b[1]));
  const Real a_2 = optimw[2] / SQR((EPSL + b[2]));

  const Real dsa = 1.0/( a_0 + a_1 + a_2 );

  rec1d_p_weno5stencils(uk[0], uk[1], uk[2], uimt, uimo, ui, uipo, uipt);
  return dsa * (a_0*uk[0] + a_1*uk[1] + a_2*uk[2]);
}

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
  Real uk[3], b[3];
  rec1d_p_JS_smoothness(b[0], b[1], b[2], uimt, uimo, ui, uipo, uipt);

  const Real db = std::abs(b[0]-b[2]);

  const Real a_0 = optimw[0] * (1.0 + db / (EPSL + b[0]));
  const Real a_1 = optimw[1] * (1.0 + db / (EPSL + b[1]));
  const Real a_2 = optimw[2] * (1.0 + db / (EPSL + b[2]));

  const Real dsa = 1.0/( a_0 + a_1 + a_2 );

  rec1d_p_weno5stencils(uk[0], uk[1], uk[2], uimt, uimo, ui, uipo, uipt);
  return dsa * (a_0*uk[0] + a_1*uk[1] + a_2*uk[2]);
}

#pragma omp declare simd
Real rec1d_p_weno5d_si(const Real uimt,
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
  Real uk[3], a[3], b[3];

  rec1d_p_JS_smoothness(b[0], b[1], b[2], uimt, uimo, ui, uipo, uipt);

  const Real phi = std::sqrt(std::abs(b[0]-2.*b[1]+b[2]));
  const Real tau = std::abs(b[0]-b[2]);

  // local descaling function
  const int r = 3;
  const Real xi = (std::abs(uimt) +
                   std::abs(uimo) +
                   std::abs(ui) +
                   std::abs(uipo) +
                   std::abs(uipt)) / (2. * (r) - 1.);

  const Real mu  = xi + W5D_SI_mu_0;
  const Real mu2 = SQR(mu);
  const Real Phi = std::min(1., phi / mu);

  const Real eps_mu2 = W5D_SI_EPSL * mu2;

  for(int j=0; j<3; ++j)
  {
    const Real Z_j = std::pow(tau / (b[j] + eps_mu2), W5D_SI_p);
    const Real Gam_j = Phi * Z_j;
    a[j] = optimw[j] * std::pow((1. + Gam_j), W5D_SI_s);
  }

  const Real dsa = 1.0/( a[0] + a[1] + a[2] );
  rec1d_p_weno5stencils(uk[0], uk[1], uk[2], uimt, uimo, ui, uipo, uipt);

  return dsa * (a[0]*uk[0] + a[1]*uk[1] + a[2]*uk[2]);
}

}
// ----------------------------------------------------------------------------

//
// :D
//
