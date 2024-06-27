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
Real rec1d_p_mp3(const Real eps,
                 const Real uimt,
                 const Real uimo,
                 const Real ui,
                 const Real uipo,
                 const Real uipt);

#pragma omp declare simd
Real rec1d_p_mp5(const Real eps,
                 const Real uimt,
                 const Real uimo,
                 const Real ui,
                 const Real uipo,
                 const Real uipt);

#pragma omp declare simd
Real rec1d_p_mp5_R(const Real eps,
                   const Real uimt,
                   const Real uimo,
                   const Real ui,
                   const Real uipo,
                   const Real uipt);

#pragma omp declare simd
Real rec1d_p_mp7(const Real eps,
                 const Real uim3,
                 const Real uim2,
                 const Real uim1,
                 const Real ui,
                 const Real uip1,
                 const Real uip2,
                 const Real uip3);

#pragma omp declare simd
Real mpnlimiter(const Real eps_mpn,
                const Real u,
                const Real uimt,
                const Real uimo,
                const Real ui,
                const Real uipo,
                const Real uipt);

#pragma omp declare simd
Real mpnlimiter(const Real eps_mpn,
                const Real u,
                const Real uim3,
                const Real uim2,
                const Real uim1,
                const Real ui,
                const Real uip1,
                const Real uip2,
                const Real uip3);

// See:
// An improved accurate monotonicity-preserving scheme for the Euler equations
// 2016; He et. al.
#pragma omp declare simd
Real mpnlimiter_R(const Real eps_mpn,
                  const Real u,
                  const Real uimt,
                  const Real uimo,
                  const Real ui,
                  const Real uipo,
                  const Real uipt);

}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------

void Reconstruction::ReconstructMP3X1(AthenaArray<Real> &z,
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

    zl_(n_tar,i+1) = rec1d_p_mp3(xorder_eps,zimt,zimo,zi,zipo,zipt);
    zr_(n_tar,i  ) = rec1d_p_mp3(xorder_eps,zipt,zipo,zi,zimo,zimt);
  }
}

void Reconstruction::ReconstructMP3X2(AthenaArray<Real> &z,
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

    zl_(n_tar,i) = rec1d_p_mp3(xorder_eps,zimt,zimo,zi,zipo,zipt);
    zr_(n_tar,i) = rec1d_p_mp3(xorder_eps,zipt,zipo,zi,zimo,zimt);
  }
}

void Reconstruction::ReconstructMP3X3(AthenaArray<Real> &z,
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

    zl_(n_tar,i) = rec1d_p_mp3(xorder_eps,zimt,zimo,zi,zipo,zipt);
    zr_(n_tar,i) = rec1d_p_mp3(xorder_eps,zipt,zipo,zi,zimo,zimt);
  }
}

// ----------------------------------------------------------------------------

void Reconstruction::ReconstructMP5X1(AthenaArray<Real> &z,
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

    zl_(n_tar,i+1) = rec1d_p_mp5(xorder_eps,zimt,zimo,zi,zipo,zipt);
    zr_(n_tar,i  ) = rec1d_p_mp5(xorder_eps,zipt,zipo,zi,zimo,zimt);
  }
}

void Reconstruction::ReconstructMP5X2(AthenaArray<Real> &z,
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

    zl_(n_tar,i) = rec1d_p_mp5(xorder_eps,zimt,zimo,zi,zipo,zipt);
    zr_(n_tar,i) = rec1d_p_mp5(xorder_eps,zipt,zipo,zi,zimo,zimt);
  }
}

void Reconstruction::ReconstructMP5X3(AthenaArray<Real> &z,
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

    zl_(n_tar,i) = rec1d_p_mp5(xorder_eps,zimt,zimo,zi,zipo,zipt);
    zr_(n_tar,i) = rec1d_p_mp5(xorder_eps,zipt,zipo,zi,zimo,zimt);
  }
}

// ----------------------------------------------------------------------------

void Reconstruction::ReconstructMP7X1(AthenaArray<Real> &z,
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
    const Real zim3 = z(n_src,k,j,i-3);
    const Real zim2 = z(n_src,k,j,i-2);
    const Real zim1 = z(n_src,k,j,i-1);
    const Real zi   = z(n_src,k,j,i);
    const Real zip1 = z(n_src,k,j,i+1);
    const Real zip2 = z(n_src,k,j,i+2);
    const Real zip3 = z(n_src,k,j,i+3);

    zl_(n_tar,i+1) = rec1d_p_mp7(xorder_eps,
                                 zim3,zim2,zim1,zi,zip1,zip2,zip3);
    zr_(n_tar,i  ) = rec1d_p_mp7(xorder_eps,
                                 zip3,zip2,zip1,zi,zim1,zim2,zim3);
  }
}

void Reconstruction::ReconstructMP7X2(AthenaArray<Real> &z,
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
    const Real zim3 = z(n_src,k,j-3,i);
    const Real zim2 = z(n_src,k,j-2,i);
    const Real zim1 = z(n_src,k,j-1,i);
    const Real zi   = z(n_src,k,j,  i);
    const Real zip1 = z(n_src,k,j+1,i);
    const Real zip2 = z(n_src,k,j+2,i);
    const Real zip3 = z(n_src,k,j+3,i);

    zl_(n_tar,i) = rec1d_p_mp7(xorder_eps,
                               zim3,zim2,zim1,zi,zip1,zip2,zip3);
    zr_(n_tar,i) = rec1d_p_mp7(xorder_eps,
                               zip3,zip2,zip1,zi,zim1,zim2,zim3);
  }
}

void Reconstruction::ReconstructMP7X3(AthenaArray<Real> &z,
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
    const Real zim3 = z(n_src,k-3,j,i);
    const Real zim2 = z(n_src,k-2,j,i);
    const Real zim1 = z(n_src,k-1,j,i);
    const Real zi   = z(n_src,k,  j,i);
    const Real zip1 = z(n_src,k+1,j,i);
    const Real zip2 = z(n_src,k+2,j,i);
    const Real zip3 = z(n_src,k+3,j,i);

    zl_(n_tar,i) = rec1d_p_mp7(xorder_eps,
                               zim3,zim2,zim1,zi,zip1,zip2,zip3);
    zr_(n_tar,i) = rec1d_p_mp7(xorder_eps,
                               zip3,zip2,zip1,zi,zim1,zim2,zim3);
  }
}

// ----------------------------------------------------------------------------

void Reconstruction::ReconstructMP5RX1(AthenaArray<Real> &z,
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

    zl_(n_tar,i+1) = rec1d_p_mp5_R(xorder_eps,zimt,zimo,zi,zipo,zipt);
    zr_(n_tar,i  ) = rec1d_p_mp5_R(xorder_eps,zipt,zipo,zi,zimo,zimt);
  }
}

void Reconstruction::ReconstructMP5RX2(AthenaArray<Real> &z,
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

    zl_(n_tar,i) = rec1d_p_mp5_R(xorder_eps,zimt,zimo,zi,zipo,zipt);
    zr_(n_tar,i) = rec1d_p_mp5_R(xorder_eps,zipt,zipo,zi,zimo,zimt);
  }
}

void Reconstruction::ReconstructMP5RX3(AthenaArray<Real> &z,
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

    zl_(n_tar,i) = rec1d_p_mp5_R(xorder_eps,zimt,zimo,zi,zipo,zipt);
    zr_(n_tar,i) = rec1d_p_mp5_R(xorder_eps,zipt,zipo,zi,zimo,zimt);
  }
}

// impl -----------------------------------------------------------------------
namespace {

#pragma omp declare simd
Real rec1d_p_mp3(const Real eps,
                 const Real uimt,
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

  static const Real cl3_0 = -1./6.;
  static const Real cl3_1 = 5./6.;
  static const Real cl3_2 = 2./6.;

  const Real ulim  = cl3_0*uimo + cl3_1*ui + cl3_2*uipo;
  return mpnlimiter(eps,ulim,uimt,uimo,ui,uipo,uipt);
}

#pragma omp declare simd
Real rec1d_p_mp5(const Real eps,
                 const Real uimt,
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

  static const Real cl5_0 = 2./60.;
  static const Real cl5_1 = -13./60.;
  static const Real cl5_2 = 47./60.;
  static const Real cl5_3 = 27./60.;
  static const Real cl5_4 = -3./60.;

  const Real ulim  = (cl5_0*uimt +
                      cl5_1*uimo +
                      cl5_2*ui +
                      cl5_3*uipo +
                      cl5_4*uipt);

  return mpnlimiter(eps,ulim,uimt,uimo,ui,uipo,uipt);
}

#pragma omp declare simd
Real rec1d_p_mp7(const Real eps,
                 const Real uim3,
                 const Real uim2,
                 const Real uim1,
                 const Real ui,
                 const Real uip1,
                 const Real uip2,
                 const Real uip3)
{
  /*
  // Computes u[i + 1/2]
  Real uimt = u [i-2];
  Real uimo = u [i-1];
  Real ui   = u [i];
  Real uipo = u [i+1];
  Real uipt = u [i+2];
  */

  static const Real cl7_0 = -3./420.;
  static const Real cl7_1 =  25./420.;
  static const Real cl7_2 = -101./420.;
  static const Real cl7_3 =  319./420.;
  static const Real cl7_4 =  214./420.;
  static const Real cl7_5 = -38./420.;
  static const Real cl7_6 =  4./420.;

  const Real ulim  = (cl7_0*uim3 +
                      cl7_1*uim2 +
                      cl7_2*uim1 +
                      cl7_3*ui +
                      cl7_4*uip1 +
                      cl7_5*uip2 +
                      cl7_6*uip3);

  return mpnlimiter(eps,ulim,uim2,uim1,ui,uip1,uip2);
}

#pragma omp declare simd
Real rec1d_p_mp5_R(const Real eps,
                   const Real uimt,
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

  static const Real cl5_0 = 2./60.;
  static const Real cl5_1 = -13./60.;
  static const Real cl5_2 = 47./60.;
  static const Real cl5_3 = 27./60.;
  static const Real cl5_4 = -3./60.;

  const Real ulim  = (cl5_0*uimt +
                      cl5_1*uimo +
                      cl5_2*ui +
                      cl5_3*uipo +
                      cl5_4*uipt);

  return mpnlimiter_R(eps,ulim,uimt,uimo,ui,uipo,uipt);
}

#pragma omp declare simd
Real mpnlimiter(const Real eps_mpn,
                const Real u,
                const Real uimt,
                const Real uimo,
                const Real ui,
                const Real uipo,
                const Real uipt)
{
  using namespace reconstruction::utils;

  static const Real oo2 = 1./2.;
  static const Real fot = 4./3.;
  static const Real alphatil = 4.;

  if (eps_mpn > 0)
  {
    const Real U_L2 = std::sqrt(SQR(uimt) + SQR(uimo) +
                                SQR(ui) +
                                SQR(uipo) + SQR(uipt));
    const Real u_MP = ui + minmod(uipo-ui, alphatil * (ui-uimo));

    // check whether we should apply limiter
    if ((u-ui)*(u-u_MP) <= eps_mpn * U_L2)
      return u;
  }

  const Real dm = uimt - 2.*uimo + ui;
  const Real d0 = uimo - 2.*ui   + uipo;
  const Real dp = ui   - 2.*uipo + uipt;

  const Real dm4p = minmod(4.*d0-dp, 4.*dp-d0, d0, dp);
  const Real dm4m = minmod(4.*d0-dm, 4.*dm-d0, d0, dm);

  // c.f. code of Suresh '97 (which has uimo)
  const Real u_ul = ui + alphatil * (ui - uimo);
  const Real u_av = oo2 * (ui + uipo);
  const Real u_md = u_av - oo2 * dm4p;
  const Real u_lc = ui + oo2 * (ui - uimo) + fot * dm4m;

  const Real u_min = std::max(min(ui, uipo, u_md), min(ui, u_ul, u_lc));
  const Real u_max = std::min(max(ui, uipo, u_md), max(ui, u_ul, u_lc));

  return u + minmod(u_min-u, u_max-u);
}

#pragma omp declare simd
Real mpnlimiter(const Real eps_mpn,
                const Real u,
                const Real uim3,
                const Real uimt,
                const Real uimo,
                const Real ui,
                const Real uipo,
                const Real uipt,
                const Real uip3)
{
  using namespace reconstruction::utils;

  static const Real oo2 = 1./2.;
  static const Real fot = 4./3.;
  static const Real alphatil = 4.;

  if (eps_mpn > 0)
  {
    const Real U_L2 = std::sqrt(SQR(uimt) + SQR(uimo) +
                                SQR(ui) +
                                SQR(uipo) + SQR(uipt));
    const Real u_MP = ui + minmod(uipo-ui, alphatil * (ui-uimo));

    // check whether we should apply limiter
    if ((u-ui)*(u-u_MP) <= eps_mpn * U_L2)
      return u;
  }

  const Real dm2 = uim3 - 2.*uimt + uimo;
  const Real dm  = uimt - 2.*uimo + ui;
  const Real d0  = uimo - 2.*ui   + uipo;
  const Real dp  = ui   - 2.*uipo + uipt;
  const Real dp2 = uipo - 2.*uipt + uip3;

  const Real dm4p = minmod(4.*d0-dp, 4.*dp-d0, d0, dp, dm, dp2);
  const Real dm4m = minmod(4.*d0-dm, 4.*dm-d0, d0, dm, dp, dm2);

  // c.f. code of Suresh '97 (which has uimo)
  const Real u_ul = ui + alphatil * (ui - uimo);
  const Real u_av = oo2 * (ui + uipo);
  const Real u_md = u_av - oo2 * dm4p;
  const Real u_lc = ui + oo2 * (ui - uimo) + fot * dm4m;

  const Real u_min = std::max(min(ui, uipo, u_md), min(ui, u_ul, u_lc));
  const Real u_max = std::min(max(ui, uipo, u_md), max(ui, u_ul, u_lc));

  return u + minmod(u_min-u, u_max-u);
}

#pragma omp declare simd
Real mpnlimiter_R(const Real eps_mpn,
                  const Real u,
                  const Real uimt,
                  const Real uimo,
                  const Real ui,
                  const Real uipo,
                  const Real uipt)
{
  using namespace reconstruction::utils;

  static const Real oo2 = 1./2.;
  static const Real fot = 4./3.;
  static const Real alphatil = 4.;

  const Real dm = uimt - 2.*uimo + ui;
  const Real d0 = uimo - 2.*ui   + uipo;
  const Real dp = ui   - 2.*uipo + uipt;

  const Real dm4p = minmod(4.*d0-dp, 4.*dp-d0, d0, dp);
  const Real dm4m = minmod(4.*d0-dm, 4.*dm-d0, d0, dm);

  // c.f. code of Suresh '97 (which has uimo)
  const Real u_ul = ui + alphatil * (ui - uimo);
  const Real u_av = oo2 * (ui + uipo);
  const Real u_md = u_av - oo2 * dm4p;
  const Real u_lc = ui + oo2 * (ui - uimo) + fot * dm4m;

  const Real u_min = std::max(min(ui, uipo, u_md), min(ui, u_ul, u_lc));
  const Real u_max = std::min(max(ui, uipo, u_md), max(ui, u_ul, u_lc));


  // TVD limit
  const Real du_mh = ui - uimo;
  const Real du_ph = uipo - ui;

  const Real u_mp = ui + minmod(uipo-ui, alphatil * (ui - uimo));

  if ((u_max-u_min) > (std::max(ui, u_mp)) - std::min(ui, u_mp))
  if ((u < u_min ) || (u_max < u))
  {
    static const Real phi_c = 2.0;

    const Real r_ph = (du_ph > 0) ? du_mh / du_ph : 0;
    const Real phi_ph = (r_ph + std::abs(r_ph)) / (1.0 + std::abs(r_ph));
    const Real u_tvd = ui + 1.0 / phi_c * phi_ph * (uipo - ui);

    return u_tvd;
  }

  // Inside [u_min, u_max], use limiter
  const Real u_re_ph = (
    ui + 0.5 * (sign(du_mh) + sign(du_ph)) *
    std::abs(du_mh * du_ph) / (std::abs(du_mh) + std::abs(du_ph) + eps_mpn)
  );

  const Real u_ph = (
    0.5 * (u + u_re_ph) -
    sign((u - u_min) * (u - u_max)) *
    0.5 * (u - u_re_ph)
  );

  return u_ph;
}


}
// ----------------------------------------------------------------------------

//
// :D
//
