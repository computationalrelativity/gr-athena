#ifndef M1_UTILS_HPP
#define M1_UTILS_HPP

// c++
// ...

// Athena++ classes headers
#include "../utils/linear_algebra.hpp"
#include "m1.hpp"
#include "m1_containers.hpp"
#include "m1_macro.hpp"

// ============================================================================
namespace M1::Assemble {
// ============================================================================

// Construction of various dense & sliced quantities
//
// Conventions for fields:
// sc: (s)calar-(f)ield
// sp: (sp)atial
// st: (s)pace-time
// Appended "_" indicates resulting scratch (in i)
// Appended "__" indicates result at a point

inline void sc_dot_sp_(
  AT_C_sca & sc_dot_sp_,
  const AT_N_vec & sp_V_A,  // alternative: flip for opposite slice
  const AT_N_vec & sp_V_B_,
  const int k, const int j,
  const int il, const int iu)
{
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sc_dot_sp_(i) = 0.0;
  }

  for (int a=0; a<N; ++a)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sc_dot_sp_(i) += sp_V_A(a,k,j,i) * sp_V_B_(a,i);
  }
}

inline void sc_dot_dense_sp_(
  AT_C_sca & sc_dot_sp_,
  const AT_N_vec & sp_V_A,
  const AT_N_vec & sp_V_B,
  const int k, const int j,
  const int il, const int iu)
{
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sc_dot_sp_(i) = sp_V_A(0,k,j,i) * sp_V_B(0,k,j,i);
  }

  for (int a=1; a<N; ++a)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sc_dot_sp_(i) += sp_V_A(a,k,j,i) * sp_V_B(a,k,j,i);
  }
}

inline Real sc_dot_dense_sp__(
  const AT_N_vec & sp_V_A,
  const AT_N_vec & sp_V_B,
  const int k, const int j, const int i)
{
  Real dot (0);
  for (int a=0; a<N; ++a)
  {
    dot += sp_V_A(a,k,j,i) * sp_V_B(a,k,j,i);
  }
  return dot;
}

inline Real sc_ddot_dense_sp__(
  const AT_N_vec & sp_V_u,
  const AT_N_sym & sp_S_dd,
  const int k, const int j, const int i)
{
  Real dot (0);
  for (int a=0; a<N; ++a)
  for (int b=0; b<N; ++b)
  {
    dot += sp_V_u(a,k,j,i) * sp_V_u(b,k,j,i) * sp_S_dd(a,b,k,j,i);
  }
  return dot;
}

// geometric ------------------------------------------------------------------
inline void sp_beta_d(
  AT_N_vec & sp_tar,
  const AT_N_vec & sp_beta_u,
  const AT_N_sym & sp_g_dd,
  M1::vars_Scratch & scratch,
  const int k, const int j,
  const int il, const int iu)
{
  LinearAlgebra::VecMetContraction(
    scratch.sp_vec_A_,
    sp_beta_u,
    sp_g_dd,
    k, j,
    il, iu);

  for (int b=0; b<N; ++b)  // spatial ranges
  for (int i=il; i<=iu; ++i)
  {
    sp_tar(b,k,j,i) = scratch.sp_vec_A_(b,i);
  }

}

// Projections between frames -------------------------------------------------

// General projection (sc_E, sp_F_d, sp_P_dd) -> J
/*
inline Real sc_J__(
  const Real & W2,
  const Real & dotFv,  // F_i v^i
  const AT_C_sca & sc_E,
  const AT_N_vec & sp_v_u,
  const AT_N_sym & sp_P_dd,
  const Real floor_J,
  const int k, const int j, const int i)
{
  Real J = sc_E(k,j,i) - 2.0 * dotFv;
  J += LinearAlgebra::InnerProductVecSym2(
        sp_v_u, sp_P_dd, k, j, i);
  return W2 * std::max(J, floor_J);
}


inline Real sc_J_tk__(
  const Real & W2,
  const Real & dotFv,  // F_i v^i
  const AT_C_sca & sc_E,
  const Real floor_J,
  const int k, const int j, const int i)
{
  const Real J__ = 3.0 / (2.0 * W2 + 1) * (
    (2.0 * W2 - 1) * sc_E(k,j,i) -
    2.0 * W2 * dotFv
  );
  return std::max(J__, floor_J);
}

inline Real sc_H_tk_d_n__(
  const Real & W,
  const Real & dotFv,  // F_d v^d
  const AT_C_sca & sc_E,
  const AT_C_sca & sc_J,
  const int k, const int j, const int i)
{
  return W * (sc_E(k,j,i) - sc_J(k,j,i) - dotFv);
}

inline void sp_H_d__(
  AT_N_vec & tar_d,
  const Real & W,
  const AT_C_sca & sc_J,
  const AT_N_vec & sp_F_d,
  const AT_N_vec & sp_v_d,
  const AT_N_vec & sp_v_u,
  const AT_N_sym & sp_P_dd,
  const int k, const int j, const int i)
{
  for (int a=0; a<N; ++a)
  {
    tar_d(a,k,j,i) = (
      sp_F_d(a,k,j,i) - sc_J(k,j,i) * sp_v_d(a,k,j,i)
    );

    for (int b=0; b<N; ++b)
    {
      tar_d(a,k,j,i) -= (
        sp_v_u(b,k,j,i) * sp_P_dd(a,b,k,j,i)
      );
    }

    tar_d(a,k,j,i) = W * tar_d(a,k,j,i);
  }
}
*/

// ----------------------------------------------------------------------------

inline void sp_norm2_(
  AT_C_sca & sp_tar_,
  const AT_N_vec & sp_V_d,  // alternative: _u &
  const AT_N_sym & sp_g_uu, // _dd
  const int k, const int j,
  const int il, const int iu)
{
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sp_tar_(i) = LinearAlgebra::InnerProductVecMetric(
      sp_V_d, sp_g_uu, k, j, i
    );
  }
}

inline Real sp_norm2__(
  const AT_N_vec & sp_V_d,  // alternative: _u &
  const AT_N_sym & sp_g_uu, // _dd
  const int k, const int j, const int i)
{
  return LinearAlgebra::InnerProductVecMetric(
      sp_V_d, sp_g_uu, k, j, i
  );
}

// ============================================================================
// Convenience methods
inline void sp_d_to_u_(
  M1 * pm1,
  AT_N_vec & sp_tar_u_,
  const AT_N_vec & sp_src_d,
  const int k, const int j,
  const int il, const int iu)
{
  LinearAlgebra::VecMetContraction(
    sp_tar_u_,
    sp_src_d,
    pm1->geom.sp_g_uu,
    k, j,
    il, iu);
}

inline void sp_dd_to_uu_(
  M1 * pm1,
  AT_N_sym & sp_tar_uu_,
  const AT_N_sym & sp_src_dd_,
  const int k, const int j,
  const int il, const int iu)
{
  const AT_N_sym & sp_g_uu = pm1->geom.sp_g_uu;

  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sp_tar_uu_(a,b,i) = 0;
  }

  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  for (int c=0; c<N; ++c)
  for (int d=0; d<N; ++d)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sp_tar_uu_(a,b,i) += sp_g_uu(a,c,k,j,i) *
                         sp_g_uu(b,d,k,j,i) *
                         sp_src_dd_(c,d,i);
  }
}

inline void sc_norm_sp_H_(
  M1 * pm1,
  AT_C_sca & sc_norm_sp_H_,
  const AT_N_vec & sp_H_d,
  const int k, const int j,
  const int il, const int iu)
{
  sp_norm2_(sc_norm_sp_H_,
            sp_H_d,
            pm1->geom.sp_g_uu,
            k, j, il, iu);
}

inline void sc_norm_sp_(
  M1 * pm1,
  AT_C_sca & sc_norm_sp_,
  const AT_N_vec & sp_V_u_,
  const AT_N_vec & sp_V_d,
  const int k, const int j,
  const int il, const int iu)
{
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sc_norm_sp_(i) = 0.0;
  }

  for (int a=0; a<N; ++a)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sc_norm_sp_(i) += sp_V_d(a,k,j,i) * sp_V_u_(a,i);
  }
}

inline void PointToDense(
  AT_N_sym & sp_tar_aa,
  AT_N_sym & sp_S_aa_,
  const int k, const int j, const int i)
{
  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  {
    sp_tar_aa(a,b,k,j,i) = sp_S_aa_(a,b,i);
  }
}

inline void PointAddToDense(
  AT_N_sym & sp_tar_aa,
  const AT_N_sym & sp_S_aa_,
  const int k, const int j, const int i)
{
  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  {
    sp_tar_aa(a,b,k,j,i) += sp_S_aa_(a,b,i);
  }
}

inline void PointArrayToAthenaTensor(
  AT_C_sca & sc_tar,
  const std::array<Real, 1> & src,
  const int k, const int j, const int i)
{
  sc_tar(k,j,i) = src[0];
}

inline void PointArrayToAthenaTensor(
  AT_N_vec & sp_tar,
  const std::array<Real, N> & src,
  const int k, const int j, const int i)
{
  for (int a=0; a<N; ++a)
  {
    sp_tar(a,k,j,i) = src[a];
  }
}

inline void PointAthenaTensorToArray(
  std::array<Real, 1> & src,
  const AT_C_sca & sc_tar,
  const int k, const int j, const int i)
{
  src[0] = sc_tar(k,j,i);
}

inline void PointAthenaTensorToArray(
  std::array<Real, N> & src,
  const AT_N_vec & sp_tar,
  const int k, const int j, const int i)
{
  for (int a=0; a<N; ++a)
  {
    src[a] = sp_tar(a,k,j,i);
  }
}

inline void CopyDenseToScratch(
  AT_N_vec & sp_tar_a_,
  const AT_N_vec & sp_S_a,
  const int k, const int j,
  const int il, const int iu)
{
  for (int a=0; a<N; ++a)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sp_tar_a_(a,i) = sp_S_a(a,k,j,i);
  }
}

inline void CopyScratchToDense(
  AT_N_vec & sp_tar_a,
  const AT_N_vec & sp_S_a_,
  const int k, const int j,
  const int il, const int iu)
{
  for (int a=0; a<N; ++a)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sp_tar_a(a,k,j,i) = sp_S_a_(a,i);
  }
}

inline void CopyDenseToScratch(
  AT_N_sca & st_tar_,
  const AT_N_sca & st_src,
  const int k, const int j,
  const int il, const int iu)
{
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_(i) = st_src(k,j,i);
  }
}

inline void CopyScratchToDense(
  AT_C_sca & st_tar,
  const AT_C_sca & st_src_,
  const int k, const int j,
  const int il, const int iu)
{
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar(k,j,i) = st_src_(i);
  }
}

inline void CopyScratchToDense(
  AT_N_sym & sp_tar_a,
  const AT_N_sym & sp_S_a_,
  const int k, const int j,
  const int il, const int iu)
{
  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sp_tar_a(a,b,k,j,i) = sp_S_a_(a,b,i);
  }
}

// ============================================================================
namespace Frames {
// ============================================================================

inline Real d_th(const AT_C_sca & sc_chi,
                 const int k, const int j, const int i)
{
  return 0.5 * (3.0 * sc_chi(k,j,i) - 1.0);
}

inline Real d_tk(const AT_C_sca & sc_chi,
                 const int k, const int j, const int i)
{
  return 1.0 - d_th(sc_chi, k, j, i);
}

// We write:
// sc_J = J_0
// st_H = H_n n^alpha + H_v v^alpha + H_F F^alpha
inline void ToFiducialExpansionCoefficients(
  M1 & pm1,
  Real & J_0,
  Real & H_n,
  Real & H_v,
  Real & H_F,
  const AT_C_sca & sc_chi,
  const AT_C_sca & sc_E,
  const AT_N_vec & sp_F_d,
  const int k,
  const int j,
  const int i)
{
  const AT_N_vec & sp_v_u = pm1.fidu.sp_v_u;
  const AT_N_vec & sp_v_d = pm1.fidu.sp_v_d;
  const AT_N_sym & sp_g_uu = pm1.geom.sp_g_uu;

  const Real W  = pm1.fidu.sc_W(k,j,i);
  const Real W2 = SQR(W);

  const Real E = sc_E(k,j,i);
  const Real dotFv = sc_dot_dense_sp__(sp_F_d, sp_v_u, k, j, i);
  const Real dotvv = sc_dot_dense_sp__(sp_v_d, sp_v_u, k, j, i);

  const Real nF2 = sp_norm2__(sp_F_d, sp_g_uu, k, j, i);
  const Real oo_nF = (nF2 > pm1.opt.fl_nF2) ? OO(std::sqrt(nF2)) : 0.0;
  const Real dotFhatv = oo_nF * dotFv;

  const Real d_th = Frames::d_th(sc_chi, k, j, i);
  const Real d_tk = Frames::d_tk(sc_chi, k, j, i);

  // ------------------------------------------------------------------------
  const Real B_0 = W2 * (E - 2.0 * dotFv);
  const Real B_th = W2 * E * SQR(dotFhatv);
  const Real B_tk = (W2 - 1.0) / (2.0 * W2 + 1.0) * (
    4.0 * W2 * dotFv + (3.0 - 2.0 * W2) * E
  );

  // coefficients appearing in vector-expansion of H
  // const Real n_0  = W * B_0 + W * (dotFv - E);
  // const Real n_th = W * B_th;
  // const Real n_tk = W * B_tk;

  const Real v_0  = W * B_0;
  const Real v_th = W * B_th;
  const Real v_tk = W * B_tk + W / (2.0 * W2 + 1.0) * (
    (3.0 - 2.0 * W2) * E + (2.0 * W2 - 1.0) * dotFv
  );

  const Real F_0 = -W;
  const Real F_th = oo_nF * W * E * dotFhatv;
  const Real F_tk = W * dotvv;

  // const Real n_H = (n_0 + d_th * n_th + d_tk * n_tk);
  const Real v_H = (v_0 + d_th * v_th + d_tk * v_tk);
  const Real F_H = (F_0 + d_th * F_th + d_tk * F_tk);

  // --------------------------------------------------------------------------
  // Populate according to comment
  // J_0 = B_0 + d_th * B_th + d_tk * B_tk;
  J_0 = std::max(
    B_0 + d_th * B_th + d_tk * B_tk,
    pm1.opt.fl_J
  );

  H_v = -v_H;
  H_F = -F_H;

  // Could do the following
  // Need additional sign: H^mu n_mu = -A in H^mu = A n^mu + B ...
  // H_n = -n_H;

  // (from H \perp u)
  H_n = H_F * dotFv + H_v * dotvv;

  // --------------------------------------------------------------------------
  // Check conventions correct (need H \perp u)
  // u^a = W * (n^a + v^a)
  /*
  if (0)
  {
    AT_N_vec & sp_F_u_ = pm1.scratch.sp_vec_A_;
    Assemble::sp_d_to_u_(&pm1, sp_F_u_, sp_F_d, k, j, i, i);

    const Real alpha = pm1.geom.sc_alpha(k,j,i);
    const Real oo_alpha = OO(alpha);

    // u_0 H^0
    Real u_d_0 = -alpha * W;
    for (int a=0; a<N; ++a)
    {
      u_d_0 += W * pm1.geom.sp_beta_u(a,k,j,i) * sp_v_d(a,k,j,i);
    }

    const Real H_u_0 = H_n * oo_alpha;

    Real dotHu = u_d_0 * H_u_0;
    for (int a=0; a<N; ++a)
    {
      const Real u_d_i = W * sp_v_d(a,k,j,i);
      const Real H_u_i = (
        H_n * (-oo_alpha * pm1.geom.sp_beta_u(a,k,j,i)) +
        H_v * sp_v_u(a,k,j,i) +
        H_F * sp_F_u_(a,i)
      );

      dotHu += u_d_i * H_u_i;
    }

    if (std::abs(dotHu) > 1e-14)
      std::printf("%.3g\n", dotHu);

  }
  */

}

inline void ToFiducial(
  M1 & pm1,
  AT_C_sca & sc_J,
  AT_D_vec & st_H_u,
  const AT_C_sca & sc_chi,
  const AT_C_sca & sc_E,
  const AT_N_vec & sp_F_d,
  const int k, const int j,
  const int il, const int iu)
{
  // typedef M1::opt_closure_variety ocv;
  // ocv ocv_cur = pm1.opt_closure.variety;

  const AT_N_vec & sp_v_u = pm1.fidu.sp_v_u;
  const AT_N_vec & sp_v_d = pm1.fidu.sp_v_d;
  const AT_N_sym & sp_g_uu = pm1.geom.sp_g_uu;
  const AT_N_vec & sp_beta_u = pm1.geom.sp_beta_u;

  for (int i=il; i<=iu; ++i)
  {

    Real J_0, H_n, H_v, H_F;

    ToFiducialExpansionCoefficients(
      pm1,
      J_0, H_n, H_v, H_F,
      sc_chi, sc_E, sp_F_d,
      k, j, i
    );

    // populate ---------------------------------------------------------------
    sc_J(k,j,i) = J_0;

    // n^0 = 1 / alpha
    // v^0 = 0 (spatial)
    // F^0 = 0 (spatial)

    const Real alpha = pm1.geom.sc_alpha(k,j,i);
    const Real oo_alpha = OO(alpha);

    st_H_u(0,k,j,i) = H_n * oo_alpha;

    for (int a=0; a<N; ++a)
    {
      Real F_u (0);
      for (int b=0; b<N; ++b)
      {
        F_u += sp_g_uu(a,b,k,j,i) * sp_F_d(b,k,j,i);
      }

      // n^i = -beta^i / alpha
      st_H_u(a+1,k,j,i) = (
        -H_n * oo_alpha * sp_beta_u(a,k,j,i) +
        H_v * sp_v_u(a,k,j,i) +
        H_F * F_u
      );
    }
  }
}

// Populate fiducial frame at a point
//
// Based on choice in polymorphism:
// (sc_J, st_H_u) are updated at (k,j,i)
// (sc_J, st_H_u, sc_n) are updated at (k,j,i)
//
// returns Gamma conversion factor
inline Real ToFiducial(
  M1 & pm1,
  AT_C_sca & sc_J,
  AT_D_vec & st_H_u,
  const AT_C_sca & sc_chi,
  const AT_C_sca & sc_E,
  const AT_N_vec & sp_F_d,
  const int k, const int j, const int i)
{
  // typedef M1::opt_closure_variety ocv;
  // ocv ocv_cur = pm1.opt_closure.variety;

  const AT_N_vec & sp_v_u = pm1.fidu.sp_v_u;
  const AT_N_vec & sp_v_d = pm1.fidu.sp_v_d;
  const AT_N_sym & sp_g_uu = pm1.geom.sp_g_uu;
  const AT_N_vec & sp_beta_u = pm1.geom.sp_beta_u;

  Real J_0, H_n, H_v, H_F;

  ToFiducialExpansionCoefficients(
    pm1,
    J_0, H_n, H_v, H_F,
    sc_chi, sc_E, sp_F_d,
    k, j, i
  );

  // populate -----------------------------------------------------------------
  sc_J(k,j,i) = J_0;

  // n^0 = 1 / alpha
  // v^0 = 0 (spatial)
  // F^0 = 0 (spatial)

  const Real alpha = pm1.geom.sc_alpha(k,j,i);
  const Real oo_alpha = OO(alpha);

  st_H_u(0,k,j,i) = H_n * oo_alpha;

  for (int a=0; a<N; ++a)
  {
    Real F_u (0);
    for (int b=0; b<N; ++b)
    {
      F_u += sp_g_uu(a,b,k,j,i) * sp_F_d(b,k,j,i);
    }

    // n^i = -beta^i / alpha
    st_H_u(a+1,k,j,i) = (
      -H_n * oo_alpha * sp_beta_u(a,k,j,i) +
      H_v * sp_v_u(a,k,j,i) +
      H_F * F_u
    );
  }

  // ------------------------------------------------------------------------
  // Check conventions correct (need H \perp u)
  // u^a = W * (n^a + v^a)
  /*
  if (0)
  {
    const Real W  = pm1.fidu.sc_W(k,j,i);

    // u_0 H^0
    Real u_d_0 = -alpha * W;
    for (int a=0; a<N; ++a)
    {
      u_d_0 += W * pm1.geom.sp_beta_u(a,k,j,i) * sp_v_d(a,k,j,i);
    }

    Real dotHu = u_d_0 * st_H_u(0,k,j,i);
    for (int a=0; a<N; ++a)
    {
      const Real u_d_i = W * sp_v_d(a,k,j,i);
      const Real H_u_i = st_H_u(1+a,k,j,i);

      dotHu += u_d_i * H_u_i;
    }

    if (std::abs(dotHu) > 1e-14)
      std::printf("%.3g\n", dotHu);
  }
  */


  // Gamma factor -------------------------------------------------------------
  // auxiliary elements
  AT_C_sca & sc_W   = pm1.fidu.sc_W;

  const Real W = sc_W(k,j,i);

  // The following two are equivalent, however, floors will potentially affect
  // results!
  const Real sc_Gam__ = (J_0 > pm1.opt.fl_J)
    ? W * (1.0 + H_n / (J_0 * W))
    : W;

  // const Real sc_Gam__ = (
  //   (sc_E(k,j,i) > pm1.opt.fl_E) &&
  //   (J_0 > pm1.opt.fl_J)
  // ) ? W * (sc_E(k,j,i) / J_0) * std::min(
  //       1.0 - sc_dot_dense_sp__(sp_F_d, sp_v_u, k, j, i),
  //       1.0 - pm1.opt.eps_E
  //     )
  //   : W;

  return sc_Gam__;
}

inline Real ToFiducial(
  M1 & pm1,
  AT_C_sca & sc_J,
  AT_D_vec & st_H_u,
  AT_C_sca & sc_n,
  const AT_C_sca & sc_chi,
  const AT_C_sca & sc_E,
  const AT_N_vec & sp_F_d,
  const AT_C_sca & sc_nG,
  const int k, const int j, const int i)
{
  const Real sc_Gam__ = ToFiducial(
    pm1,
    sc_J, st_H_u,
    sc_chi, sc_E, sp_F_d,
    k, j, i
  );

  /*
  AT_C_sca & sc_nG_ = const_cast<AT_C_sca &>(sc_nG);
  sc_nG_(k,j,i) = std::max(pm1.opt.fl_nG, sc_nG(k,j,i));
  */
  sc_n(k,j,i) = sc_nG(k,j,i) / sc_Gam__;

  return sc_Gam__;
}

inline Real sc_Gam__(
  M1 & pm1,
  AT_C_sca & sc_J,
  AT_D_vec & st_H_u,
  const int k, const int j, const int i
)
{
  Real ret_sc_Gam__ = 1.0;
  const Real alpha = pm1.geom.sc_alpha(k,j,i);

  AT_C_sca & sc_W   = pm1.fidu.sc_W;

  const Real W = sc_W(k,j,i);

  const Real J_0 = sc_J(k,j,i);
  const Real H_n = st_H_u(0,k,j,i) * alpha;

  ret_sc_Gam__ = (J_0 > pm1.opt.fl_J)
    ? W * (1.0 + H_n / (J_0 * W))
    : W;
  return ret_sc_Gam__;
}

inline Real sp_f_u__(
  M1 & pm1,
  AT_C_sca & sc_J,
  AT_D_vec & st_H_u,
  const int dir,
  const int k, const int j, const int i)
{
  const AT_N_sca & sc_W   = pm1.fidu.sc_W;
  const AT_N_vec & sp_v_u = pm1.fidu.sp_v_u;

  const AT_N_sca & sc_alpha  = pm1.geom.sc_alpha;
  const AT_N_vec & sp_beta_u = pm1.geom.sp_beta_u;

  return (
    sc_W(k,j,i) * (sp_v_u(dir,k,j,i) - sp_beta_u(dir,k,j,i) / sc_alpha(k,j,i))
    + ((sc_J(k,j,i) > pm1.opt.fl_J) ? st_H_u(1+dir,k,j,i) / sc_J(k,j,i) : 0.0)
  );

  // return (
  //   sc_W(k,j,i) * (sp_v_u(dir,k,j,i) - sp_beta_u(dir,k,j,i) / sc_alpha(k,j,i))
  //   + ((sc_J(k,j,i) > 0) ? st_H_u(1+dir,k,j,i) / sc_J(k,j,i) : 0.0)
  // );

}

template <bool overwrite_slice>
inline void sp_P_th_dd_(
  M1 & pm1,
  AT_N_sym & sp_P_th_dd_,
  const AT_C_sca & sc_E,
  const AT_N_vec & sp_F_d,
  const int k, const int j,
  const int il, const int iu)
{

  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    const Real nF2 = Assemble::sp_norm2__(sp_F_d, pm1.geom.sp_g_uu, k, j, i);
    const Real fac = (nF2 > pm1.opt.fl_nF2) ? sc_E(k,j,i) / nF2 : 0.0;

    for (int a=0; a<N; ++a)
    for (int b=a; b<N; ++b)
    {
      const Real res_dd__ = fac * sp_F_d(a,k,j,i) * sp_F_d(b,k,j,i);
      if (overwrite_slice)
      {
        sp_P_th_dd_(a,b,i) = res_dd__;
      }
      else
      {
        sp_P_th_dd_(a,b,i) += res_dd__;
      }
    }
  }
}

template <bool overwrite_slice>
inline void sp_P_tk_dd_(
  M1 & pm1,
  AT_N_sym & sp_P_tk_dd_,
  const AT_C_sca & sc_E,
  const AT_N_vec & sp_F_d,
  const int k, const int j,
  const int il, const int iu)
{
  AT_C_sca & sc_W   = pm1.fidu.sc_W;
  AT_N_vec & sp_v_d = pm1.fidu.sp_v_d;
  AT_N_vec & sp_v_u = pm1.fidu.sp_v_u;
  AT_N_sym & sp_g_dd = pm1.geom.sp_g_dd;

  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    const Real W    = sc_W(k,j,i);
    const Real oo_W = 1.0 / W;
    const Real W2   = SQR(W);

    const Real dotFv = Assemble::sc_dot_dense_sp__(sp_F_d,
                                                   sp_v_u,
                                                   k, j, i);

    Real J_tk = 3.0 / (2.0 * W2 + 1.0) * (
      (2.0 * W2 - 1.0) * sc_E(k,j,i) - 2.0 * W2 * dotFv
    );

    const Real fac_H_tk = W /  (2.0 * W2 + 1.0) * (
      (4.0 * W2 + 1.0) * dotFv - 4.0 * W2 * sc_E(k,j,i)
    );

    for (int a=0; a<N; ++a)
    {
      const Real H_a_tk = oo_W * sp_F_d(a,k,j,i) +
                          fac_H_tk * sp_v_d(a,k,j,i);

      for (int b=a; b<N; ++b)
      {
        const Real H_b_tk = oo_W * sp_F_d(b,k,j,i) +
                            fac_H_tk * sp_v_d(b,k,j,i);

        const Real res_dd__ = (
          4.0 * ONE_3RD * W2 * J_tk * sp_v_d(a,k,j,i) * sp_v_d(b,k,j,i) +
          W * (sp_v_d(a,k,j,i) * H_b_tk + sp_v_d(b,k,j,i) * H_a_tk) +
          ONE_3RD * J_tk * sp_g_dd(a,b,k,j,i)
        );

        if (overwrite_slice)
        {
          sp_P_tk_dd_(a,b,i) = res_dd__;
        }
        else
        {
          sp_P_tk_dd_(a,b,i) += res_dd__;
        }
      }

    }
  }
}

inline void sp_P_dd_(
  M1 & pm1,
  AT_N_sym & sp_P_dd_,
  const AT_C_sca & sc_chi,
  const AT_C_sca & sc_E,
  const AT_N_vec & sp_F_d,
  const int k, const int j,
  const int il, const int iu)
{
  // prepare scratches
  AT_N_sym & sp_P_th_dd_ = pm1.scratch.sp_P_th_dd_;
  AT_N_sym & sp_P_tk_dd_ = pm1.scratch.sp_P_tk_dd_;

  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sp_P_dd_(a,b,i) = 0;
  }

  for (int i=il; i<=iu; ++i)
  {
    // chi \in [1/3, 1];
    // 1/3 is thick limit, 1 is thin limit

    const Real chi__ = sc_chi(k,j,i);
    const bool tk_cpt = chi__ < 1;
    const bool th_cpt = chi__ > ONE_3RD;

    const Real d_th__ = th_cpt ? d_th(sc_chi, k, j, i) : 0.0;
    const Real d_tk__ = tk_cpt ? d_tk(sc_chi, k, j, i) : 0.0;

    if (th_cpt)
    {
      Assemble::Frames::sp_P_th_dd_<true>(
        pm1, sp_P_th_dd_, sc_E, sp_F_d, k, j, i, i
      );

      for (int a=0; a<N; ++a)
      for (int b=a; b<N; ++b)
      {
        sp_P_dd_(a,b,i) += d_th__ * sp_P_th_dd_(a,b,i);
      }
    }

    if (tk_cpt)
    {
      Assemble::Frames::sp_P_tk_dd_<true>(
        pm1, sp_P_tk_dd_, sc_E, sp_F_d, k, j, i, i
      );

      for (int a=0; a<N; ++a)
      for (int b=a; b<N; ++b)
      {
        sp_P_dd_(a,b,i) += d_tk__ * sp_P_tk_dd_(a,b,i);
      }
    }
  }
}

inline void sources_sc_E_sp_F_d(
  M1 & pm1,
  AT_C_sca & S_sc_E,
  AT_N_vec & S_sp_F_d,
  const AT_C_sca & sc_chi,
  const AT_C_sca & sc_E,
  const AT_N_vec & sp_F_d,
  const AT_C_sca & sc_eta,
  const AT_C_sca & sc_kap_a,
  const AT_C_sca & sc_kap_s,
  const int k, const int j, const int i)
{
  // Required quantities
  const Real W = pm1.fidu.sc_W(k,j,i);
  const Real eta = sc_eta(k,j,i);
  const Real kap_a = sc_kap_a(k,j,i);
  const Real kap_s = sc_kap_s(k,j,i);
  const Real kap_as = kap_a + sc_kap_s(k,j,i);

  const Real alpha    = pm1.geom.sc_alpha(k,j,i);
  const Real oo_alpha = OO(alpha);
  const Real sqrt_det_g = pm1.geom.sc_sqrt_det_g(k,j,i);

  const AT_N_vec & sp_v_d = pm1.fidu.sp_v_d;
  const AT_N_vec & sp_v_u = pm1.fidu.sp_v_u;
  const AT_N_vec & sp_beta_d = pm1.geom.sp_beta_d;

  // Prepare H^alpha

  // We write:
  // sc_J = J_0
  // st_H = H_n n^alpha + H_v v^alpha + H_F F^alpha
  Real J_0, H_n, H_v, H_F;

  Assemble::Frames::ToFiducialExpansionCoefficients(
    pm1,
    J_0, H_n, H_v, H_F,
    sc_chi, sc_E, sp_F_d,
    k, j, i
  );

  // The following two are equivalent, though, floors may affect near-zero
  // regions.
  S_sc_E(k,j,i) = -alpha * (
    H_n * kap_as +
    (J_0 * kap_a - eta * sqrt_det_g) * W
  );

  // const Real dotFv = Assemble::sc_dot_dense_sp__(sp_F_d, sp_v_u, k, j, i);

  // S_sc_E(k,j,i) = alpha * (
  //   W * (eta * sqrt_det_g + kap_s * J_0 -
  //        kap_as * (sc_E(k,j,i)) - dotFv)
  // );

  for (int a=0; a<N; ++a)
  {
    // S_sp_F_d(a,k,j,i) = alpha * (
    //   (sqrt_det_g * eta - kap_a * J_0) * W * sp_v_d(a,k,j,i) -
    //   kap_as * (// -H_n * oo_alpha * sp_beta_d(a,k,j,i)
    //             +H_v * sp_v_d(a,k,j,i)
    //             +H_F * sp_F_d(a,k,j,i) )
    // );

    S_sp_F_d(a,k,j,i) = -alpha * (
      sp_F_d(a,k,j,i) * H_F * kap_as +
      sp_v_d(a,k,j,i) * (
        H_v * kap_as +
        (J_0 * kap_a - eta * sqrt_det_g) * W
      )
    );
  }
}

inline void sources_sc_nG(
  M1 & pm1,
  AT_C_sca & S_sc_nG,
  const AT_C_sca & sc_n,
  const AT_C_sca & sc_eta_0,
  const AT_C_sca & sc_kap_a_0,
  const int k, const int j, const int i)
{
  S_sc_nG(k,j,i) = pm1.geom.sc_alpha(k,j,i) * (
    pm1.geom.sc_sqrt_det_g(k,j,i) * sc_eta_0(k,j,i) -
    sc_kap_a_0(k,j,i) * sc_n(k,j,i)
  );
}

inline void Jacobian_sc_E_sp_F_d(
  M1 & pm1,
  AA & J,                               // Storage for Jacobian
  const AT_C_sca & sc_chi,
  const AT_C_sca & sc_E,
  const AT_N_vec & sp_F_d,
  const AT_C_sca & sc_kap_a,
  const AT_C_sca & sc_kap_s,
  const int k, const int j, const int i)
{
  // New implementation -------------------------------------------------------
  // J(a,b) := dSc_a / dX_b where X=(E, F_i)
  // [Sc_n, Sc_i] = alpha * [-n_a S^a, gamma_{ib} S^b]

  // scratch quantities
  AT_C_sca & sc_dJ_dE_ = pm1.scratch.sc_dJ_dE_;
  AT_N_vec & sp_dJ_dF_d_ = pm1.scratch.sp_dJ_dF_d_;
  AT_N_vec & sp_dH_d_dE_ = pm1.scratch.sp_dH_d_dE_;
  AT_N_bil & sp_dH_d_dF_d_ = pm1.scratch.sp_dH_d_dF_d_;
  AT_N_vec & sp_F_u_ = pm1.scratch.sp_F_u_;

  // other quantities
  const Real d_th = Assemble::Frames::d_th(sc_chi, k, j, i);
  const Real d_tk = Assemble::Frames::d_tk(sc_chi, k, j, i);

  const Real kap_a = sc_kap_a(k,j,i);
  const Real kap_s = sc_kap_s(k,j,i);
  const Real kap_as = (kap_a + kap_s);

  const Real alpha = pm1.geom.sc_alpha(k,j,i);
  const AT_N_sym & sp_g_uu = pm1.geom.sp_g_uu;

  const AT_N_vec & sp_v_u = pm1.fidu.sp_v_u;
  const AT_N_vec & sp_v_d = pm1.fidu.sp_v_d;

  const Real W = pm1.fidu.sc_W(k,j,i);
  const Real W2 = SQR(W);
  const Real W3 = W * W2;

  const Real E = sc_E(k,j,i);
  const Real dotFv = Assemble::sc_dot_dense_sp__(sp_F_d, sp_v_u, k, j, i);
  const Real dotvv = Assemble::sc_dot_dense_sp__(sp_v_d, sp_v_u, k, j, i);
  Assemble::sp_d_to_u_(&pm1, sp_F_u_, sp_F_d, k, j, i, i);

  const Real nF2 = Assemble::sp_norm2__(sp_F_d, sp_g_uu, k, j, i);

  const Real oo_nF2 = (nF2 > pm1.opt.fl_nF2) ? OO(nF2) : 0.0;
  const Real oo_nF  = (nF2 > pm1.opt.fl_nF2) ? std::sqrt(oo_nF2) : 0.0;
  const Real dotFhatv = oo_nF * dotFv;

  // derivative terms ---------------------------------------------------------
  sc_dJ_dE_(i) = (
    W2 * (1.0 + d_th * SQR(dotFhatv)) +
    d_tk * (3.0 - 2.0 * W2) * (W2 - 1.0) / (1.0 + 2.0 * W2)
  );

  // sp_dJ_dF_d_ factors
  const Real fac_a_sp_dJ_dF_d_ = 2.0 * W2 * (
      -1.0 + d_th * E * oo_nF * dotFhatv +
      2.0 * d_tk * (W2 - 1.0) / (1.0 + 2 * W2)
  );
  const Real fac_b_sp_dJ_dF_d_ = -2.0 * d_th * (
    W2 * E * oo_nF * SQR(dotFhatv)
  );

  // sp_dH_d_dE_ factors
  const Real fac_a_sp_dH_d_dE_ = W3 * (
    -1.0 - d_th * SQR(dotFhatv) + d_tk * (2.0 * W2 - 3.0) / (1.0 + 2.0 * W2)
  );

  const Real fac_b_sp_dH_d_dE_ = (
    -d_th * W * dotFhatv
  );

  for (int a=0; a<N; ++a)
  {
    sp_dJ_dF_d_(a,i) = (
      fac_a_sp_dJ_dF_d_ * sp_v_u(a,k,j,i) +
      fac_b_sp_dJ_dF_d_ * (oo_nF * sp_F_u_(a,i))
    );

    sp_dH_d_dE_(a,i) = (
      fac_a_sp_dH_d_dE_ * sp_v_d(a,k,j,i) +
      fac_b_sp_dH_d_dE_ * (oo_nF * sp_F_d(a,k,j,i))
    );
  }

  // sp_dH_d_dF_d_ factors
  const Real fac_a_sp_dH_d_dF_d_ = W * (
    1.0 - d_th * E * oo_nF * dotFhatv - d_tk * dotvv
  );

  const Real fac_b_sp_dH_d_dF_d_ = 2.0 * W3 * (
    1.0 - d_th * E * oo_nF * dotFhatv - d_tk * (
      dotvv + OO(2.0 * W2 * (1.0 + 2.0 * W2))
    )
  );

  const Real fac_c_sp_dH_d_dF_d_ = 2.0 * d_th * W * E *oo_nF * dotFhatv;

  const Real fac_d_sp_dH_d_dF_d_ = (
    2.0 * d_th * W3 * E * oo_nF * SQR(dotFhatv)
  );

  const Real fac_e_sp_dH_d_dF_d_ = (
    -d_th * W * E * oo_nF
  );

  for (int a=0; a<N; ++a)
  for (int b=0; b<N; ++b)
  {
    sp_dH_d_dF_d_(a,b,i) = (
      fac_a_sp_dH_d_dF_d_ * (a == b) +
      fac_b_sp_dH_d_dF_d_ * sp_v_d(a,k,j,i) * sp_v_u(b,k,j,i) +
      fac_c_sp_dH_d_dF_d_ * (oo_nF2 * sp_F_d(a,k,j,i) * sp_F_u_(b,i)) +
      fac_d_sp_dH_d_dF_d_ * sp_v_d(a,k,j,i) * (oo_nF * sp_F_u_(b,i)) +
      fac_e_sp_dH_d_dF_d_ * (oo_nF * sp_F_d(a,k,j,i)) * sp_v_u(b,k,j,i)
    );
  }

  // populate Jacobian --------------------------------------------------------
  J(0,0) = -alpha * W * (kap_as - kap_s * sc_dJ_dE_(i));

  for (int b=0; b<N; ++b)
  {
    J(0,1+b) = alpha * W * (kap_s * sp_dJ_dF_d_(b,i) +
                            kap_as * sp_v_u(b,k,j,i));

    J(1+b,0) = -alpha * (kap_as * sp_dH_d_dE_(b,i) +
                          W * kap_a * sc_dJ_dE_(i) * sp_v_d(b,k,j,i));
  }

  for (int a=0; a<N; ++a)
  for (int b=0; b<N; ++b)
  {
    J(1+a,1+b) = -alpha * (
      kap_as * sp_dH_d_dF_d_(a,b,i) +
      W * kap_a * sp_v_d(a,k,j,i) * sp_dJ_dF_d_(b,i)
    );
  }

  // [Debug] static fluid: ----------------------------------------------------
  if (0)
  {
    const Real kap_a = sc_kap_a(k,j,i);
    const Real kap_s = sc_kap_s(k,j,i);
    const Real kap_as = (kap_a + kap_s);

    const Real alpha = pm1.geom.sc_alpha(k,j,i);

    J(0,0) = -alpha * kap_a;
    for (int a=0; a<N; ++a)
    for (int b=0; b<N; ++b)
    {
      J(a+1,b+1) = -alpha * kap_as * (a==b);
    }
  }
  // --------------------------------------------------------------------------

  /*
  // Old implementation -------------------------------------------------------
  const Real W = pm1.fidu.sc_W(k,j,i);
  const Real W2 = SQR(W);
  const Real W3 = W * W2;

  const Real kap_a = sc_kap_a(k,j,i);
  const Real kap_s = sc_kap_s(k,j,i);
  const Real kap_as = (kap_a + kap_s);

  const Real alpha = pm1.geom.sc_alpha(k,j,i);

  // P_dd thick (tk) and thin (tn) factors
  const Real d_tk = 3.0 * 0.5 * (1.0 - sc_chi(k,j,i));
  const Real d_tn = 1.0 - d_tk;

  const Real dotFv = Assemble::sc_dot_dense_sp__(sp_F_d,
                                                 pm1.fidu.sp_v_u,
                                                 k, j, i);

  const Real dotvv = Assemble::sc_dot_dense_sp__(pm1.fidu.sp_v_d,
                                                 pm1.fidu.sp_v_u,
                                                 k, j, i);

  AT_N_vec & sp_F_u_ = pm1.scratch.sp_vec_A_;

  Assemble::sp_d_to_u_(&pm1, sp_F_u_, sp_F_d, k, j, i, i);
  Real dotFF = 0;

  for (int a=0; a<N; ++a)
  {
    dotFF += sp_F_u_(a,i) * sp_F_d(a,k,j,i);
  }

  // J(I,J) ~ D[S_I,(E,F_d)_J]

  // sc_Stil_1 ----------------------------------------------------------------

  // D_E
  J(0,0) = alpha * W * (
    -kap_as + kap_s * W2
  );

  const Real fac_J_0a = alpha * W * (
    kap_as - 2.0 * kap_s * W2
  );

  // D_F_d
  for (int a=0; a<N; ++a)
  {
    J(0,1+a) = fac_J_0a * pm1.fidu.sp_v_u(a,k,j,i);
  }

  // sc_Stil_1pa --------------------------------------------------------------
  for (int a=0; a<N; ++a)
  {
    // D_E
    J(1+a,0) = alpha * kap_s * W3 * pm1.fidu.sp_v_d(a,k,j,i);

    // D_F_d
    for (int b=0; b<N; ++b)
    {
      J(1+a,1+b) = -alpha * W * (
        (a == b) * kap_as +
        2.0 * kap_s * W2 * pm1.fidu.sp_v_u(b,k,j,i) * pm1.fidu.sp_v_d(a,k,j,i)
      );
    }
  }

  // thin correction to Jacobian ----------------------------------------------
  if ((dotFF > 0) && (d_tn > 0))
  {
    J(0,0) += d_tn * alpha * W3 * kap_s * SQR(dotFv) / dotFF;

    for (int b=0; b<N; ++b)
    {
      J(0,1+b) += d_tn * 2.0 * alpha * dotFv * sc_E(k,j,i) * kap_s * W3 * (
        -dotFv * sp_F_u_(b,i) +
        dotFF * pm1.fidu.sp_v_u(b,k,j,i)
      ) / SQR(dotFF);
    }

    for (int a=0; a<N; ++a)
    {
      J(1+a,0) += d_tn * alpha * dotFv * W * (
        sp_F_d(a,k,j,i) * kap_as +
        W2 * dotFv * kap_s * pm1.fidu.sp_v_d(a,k,j,i)
      ) / dotFF;

      for (int b=0; b<N; ++b)
      {
        J(1+a,1+b) += d_tn * alpha * sc_E(k,j,i) * W * (
          -2.0 * dotFv * sp_F_u_(b,i) *
          (sp_F_d(a,k,j,i) * kap_as +
           W2 * dotFv * kap_s * pm1.fidu.sp_v_d(a,k,j,i)) +
          dotFF * ((a==b) * dotFv * kap_as +
                   pm1.fidu.sp_v_u(b,k,j,i) * (
                    sp_F_d(a,k,j,i) * kap_as +
                    2.0 * W2 * dotFv * kap_s * pm1.fidu.sp_v_d(a,k,j,i)
                   ))
        ) / SQR(dotFF);
      }
    }
  }

  // thick correction to Jacobian ---------------------------------------------
  if (d_tk > 0)
  {
    J(0,0) += d_tk * alpha * dotvv * kap_s * W3 * (
      -1.0 + (2.0 - 4.0 * dotvv) * W2
    ) / (1.0 + 2.0 * W2);

    for (int b=0; b<N; ++b)
    {
      J(0,1+b) += d_tk * 2.0 * alpha * dotvv * kap_s * W3 *
                  pm1.fidu.sp_v_u(b,k,j,i) * (
                    1.0 + (1.0 + dotvv) * W2
                  ) / (1.0 + 2.0 * W2);
    }

    for (int a=0; a<N; ++a)
    {
      J(1+a,0) += -d_tk * alpha * pm1.fidu.sp_v_d(a,k,j,i) * W * (
        1.0 + (-2.0 + 4.0 * dotvv) * W2
      ) * (
        kap_as + dotvv * kap_s * W2
      ) / (1.0 + 2.0 * W2);

      for (int b=0; b<N; ++b)
      {
        J(1+a,1+b) += d_tk * alpha * W * (
          (a == b) * dotvv * kap_as +
          pm1.fidu.sp_v_u(b,k,j,i) * pm1.fidu.sp_v_d(a,k,j,i) * (
            kap_as + 2.0 * dotvv * kap_as * W2 +
            2.0 * dotvv * kap_s * W2 * (1.0 + (1.0 + dotvv) * W2)
          ) / (1.0 + 2.0 * W2)
        );
      }
    }
  }

  // --------------------------------------------------------------------------
  */

}

// ============================================================================
namespace D1 {
// ============================================================================

// Derivatives with respect to Chi

// First derivatives (wrt. Chi);
inline Real d_th(const AT_C_sca & sc_chi,
                 const int k, const int j, const int i)
{
  return 1.5;
}

inline Real d_tk(const AT_C_sca & sc_chi,
                 const int k, const int j, const int i)
{
  return -1.5;
}

// We write:
// sc_J = J_0
// st_H = H_n n^alpha + H_v v^alpha + H_F F^alpha
inline void ToFiducialExpansionCoefficients(
  M1 & pm1,
  Real & J_0,
  Real & H_n,
  Real & H_v,
  Real & H_F,
  Real & dJ_dchi_0,
  Real & dH_dchi_n,
  Real & dH_dchi_v,
  Real & dH_dchi_F,
  const AT_C_sca & sc_chi,
  const AT_C_sca & sc_E,
  const AT_N_vec & sp_F_d,
  const int k,
  const int j,
  const int i)
{
  const AT_N_vec & sp_v_u = pm1.fidu.sp_v_u;
  const AT_N_vec & sp_v_d = pm1.fidu.sp_v_d;
  const AT_N_sym & sp_g_uu = pm1.geom.sp_g_uu;

  const Real W  = pm1.fidu.sc_W(k,j,i);
  const Real W2 = SQR(W);

  const Real E = sc_E(k,j,i);
  const Real dotFv = sc_dot_dense_sp__(sp_F_d, sp_v_u, k, j, i);
  const Real dotvv = sc_dot_dense_sp__(sp_v_d, sp_v_u, k, j, i);

  const Real nF2 = sp_norm2__(sp_F_d, sp_g_uu, k, j, i);
  const Real oo_nF = (nF2 > pm1.opt.fl_nF2) ? OO(std::sqrt(nF2)) : 0.0;
  const Real dotFhatv = oo_nF * dotFv;

  const Real d_th = Assemble::Frames::d_th(sc_chi, k, j, i);
  const Real d_tk = Assemble::Frames::d_tk(sc_chi, k, j, i);

  const Real dd_th_dchi = Assemble::Frames::D1::d_th(sc_chi, k, j, i);
  const Real dd_tk_dchi = Assemble::Frames::D1::d_tk(sc_chi, k, j, i);

  // ------------------------------------------------------------------------
  const Real B_0 = W2 * (E - 2.0 * dotFv);
  const Real B_th = W2 * E * SQR(dotFhatv);
  const Real B_tk = (W2 - 1.0) / (2.0 * W2 + 1.0) * (
    4.0 * W2 * dotFv + (3.0 - 2.0 * W2) * E
  );

  // coefficients appearing in vector-expansion of H
  const Real v_0  = W * B_0;
  const Real v_th = W * B_th;
  const Real v_tk = W * B_tk + W / (2.0 * W2 + 1.0) * (
    (3.0 - 2.0 * W2) * E + (2.0 * W2 - 1.0) * dotFv
  );

  const Real F_0 = -W;
  const Real F_th = oo_nF * W * E * dotFhatv;
  const Real F_tk = W * dotvv;

  const Real v_H = (v_0 + d_th * v_th + d_tk * v_tk);
  const Real F_H = (F_0 + d_th * F_th + d_tk * F_tk);

  const Real d_v_H_dchi = (dd_th_dchi * v_th + dd_tk_dchi * v_tk);
  const Real d_F_H_dchi = (dd_th_dchi * F_th + dd_tk_dchi * F_tk);

  // --------------------------------------------------------------------------
  // Populate according to comment
  // J_0 = B_0 + d_th * B_th + d_tk * B_tk;
  J_0 = std::max(
    B_0 + d_th * B_th + d_tk * B_tk,
    pm1.opt.fl_J
  );

  H_v = -v_H;
  H_F = -F_H;

  // (from H \perp u)
  H_n = H_F * dotFv + H_v * dotvv;

  // Derivative terms
  dJ_dchi_0 = dd_th_dchi * B_th + dd_tk_dchi * B_tk;
  dH_dchi_v = -d_v_H_dchi;
  dH_dchi_F = -d_F_H_dchi;
  dH_dchi_n = dH_dchi_F * dotFv + dH_dchi_v * dotvv;
}
// ============================================================================
} // namespace M1::Assemble::Frames::D1
// ============================================================================

// ============================================================================
} // namespace M1::Assemble::Frames
// ============================================================================


// ============================================================================
namespace SpaceTime {
// ============================================================================

inline void st_x_sp_d_(
  AT_D_vec & st_tar_d_,
  const AT_N_vec & sp_V_d,
  const AT_N_vec & sp_beta_u,
  const int k, const int j,
  const int il, const int iu)
{
  // V_0 = g_0i V^i = beta_i V^i = beta^i V_i

  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_d_(0,i) = 0;
  }

  for (int a=0; a<D-1; ++a)  // spatial ranges
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_d_(0,i) += sp_beta_u(a,k,j,i) * sp_V_d(a,k,j,i);
  }

  for (int a=0; a<D-1; ++a)  // spatial ranges
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_d_(a+1,i) = sp_V_d(a,k,j,i);
  }
}

inline void st_x_sp_dd_(
  AT_D_sym & st_tar_dd_,
  const AT_N_sym & sp_S_dd,
  const AT_N_vec & sp_beta_u,
  const int k, const int j,
  const int il, const int iu)
{
  // S_00 = g_0i g_k0 S^ik = beta^i beta^k S_ik
  // S_0i = g_0j g_ki S^jk = beta_j S_i^j = beta^j S_ij

  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_dd_(0,0,i) = 0;
  }

  for (int a=0; a<N; ++a)  // spatial ranges
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_dd_(a+1,0,i) = 0;
  }

  for (int a=0; a<N; ++a)  // spatial ranges
  for (int b=0; b<N; ++b)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_dd_(0,0,i) += sp_S_dd(a,b,k,j,i) *
                         sp_beta_u(a,k,j,i) *
                         sp_beta_u(b,k,j,i);
  }

  for (int a=0; a<N; ++a)  // spatial ranges
  for (int b=0; b<N; ++b)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_dd_(0,a+1,i) += sp_S_dd(a,b,k,j,i) *
                           sp_beta_u(b,k,j,i);
  }

  // S_ij
  for (int a=0; a<N; ++a)  // spatial ranges
  for (int b=a; b<N; ++b)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_dd_(a+1,b+1,i) = sp_S_dd(a,b,k,j,i);
  }
}

// ============================================================================
} // namespace M1::Assemble::SpaceTime
// ============================================================================


// ============================================================================
}  // M1::Assemble
// ============================================================================

#endif // M1_UTILS_HPP

//
// :D
//