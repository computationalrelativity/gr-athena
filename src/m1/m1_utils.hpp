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
  const Real oo_nF = (nF2 > 0) ? OO(std::sqrt(nF2)) : 0.0;
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
  const Real n_0  = W * B_0 + W * (dotFv - E);
  const Real n_th = W * B_th;
  const Real n_tk = W * B_tk;

  const Real v_0  = W * B_0;
  const Real v_th = n_th;
  const Real v_tk = W * B_tk + W / (2.0 * W2 + 1.0) * (
    (3.0 - 2.0 * W2) * E + (2.0 * W2 - 1.0) * dotFv
  );

  const Real F_0 = -W;
  const Real F_th = oo_nF * W * E * dotFhatv;
  const Real F_tk = W * dotvv;

  const Real n_H = (n_0 + d_th * n_th + d_tk * n_tk);
  const Real v_H = (v_0 + d_th * v_th + d_tk * v_tk);
  const Real F_H = (F_0 + d_th * F_th + d_tk * F_tk);

  // --------------------------------------------------------------------------
  // Populate according to comment
  J_0 = (B_0 + d_th * B_th + d_tk * B_tk);
  H_n = -n_H;  // Need additional sign: H^mu n_mu = -A in H^mu = A n^mu + B ...
  H_v = -v_H;
  H_F = -F_H;
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
    const Real fac = (nF2 > 0) ? sc_E(k,j,i) / nF2
                               : 0.0;

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