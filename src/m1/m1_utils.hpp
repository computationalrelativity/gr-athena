#ifndef M1_UTILS_HPP
#define M1_UTILS_HPP

// c++
// ...

// Athena++ classes headers
// #include "../athena.hpp"
// #include "../athena_arrays.hpp"
// #include "../athena_tensor.hpp"
// #include "../mesh/mesh.hpp"
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
// Appended "_" indicates scratch (in i)
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

inline void sc_dot_st_(
  AT_C_sca & sc_dot_st_,
  const AT_D_vec & st_V_A,  // alternative: flip for opposite slice
  const AT_D_vec & st_V_B_,
  const int k, const int j,
  const int il, const int iu)
{
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sc_dot_st_(i) = 0.0;
  }

  for (int a=0; a<D; ++a)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sc_dot_st_(i) += st_V_A(a,k,j,i) * st_V_B_(a,i);
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

inline void st_beta_u_(
  AT_D_vec & st_tar_,
  const AT_N_vec & sp_beta_u,
  const int k, const int j,
  const int il, const int iu)
{
  for (int i=il; i<=iu; ++i)
  {
    st_tar_(0,i) = 0;
  }

  for (int b=0; b<D-1; ++b)  // spatial ranges
  for (int i=il; i<=iu; ++i)
  {
    st_tar_(b+1,i) = sp_beta_u(b,k,j,i);
  }
}

inline void st_g_dd_(
  AT_D_sym & st_tar_,
  const AT_N_sym & sp_g_dd,
  const AT_C_sca & sc_alpha,
  const AT_N_vec & sp_beta_d,
  const AT_N_vec & sp_beta_u,
  M1::vars_Scratch & scratch,
  const int k, const int j,
  const int il, const int iu)
{
  LinearAlgebra::Assemble_ST_Metric_dd(
    st_tar_, sp_g_dd, sc_alpha, sp_beta_d, sp_beta_u,
    scratch.sc_A_,
    k, j, il, iu);
}

inline void st_g_uu_(
  AT_D_sym & st_tar_,
  const AT_N_sym & sp_g_uu,
  const AT_C_sca & sc_alpha,
  const AT_N_vec & sp_beta_u,
  M1::vars_Scratch & scratch,
  const int k, const int j,
  const int il, const int iu)
{
  LinearAlgebra::Assemble_ST_Metric_uu(
    st_tar_, sp_g_uu, sc_alpha, sp_beta_u,
    scratch.sc_A_,
    k, j, il, iu);
}

// normal contra / cov
inline void st_n_u_(
  AT_D_vec & st_tar_,
  const AT_C_sca & sc_alpha,
  const AT_N_vec & sp_beta_u,
  const int k, const int j,
  const int il, const int iu)
{
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_(0,i) = 1.0 / sc_alpha(k,j,i);
  }

  for (int a=0; a<D-1; ++a)  // spatial ranges
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_(a+1,i) = -sp_beta_u(a,k,j,i) * st_tar_(0,i);
  }
}

inline void st_n_d_(
  AT_D_vec & st_tar_,
  const AT_C_sca & sc_alpha,
  const int k, const int j,
  const int il, const int iu)
{
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_(0,i) = -sc_alpha(k,j,i);
  }

  for (int a=0; a<D-1; ++a)  // spatial ranges
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_(a+1,i) = 0;
  }
}

// projector
inline void st_P_ud_(
  AT_D_bil & st_tar_,
  const AT_D_vec & st_n_u_,
  const AT_D_vec & st_n_d_,
  const int k, const int j,
  const int il, const int iu)
{
  for (int a=0; a<D; ++a)
  for (int b=0; b<D; ++b)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_(a,b,i) = (a==b) + st_n_u_(a,i)*st_n_d_(b,i);
  }
}

// project: both directions tangent
inline void st_proj_PP(
  AT_D_sym & st_tar_dd_,
  const AT_D_bil & st_P_ud_,
  const AT_D_sym & st_src_dd_,
  const int k, const int j,
  const int il, const int iu)
{

  for (int a=0; a<D; ++a)
  for (int b=a; b<D; ++b)
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      st_tar_dd_(a,b,i) = 0;
    }

    for (int c=0; c<D; ++c)
    for (int d=0; d<D; ++d)
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      st_tar_dd_(a,b,i) += st_P_ud_(c,a,i) * st_P_ud_(d,b,i) *
                           st_src_dd_(c,d,i);
    }
  }
}

// project: one direction orthogonal
inline void st_proj_oP(
  AT_D_vec & st_tar_d_,
  const AT_D_vec & st_o_u_,
  const AT_D_bil & st_P_ud_,
  const AT_D_sym & st_src_dd_,
  const int k, const int j,
  const int il, const int iu)
{
  for (int a=0; a<D; ++a)
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      st_tar_d_(a,i) = 0;
    }

    for (int c=0; c<D; ++c)
    for (int d=0; d<D; ++d)
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      st_tar_d_(a,i) += st_P_ud_(c,a,i) * st_o_u_(d,i) *
                        st_src_dd_(c,d,i);
    }

  }
}

inline void st_proj_oo(
  AT_C_sca & st_tar_,
  const AT_D_vec & st_o_u_,
  const AT_D_sym & st_src_dd_,
  const int k, const int j,
  const int il, const int iu)
{
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_(i) = 0;
  }

  for (int c=0; c<D; ++c)
  for (int d=0; d<D; ++d)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_(i) += st_o_u_(c,i) * st_o_u_(d,i) *
                  st_src_dd_(c,d,i);
  }
}

// Hydro ----------------------------------------------------------------------
inline void st_w_u_u_(
  AT_D_vec & st_tar_u_,
  const AT_C_sca & sc_alpha,
  const AT_C_sca & sc_W,
  const AT_N_vec & sp_w_util_u,
  const int k, const int j,
  const int il, const int iu)
{

  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_u_(0,i) = sc_W(k,j,i) / sc_alpha(k,j,i);
  }


  for (int a=0; a<D-1; ++a)  // spatial ranges
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_u_(a+1,i) = sp_w_util_u(a,k,j,i);
  }
}

// Radiation ------------------------------------------------------------------
inline void st_F_d_(
  AT_D_vec & st_tar_d_,
  const AT_N_vec & sp_F_d,
  const AT_N_vec & sp_beta_u,
  const int k, const int j,
  const int il, const int iu)
{
  // F_0 = g_0i F^i = beta_i F^i = beta^i F_i

  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_d_(0,i) = 0;
  }

  for (int a=0; a<D-1; ++a)  // spatial ranges
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_d_(0,i) += sp_beta_u(a,k,j,i) * sp_F_d(a,k,j,i);
  }

  for (int a=0; a<D-1; ++a)  // spatial ranges
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_d_(a+1,i) = sp_F_d(a,k,j,i);
  }
}

inline void st_H_d_(
  AT_D_vec & st_tar_d_,
  const AT_C_sca & sc_H_t,
  const AT_N_vec & sp_H_d,
  const int k, const int j,
  const int il, const int iu)
{
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_d_(0,i) = sc_H_t(k,j,i);
  }

  for (int a=0; a<D-1; ++a)  // spatial ranges
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_d_(a+1,i) = sp_H_d(a,k,j,i);
  }
}

inline void st_P_dd_(
  AT_D_vec & st_tar_dd_,
  const AT_N_vec & sp_P_dd,
  const AT_N_vec & sp_beta_u,
  const int k, const int j,
  const int il, const int iu)
{
  // P_00 = g_0i g_k0 P^ik = beta^i beta^k P_ik
  // P_0i = g_0j g_ki P^jk = beta_j P_i^j = beta^j P_ij

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
    st_tar_dd_(0,0,i) += sp_P_dd(a,b,k,j,i) *
                         sp_beta_u(a,k,j,i) *
                         sp_beta_u(b,k,j,i);
  }

  for (int a=0; a<N; ++a)  // spatial ranges
  for (int b=0; b<N; ++b)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_dd_(0,a+1,i) += sp_P_dd(a,b,k,j,i) *
                           sp_beta_u(b,k,j,i);
  }

  // P_ij
  for (int a=0; a<N; ++a)  // spatial ranges
  for (int b=a; b<N; ++b)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_dd_(a+1,b+1,i) = sp_P_dd(a,b,k,j,i);
  }
}

// Eq.(2) of [1]
inline void st_T_rad_dd_(
  AT_D_sym & st_tar_dd_,
  const AT_D_vec & st_u_d_,
  const AT_C_sca & sc_J,
  const AT_D_vec & st_H_d_,
  const AT_D_sym & st_K_dd_,
  const int k, const int j,
  const int il, const int iu)
{
  for (int a=0; a<D; ++a)
  for (int b=a; b<D; ++b)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_dd_(a,b,i) = (
      sc_J(k,j,i) * st_u_d_(a,i) * st_u_d_(b,i) +
      st_H_d_(a,i) * st_u_d_(b,i) +
      st_u_d_(a,i) * st_H_d_(b,i) +
      st_K_dd_(a,b,i)
    );
  }
}

// K_{alpha beta} is (projected) linear functional of (T_rad)_{alpha beta}
inline void st_K_o_T_rad_dd_(
  AT_D_sym & st_tar_dd_,
  const AT_D_bil & st_P_ud_,  // N.B. use fiducial projector here
  const AT_D_sym & st_T_rad_,
  const int k, const int j,
  const int il, const int iu)
{
  st_proj_PP(
    st_tar_dd_, st_P_ud_, st_T_rad_,
    k, j, il, iu
  );
}

// H_{alpha} is (projected) linear function of (T_rad)_{alpha beta}
inline void st_H_o_T_rad_d_(
  AT_D_vec & st_tar_d_,
  const AT_D_vec & st_u_u_,   // N.B. use suitably mapped fiducial velocity
  const AT_D_bil & st_P_ud_,  // N.B. use fiducial projector here
  const AT_D_sym & st_T_rad_,
  const int k, const int j,
  const int il, const int iu)
{
  st_proj_oP(
    st_tar_d_, st_u_u_, st_P_ud_, st_T_rad_,
    k, j, il, iu
  );

  // convention <u,u> = -1
  for (int a=0; a<D; ++a)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_d_(a,i) = -st_tar_d_(a,i);
  }
}

// J is (projected) linear function of (T_rad)_{alpha beta}
inline void st_J_o_T_rad_(
  AT_C_sca & st_tar_,
  const AT_D_vec & st_u_u_,   // N.B. use suitably mapped fiducial velocity
  const AT_D_sym & st_T_rad_,
  const int k, const int j,
  const int il, const int iu)
{
  st_proj_oo(
    st_tar_, st_u_u_, st_T_rad_,
    k, j, il, iu
  );
}

// f^alpha in Eq.(21) of [1]
inline void st_f_u_(
  AT_D_vec & st_tar_u_,
  const AT_D_vec & st_u_u_,
  const AT_D_vec & st_H_u_,
  const AT_C_sca & sc_J,
  const AT_C_sca & sc_norm_st_H_,
  const Real eps_f_J,
  const int k, const int j,
  const int il, const int iu)
{
  for (int a=0; a<D; ++a)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_u_(a,i) = (
      st_u_u_(a,i) +
      ((sc_J(k,j,i) > eps_f_J * sc_norm_st_H_(i)) ? st_H_u_(a,i) / sc_J(k,j,i)
                                                  : 0.0)
    );
  }
}

inline void sp_f_u_(
  AT_N_vec & sp_tar_u_,
  const AT_C_sca & sc_alpha,
  const AT_N_vec & sp_beta_u,
  const AT_C_sca & sc_W,
  const AT_N_vec & sp_v_u,
  const AT_N_vec & sp_H_u_,
  const AT_C_sca & sc_norm_sp_H_,
  const AT_C_sca & sc_J,
  const Real eps_J,
  const int a,
  const int k, const int j,
  const int il, const int iu)
{
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sp_tar_u_(a,i) = (
      sc_W(k,j,i) * (sp_v_u(a,k,j,i) - sp_beta_u(a,k,j,i) / sc_alpha(k,j,i)) +
      ((sc_J(k,j,i) > eps_J * sc_norm_sp_H_(i))
        ? sp_H_u_(a,i) / sc_J(k,j,i)
        : 0.0)
    );
  }
}

// Gamma in Eq.(24) of [1]
inline void sc_G_(
  AT_C_sca & sc_tar_,
  const AT_C_sca & sc_W,
  const AT_C_sca & sc_E,
  const AT_C_sca & sc_J,
  const AT_D_vec & st_F_d_,
  const AT_D_vec & st_v_u,
  M1::vars_Scratch & scratch,
  const Real floor_E,
  const Real floor_J,
  const Real eps_E,
  const int k, const int j,
  const int il, const int iu)
{
  // could be rewritten as branchless.. probably not worth it

  sc_dot_st_(scratch.sc_A_, st_v_u, st_F_d_, k, j, il, iu);

  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    if ((sc_E(k,j,i) > floor_E) && (sc_J(k,j,i) > floor_J))
    {
      sc_tar_(i) = sc_W(k,j,i) * sc_E(k,j,i) / sc_J(k,j,i) * (
        1 - std::min(scratch.sc_A_(i) / sc_E(k,j,i), 1.0-eps_E)
      );
    }
    else
    {
      sc_tar_(i) = 1.0;
    }
  }
}

inline void sc_G_(
  AT_C_sca & sc_tar_,
  const AT_C_sca & sc_W,
  const AT_C_sca & sc_E,
  const AT_C_sca & sc_J,
  const AT_C_sca & sc_dotFv_,
  const Real floor_E,
  const Real floor_J,
  const Real eps_E,
  const int k, const int j,
  const int il, const int iu)
{
  // could be rewritten as branchless.. probably not worth it
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    if ((sc_E(k,j,i) > floor_E) && (sc_J(k,j,i) > floor_J))
    {
      sc_tar_(i) = sc_W(k,j,i) * sc_E(k,j,i) / sc_J(k,j,i) * (
        1 - std::min(sc_dotFv_(i) / sc_E(k,j,i), 1.0-eps_E)
      );
    }
    else
    {
      sc_tar_(i) = 1.0;
    }
  }
}

inline Real sc_G_(
  const Real & W,
  const Real & E,
  const Real & J,
  const Real & dotFv_,
  const Real floor_E,
  const Real floor_J,
  const Real eps_E)
{
  if ((E > floor_E) && (J > floor_J))
  {
    return W * E / J * (
      1 - std::min(dotFv_ / E, 1.0-eps_E)
    );
  }
  return 1.0;
}

inline Real sc_J__(
  const Real & W2,
  const Real & dotFv,  // F_d v^d
  const AT_C_sca & sc_E,
  const AT_N_vec & sp_v_u,
  const AT_N_sym & sp_P_dd,
  const int k, const int j, const int i)
{
  Real J = sc_E(k,j,i) - 2.0 * dotFv;
  for (int a=0; a<N; ++a)
  for (int b=0; b<N; ++b)
  {
    J += sp_P_dd(a,b,k,j,i) * sp_v_u(a,k,j,i) * sp_v_u(b,k,j,i);
  }

  return W2 * J;
}

inline Real sc_H_t__(
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

inline Real sc_H2_st__(
  const AT_C_sca & sc_H_t,
  const AT_N_vec & sp_H_d,
  const AT_N_sym & sp_g_uu,
  const int k, const int j, const int i)
{
  Real sc_H2_st = -SQR(sc_H_t(k,j,i));
  for (int a=0; a<N; ++a)
  for (int b=0; b<N; ++b)
  {
    sc_H2_st += sp_g_uu(a,b,k,j,i) * sp_H_d(a,k,j,i) * sp_H_d(b,k,j,i);
  }
  return sc_H2_st;
}

inline void st_norm2_(
  AT_C_sca & st_tar_,
  const AT_D_vec & st_V_d_,  // alternative: _u_ &
  const AT_D_sym & st_g_uu_, // _dd_
  const int k, const int j,
  const int il, const int iu)
{
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_(i) = LinearAlgebra::InnerProductSlicedVecMetric(
      st_V_d_, st_g_uu_, i
    );
  }
}

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

inline void st_vec_from_t_sp(
  AT_D_vec & st_tar_,
  const AT_C_sca & cpt_ut,  // alternative: _dt &
  const AT_N_vec & vec_u,   // _d
  const int k, const int j,
  const int il, const int iu)
{
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_(0,i) = cpt_ut(k,j,i);
  }

  for (int a=0; a<N; ++a)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_tar_(a+1,i) = vec_u(a,k,j,i);
  }
};

// source related expressions -------------------------------------------------
/*
inline void st_S_u_(
  AT_D_vec & st_S_u_,
  const AT_C_sca & sc_eta,
  const AT_C_sca & sc_kap_a,
  const AT_C_sca & sc_kap_s,
  const AT_C_sca & sc_J,
  const AT_D_vec & st_u_u_,
  const AT_D_vec & st_H_u_,
  const int k, const int j,
  const int il, const int iu)
{
  for (int a=0; a<D; ++a)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    st_S_u_(a,i) = (
      (sc_eta(k,j,i) - sc_kap_a(k,j,i) * sc_J(k,j,i)) * st_u_u_(a,i) -
      (sc_kap_a(k,j,i) + sc_kap_s(k,j,i)) * st_H_u_(a,i)
    );
  }
}

// Eq.(29) of [1]
inline void src_mat_nG_(
  AT_C_sca & sc_tar_,
  const AT_C_sca & sc_nG,
  const AT_C_sca & sc_G_,
  const AT_C_sca & sc_eta,
  const AT_C_sca & sc_kap_a,
  const AT_C_sca & sc_sqrt_det_g,
  const AT_C_sca & sc_alpha,
  const int k, const int j,
  const int il, const int iu)
{
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sc_tar_(i) = sc_alpha(k,j,i) * (
      sc_sqrt_det_g(k,j,i) * sc_eta(k,j,i) -
      sc_kap_a(k,j,i) * sc_nG(k,j,i) / sc_G_(i)
    );
  }
}

inline void src_mat_E_(
  AT_C_sca & sc_tar_,
  const AT_D_vec & st_S_u_,
  const AT_C_sca & sc_sqrt_det_g,
  const AT_C_sca & sc_alpha,
  const AT_D_vec & st_n_d_,
  const int k, const int j,
  const int il, const int iu)
{
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sc_tar_(i) = 0.0;
  }

  for (int a=0; a<D; ++a)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sc_tar_(i) -= sc_alpha(k,j,i) * sc_sqrt_det_g(k,j,i) * (
      st_S_u_(a,i) * st_n_d_(a,i)
    );
  }
}

inline void src_mat_F_d_(
  AT_N_vec & sc_tar_d_,
  const AT_D_vec & st_S_u_,
  const AT_C_sca & sc_sqrt_det_g,
  const AT_C_sca & sc_alpha,
  const AT_N_sym & sp_g_dd,
  const AT_D_vec & st_n_d_,
  const int k, const int j,
  const int il, const int iu)
{
  for (int a=0; a<D; ++a)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sc_tar_d_(a,i) = 0.0;
  }

  for (int b=0; b<N; ++b)
  for (int a=0; a<D; ++a)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sc_tar_d_(b,i) += sc_alpha(k,j,i) * sc_sqrt_det_g(k,j,i) * (
      st_S_u_(a,i) * (sp_g_dd(b,a,k,j,i) + st_n_d_(b,i)*st_n_d_(a,i))
    );
  }

}

// See Eq.(30) of [1]
inline void src_geom_E_(
  AT_C_sca & sc_tar_,
  const AT_N_vec & sp_F_u_,
  const AT_N_sym & sp_P_uu_,
  const AT_C_sca & sc_alpha,
  const AT_N_D1sca & sp_dalpha_d,
  const AT_N_sym & sp_K_dd,
  const int k, const int j,
  const int il, const int iu)
{
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sc_tar_(i) = 0.0;
  }

  for (int a=0; a<N; ++a)
  for (int b=0; b<N; ++b)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sc_tar_(i) += sc_alpha(k,j,i) * (
      sp_P_uu_(a,b,i) * sp_K_dd(a,b,k,j,i)
    );
  }

  for (int a=0; a<N; ++a)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sc_tar_(i) -= (
      sp_F_u_(a,i) * sp_dalpha_d(a,k,j,i)
    );
  }
}

inline void src_geom_F_d_(
  AT_N_vec & sc_tar_d_,
  const AT_C_sca & sc_E,
  const AT_N_vec & sp_F_d,
  const AT_N_sym & sp_P_uu_,
  const AT_C_sca & sc_alpha,
  const AT_N_D1sca & sp_dalpha_d,
  const AT_N_D1vec & sp_dbeta_du,
  const AT_N_D1sym & sp_dg_ddd,
  const int k, const int j,
  const int il, const int iu)
{
  for (int a=0; a<N; ++a)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sc_tar_d_(a,i) = -sp_dalpha_d(a,k,j,i) * sc_E(k,j,i);
  }

  for (int a=0; a<N; ++a)
  for (int b=0; b<N; ++b)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sc_tar_d_(a,i) += (
      sp_F_d(b,k,j,i) * sp_dbeta_du(b,a,k,j,i)
    );
  }

  for (int a=0; a<N; ++a)
  for (int b=0; b<N; ++b)
  for (int c=0; c<N; ++c)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sc_tar_d_(a,i) += 0.5 * sc_alpha(k,j,i) * (
      sp_P_uu_(b,c,i) * sp_dg_ddd(a,b,c,k,j,i)
    );
  }

}
*/
// // Flux related expressions ---------------------------------------------------
// // See Eq.(28) of [1]
// inline void flux_nG_(
//   AT_C_sca & sc_tar_,
//   const int dir,
//   const AT_C_sca & sc_nG,
//   const AT_C_sca & sc_G_,
//   const AT_D_vec & st_f_u_,
//   const AT_C_sca & sc_alpha,
//   const int k, const int j,
//   const int il, const int iu)
// {
//   #pragma omp simd
//   for (int i=il; i<=iu; ++i)
//   {
//     sc_tar_(i) = sc_alpha(k,j,i) * sc_nG(k,j,i) / sc_G_(i) * st_f_u_(1+dir,i);
//   }
// }

// inline void flux_E_(
//   AT_C_sca & sc_tar_,
//   const int dir,
//   const AT_C_sca & sc_E,
//   const AT_N_vec & sp_F_d,
//   const AT_C_sca & sc_alpha,
//   const AT_N_vec & sp_beta_u,
//   const AT_N_sym & sp_g_uu,
//   const int k, const int j,
//   const int il, const int iu)
// {
//   #pragma omp simd
//   for (int i=il; i<=iu; ++i)
//   {
//     sc_tar_(i) = -sp_beta_u(dir,k,j,i) * sc_E(k,j,i);
//   }

//   for (int a=0; a<N; ++a)
//   #pragma omp simd
//   for (int i=il; i<=iu; ++i)
//   {
//     sc_tar_(i) += sc_alpha(k,j,i) *
//                   sp_g_uu(dir,a,k,j,i) * sp_F_d(a,k,j,i);
//   }
// }

// inline void flux_F_d_(
//   AT_N_vec & sc_tar_d_,
//   const int dir,
//   const AT_N_vec & sp_F_d,
//   const AT_N_sym & sp_P_dd,
//   const AT_C_sca & sc_alpha,
//   const AT_N_vec & sp_beta_u,
//   const AT_N_sym & sp_g_uu,
//   const int k, const int j,
//   const int il, const int iu)
// {

//   for (int a=0; a<N; ++a)
//   #pragma omp simd
//   for (int i=il; i<=iu; ++i)
//   {
//     sc_tar_d_(a,i) = -sp_beta_u(dir,k,j,i) * sp_F_d(a,k,j,i);
//   }

//   for (int a=0; a<N; ++a)
//   for (int b=0; b<N; ++b)
//   #pragma omp simd
//   for (int i=il; i<=iu; ++i)
//   {
//     sc_tar_d_(a,i) += sc_alpha(k,j,i) *
//                       sp_g_uu(dir,b,k,j,i) * sp_P_dd(b,a,k,j,i);
//   }


// }

// ============================================================================
// Convenience methods

// inline void st_g_uu_(
//   M1 * pm1,
//   const int k, const int j,
//   const int il, const int iu)
// {
//   st_g_uu_(pm1->geom.st_g_uu_,
//            pm1->geom.sp_g_uu,
//            pm1->geom.sc_alpha,
//            pm1->geom.sp_beta_u,
//            pm1->scratch,
//            k, j, il, iu);
// }

// inline void st_F_d_(
//   M1 * pm1,
//   const int ix_g, const int ix_s,
//   const int k, const int j,
//   const int il, const int iu)
// {
//   st_F_d_(pm1->lab.st_F_d_,
//           pm1->lab.sp_F_d(ix_g,ix_s),
//           pm1->geom.sp_beta_u,
//           k, j, il, iu);
// }

// inline void sc_norm_st_F_(
//   M1 * pm1,
//   const int ix_g, const int ix_s,
//   const int k, const int j,
//   const int il, const int iu)
// {
//   Norm_st_(pm1->lab.sc_norm_st_F_,
//            pm1->lab.st_F_d_,
//            pm1->geom.st_g_uu_,
//            k, j, il, iu);
// }

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
  const AT_N_sym & sp_src_dd,
  const int k, const int j,
  const int il, const int iu)
{
  LinearAlgebra::SymMetContraction(
    sp_tar_uu_,
    sp_src_dd,
    pm1->geom.sp_g_uu,
    k, j,
    il, iu);
}

inline void st_g_dd_(
  M1 * pm1,
  AT_D_sym & st_tar_dd_,
  const int k, const int j,
  const int il, const int iu)
{
  LinearAlgebra::Assemble_ST_Metric_dd(
    st_tar_dd_,
    pm1->geom.sp_g_dd,
    pm1->geom.sc_alpha,
    pm1->geom.sp_beta_d,
    pm1->geom.sp_beta_u,
    pm1->scratch.sc_A_,
    k, j, il, iu);
}

inline void st_g_uu_(
  M1 * pm1,
  AT_D_sym & st_tar_uu_,
  const int k, const int j,
  const int il, const int iu)
{
  LinearAlgebra::Assemble_ST_Metric_uu(
    st_tar_uu_,
    pm1->geom.sp_g_uu,
    pm1->geom.sc_alpha,
    pm1->geom.sp_beta_u,
    pm1->scratch.sc_A_,
    k, j, il, iu);
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

inline void sp_f_u_(
  M1 * pm1,
  AT_N_vec & sp_tar_u_,
  const AT_N_vec & sp_H_u_,
  const AT_C_sca & sc_norm_sp_H_,
  const AT_C_sca & sc_J,
  const int a,
  const int k, const int j,
  const int il, const int iu)
{
  sp_f_u_(
    sp_tar_u_,
    pm1->geom.sc_alpha,
    pm1->geom.sp_beta_u,
    pm1->fidu.sc_W,
    pm1->fidu.sp_v_u,
    sp_H_u_,
    sc_norm_sp_H_,
    sc_J,
    pm1->opt.eps_J,
    a,
    k, j, il, iu
  );
}

inline void st_F_d_(
  M1 * pm1,
  AT_D_vec & st_tar_d_,
  const AT_N_vec & sp_F_d,
  const int k, const int j,
  const int il, const int iu)
{
  st_F_d_(st_tar_d_,
          sp_F_d,
          pm1->geom.sp_beta_u,
          k, j,
          il, iu);
}

inline void sc_G_(
  M1 * pm1,
  AT_C_sca & sc_tar_,
  const AT_C_sca & sc_E,
  const AT_C_sca & sc_J,
  const AT_D_vec & st_F_d_,
  const int k, const int j,
  const int il, const int iu)
{
  sc_G_(sc_tar_,
        pm1->fidu.sc_W,
        sc_E,
        sc_J,
        st_F_d_,
        pm1->fidu.st_v_u,
        pm1->scratch,
        pm1->opt.fl_E,
        pm1->opt.fl_J,
        pm1->opt.eps_E,
        k, j, il, iu);
}

// given data over an i strip populate compatible dense quantity
inline void ScratchToDense(
  AT_N_sym & sp_tar_aa,
  AT_N_sym & sp_S_aa_,
  const int k, const int j,
  const int il, const int iu)
{
  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sp_tar_aa(a,b,k,j,i) = sp_S_aa_(a,b,i);
  }
}

inline void ScratchAddToDense(
  AT_N_sym & sp_tar_aa,
  const AT_N_sym & sp_S_aa_,
  const int k, const int j,
  const int il, const int iu)
{
  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sp_tar_aa(a,b,k,j,i) += sp_S_aa_(a,b,i);
  }
}


// ============================================================================
}  // M1::Assemble
// ============================================================================

#endif // M1_UTILS_HPP

