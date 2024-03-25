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
    scratch.sp_vec_,
    sp_beta_u,
    sp_g_dd,
    k, j,
    il, iu);

  for (int b=0; b<N; ++b)  // spatial ranges
  for (int i=il; i<=iu; ++i)
  {
    sp_tar(b,k,j,i) = scratch.sp_vec_(b,i);
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
    scratch.sc_,
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
    scratch.sc_,
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

// Gamma in Eq.(24) of [1]
inline void sc_G_(
  AT_C_sca & st_tar_,
  const AT_C_sca & sc_W,
  const AT_C_sca & sc_E,
  const AT_C_sca & sc_J,
  const AT_D_vec & st_F_d_,
  const AT_D_vec & st_v_u_,
  const Real floor_rad_E,
  const Real eps_rad_E,
  const int k, const int j,
  const int il, const int iu)
{
  // could be rewritten as branchless.. probably not worth it

  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    if ((sc_E(k,j,i) > floor_rad_E) && (sc_J(k,j,i) > floor_rad_E))
    {
      st_tar_(i) = sc_W(k,j,i) * sc_E(k,j,i) / sc_J(k,j,i) * (
        1 - std::min(LinearAlgebra::Dot(st_F_d_, st_v_u_, i) / sc_E(k,j,i),
                     1.0-eps_rad_E)
      );
    }
    else
    {
      st_tar_(i) = 1.0;
    }
  }
}

inline void Norm_st_(
  AT_C_sca & st_tar_,
  const AT_D_vec & st_V_d_,  // alternative u_, dd_
  const AT_D_sym & st_g_uu_,
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

// ============================================================================
}  // M1::Assemble
// ============================================================================

#endif // M1_UTILS_HPP

