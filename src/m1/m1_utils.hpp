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
// #include "m1_containers.hpp"
// #include "m1_macro.hpp"
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
  const AT_N_sca & sc_alpha,
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
  const AT_N_sca & sc_alpha,
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
  const AT_N_sca & sc_alpha,
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
  const AT_N_sca & sc_alpha,
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
  AT_D_sca & st_tar_,
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

)
{

}

// Radiation ------------------------------------------------------------------
// ...

// ============================================================================
}  // M1::Assemble
// ============================================================================

#endif // M1_UTILS_HPP

