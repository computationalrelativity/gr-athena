#ifndef M1_SOURCES_HPP
#define M1_SOURCES_HPP

// c++
// ...

// Athena++ classes headers
#include "m1.hpp"
#include "m1_containers.hpp"
#include "m1_macro.hpp"
#include "m1_utils.hpp"
#include "m1_calc_update.hpp"

// ============================================================================
namespace M1::Sources {
// ============================================================================

// Given an input state vector, prepare sources [based on conserved vars.] ----

// Set sources to zero at a point.
// Modifies S.{sc_nG, sc_E, sp_F_d}
void SetMatterSourceZero(
  Update::SourceMetaVector & S,
  const int k, const int j, const int i);

// PrepareMatterSource_E_F_d requires the following internal vectors:
// V.{sc_kap_a, sc_kap_s, sc_eta, sp_F_d, sp_P_dd}
//
// And utilizes:
// V.{sc_E, sp_F_d}
//
// This means closures need to be applied prior.
void PrepareMatterSource_E_F_d(
  M1 & pm1,
  const Update::StateMetaVector & V,
  Update::SourceMetaVector & S,
  const int k, const int j, const int i);

// Cached overload: forwards DotProductCache to the underlying
// sources_sc_E_sp_F_d cached variant, avoiding redundant dot-product
// and Lorentz factor recomputation within the Newton iteration.
inline void PrepareMatterSource_E_F_d(
  M1 & pm1,
  const Update::StateMetaVector & V,
  Update::SourceMetaVector & S,
  const Assemble::Frames::DotProductCache & cache,
  const int k, const int j, const int i)
{
  Assemble::Frames::sources_sc_E_sp_F_d(
    pm1,
    S.sc_E,
    S.sp_F_d,
    V.sc_chi,
    V.sc_E,
    V.sp_F_d,
    V.sc_eta,
    V.sc_kap_a,
    V.sc_kap_s,
    cache,
    k, j, i
  );
}

// PrepareMatterSource_nG requires the following internal vectors:
// V.{sc_eta_0, sc_kap_a_0}
//
// And utilizes:
// V.{sc_n}
//
// This means n needs to be previously prepared-
// For this we solve for nG* (and hence n*) directly, then, get the compatible
// source here.
void PrepareMatterSource_nG(
  M1 & pm1,
  const Update::StateMetaVector & V,
  Update::SourceMetaVector & S,
  const int k, const int j, const int i);

// Given sc_nG and a StateMetaVector construct sc_n [i.e. prepare Gam]
void Prepare_n_from_nG(
  M1 & pm1,
  const Update::StateMetaVector & V,
  const int k, const int j, const int i);

// ============================================================================
namespace Minerbo {
// ============================================================================

// Source term Jacobian in terms of \sqrt{g} densitized variables.
//
// See also 3.2 of [1].
void PrepareMatterSourceJacobian_E_F_d(
  M1 & pm1,
  const Real dt,
  AA & J,                               // Storage for Jacobian
  const Update::StateMetaVector & C,    // current step
  const int k, const int j, const int i
);

// Cached overload: forwards DotProductCache to the underlying
// Jacobian_sc_E_sp_F_d cached variant.
inline void PrepareMatterSourceJacobian_E_F_d(
  M1 & pm1,
  const Real dt,
  AA & J,                               // Storage for Jacobian
  const Update::StateMetaVector & C,    // current step
  const Assemble::Frames::DotProductCache & cache,
  const int k, const int j, const int i)
{
  Assemble::Frames::Jacobian_sc_E_sp_F_d(
    pm1,
    J,
    C.sc_chi, C.sc_E, C.sp_F_d, C.sc_kap_a, C.sc_kap_s,
    cache,
    k, j, i
  );
}

// ============================================================================
} // namespace M1::Sources::Minerbo
// ============================================================================

inline void PrepareMatterSourceZJacobian_E_F_d(
  M1 & pm1,
  const Real dt,
  M1::opt_closure_variety ocv,
  AA & J,                               // Storage for Jacobian
  const Update::StateMetaVector & C,    // current step
  const int k, const int j, const int i)
{
  switch (ocv)
  {
    // BD: TODO - additional closures would need extension here
    // By default, Newton-style method is only applied when using Minerbo
    // closure.
    // case M1::opt_closure_variety::thin:
    // {
    //   assert(false);
    // }
    // case M1::opt_closure_variety::thick:
    // {
    //   assert(false);
    // }
    default:
    {
      Minerbo::PrepareMatterSourceJacobian_E_F_d(pm1, dt, J, C, k, j, i);
    }
  }

  // Actually need Jacobian for Z and not the sources -------------------------
  const int N_SYS = J.GetDim1();
  for (int a=0; a<N_SYS; ++a)
  for (int b=0; b<N_SYS; ++b)
  {
    J(a,b) = (a==b) - dt * J(a,b);
  }
}

// ============================================================================
namespace Limiter {
// ============================================================================

// Given (C, P, I) StateMetaVector collection together with source S, prepare
// source limiting mask.
//
// Note:
// StateMetaVector C is modified - it is not required for solution assembly.
void Prepare(
  M1 * pm1,
  const Real dt,
  AT_C_sca & theta,
  const M1::vars_Lab & U_C,
  const M1::vars_Lab & U_P,
  const M1::vars_Lab & U_I,
  const M1::vars_Source & U_S,
  const int kl, const int ku,
  const int jl, const int ju,
  const int il, const int iu
);

void Apply(
  M1 * pm1,
  const Real dt,
  AT_C_sca & theta,
  M1::vars_Lab & U_C,
  const M1::vars_Lab & U_P,
  const M1::vars_Lab & U_I,
  M1::vars_Source & U_S,
  const int kl, const int ku,
  const int jl, const int ju,
  const int il, const int iu
);

void PrepareFull(
  M1 * pm1,
  const Real dt,
  AT_C_sca & theta,
  const M1::vars_Lab & U_C,
  const M1::vars_Lab & U_P,
  const M1::vars_Lab & U_I,
  const M1::vars_Source & U_S,
  const int kl, const int ku,
  const int jl, const int ju,
  const int il, const int iu
);

void ApplyFull(
  M1 * pm1,
  const Real dt,
  AT_C_sca & theta,
  M1::vars_Lab & U_C,
  const M1::vars_Lab & U_P,
  const M1::vars_Lab & U_I,
  M1::vars_Source & U_S,
  const int kl, const int ku,
  const int jl, const int ju,
  const int il, const int iu
);

// Check whether the current sources / state give rise to a physical solution
// with respect to the hydro / species sector.
void CheckPhysicalFallback(
  M1 * pm1,
  const Real dt,
  const M1::vars_Source & U_S,
  const int kl, const int ku,
  const int jl, const int ju,
  const int il, const int iu
);

// ============================================================================
} // namespace M1::Sources::Limiter
// ============================================================================

// ============================================================================
} // namespace M1::Sources
// ============================================================================


#endif // M1_SOURCES_HPP

