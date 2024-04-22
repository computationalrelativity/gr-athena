#ifndef M1_CALC_UPDATE_HPP
#define M1_CALC_UPDATE_HPP

// c++
// ...

// Athena++ classes headers
#include "m1.hpp"
#include "m1_containers.hpp"
#include "m1_macro.hpp"
#include "m1_utils.hpp"

// ============================================================================
namespace M1::Update {
// ============================================================================

// structure that stores state information and can be utilized in GSL
struct StateMetaVector {
  M1 & pm1;
  const int ix_g;
  const int ix_s;

  AT_C_sca & sc_nG;
  AT_C_sca & sc_E;
  AT_N_vec & sp_F_d;

  AT_C_sca & sc_n;
  AT_C_sca & sc_chi;
  AT_C_sca & sc_xi;
  AT_N_sym & sp_P_dd;

  AT_C_sca & sc_J;
  AT_C_sca & sc_H_t;
  AT_N_vec & sp_H_d;

  AT_C_sca & sc_eta_0;
  AT_C_sca & sc_kap_a_0;

  AT_C_sca & sc_eta;
  AT_C_sca & sc_kap_a;
  AT_C_sca & sc_kap_s;
};

StateMetaVector PopulateStateMetaVector(
  M1 & pm1, M1::vars_Lab & vlab,
  const int ix_g, const int ix_s);

void AddSourceMatter(
  M1 & pm1,
  const StateMetaVector & C,  // state to utilize
  StateMetaVector & I,        // add source here
  const int k, const int j, const int i);

inline void ApplyFloors(
  M1 & pm1, StateMetaVector & C,
  const int k, const int j, const int i)
{
  C.sc_E(k,j,i) = std::max(C.sc_E(k,j,i), pm1.opt.fl_E);
}

// Require \Vert F \Vert_\gamma \leq E
//
// Enforce this by setting: F_i -> F_i / (\Vert F \Vert_\gamma / E)
inline void EnforceCausality(
  M1 & pm1, StateMetaVector & C,
  const int k, const int j, const int i)
{
  const Real norm2F = Assemble::sp_norm2__(C.sp_F_d, pm1.geom.sp_g_uu,
                                           k, j, i);

  if (norm2F > 0)
  if (SQR(C.sc_E(k,j,i)) < norm2F)
  {
    const Real normF = std::sqrt(norm2F);
    const Real fac = normF / C.sc_E(k,j,i);

    for (int a=0; a<N; ++a)
    {
      C.sp_F_d(a,k,j,i) = C.sp_F_d(a,k,j,i) / fac;
    }
  }
}

// ============================================================================
} // namespace M1::Update
// ============================================================================


#endif // M1_CALC_UPDATE_HPP

