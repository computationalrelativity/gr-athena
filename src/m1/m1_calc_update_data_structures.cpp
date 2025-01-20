// C++ standard headers
#include <iostream>

// Athena++ headers
#include "m1.hpp"
#include "m1_calc_update.hpp"

// ============================================================================
namespace M1::Update {
// ============================================================================

StateMetaVector ConstructStateMetaVector(
  M1 & pm1, M1::vars_Lab & vlab,
  const int ix_g, const int ix_s)
{
  return StateMetaVector {
    pm1,
    ix_g,
    ix_s,
    // state-vector dependent
    vlab.sc_nG( ix_g,ix_s),
    vlab.sc_E(  ix_g,ix_s),
    vlab.sp_F_d(ix_g,ix_s),
    // group-dependent, but common
    pm1.rad.sc_n(       ix_g,ix_s),
    pm1.lab_aux.sc_chi( ix_g,ix_s),
    pm1.lab_aux.sc_xi(  ix_g,ix_s),
    pm1.lab_aux.sp_P_dd(ix_g,ix_s),
    // Lagrangian frame
    pm1.rad.sc_J(  ix_g,ix_s),
    pm1.rad.st_H_u(ix_g,ix_s),
    // opacities
    pm1.radmat.sc_eta_0(  ix_g,ix_s),
    pm1.radmat.sc_kap_a_0(ix_g,ix_s),
    pm1.radmat.sc_eta(  ix_g,ix_s),
    pm1.radmat.sc_kap_a(ix_g,ix_s),
    pm1.radmat.sc_kap_s(ix_g,ix_s),
    // averages
    pm1.radmat.sc_avg_nrg(ix_g,ix_s)
  };
}

SourceMetaVector ConstructSourceMetaVector(
  M1 & pm1, M1::vars_Source & vsrc,
  const int ix_g, const int ix_s)
{
  return SourceMetaVector {
    pm1,
    ix_g,
    ix_s,
    vsrc.sc_nG( ix_g,ix_s),
    vsrc.sc_E(  ix_g,ix_s),
    vsrc.sp_F_d(ix_g,ix_s),
  };
}

// ============================================================================
} // namespace M1::Update
// ============================================================================

//
// :D
//