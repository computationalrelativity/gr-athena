// C++ standard headers
#include <iostream>

// Athena++ headers

#include "m1.hpp"

#if !(M1_NO_WEAKRATES)
#include "opacities/weakrates/m1_opacities_weakrates.hpp"
#endif
#include "opacities/m1_opacities.hpp"

#include "m1_sources.hpp"
#include "m1_set_equilibrium.hpp"
#include "../hydro/hydro.hpp"
#include "../scalars/scalars.hpp"

// BD: TODO - Direct enforcement of equilibrium is never performed.
//            The following logic is scratch (won't compile) & left for
//            reference purposes.

// ============================================================================
namespace M1::Equilibrium {
// ============================================================================

void SetEquilibrium(
  M1 & pm1,
  Update::StateMetaVector & C,
  Update::SourceMetaVector & S,
  Closures::ClosureMetaVector & CL,
  const int k,
  const int j,
  const int i)
{
#ifdef DBG_ENFORCE_EQUILIBRIUM
  using namespace Update;
  using namespace Sources;
  using namespace Closures;

  Opacities::Opacities * popac = pm1.popac;
  typedef Opacities::Opacities::opt_opacity_variety oov;

  assert((pm1.N_GRPS == 1) && (pm1.N_SPCS <= 4));
  assert(((popac->opt.opacity_variety == oov::weakrates) &&
          (pm1.N_SPCS > 1)) ||
         ((popac->opt.opacity_variety == oov::photon) &&
          (pm1.N_SPCS == 1)) );

  // Reset sources
  SetMatterSourceZero(S, k, j, i);

  // Compute the optically thick weak equilibrium -----------------------------
  Hydro * ph          = pm1.pmy_block->phydro;
  PassiveScalars * ps = pm1.pmy_block->pscalars;

  const Real w_rho = ph->w(IDN,k,j,i);
  const Real w_T   = ph->temperature(k,j,i);
  const Real w_Y_e = ps->r(0,k,j,i);

  // Short-circuit at low density
  if (w_rho < pm1.opt_solver.eql_rho_min)
  {
    return;
  }

  // [[n_nue, n_nua, n_nux, n_nux]
  //  [e_nue, e_nua, e_nux, e_nux]]
  static const int N_nu      = 2;
  static const int N_nu_spcs = 4;

  AA nudens(N_nu, N_nu_spcs);
  nudens.Fill(0.0);

  popac->CalculateEquilbriumDensity(w_rho, w_T, w_Y_e, nudens);

  // Set equilibrium in fiducial frame (sc_J, st_H_d={sc_H_t, sp_H_d})
  // Reconstruct Eulerian: (sc_E, sp_F_d, sp_P_dd)
  M1::vars_Rad & rad = pm1.rad;
  const Real sc_sqrt_det_g = pm1.geom.sc_sqrt_det_g(k,j,i);
  const Real W = pm1.fidu.sc_W(k,j,i);
  const Real W2 = SQR(W);

  for (int ix_g=0; ix_g<pm1.N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<pm1.N_SPCS; ++ix_s)
  {
    /*
    C.sc_J(ix_g,ix_s)(k,j,i) = nudens(1,ix_s) * sc_sqrt_det_g;
    C.sc_H_t(ix_g,ix_s)(k,j,i) = 0;
    for (int a=0; a<N; ++a)
    {
      C.sp_H_d(ix_g,ix_s)(a,k,j,i) = 0;
    }
    */


    // (sc_J, st_H_d) -> (sc_E, sp_F_d) reduces to:
    C.sc_E(k,j,i) = W2 * C.sc_J(k,j,i);
    for (int a=0; a<N; ++a)
    {
      C.sp_F_d(a,k,j,i) = W2 * pm1.fidu.sp_v_d(a,k,j,i) * C.sc_J(k,j,i);
    }

    // Get sp_P_dd from thick closure relation:
    CL.ClosureThick(k,j,i);

    // Prepare neutrino number density
    if (pm1.N_SPCS > 1)
    {
      const Real sc_n = 1;
      rad.sc_n(ix_g, ix_s)(k,j,i) = nudens(0, ix_s) * sc_sqrt_det_g;
      // BD: TODO - add rad_N_floor as opt
      // C.sc_nG(k,j,i) = std::max(W * sc_n, rad_N_floor);
      // BD: TODO - compute G and consequently n := nG / G
    }
  }

#endif // DBG_ENFORCE_EQUILIBRIUM

}

// ============================================================================
} // namespace M1::Equilibrium
// ============================================================================


//
// :D
//