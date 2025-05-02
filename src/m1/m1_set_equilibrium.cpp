// C++ standard headers
#include <iostream>

// Athena++ headers

#include "m1.hpp"

#if !(M1_NO_WEAKRATES)
#include "opacities/weakrates/m1_opacities_weakrates.hpp"
#endif
#include "opacities/m1_opacities.hpp"

#include "m1_set_equilibrium.hpp"
#include "m1_sources.hpp"
#include "m1_calc_closure.hpp"
#include "../hydro/hydro.hpp"
#include "../scalars/scalars.hpp"

// ============================================================================
namespace M1::Equilibrium {
// ============================================================================

void SetEquilibrium(
  M1 & pm1,
  M1::vars_Lab & U_C,
  M1::vars_Source & U_S,
  const int k,
  const int j,
  const int i)
{
#if FLUID_ENABLED
  using namespace Update;
  using namespace Sources;
  // using namespace Closures;

  Opacities::Opacities * popac = pm1.popac;
  typedef Opacities::Opacities::opt_opacity_variety oov;

  assert((pm1.N_GRPS == 1) && (pm1.N_SPCS <= 4));
  assert(((popac->opt.opacity_variety == oov::weakrates) &&
          (pm1.N_SPCS > 1)) ||
         ((popac->opt.opacity_variety == oov::photon) &&
          (pm1.N_SPCS == 1)) );

  // Compute the optically thick weak equilibrium -----------------------------
  MeshBlock * pmb       = pm1.pmy_block;
  Hydro * ph            = pmb->phydro;
  EquationOfState *peos = pmb->peos;

  const Real w_rho = pm1.hydro.sc_w_rho(k,j,i);
  const Real w_p   = pm1.hydro.sc_w_p(k,j,i);
  const Real w_Y_e = pm1.hydro.sc_w_Ye(k,j,i);

  Real Y[MAX_SPECIES] = {0.0};
  Y[0] = pm1.hydro.sc_w_Ye(k,j,i);

  Real const nb = w_rho / (peos->GetEOS().GetBaryonMass());

  const Real w_T = pm1.hydro.sc_T(k,j,i);

  // Short-circuit at low density
  // Convert from code units to CGS for this comparison?
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

  // Opticall thick weak equilibrium
  popac->CalculateEquilbriumDensity(w_rho, w_T, w_Y_e, nudens);

  // Set equilibrium in fiducial frame (sc_J, st_H_d={sc_H_t, sp_H_d}, sc_n)
  // Reconstruct Eulerian: (sc_E, sp_F_d, sc_nG)
  const Real sc_sqrt_det_g = pm1.geom.sc_sqrt_det_g(k,j,i);
  const Real W = pm1.fidu.sc_W(k,j,i);
  const Real W2 = SQR(W);

  for (int ix_g=0; ix_g<pm1.N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<pm1.N_SPCS; ++ix_s)
  {
    // aliases ----------------------------------------------------------------
    StateMetaVector C = ConstructStateMetaVector(pm1, U_C, ix_g, ix_s);
    SourceMetaVector S = ConstructSourceMetaVector(pm1, U_S, ix_g, ix_s);

    AT_C_sca & sc_J   = C.sc_J;
    AT_D_vec & st_H_u = C.st_H_u;
    AT_C_sca & sc_n   = C.sc_n;

    AT_C_sca & sc_E    = C.sc_E;
    AT_N_vec & sp_F_d  = C.sp_F_d;
    AT_C_sca & sc_nG   = C.sc_nG;

    AT_C_sca & sc_chi  = C.sc_chi;
    AT_C_sca & sc_xi   = C.sc_xi;

    // kill sources
    SetMatterSourceZero(S, k, j, i);
    S.sc_nG(k,j,i) = 0;

    // impose thick closure
    Closures::EddingtonFactors::ThickLimit(sc_xi(k,j,i), sc_chi(k,j,i));
    // ------------------------------------------------------------------------

    sc_J(k,j,i) = nudens(1,ix_s) * sc_sqrt_det_g;

    for (int a=0; a<D; ++a)
    {
      st_H_u(a,k,j,i) = 0;
    }

    // (sc_J, st_H_d) -> (sc_E, sp_F_d) reduces to:
    sc_E(k,j,i) = W2 * sc_J(k,j,i);
    for (int a=0; a<N; ++a)
    {
      sp_F_d(a,k,j,i) = W2 * pm1.fidu.sp_v_d(a,k,j,i) * sc_J(k,j,i);
    }

    // Prepare neutrino number density
    if (pm1.N_SPCS > 1)
    {
      sc_n(k,j,i) = nudens(0, ix_s) * sc_sqrt_det_g;
      // floors
      sc_nG(k,j,i) = std::max(W * sc_n(k,j,i), pm1.opt.fl_nG);
      sc_n(k,j,i) = sc_nG(k,j,i) / W;  // propagate back
    }

    // Fix strategy
    pm1.ev_strat.masks.solution_regime(ix_g,ix_s,k,j,i) = (
      M1::evolution_strategy::opt_solution_regime::equilibrium
    );

    // Ensure update preserves energy non-negativity
    EnforcePhysical_E_F_d(pm1, C, k, j, i);
    EnforcePhysical_nG(   pm1, C, k, j, i);
  }

#endif // FLUID_ENABLED
}

// ============================================================================
} // namespace M1::Equilibrium
// ============================================================================


//
// :D
//