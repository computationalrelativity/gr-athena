// C++ standard headers
#include <iostream>
#include <array>

// Athena++ headers

#include "m1.hpp"
#include "m1_calc_update.hpp"
#include "m1_integrators.hpp"
#include "m1_utils.hpp"

#if !(M1_NO_WEAKRATES)
#include "opacities/weakrates/m1_opacities_weakrates.hpp"
#endif
#include "opacities/m1_opacities.hpp"

#include "m1_set_equilibrium.hpp"
#include "m1_sources.hpp"
#include "m1_calc_closure.hpp"
#include "m1_integrators.hpp"
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

  // Optically thick weak equilibrium
  popac->CalculateEquilibriumDensity(w_rho, w_T, w_Y_e, nudens);

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

    AT_C_sca & sc_avg_nrg = pm1.radmat.sc_avg_nrg(ix_g, ix_s);

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
    sc_E(k,j,i) = ONE_3RD * sc_J(k,j,i) * (
      4.0 * W2 - 1.0
    );

    for (int a=0; a<N; ++a)
    {
      sp_F_d(a,k,j,i) = 4.0 * ONE_3RD * W2 *
                        pm1.fidu.sp_v_d(a,k,j,i) * sc_J(k,j,i);
    }

    Real dotFv__ (0.0);
    for (int a=0; a<N; ++a)
    {
      dotFv__ += sp_F_d(a,k,j,i) * pm1.fidu.sp_v_u(a,k,j,i);
    }
    const Real Gamma__ = W / sc_J(k,j,i) * (
      sc_E(k,j,i) - dotFv__
    );

    // Prepare neutrino number density
    if (pm1.N_SPCS > 1)
    {
      sc_n(k,j,i) = nudens(0, ix_s) * sc_sqrt_det_g;
      // floors (here Gamma = W as H_n is 0)
      sc_nG(k,j,i) = std::max(Gamma__ * sc_n(k,j,i), pm1.opt.fl_nG);
      sc_n(k,j,i) = sc_nG(k,j,i) / Gamma__;  // propagate back
    }

    // Changes to M1 state vector- propagate back to average energy
    sc_avg_nrg(k,j,i) = sc_J(k,j,i) / sc_n(k,j,i);

    // Ensure update preserves energy non-negativity
    EnforcePhysical_E_F_d(pm1, C, k, j, i);
    EnforcePhysical_nG(   pm1, C, k, j, i);

    // Fix strategy / sources -------------------------------------------------
    pm1.SetMaskSolutionRegime(M1::M1::t_sln_r::equilibrium,ix_g,ix_s,k,j,i);
    if (pm1.opt_solver.equilibrium_sources)
    {
      // source terms (entering coupling)
      pm1.SetMaskSourceTreatment(M1::M1::t_src_t::full,ix_g,ix_s,k,j,i);
      ::M1::Sources::PrepareMatterSource_E_F_d(pm1, C, S, k, j, i);
      ::M1::Sources::PrepareMatterSource_nG(   pm1, C, S, k, j, i);
    }
    else
    {
      pm1.SetMaskSourceTreatment(M1::M1::t_src_t::set_zero,ix_g,ix_s,k,j,i);
    }
  }

#endif // FLUID_ENABLED


}

void SetEquilibrium_n_nG(
  M1 & pm1,
  const Real dt,
  Update::StateMetaVector & C,
  Update::StateMetaVector & P,
  Update::StateMetaVector & I,
  Update::SourceMetaVector & S,
  Closures::ClosureMetaVector & CL_C,
  const int k,
  const int j,
  const int i
)
{
  using namespace Update;
  using namespace Sources;

  // Short-circuit at low density
  // Convert from code units to CGS for this comparison?
  const Real w_rho = pm1.hydro.sc_w_rho(k,j,i);
  if (w_rho < pm1.opt_solver.eql_rho_min)
  {
    return;
  }

  // eql fall-back should be disabled -----------------------------------------
  Real opt_flux_lo_fallback_E = pm1.opt.flux_lo_fallback_E;
  Real opt_flux_lo_fallback_nG = pm1.opt.flux_lo_fallback_nG;
  if (pm1.opt.flux_lo_fallback_eql_ho)
  {
    pm1.opt.flux_lo_fallback_E = false;
    pm1.opt.flux_lo_fallback_nG = false;
  }
  // --------------------------------------------------------------------------

  const Real sc_sqrt_det_g = pm1.geom.sc_sqrt_det_g(k,j,i);

  const int ix_g = C.ix_g;
  const int ix_s = C.ix_s;

  // extract field components / prepare frame ---------------------------------
  // U_new:
  AT_C_sca & sc_J   = C.sc_J;
  AT_D_vec & st_H_u = C.st_H_u;
  AT_C_sca & sc_n   = C.sc_n;

  AT_C_sca & sc_E    = C.sc_E;
  AT_N_vec & sp_F_d  = C.sp_F_d;
  AT_C_sca & sc_nG   = C.sc_nG;

  AT_C_sca & sc_chi  = C.sc_chi;
  AT_C_sca & sc_xi   = C.sc_xi;

  AT_C_sca & sc_avg_nrg = pm1.radmat.sc_avg_nrg(ix_g, ix_s);


  // need to retain additional candidate (star) state
  Real T_E;
  Real T_nG;
  std::array<Real, N> T_F_d;

  T_E = P.sc_E(k,j,i);
  T_nG = P.sc_nG(k,j,i);
  for (int n=0; n<N; ++n)
  {
    T_F_d[n] = P.sp_F_d(n,k,j,i);
  }

  // --------------------------------------------------------------------------

  // impose thick closure
  Closures::EddingtonFactors::ThickLimit(sc_xi(k,j,i), sc_chi(k,j,i));

  // If required, evolve (E,F_d) in absence of sources
  if (pm1.opt_solver.equilibrium_evolve)
  {
    Sources::SetMatterSourceZero(S, k, j, i);
    Integrators::Explicit::StepExplicit_E_F_d(pm1, dt, C, P, I, S, k, j, i);

    // Retain advected
    T_E = sc_E(k,j,i);

    for (int n=0; n<N; ++n)
    {
      T_F_d[n] = sp_F_d(n,k,j,i);
    }

    if (!pm1.opt_solver.equilibrium_evolve_use_euler)
    {
      // Sources::SetMatterSourceZero(S, k, j, i);
      // Integrators::Explicit::StepExplicit_E_F_d(pm1, dt, C, P, I, S, k, j, i);

      CL_C.Closure(k, j, i);

      // Evolve fiducial frame; prepare (J, H^alpha):
      // We write:
      // sc_J   = J_0
      // st_H^a = H_n n^a + H_v v^a + H_F F^a
      const Real W  = pm1.fidu.sc_W(k,j,i);
      const Real W2 = SQR(W);

      const AT_N_vec & sp_v_d = pm1.fidu.sp_v_d;
      const AT_N_vec & sp_v_u = pm1.fidu.sp_v_u;

      const AT_C_sca & sc_alpha  = pm1.geom.sc_alpha;
      const AT_N_vec & sp_beta_u = pm1.geom.sp_beta_u;

      Real J_0, H_n, H_v, H_F;

      Assemble::Frames::ToFiducialExpansionCoefficients(
        pm1,
        J_0, H_n, H_v, H_F,
        C.sc_chi, C.sc_E, C.sp_F_d,
        k, j, i
      );

      // populate spatial projection of fluid frame rad. flux.
      AT_N_vec & sp_H_d_ = pm1.scratch.sp_vec_A_;
      for (int a=0; a<N; ++a)
      {
        sp_H_d_(a,i) = H_v * sp_v_d(a,k,j,i) + H_F * C.sp_F_d(a,k,j,i);
      }
      // ----------------------------------------------------------------------


      // Evolve fluid frame quantities
      const Real sqrt_deg_g__ = pm1.geom.sc_sqrt_det_g(k,j,i);
      const Real kap_as = C.sc_kap_a(k,j,i) + C.sc_kap_s(k,j,i);

      J_0 = (J_0 * W + dt * sqrt_deg_g__ * C.sc_eta(k,j,i)) /
            (W + dt * C.sc_kap_a(k,j,i));

      Real H_n__ = W * H_n / (W + dt * kap_as);

      for (int a=0; a<N; ++a)
      {
        sp_H_d_(a,i) = W * sp_H_d_(a,i) / (W + dt * kap_as);

      }

      // Project back assuming thick limit ------------------------------------
      Closures::EddingtonFactors::ThickLimit(
        CL_C.sc_xi(k,j,i), CL_C.sc_chi(k,j,i)
      );

      // (sc_J, st_H_d) -> (sc_E, sp_F_d) reduces to:
      sc_E(k,j,i) = ONE_3RD * J_0 * (
        4.0 * W2 - 1.0
      ) + 2.0 * W * H_n__;


      Real dotFv (0.0);
      for (int a=0; a<N; ++a)
      {
        sp_F_d(a,k,j,i) = 4.0 * ONE_3RD * W2 *
                          pm1.fidu.sp_v_d(a,k,j,i) * J_0;
        sp_F_d(a,k,j,i) += W * (H_n__ * pm1.fidu.sp_v_d(a,k,j,i) +
                                sp_H_d_(a,i));

        dotFv += sp_F_d(a,k,j,i) * pm1.fidu.sp_v_u(a,k,j,i);
      }

      // N.B. THC does the following:
      /*
      C.sc_E(k,j,i) = J_0;

      for (int a=0; a<N; ++a)
      {
        C.sp_F_d(a,k,j,i) = sp_H_d_(a,i);
      }
      */
    }


  }

  // Prepare sources if required
  if (pm1.opt_solver.equilibrium_src_E_F_d)
  {
    if (pm1.opt_solver.equilibrium_use_diff_src)
    {
      // new - star
      S.sc_E(k,j,i) = (sc_E(k,j,i) - T_E) / dt;
      for (int n=0; n<N; ++n)
      {
        S.sp_F_d(n,k,j,i) = (sp_F_d(n,k,j,i) - T_F_d[n]) / dt;
      }
    }
    else
    {
      Assemble::Frames::sources_sc_E_sp_F_d(
        pm1,
        S.sc_E,
        S.sp_F_d,
        C.sc_chi,
        C.sc_E,
        C.sp_F_d,
        C.sc_eta,
        C.sc_kap_a,
        C.sc_kap_s,
        k, j, i
      );
    }
  }
  else
  {
    Sources::SetMatterSourceZero(S, k, j, i);
  }

  // Construct C based on modified S
  if (pm1.opt_solver.equilibrium_src_E_F_d &&
      pm1.opt_solver.equilibrium_use_diff_src)
  {
    Integrators::Explicit::StepExplicit_E_F_d(pm1, dt, C, P, I, S, k, j, i);
  }

  // now handle species -------------------------------------------------------

  // Irrespective of the above, we now need the fiducial frame
  // n will be overwitten below and is not required for J, H
  Real sc_Gam__ = Assemble::Frames::ToFiducial(
    pm1,
    sc_J, st_H_u, sc_n,
    sc_chi,
    sc_E, sp_F_d, sc_nG,
    k, j, i
  );

  /*
  // Fiducial frame calculation: ----------------------------------------------

  // update in accordance with average energies prepared in opac.
  sc_n(k,j,i)  = sc_J(k,j,i) / sc_avg_nrg(k,j,i);
  sc_nG(k,j,i) = sc_Gam__ * sc_n(k,j,i);

  // Flooring
  EnforcePhysical_nG(pm1, C, k, j, i);
  sc_n(k,j,i) = sc_nG(k,j,i) / sc_Gam__;  // propagate back
  */

  // Eulerian frame calculation (use advanced E, F_d) -------------------------
  Real dotFv (0.0);
  for (int a=0; a<N; ++a)
  {
    dotFv += sp_F_d(a,k,j,i) * pm1.fidu.sp_v_u(a,k,j,i);
  }
  const Real W = pm1.fidu.sc_W(k,j,i);

  // Original expression for seeding average energy
  // sc_avg_nrg(k,j,i) = W / sc_nG(k,j,i) * (sc_E(k,j,i) - dotFv);

  if (pm1.opt.retain_equilibrium)
  {
    sc_avg_nrg(k,j,i) = (
      pm1.eql.sc_J(ix_g,ix_s)(k,j,i) /
      pm1.eql.sc_n(ix_g,ix_s)(k,j,i)
    );
    // sc_nG(k,j,i) = pm1.eql.sc_n(C.ix_g,C.ix_s)(k,j,i) * sc_Gam__;
  }

  // update in accordance with average energies prepared in opac.
  sc_nG(k,j,i) = W / sc_avg_nrg(k,j,i) * (sc_E(k,j,i) - dotFv);

  EnforcePhysical_nG(pm1, C, k, j, i);
  sc_n(k,j,i) = sc_nG(k,j,i) / sc_Gam__;  // propagate back

  // prepare eql. adjust sources
  if (pm1.opt_solver.equilibrium_src_nG)
  {
    if (pm1.opt_solver.equilibrium_use_diff_src)
    {
      S.sc_nG(k,j,i) = (sc_nG(k,j,i) - T_nG) / dt;
    }
    else
    {
      Assemble::Frames::sources_sc_nG(
        pm1,
        S.sc_nG,
        C.sc_n, C.sc_eta_0, C.sc_kap_a_0, k, j, i
      );
    }
  }
  else
  {
    S.sc_nG(k,j,i) = 0;
  }

  // If required, evolve N explicitly based on eql. adjusted sources
  if (pm1.opt_solver.equilibrium_evolve)
  {
    Integrators::Explicit::StepExplicit_nG(pm1, dt, C, P, I, S, k, j, i);
    sc_n(k,j,i) = sc_nG(k,j,i) / sc_Gam__;
  }

  // eql fall-back should be disabled -----------------------------------------
  pm1.opt.flux_lo_fallback_E = opt_flux_lo_fallback_E;
  pm1.opt.flux_lo_fallback_nG = opt_flux_lo_fallback_nG;
  // --------------------------------------------------------------------------
}

/*
void SetEquilibrium_E_F_d_n_nG(
  M1 & pm1,
  Update::StateMetaVector & C,
  Update::StateMetaVector & P,
  Update::SourceMetaVector & S,
  const int k,
  const int j,
  const int i,
  const bool construct_src_nG,
  const bool construct_src_E_F_d,
  const bool use_diff_src
)
{
  using namespace Update;
  using namespace Sources;

  Opacities::Opacities * popac = pm1.popac;
  typedef Opacities::Opacities::opt_opacity_variety oov;

  // Compute the optically thick weak equilibrium -----------------------------
  MeshBlock * pmb       = pm1.pmy_block;
  Hydro * ph            = pmb->phydro;
  EquationOfState *peos = pmb->peos;

  const Real w_rho = pm1.hydro.sc_w_rho(k,j,i);
  const Real w_p   = pm1.hydro.sc_w_p(k,j,i);
  const Real w_Y_e = pm1.hydro.sc_w_Ye(k,j,i);
  const Real w_T   = pm1.hydro.sc_T(k,j,i);

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

  // Optically thick weak equilibrium
  popac->CalculateEquilibriumDensity(w_rho, w_T, w_Y_e, nudens);

  // Set equilibrium in fiducial frame (sc_J, st_H_d={sc_H_t, sp_H_d}, sc_n)
  // Reconstruct Eulerian: (sc_E, sp_F_d, sc_nG)
  const Real sc_sqrt_det_g = pm1.geom.sc_sqrt_det_g(k,j,i);
  const Real W = pm1.fidu.sc_W(k,j,i);
  const Real W2 = SQR(W);

  // Fix species and group ----------------------------------------------------
  const int ix_g = C.ix_g;
  const int ix_s = C.ix_s;

  // extract field components / prepare frame ---------------------------------
  AT_C_sca & sc_J   = C.sc_J;
  AT_D_vec & st_H_u = C.st_H_u;
  AT_C_sca & sc_n   = C.sc_n;

  AT_C_sca & sc_E    = C.sc_E;
  AT_N_vec & sp_F_d  = C.sp_F_d;
  AT_C_sca & sc_nG   = C.sc_nG;

  AT_C_sca & sc_chi  = C.sc_chi;
  AT_C_sca & sc_xi   = C.sc_xi;

  AT_C_sca & sc_avg_nrg = pm1.radmat.sc_avg_nrg(ix_g, ix_s);

  // impose thick closure
  Closures::EddingtonFactors::ThickLimit(sc_xi(k,j,i), sc_chi(k,j,i));

  // --------------------------------------------------------------------------
  sc_J(k,j,i) = nudens(1,ix_s) * sc_sqrt_det_g;

  for (int a=0; a<D; ++a)
  {
    st_H_u(a,k,j,i) = 0;
  }

  // (sc_J, st_H_d) -> (sc_E, sp_F_d) reduces to:
  sc_E(k,j,i) = ONE_3RD * sc_J(k,j,i) * (
    4.0 * W2 - 1.0
  );

  for (int a=0; a<N; ++a)
  {
    sp_F_d(a,k,j,i) = 4.0 * ONE_3RD * W2 *
                      pm1.fidu.sp_v_d(a,k,j,i) * sc_J(k,j,i);
  }

  // compute from average -----------------------------------------------------
  sc_n(k,j,i) = sc_J(k,j,i) / sc_avg_nrg(k,j,i);

  // here Gamma = W as H_n is 0
  sc_nG(k,j,i) = W * sc_n(k,j,i);

  // Ensure update preserves energy non-negativity
  EnforcePhysical_E_F_d(pm1, C, k, j, i);
  EnforcePhysical_nG(pm1, C, k, j, i);
  sc_n(k,j,i) = sc_nG(k,j,i) / W;  // propagate back

  // source terms (entering coupling) -----------------------------------------
  if (construct_src_nG)
  {
    if (use_diff_src)
    {
      // new - star
      S.sc_nG(k,j,i) = sc_nG(k,j,i) - P.sc_nG(k,j,i);
    }
    else
    {
      ::M1::Sources::PrepareMatterSource_nG(pm1, C, S, k, j, i);
    }
  }

  if (construct_src_E_F_d)
  {
    if (use_diff_src)
    {
      // new - star
      S.sc_E(k,j,i) = sc_E(k,j,i) - P.sc_E(k,j,i);
      for (int n=0; n<N; ++n)
      {
        S.sp_F_d(n,k,j,i) = sp_F_d(n,k,j,i) - P.sp_F_d(n,k,j,i);
      }
    }
    else
    {
      ::M1::Sources::PrepareMatterSource_E_F_d(pm1, C, S, k, j, i);
    }
  }
}
*/

void SetEquilibrium_E_F_d_n_nG(
  M1 & pm1,
  const Real dt,
  Update::StateMetaVector & C,
  Update::StateMetaVector & P,
  Update::StateMetaVector & I,
  Update::SourceMetaVector & S,
  Closures::ClosureMetaVector & CL_C,
  const int k,
  const int j,
  const int i
)
{
  using namespace Update;
  using namespace Sources;

  Opacities::Opacities * popac = pm1.popac;
  typedef Opacities::Opacities::opt_opacity_variety oov;

  // Compute the optically thick weak equilibrium -----------------------------
  MeshBlock * pmb       = pm1.pmy_block;
  Hydro * ph            = pmb->phydro;
  EquationOfState *peos = pmb->peos;

  const Real w_rho = pm1.hydro.sc_w_rho(k,j,i);

  // Short-circuit at low density
  // Convert from code units to CGS for this comparison?
  if (w_rho < pm1.opt_solver.eql_rho_min)
  {
    return;
  }

  assert(pm1.opt.retain_equilibrium);

  // eql fall-back should be disabled -----------------------------------------
  Real opt_flux_lo_fallback_E = pm1.opt.flux_lo_fallback_E;
  Real opt_flux_lo_fallback_nG = pm1.opt.flux_lo_fallback_nG;
  if (pm1.opt.flux_lo_fallback_eql_ho)
  {
    pm1.opt.flux_lo_fallback_E = false;
    pm1.opt.flux_lo_fallback_nG = false;
  }
  // --------------------------------------------------------------------------

  // Set equilibrium in fiducial frame (sc_J, st_H_d={sc_H_t, sp_H_d}, sc_n)
  const Real sc_sqrt_det_g = pm1.geom.sc_sqrt_det_g(k,j,i);
  const Real W = pm1.fidu.sc_W(k,j,i);
  const Real W2 = SQR(W);

  // Fix species and group ----------------------------------------------------
  const int ix_g = C.ix_g;
  const int ix_s = C.ix_s;

  // extract field components / prepare frame ---------------------------------
  AT_C_sca & sc_J   = C.sc_J;
  AT_D_vec & st_H_u = C.st_H_u;
  AT_C_sca & sc_n   = C.sc_n;

  AT_C_sca & sc_E    = C.sc_E;
  AT_N_vec & sp_F_d  = C.sp_F_d;
  AT_C_sca & sc_nG   = C.sc_nG;

  AT_C_sca & sc_chi  = C.sc_chi;
  AT_C_sca & sc_xi   = C.sc_xi;

  AT_C_sca & sc_avg_nrg = pm1.radmat.sc_avg_nrg(ix_g, ix_s);

  // need to retain additional candidate (star) state
  Real T_E;
  Real T_nG;
  std::array<Real, N> T_F_d;


  // evolved state without collisional term: ----------------------------------
  const bool use_inh = true;
  const Real dt_fac = use_inh * dt;

  T_E = P.sc_E(k,j,i) + dt_fac * I.sc_E(k,j,i);
  T_nG = P.sc_nG(k,j,i) + dt_fac * I.sc_nG(k,j,i);
  for (int n=0; n<N; ++n)
  {
    T_F_d[n] = P.sp_F_d(n,k,j,i) + dt_fac * I.sp_F_d(n,k,j,i);
  }

  // impose thick closure
  Closures::EddingtonFactors::ThickLimit(sc_xi(k,j,i), sc_chi(k,j,i));

  // --------------------------------------------------------------------------
  // read directly from opac.
  sc_J(k,j,i) = pm1.eql.sc_J(C.ix_g,C.ix_s)(k,j,i);

  // by definition of eql
  for (int a=0; a<D; ++a)
  {
    st_H_u(a,k,j,i) = 0;
  }

  // (sc_J, st_H_d) -> (sc_E, sp_F_d) reduces to:
  sc_E(k,j,i) = ONE_3RD * sc_J(k,j,i) * (
    4.0 * W2 - 1.0
  );


  Real dotFv (0.0);
  for (int a=0; a<N; ++a)
  {
    sp_F_d(a,k,j,i) = 4.0 * ONE_3RD * W2 *
                      pm1.fidu.sp_v_d(a,k,j,i) * sc_J(k,j,i);

    dotFv += sp_F_d(a,k,j,i) * pm1.fidu.sp_v_u(a,k,j,i);
  }

  Real eq_sc_Gam__ = pm1.fidu.sc_W(k,j,i) / sc_J(k,j,i) *
                     (sc_E(k,j,i) - dotFv);

  // Prepare sources if required ----------------------------------------------
  if (pm1.opt_solver.equilibrium_src_E_F_d)
  {
    if (pm1.opt_solver.equilibrium_use_diff_src)
    {
      // new - star
      S.sc_E(k,j,i) = (sc_E(k,j,i) - T_E) / dt;
      for (int n=0; n<N; ++n)
      {
        S.sp_F_d(n,k,j,i) = (sp_F_d(n,k,j,i) - T_F_d[n]) / dt;
      }
    }
    else
    {
      // ::M1::Sources::PrepareMatterSource_E_F_d(pm1, C, S, k, j, i);

      Assemble::Frames::sources_sc_E_sp_F_d(
        pm1,
        S.sc_E,
        S.sp_F_d,
        C.sc_chi,
        C.sc_E,
        C.sp_F_d,
        C.sc_eta,
        C.sc_kap_a,
        C.sc_kap_s,
        k, j, i
      );
    }
  }
  else
  {
    Sources::SetMatterSourceZero(S, k, j, i);
  }

  // If required, evolve (E,F_d); source treatment based on above -------------
  Real sc_Gam__ = eq_sc_Gam__;

  if (pm1.opt_solver.equilibrium_evolve)
  {
    Integrators::Explicit::StepExplicit_E_F_d(pm1, dt, C, P, I, S, k, j, i);


    // fiducial frame of evolved state
    sc_Gam__ = Assemble::Frames::ToFiducial(
      pm1,
      sc_J, st_H_u, sc_n,
      sc_chi,
      sc_E, sp_F_d, sc_nG,
      k, j, i
    );
  }

  if (!pm1.opt_solver.equilibrium_use_thick)
  {
    // Need to compute closure
    CL_C.Closure(k,j,i);
  }

  // update in accordance with average energies prepared in opac. -------------
  // sc_n(k,j,i)  = sc_J(k,j,i) / sc_avg_nrg(k,j,i);
  // sc_nG(k,j,i) = sc_Gam__ * sc_n(k,j,i);

  // read directly from opac.
  sc_n(k,j,i)  = pm1.eql.sc_n(C.ix_g,C.ix_s)(k,j,i);
  sc_nG(k,j,i) = sc_n(k,j,i) * eq_sc_Gam__;

  // std::printf("%.3e %.3e\n", std::abs(1 - sc_Gam__ / eq_sc_Gam__), sc_Gam__);

  // update based on what would be average energy and updated fid.
  // const Real avg_nrg = (pm1.eql.sc_J(ix_g,ix_s)(k,j,i) /
  //                       pm1.eql.sc_n(ix_g,ix_s)(k,j,i));
  // sc_nG(k,j,i) = sc_Gam__ * sc_J(k,j,i) / avg_nrg;

  // // Flooring
  // EnforcePhysical_nG(pm1, C, k, j, i);
  // sc_n(k,j,i) = sc_nG(k,j,i) / eq_sc_Gam__;  // propagate back

  // prepare eql. adjust sources
  if (pm1.opt_solver.equilibrium_src_nG)
  {
    if (pm1.opt_solver.equilibrium_use_diff_src)
    {
      S.sc_nG(k,j,i) = (sc_nG(k,j,i) - T_nG) / dt;
    }
    else
    {
      // ::M1::Sources::PrepareMatterSource_nG(pm1, C, S, k, j, i);

      Assemble::Frames::sources_sc_nG(
        pm1,
        S.sc_nG,
        C.sc_n, C.sc_eta_0, C.sc_kap_a_0, k, j, i
      );
    }
  }
  else
  {
    S.sc_nG(k,j,i) = 0;
  }

  // If required, evolve N explicitly based on eql. adjusted sources
  if (pm1.opt_solver.equilibrium_evolve)
  {
    Integrators::Explicit::StepExplicit_nG(pm1, dt, C, P, I, S, k, j, i);
    // propagate back (N.B. using evolved Gamma factor)
    sc_n(k,j,i) = sc_nG(k,j,i) / sc_Gam__;
  }

  // eql fall-back should be disabled -----------------------------------------
  pm1.opt.flux_lo_fallback_E = opt_flux_lo_fallback_E;
  pm1.opt.flux_lo_fallback_nG = opt_flux_lo_fallback_nG;
  // --------------------------------------------------------------------------
}

void MapReferenceEquilibrium(
  M1 & pm1,
  M1::vars_Eql & eq,
  Update::StateMetaVector & C,
  const int k,
  const int j,
  const int i
)
{

  const Real sc_sqrt_det_g = pm1.geom.sc_sqrt_det_g(k,j,i);
  const Real W = pm1.fidu.sc_W(k,j,i);
  const Real W2 = SQR(W);

  const Real eq_sc_J = eq.sc_J(C.ix_g,C.ix_s)(k,j,i);
  const Real eq_sc_n = eq.sc_n(C.ix_g,C.ix_s)(k,j,i);

  // inject reference values
  C.sc_J(k,j,i) = eq_sc_J;
  C.sc_n(k,j,i) = eq_sc_n;

  for (int a=0; a<D; ++a)
  {
    C.st_H_u(a,k,j,i) = 0;
  }

  // (sc_J, st_H_d) -> (sc_E, sp_F_d) reduces to:
  C.sc_E(k,j,i) = ONE_3RD * C.sc_J(k,j,i) * (
    4.0 * W2 - 1.0
  );

  for (int a=0; a<N; ++a)
  {
    C.sp_F_d(a,k,j,i) = 4.0 * ONE_3RD * W2 *
                        pm1.fidu.sp_v_d(a,k,j,i) * C.sc_J(k,j,i);
  }

  Real dotFv__ (0.0);
  for (int a=0; a<N; ++a)
  {
    dotFv__ += C.sp_F_d(a,k,j,i) * pm1.fidu.sp_v_u(a,k,j,i);
  }
  const Real Gamma__ = W / C.sc_J(k,j,i) * (
    C.sc_E(k,j,i) - dotFv__
  );

  // Ensure update preserves energy non-negativity
  EnforcePhysical_E_F_d(pm1, C, k, j, i);
  EnforcePhysical_nG(   pm1, C, k, j, i);

  // Prepare neutrino number density
  if (pm1.N_SPCS > 1)
  {
    C.sc_nG(k,j,i) = std::max(Gamma__ * C.sc_n(k,j,i), pm1.opt.fl_nG);
    C.sc_n(k,j,i) = C.sc_nG(k,j,i) / Gamma__;  // propagate back
  }

  pm1.radmat.sc_avg_nrg(C.ix_g,C.ix_s)(k,j,i) = W / C.sc_nG(k,j,i) * (
    C.sc_E(k,j,i) - dotFv__
  );

}

// ============================================================================
} // namespace M1::Equilibrium
// ============================================================================


//
// :D
//
