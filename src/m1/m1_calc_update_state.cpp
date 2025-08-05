// C++ standard headers
#include <iomanip>
#include <iostream>

// Athena++ headers
#include "m1.hpp"
#include "m1_calc_closure.hpp"
#include "m1_macro.hpp"
#include "m1_utils.hpp"
#include "m1_calc_update.hpp"
#include "m1_sources.hpp"
#include "m1_integrators.hpp"
#include "m1_set_equilibrium.hpp"

#if FLUID_ENABLED
#include "../eos/eos.hpp"
#endif

// External libraries
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_vector.h>


// ============================================================================
namespace M1::State {
// ============================================================================

void SlopeFallback(
  M1 * pm1,
  const Real dt,
  M1::vars_Lab & U_C,
  const M1::vars_Lab & U_P,
  const M1::vars_Lab & U_I,
  M1::vars_Source & U_S)
{
  using namespace Update;
  using namespace Closures;

  // Requires LO flux fallback
  assert(pm1->opt.flux_lo_fallback);

  const Real rat_sl_sc_E   = pm1->opt_solver.fb_rat_sl_E;
  const Real rat_sl_sp_F_d = pm1->opt_solver.fb_rat_sl_F_d;
  const Real rat_sl_sc_nG  = pm1->opt_solver.fb_rat_sl_nG;

  // This is ugly; define left / right limits (derived from MLOOP macro)
  const int IL = M1_IX_IL-M1_MSIZEI;
  const int IU = M1_IX_IU+M1_MSIZEI;

  const int JL = M1_IX_JL-M1_MSIZEJ;
  const int JU = M1_IX_JU+M1_MSIZEJ;

  const int KL = M1_IX_KL-M1_MSIZEK;
  const int KU = M1_IX_KU+M1_MSIZEK;

  // hybridize (into ho) pp corrected fluxes
  AA & mask_pp = pm1->ev_strat.masks.pp;

  for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<pm1->N_SPCS; ++ix_s)
  {
    // mask species index if using per-species
    const int ix_ms = (pm1->opt.flux_lo_fallback_species)
      ? ix_s
      : 0;

    M1::vars_Lab & C = const_cast<M1::vars_Lab &>(U_C);
    M1::vars_Lab & P = const_cast<M1::vars_Lab &>(U_P);
    M1::vars_Lab & I = const_cast<M1::vars_Lab &>(U_I);

    // Solution at next time
    AT_C_sca & C_sc_E = C.sc_E(ix_g,ix_s);
    AT_N_vec & C_sp_F_d = C.sp_F_d(ix_g,ix_s);
    AT_C_sca & C_sc_nG = C.sc_nG(ix_g,ix_s);

    Real rat_L, rat_R;
    Real d_p1, d_0, d_m1;

    M1_MLOOP3(k, j, i)
    if (pm1->MaskGet(k, j, i))
    {
      // hybridization factor starts with HO
      Real & hf = mask_pp(ix_ms,k,j,i);

      // D_i[f] := f_{i+1} - f_{i}

      if (hf == 1.0) continue;  // already LO, short-circuit

      if (rat_sl_sc_E > 0)
      {
        // x-dir --------------------------------------------------------------
        d_p1 = C_sc_E(k,j,i+2) - C_sc_E(k,j,i+1);
        d_0  = C_sc_E(k,j,i+1) - C_sc_E(k,j,i);
        d_m1 = C_sc_E(k,j,i)   - C_sc_E(k,j,i-1);

        rat_L = d_m1 / d_0;
        rat_R = d_0 / d_p1;

        // if idx in range, and rat too sharp, fall-back to LO
        hf = (i>IL) ? (
          (rat_L < rat_sl_sc_E) ? 1.0 : hf
        ) : hf;

        hf = (i<IU-1) ? (
          (rat_R < rat_sl_sc_E) ? 1.0 : hf
        ) : hf;

        // y-dir --------------------------------------------------------------
        d_p1 = C_sc_E(k,j+2,i) - C_sc_E(k,j+1,i);
        d_0  = C_sc_E(k,j+1,i) - C_sc_E(k,j,i);
        d_m1 = C_sc_E(k,j,i)   - C_sc_E(k,j-1,i);

        rat_L = d_m1 / d_0;
        rat_R = d_0 / d_p1;

        hf = (j>JL) ? (
          (rat_L < rat_sl_sc_E) ? 1.0 : hf
        ) : hf;

        hf = (j<JU-1) ? (
          (rat_R < rat_sl_sc_E) ? 1.0 : hf
        ) : hf;

        // z-dir --------------------------------------------------------------
        d_p1 = C_sc_E(k+2,j,i) - C_sc_E(k+1,j,i);
        d_0  = C_sc_E(k+1,j,i) - C_sc_E(k,j,i);
        d_m1 = C_sc_E(k,j,i)   - C_sc_E(k-1,j,i);

        rat_L = d_m1 / d_0;
        rat_R = d_0 / d_p1;

        // if idx in range, and rat too sharp, fall-back to LO
        hf = (k>KL) ? (
          (rat_L < rat_sl_sc_E) ? 1.0 : hf
        ) : hf;

        // if idx in range, and rat too sharp, fall-back to LO
        hf = (k<KU-1) ? (
          (rat_R < rat_sl_sc_E) ? 1.0 : hf
        ) : hf;
      }

      if (hf == 1.0) continue;

      if (rat_sl_sc_nG > 0)
      {
        // x-dir --------------------------------------------------------------
        d_p1 = C_sc_nG(k,j,i+2) - C_sc_nG(k,j,i+1);
        d_0  = C_sc_nG(k,j,i+1) - C_sc_nG(k,j,i);
        d_m1 = C_sc_nG(k,j,i)   - C_sc_nG(k,j,i-1);

        rat_L = d_m1 / d_0;
        rat_R = d_0 / d_p1;

        // if idx in range, and rat too sharp, fall-back to LO
        hf = (i>IL) ? (
          (rat_L < rat_sl_sc_nG) ? 1.0 : hf
        ) : hf;

        // if idx in range, and rat too sharp, fall-back to LO
        hf = (i<IU-1) ? (
          (rat_R < rat_sl_sc_nG) ? 1.0 : hf
        ) : hf;

        // y-dir --------------------------------------------------------------
        d_p1 = C_sc_nG(k,j+2,i) - C_sc_nG(k,j+1,i);
        d_0  = C_sc_nG(k,j+1,i) - C_sc_nG(k,j,i);
        d_m1 = C_sc_nG(k,j,i)   - C_sc_nG(k,j-1,i);

        rat_L = d_m1 / d_0;
        rat_R = d_0 / d_p1;

        // if idx in range, and rat too sharp, fall-back to LO
        hf = (j>JL) ? (
          (rat_L < rat_sl_sc_nG) ? 1.0 : hf
        ) : hf;

        // if idx in range, and rat too sharp, fall-back to LO
        hf = (j<JU-1) ? (
          (rat_R < rat_sl_sc_nG) ? 1.0 : hf
        ) : hf;

        // z-dir --------------------------------------------------------------
        d_p1 = C_sc_nG(k+2,j,i) - C_sc_nG(k+1,j,i);
        d_0  = C_sc_nG(k+1,j,i) - C_sc_nG(k,j,i);
        d_m1 = C_sc_nG(k,j,i)   - C_sc_nG(k-1,j,i);

        rat_L = d_m1 / d_0;
        rat_R = d_0 / d_p1;

        // if idx in range, and rat too sharp, fall-back to LO
        hf = (k>KL) ? (
          (rat_L < rat_sl_sc_nG) ? 1.0 : hf
        ) : hf;

        // if idx in range, and rat too sharp, fall-back to LO
        hf = (k<KU-1) ? (
          (rat_R < rat_sl_sc_nG) ? 1.0 : hf
        ) : hf;
      }

      if (hf == 1.0) continue;

      if (rat_sl_sp_F_d > 0)
      for (int a=0; a<N; ++a)
      {
        // x-dir --------------------------------------------------------------
        d_p1 = C_sp_F_d(a,k,j,i+2) - C_sp_F_d(a,k,j,i+1);
        d_0  = C_sp_F_d(a,k,j,i+1) - C_sp_F_d(a,k,j,i);
        d_m1 = C_sp_F_d(a,k,j,i)   - C_sp_F_d(a,k,j,i-1);

        rat_L = d_m1 / d_0;
        rat_R = d_0 / d_p1;

        // if idx in range, and rat too sharp, fall-back to LO
        hf = (i>IL) ? (
          (rat_L < rat_sl_sp_F_d) ? 1.0 : hf
        ) : hf;

        // if idx in range, and rat too sharp, fall-back to LO
        hf = (i<IU-1) ? (
          (rat_R < rat_sl_sp_F_d) ? 1.0 : hf
        ) : hf;

        // y-dir --------------------------------------------------------------
        d_p1 = C_sp_F_d(a,k,j+2,i) - C_sp_F_d(a,k,j+1,i);
        d_0  = C_sp_F_d(a,k,j+1,i) - C_sp_F_d(a,k,j,i);
        d_m1 = C_sp_F_d(a,k,j,i)   - C_sp_F_d(a,k,j-1,i);

        rat_L = d_m1 / d_0;
        rat_R = d_0 / d_p1;

        // if idx in range, and rat too sharp, fall-back to LO
        hf = (j>JL) ? (
          (rat_L < rat_sl_sp_F_d) ? 1.0 : hf
        ) : hf;

        // if idx in range, and rat too sharp, fall-back to LO
        hf = (j<JU-1) ? (
          (rat_R < rat_sl_sp_F_d) ? 1.0 : hf
        ) : hf;

        // z-dir --------------------------------------------------------------
        d_p1 = C_sp_F_d(a,k+2,j,i) - C_sp_F_d(a,k+1,j,i);
        d_0  = C_sp_F_d(a,k+1,j,i) - C_sp_F_d(a,k,j,i);
        d_m1 = C_sp_F_d(a,k,j,i)   - C_sp_F_d(a,k-1,j,i);

        rat_L = d_m1 / d_0;
        rat_R = d_0 / d_p1;

        // if idx in range, and rat too sharp, fall-back to LO
        hf = (k>KL) ? (
          (rat_L < rat_sl_sp_F_d) ? 1.0 : hf
        ) : hf;

        // if idx in range, and rat too sharp, fall-back to LO
        hf = (k<KU-1) ? (
          (rat_R < rat_sl_sp_F_d) ? 1.0 : hf
        ) : hf;
      }

    }
  }

}

// ============================================================================
} // namespace M1::State
// ============================================================================


// ============================================================================
namespace M1 {
// ============================================================================

void M1::ResetEvolutionStrategy()
{
  AthenaArray<t_sln_r> & mask_sln_r = pm1->ev_strat.masks.solution_regime;
  AthenaArray<t_src_t> & mask_src_t = pm1->ev_strat.masks.source_treatment;
  mask_sln_r.Fill(t_sln_r::noop);
  mask_src_t.Fill(t_src_t::noop);
}

void M1::PrepareEvolutionStrategy(const Real dt,
                                  const Real kap_a,
                                  const Real kap_s,
                                  const Real rho,
                                  t_sln_r & mask_sln_r,
                                  t_src_t & mask_src_t)
{
  // Equilibrium detected in Weak-rates; short-circuit
  if ((mask_sln_r == t_sln_r::equilibrium) ||
      (mask_sln_r == t_sln_r::equilibrium_wr))
  {
    if (pm1->opt_solver.equilibrium_sources)
    {
      mask_src_t = t_src_t::full;
    }
    else
    {
      mask_src_t = t_src_t::set_zero;
    }
    return;
  }

  // equilibrium regime (additional thick limiter)
  if((opt_solver.src_lim_thick > 0) &&
     (SQR(dt) * (kap_a * (kap_a + kap_s)) >
      SQR(opt_solver.src_lim_thick)) &&
     (rho >= pm1->opt_solver.eql_rho_min))
  {
    mask_sln_r = t_sln_r::equilibrium;
    if (pm1->opt_solver.equilibrium_sources)
    {
      mask_src_t = t_src_t::full;
    }
    else
    {
      mask_src_t = t_src_t::set_zero;
    }
  }
  // scattering regime
  else if((opt_solver.src_lim_scattering > 0) &&
          (dt * kap_s > opt_solver.src_lim_scattering))
  {
    mask_sln_r = t_sln_r::scattering;
    mask_src_t = t_src_t::set_zero;
  }
  // Non-stiff regime
  else if ((dt * kap_a < 1) &&
           (dt * kap_s < 1))
  {
    mask_sln_r = t_sln_r::non_stiff;
    mask_src_t = t_src_t::full;
  }
  // stiff refime
  else
  {
    mask_sln_r = t_sln_r::stiff;
    mask_src_t = t_src_t::full;
  }
}

void M1::PrepareEvolutionStrategyCommon(const Real dt)
{
  AthenaArray<t_sln_r> & mask_sln_r = pm1->ev_strat.masks.solution_regime;
  AthenaArray<t_src_t> & mask_src_t = pm1->ev_strat.masks.source_treatment;

  for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
  {
    M1_MLOOP3(k, j, i)
    if (pm1->MaskGet(k, j, i))
    {
      // First check if already in equilibrium from weak rates
      if ((mask_sln_r(ix_g, k, j, i) == t_sln_r::equilibrium) ||
          (mask_sln_r(ix_g, k, j, i) == t_sln_r::equilibrium_wr))
      {
        if (pm1->opt_solver.equilibrium_sources)
        {
          mask_src_t(ix_g, k, j, i) = t_src_t::full;
        }
        else
        {
          mask_src_t(ix_g, k, j, i) = t_src_t::set_zero;
        }
        continue;
      }

      // Check opacity-based regimes using maximum opacity across all species
      Real max_kap_prod = 0.0;
      Real max_kap_s = 0.0;

      bool non_stiff = true;

      const Real rho = hydro.sc_w_rho(k,j,i);

      // Find maximum opacities across all species
      for (int ix_s=0; ix_s<pm1->N_SPCS; ++ix_s)
      {
        const Real kap_a = pm1->radmat.sc_kap_a(ix_g,ix_s)(k,j,i);
        const Real kap_s = pm1->radmat.sc_kap_s(ix_g,ix_s)(k,j,i);

        max_kap_prod = std::max(
          max_kap_prod,
          kap_a * (kap_a + kap_s)
        );

        non_stiff = non_stiff && (dt * kap_a < 1) && (dt * kap_s < 1);
        max_kap_s = std::max(max_kap_s, kap_s);
      }

      // Determine the regime based on maximum opacities
      // Priority: thick (equilibrium) > scattering > non-stiff > stiff

      // Check thick regime
      if ((opt_solver.src_lim_thick > 0) &&
          (SQR(dt) * max_kap_prod >
           SQR(opt_solver.src_lim_thick)) &&
          (rho >= pm1->opt_solver.eql_rho_min))
      {
        mask_sln_r(ix_g, k, j, i) = t_sln_r::equilibrium;
        if (pm1->opt_solver.equilibrium_sources)
        {
          mask_src_t(ix_g, k, j, i) = t_src_t::full;
        }
        else
        {
          mask_src_t(ix_g, k, j, i) = t_src_t::set_zero;
        }
      }
      // Check scattering regime
      else if ((opt_solver.src_lim_scattering > 0) &&
               (dt * max_kap_s > opt_solver.src_lim_scattering))
      {
        mask_sln_r(ix_g, k, j, i) = t_sln_r::scattering;
        mask_src_t(ix_g, k, j, i) = t_src_t::set_zero;
      }
      // Check non-stiff regime
      else if (non_stiff)
      {
        mask_sln_r(ix_g, k, j, i) = t_sln_r::non_stiff;
        mask_src_t(ix_g, k, j, i) = t_src_t::full;
      }
      // stiff regime
      else
      {
        mask_sln_r(ix_g, k, j, i) = t_sln_r::stiff;
        mask_src_t(ix_g, k, j, i) = t_src_t::full;
      }
    }
  }
}

void M1::PrepareEvolutionStrategy(const Real dt)
{
  // If we reduce to common then we short-circuit this function
  if (opt_solver.solver_reduce_to_common)
  {
    PrepareEvolutionStrategyCommon(dt);
    return;
  }

  AthenaArray<t_sln_r> & mask_sln_r = pm1->ev_strat.masks.solution_regime;
  AthenaArray<t_src_t> & mask_src_t = pm1->ev_strat.masks.source_treatment;

  // Revert to stiff / kill sources, as required ------------------------------
  for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<pm1->N_SPCS; ++ix_s)
  M1_MLOOP3(k, j, i)
  if (pm1->MaskGet(k, j, i))
  {
    const Real kap_a = pm1->radmat.sc_kap_a(ix_g,ix_s)(k,j,i);
    const Real kap_s = pm1->radmat.sc_kap_s(ix_g,ix_s)(k,j,i);

    PrepareEvolutionStrategy(dt, kap_a, kap_s,
                             hydro.sc_w_rho(k,j,i),
                             mask_sln_r(ix_g, ix_s, k, j, i),
                             mask_src_t(ix_g, ix_s, k, j, i));
  }

  // Treat species uniformly, if required -------------------------------------
  // dead-code
  /*
  if (opt_solver.solver_reduce_to_common)
  {
    auto reduce_all = [&](const int ix_g, const int k, const int j, const int i)
    {
      // First pass: determine the most restrictive regime across all species
      bool any_equilibrium = false;
      bool any_stiff = false;
      bool any_scattering = false;
      bool any_nonstiff = false;

      for (int ix_s = 0; ix_s < N_SPCS; ++ix_s)
      {
        t_sln_r regime = mask_sln_r(ix_g, ix_s, k, j, i);

        if (regime == t_sln_r::equilibrium)
        {
          any_equilibrium = true;
        }
        else if (regime == t_sln_r::scattering)
        {
          any_scattering = true;
        }
        else if (regime == t_sln_r::stiff)
        {
          any_stiff = true;
        }
        else if (regime == t_sln_r::non_stiff)
        {
          any_nonstiff = true;
        }
      }

      // Second pass: apply the most restrictive regime to all species
      // Priority: equilibrium > scattering > stiff > non_stiff
      t_sln_r common_regime;
      t_src_t common_treatment;

      if (any_equilibrium)
      {
        common_regime = t_sln_r::equilibrium;
        common_treatment = t_src_t::set_zero;
      }
      else if (any_scattering)
      {
        common_regime = t_sln_r::scattering;
        common_treatment = t_src_t::set_zero;
      }
      else if (any_stiff)
      {
        common_regime = t_sln_r::stiff;
        common_treatment = t_src_t::full;
      }
      else if (any_nonstiff)
      {
        common_regime = t_sln_r::non_stiff;
        common_treatment = t_src_t::full;
      }
      else
      {
        common_regime = t_sln_r::noop;
        common_treatment = t_src_t::noop;
      }

      // Apply the common regime to all species for this energy group
      for (int ix_s = 0; ix_s < N_SPCS; ++ix_s)
      {
        mask_sln_r(ix_g, ix_s, k, j, i) = common_regime;
        mask_src_t(ix_g, ix_s, k, j, i) = common_treatment;

            // set thick limit
            if (common_regime== t_sln_r::equilibrium)
          for (int ix_s=0; ix_s<3; ++ix_s)
          {
            pm1->lab_aux.sc_chi(ix_g,ix_s)(k,j,i) = ONE_3RD;
            pm1->lab_aux.sc_xi(ix_g,ix_s)(k,j,i) = 0;
          }
      }
    };

    for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
    M1_MLOOP3(k, j, i)
    if (pm1->MaskGet(k, j, i))
    {
      reduce_all(ix_g, k, j, i);
    }

  }
  */

}

void M1::CalcUpdate(const int stage,
                    Real const dt,
                    AA & u_pre,
                    AA & u_cur,
		                AA & u_inh,
                    AA & u_src)
{
  using namespace Update;
  using namespace Sources;
  using namespace Integrators;
  using namespace Closures;

  // can we short-circuit? ----------------------------------------------------
  if (0)
  if (pm1->opt.flux_lo_fallback && pmy_block->NeighborBlocksSameLevel())
  {
    assert(!opt.flux_lo_fallback_species);  // B.D. needs fix for species
    AA & mask_pp = pm1->ev_strat.masks.pp;
    Real max_theta = 0.0;
    M1_MLOOP3(k, j, i)
    {
      max_theta = std::min(max_theta, mask_pp(k,j,i));
    }

    if (max_theta == 0.0)
    {
      return;
    }
  }

  // setup aliases ------------------------------------------------------------
  vars_Lab U_P { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u_pre, U_P);

  vars_Lab U_C { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u_cur, U_C);

  vars_Lab U_I { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u_inh, U_I);

  vars_Source U_S { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesSource(u_src, U_S);

  // local settings -----------------------------------------------------------
  const bool use_full_limiter = opt_solver.full_lim >= 0;
  const bool use_src_limiter = opt_solver.src_lim >= 0;
  const bool use_fb_lo_matter = opt.flux_lo_fallback && (
    opt_solver.flux_lo_fallback_tau_min > -1 ||
    ((opt_solver.flux_lo_fallback_Ye_min > -1) &&
     (opt_solver.flux_lo_fallback_Ye_max > -1))
  );

  const bool use_fb_slope = (
    (opt_solver.fb_rat_sl_E > 0) ||
    (opt_solver.fb_rat_sl_F_d > 0) ||
    (opt_solver.fb_rat_sl_nG > 0)
  );

  // construct unlimited solution ---------------------------------------------
  // uses ev_strat.masks.{solution_regime, source_treatment} internally
  DispatchIntegrationMethod(*this, dt, U_C, U_P, U_I, U_S);

  // check whether current sources give physical matter coupling --------------
  if (use_fb_lo_matter)
  {
    Sources::Limiter::CheckPhysicalFallback(this, dt, U_S);
  }

  // check whether solution is developing too rapidly -------------------------
  if (use_fb_slope)
  {
    State::SlopeFallback(this, dt, U_C, U_P, U_I, U_S);
  }

  // prepare source & apply limiting mask -------------------------------------
  if (use_full_limiter)
  {
    AT_C_sca & theta = sources.theta;
    Sources::Limiter::PrepareFull(this, dt, theta, U_C, U_P, U_I, U_S);
    Sources::Limiter::ApplyFull(this, dt, theta, U_C, U_P, U_I, U_S);
  }

  if (use_src_limiter)
  {
    AT_C_sca & theta = sources.theta;

    Sources::Limiter::Prepare(this, dt, theta, U_C, U_P, U_I, U_S);

    // adjust matter sources with theta mask and construct limited solution
    Sources::Limiter::Apply(this, dt, theta, U_C, U_P, U_I, U_S);
  }

  // should we enforce the equilibrium ? --------------------------------------
  if (opt_solver.equilibrium_enforce ||
      (opt_solver.equilibrium_initial && (pmy_mesh->time == 0)) &&
      (stage == 1))
  {
    M1_MLOOP3(k, j, i)
    if (MaskGet(k, j, i))
    // if (MaskGetHybridize(k,j,i))
    {
      bool equilibriate = false;
      for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
      for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
      {
        if (pm1->GetMaskSolutionRegime(ix_g,ix_s,k,j,i) ==
            t_sln_r::equilibrium)
        {
          equilibriate = true;
          continue;
        }
      }

      if (equilibriate)
      {
        if (pm1->opt.retain_equilibrium)
        {
          for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
          for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
          {
            StateMetaVector C = ConstructStateMetaVector(*pm1, U_C, ix_g, ix_s);
            StateMetaVector P = ConstructStateMetaVector(*pm1, U_P, ix_g, ix_s);
            StateMetaVector I = ConstructStateMetaVector(*pm1, U_I, ix_g, ix_s);

            SourceMetaVector S = ConstructSourceMetaVector(*pm1, U_S, ix_g, ix_s);

            Equilibrium::MapReferenceEquilibrium(*pm1, pm1->eql, C, k, j, i);

            S.sc_E(k,j,i) = C.sc_E(k,j,i) - (
              P.sc_E(k,j,i) + dt * I.sc_E(k,j,i)
            );
            S.sc_E(k,j,i) /= dt;

            S.sc_nG(k,j,i) = C.sc_nG(k,j,i) - (
              P.sc_nG(k,j,i) + dt * I.sc_nG(k,j,i)
            );
            S.sc_nG(k,j,i) /= dt;

            for (int a=0; a<N; ++a)
            {
              S.sp_F_d(a,k,j,i) = C.sp_F_d(a,k,j,i) - (
                P.sp_F_d(a,k,j,i) + dt * I.sp_F_d(a,k,j,i)
              );

              S.sp_F_d(a,k,j,i) /= dt;
            }
          }
        }
        else
        {
          Equilibrium::SetEquilibrium(*this, U_C, U_S, k, j, i);

          // Freeze state of point (handles subsequent CalcUpdate calls)
          // for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
          // for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
          // {
          //   pm1->SetMaskSolutionRegime(t_sln_r::noop, ix_g, ix_s, k, j, i);
          // }
        }
      }
    }
  }

  // rescale sources by dt for convenience in M1+N0 -> GR-evo coupling --------
  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    SourceMetaVector S = ConstructSourceMetaVector(*this, U_S, ix_g, ix_s);

    M1_MLOOP3(k, j, i)
    if (MaskGet(k, j, i))
    // if (MaskGetHybridize(k,j,i))
    {
      // S -> dt * S for (nG, E, F_d) components
      InPlaceScalarMul_nG_E_F_d(dt, S, k, j, i);
    }
  }

  // assemble remaining auxiliary quantities ----------------------------------
  // BD: TODO: double check / refactor to new task & check nstages there
  const int nstages = 2;

  if ((stage==nstages) &&
      (N_GRPS == 1) && (N_SPCS == 3))
  {
    const AT_C_sca & C_sc_E_00 = U_C.sc_E(0,0);
    const AT_C_sca & C_sc_E_01 = U_C.sc_E(0,1);
    const AT_C_sca & C_sc_E_02 = U_C.sc_E(0,2);

    const AT_C_sca & P_sc_E_00 = U_P.sc_E(0,0);
    const AT_C_sca & P_sc_E_01 = U_P.sc_E(0,1);
    const AT_C_sca & P_sc_E_02 = U_P.sc_E(0,2);

    const AT_C_sca & I_sc_E_00 = U_I.sc_E(0,0);
    const AT_C_sca & I_sc_E_01 = U_I.sc_E(0,1);
    const AT_C_sca & I_sc_E_02 = U_I.sc_E(0,2);

#if FLUID_ENABLED && USETM
    const AT_C_sca & C_sc_nG_00 = U_C.sc_nG(0,0);
    const AT_C_sca & C_sc_nG_01 = U_C.sc_nG(0,1);
    // const AT_C_sca & C_sc_nG_02 = U_C.sc_nG(0,2);

    const AT_C_sca & P_sc_nG_00 = U_P.sc_nG(0,0);
    const AT_C_sca & P_sc_nG_01 = U_P.sc_nG(0,1);
    // const AT_C_sca & P_sc_nG_02 = U_P.sc_nG(0,2);

    const AT_C_sca & I_sc_nG_00 = U_I.sc_nG(0,0);
    const AT_C_sca & I_sc_nG_01 = U_I.sc_nG(0,1);
    // const AT_C_sca & I_sc_nG_02 = U_I.sc_nG(0,2);

    const Real mb = pmy_block->peos->GetEOS().GetRawBaryonMass();
#endif

    M1_ILOOP3(k, j, i)
    if (MaskGet(k, j, i))
    {
      const Real E_star_00__ = P_sc_E_00(k,j,i) + dt * I_sc_E_00(k,j,i);
      const Real DE_00 = C_sc_E_00(k,j,i) - E_star_00__;
      const Real E_star_01__ = P_sc_E_01(k,j,i) + dt * I_sc_E_01(k,j,i);
      const Real DE_01 = C_sc_E_01(k,j,i) - E_star_01__;
      const Real E_star_02__ = P_sc_E_02(k,j,i) + dt * I_sc_E_02(k,j,i);
      const Real DE_02 = C_sc_E_02(k,j,i) - E_star_02__;

      net.heat(k,j,i) = DE_00 + DE_01 + DE_02;
#if FLUID_ENABLED && USETM
      const Real nG_star_00__ = P_sc_nG_00(k,j,i) + dt * I_sc_nG_00(k,j,i);
      const Real DN_00 = C_sc_nG_00(k,j,i) - nG_star_00__;
      const Real nG_star_01__ = P_sc_nG_01(k,j,i) + dt * I_sc_nG_01(k,j,i);
      const Real DN_01 = C_sc_nG_01(k,j,i) - nG_star_01__;
      // const Real nG_star_02__ = P_sc_nG_02(k,j,i) + dt * I_sc_nG_02(k,j,i);
      // const Real DN_02 = C_sc_nG_02(k,j,i) - nG_star_02__;

      net.abs(k,j,i) = mb * (-DN_00 + DN_01);
#endif
    }

  }

}

// ============================================================================
} // namespace M1
// ============================================================================

//
// :D
//
