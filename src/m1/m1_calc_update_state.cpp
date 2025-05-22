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

// External libraries
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_vector.h>

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
                                  t_sln_r & mask_sln_r,
                                  t_src_t & mask_src_t)
{
  // Equilibrium detected in Weak-rates; short-circuit
  if (mask_sln_r == t_sln_r::equilibrium)
  {
    mask_src_t = t_src_t::set_zero;
    return;
  }

  // equilibrium regime (additional thick limiter)
  if((opt_solver.src_lim_thick > 0) &&
     (SQR(dt) * (kap_a * (kap_a + kap_s)) >
      SQR(opt_solver.src_lim_thick)))
  {
    mask_sln_r = t_sln_r::equilibrium;
    mask_src_t = t_src_t::set_zero;
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
      if (mask_sln_r(ix_g, k, j, i) == t_sln_r::equilibrium)
      {
        // Already in equilibrium, ensure source treatment is consistent
        mask_src_t(ix_g, k, j, i) = t_src_t::set_zero;
        continue;
      }

      // Check opacity-based regimes using maximum opacity across all species
      Real max_kap_prod = 0.0;
      Real max_kap_s = 0.0;

      bool non_stiff = true;

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
           SQR(opt_solver.src_lim_thick)))
      {
        mask_sln_r(ix_g, k, j, i) = t_sln_r::equilibrium;
        mask_src_t(ix_g, k, j, i) = t_src_t::set_zero;
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

  // construct unlimited solution ---------------------------------------------
  // uses ev_strat.masks.{solution_regime, source_treatment} internally
  DispatchIntegrationMethod(*this, dt, U_C, U_P, U_I, U_S);

  // prepare source limiting mask ---------------------------------------------
  if (opt_solver.src_lim >= 0)
  {
    AT_C_sca & theta = sources.theta;

    Sources::Limiter::Prepare(this, dt, theta, U_C, U_P, U_I, U_S);

    // adjust matter sources with theta mask and construct limited solution
    Sources::Limiter::Apply(this, dt, theta, U_C, U_P, U_I, U_S);
  }

  // should we enforce the equilibirum ? --------------------------------------
  if (opt_solver.equilibrium_enforce ||
      (opt_solver.equilibrium_initial && (pmy_mesh->time == 0)))
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
        Equilibrium::SetEquilibrium(*this, U_C, U_S, k, j, i);

        // Freeze state of point (handles subsequent CalcUpdate calls)
        for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
        for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
        {
          pm1->SetMaskSolutionRegime(t_sln_r::noop, ix_g, ix_s, k, j, i);
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
  // BD: TODO - shift elsewhere
  // for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  // for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  // {
  //   StateMetaVector C = ConstructStateMetaVector(*this, U_C, ix_g, ix_s);

  //   M1_ILOOP3(k, j, i)
  //   if (MaskGet(k, j, i))
  //   {
  //     // BD: TODO - net.abs, net.heat
  //   }
  // }
}

// ============================================================================
} // namespace M1
// ============================================================================

//
// :D
//
