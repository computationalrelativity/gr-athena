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

// External libraries
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_vector.h>

// ============================================================================
namespace M1 {
// ============================================================================

void M1::PrepareEvolutionStrategy(const Real dt,
                                  const Real kap_a,
                                  const Real kap_s,
                                  t_sln_r & mask_sln_r,
                                  t_src_t & mask_src_t)
{
  // Non-stiff regime
  if ((dt * kap_a < 1) &&
      (dt * kap_s < 1))
  {
    mask_sln_r = t_sln_r::non_stiff;
    mask_src_t = t_src_t::full;
  }
  // equilibrium regime
  else if((opt_solver.src_lim_thick > 0) &&
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
  // stiff refime
  else
  {
    mask_sln_r = t_sln_r::stiff;
    mask_src_t = t_src_t::full;
  }
}

void M1::PrepareEvolutionStrategy(const Real dt)
{
  AthenaArray<t_sln_r> & mask_sln_r = pm1->ev_strat.masks.solution_regime;
  AthenaArray<t_src_t> & mask_src_t = pm1->ev_strat.masks.source_treatment;

  mask_sln_r.Fill(t_sln_r::noop);
  mask_src_t.Fill(t_src_t::noop);

  // Revert to stiff / kill sources, as required ------------------------------
  for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<pm1->N_SPCS; ++ix_s)
  M1_ILOOP3(k, j, i)
  if (pm1->MaskGet(k, j, i))
  {
    const Real kap_a = pm1->radmat.sc_kap_a(ix_g,ix_s)(k,j,i);
    const Real kap_s = pm1->radmat.sc_kap_s(ix_g,ix_s)(k,j,i);

    PrepareEvolutionStrategy(dt, kap_a, kap_s,
                             mask_sln_r(ix_g, ix_s, k, j, i),
                             mask_src_t(ix_g, ix_s, k, j, i));
  }

  // Treat species uniformly, if required -------------------------------------
  if (opt_solver.solver_reduce_to_common)
  {
    auto reduce_all = [&](const int ix_g, const int k, const int j, const int i)
    {

    };

    auto reduce_any = [&]()
    {

    };

    // BD: TODO - need to implement reduction
    assert(false);
  }

}

// ----------------------------------------------------------------------------
// Function to update the state vector
void M1::CalcUpdateNew(Real const dt,
                       AA & u_pre,
                       AA & u_cur,
		                   AA & u_inh,
                       AA & u_src)
{
  using namespace Update;
  using namespace Sources;
  using namespace Integrators;
  using namespace Closures;

  // setup aliases ------------------------------------------------------------
  vars_Lab U_P { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u_pre, U_P);

  vars_Lab U_C { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u_cur, U_C);

  vars_Lab U_I { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u_inh, U_I);

  vars_Source U_S { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesSource(u_src, U_S);

  // evolution strategy / source treatment ------------------------------------
  PrepareEvolutionStrategy(dt);

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

  // rescale sources by dt for convenience in M1+N0 -> GR-evo coupling --------
  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    SourceMetaVector S = ConstructSourceMetaVector(*this, U_S, ix_g, ix_s);

    M1_ILOOP3(k, j, i)
    if (MaskGet(k, j, i))
    {
      // S -> dt * S for (nG, E, F_d) components
      InPlaceScalarMul_nG_E_F_d(dt, S, k, j, i);
    }
  }

  // assemble remaining auxiliary quantities ----------------------------------
  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    StateMetaVector C = ConstructStateMetaVector(*this, U_C, ix_g, ix_s);

    M1_ILOOP3(k, j, i)
    if (MaskGet(k, j, i))
    {
      // BD: TODO - avg_nrg, net.abs, net.heat
    }
  }


}

// ============================================================================
} // namespace M1
// ============================================================================

//
// :D
//