// C++ standard headers
#include <cmath>
#include <iostream>

// Athena++ headers
#include "m1_sources.hpp"
#include "m1_calc_closure.hpp"
#include "m1_calc_update.hpp"
#include "m1_utils.hpp"

#if FLUID_ENABLED
#include "../hydro/hydro.hpp"
#include "../eos/eos.hpp"
#include "../scalars/scalars.hpp"
#endif // FLUID_ENABLED

// ============================================================================
namespace M1::Sources {
// ============================================================================

void SetMatterSourceZero(
  Update::SourceMetaVector & S,
  const int k, const int j, const int i)
{
  // S.sc_nG(k,j,i) = 0;
  S.sc_E(k,j,i) = 0;

  for (int a=0; a<N; ++a)
  {
    S.sp_F_d(a,k,j,i) = 0;
  }
}

void PrepareMatterSource_E_F_d(
  M1 & pm1,
  const Update::StateMetaVector & V,
  Update::SourceMetaVector & S,
  const int k, const int j, const int i)
{
  // Extract source mask value for potential short-circuit treatment ----------
  typedef M1::evolution_strategy::opt_source_treatment ost;
  ost st = pm1.GetMaskSourceTreatment(0,0,k,j,i);

  if (st == ost::set_zero)
  {
    SetMatterSourceZero(S, k, j, i);
    return;
  }
  // --------------------------------------------------------------------------

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
    k, j, i
  );
}

void PrepareMatterSource_nG(
  M1 & pm1,
  const Update::StateMetaVector & V,
  Update::SourceMetaVector & S,
  const int k, const int j, const int i)
{
  // Extract source mask value for potential short-circuit treatment ----------
  typedef M1::evolution_strategy::opt_source_treatment ost;
  ost st = pm1.GetMaskSourceTreatment(0,0,k,j,i);

  if (st == ost::set_zero)
  {
    S.sc_nG(k,j,i) = 0;
    return;
  }
  // --------------------------------------------------------------------------

  Assemble::Frames::sources_sc_nG(
    pm1,
    S.sc_nG,
    V.sc_n, V.sc_eta_0, V.sc_kap_a_0, k, j, i
  );
}

// Given nG and a StateMetaVector construct n [i.e. prepare Gam]
void Prepare_n_from_nG(
  M1 & pm1,
  const Update::StateMetaVector & V,
  const int k, const int j, const int i)
{
  Assemble::Frames::ToFiducial(
    pm1,
    V.sc_J, V.st_H_u, V.sc_n,
    V.sc_chi,
    V.sc_E, V.sp_F_d, V.sc_nG,
    k, j, i
  );
}

// ============================================================================
namespace Minerbo {
// ============================================================================

void PrepareMatterSourceJacobian_E_F_d(
  ::M1::M1 & pm1,
  const Real dt,
  AA & J,                               // Storage for Jacobian
  const Update::StateMetaVector & C,    // current step
  const int k, const int j, const int i)
{
  Assemble::Frames::Jacobian_sc_E_sp_F_d(
    pm1,
    J,
    C.sc_chi, C.sc_E, C.sp_F_d, C.sc_kap_a, C.sc_kap_s,
    k, j, i
  );
}

// ============================================================================
} // namespace M1::Sources::Minerbo
// ============================================================================

// ============================================================================
namespace Limiter {
// ============================================================================

void Prepare(
  M1 * pm1,
  const Real dt,
  AT_C_sca & theta,
  const M1::vars_Lab & U_C,
  const M1::vars_Lab & U_P,
  const M1::vars_Lab & U_I,
  const M1::vars_Source & U_S)
{
  // parameters & aliases------------------------------------------------------
  MeshBlock * pmb = pm1->pmy_block;
#if FLUID_ENABLED
  const Real mb_raw = pmb->peos->GetEOS().GetRawBaryonMass();
#endif

  const Real par_src_lim = pm1->opt_solver.src_lim;
  const Real par_src_Ye_min = pm1->opt_solver.src_lim_Ye_min;
  const Real par_src_Ye_max = pm1->opt_solver.src_lim_Ye_max;

  M1::vars_Lab & C = const_cast<M1::vars_Lab &>(U_C);
  M1::vars_Lab & P = const_cast<M1::vars_Lab &>(U_P);
  M1::vars_Lab & I = const_cast<M1::vars_Lab &>(U_I);

  theta.Fill(1.0);
  // --------------------------------------------------------------------------

  M1_MLOOP3(k, j, i)
  if (pm1->MaskGet(k, j, i))
  // if (pm1->MaskGetHybridize(k,j,i))
  {
    Real & theta__ = theta(k,j,i);

#if FLUID_ENABLED
    const Real & tau = pmb->phydro->u(IEN, k, j, i);
    const Real & w_Y_e = pm1->hydro.sc_w_Ye(0,k,j,i);

    const Real D = (
      pm1->hydro.sc_W(k,j,i) *
      pm1->hydro.sc_w_rho(k,j,i) *
      pm1->geom.sc_sqrt_det_g(k,j,i)
    );

    Real Dtau_sum (0);
    Real DDYe (0);
#endif // FLUID_ENABLED

    for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
    for (int ix_s=0; ix_s<pm1->N_SPCS; ++ix_s)
    {
      // these points have zero sources
      /*
      if (pm1->GetMaskSolutionRegime(ix_g,ix_s,k,j,i) ==
          M1::M1::t_sln_r::equilibrium)
      {
        theta__ = 0;
        continue;
      }
      */

      // C = P + dt (I + S[C])
      // => dt S[C] = C - (P + dt I)
      //
      // N.B.
      // We take the difference rather than value directly from the sources;
      // This is because values may differ based on solution regime selected
      const Real Estar = (
        P.sc_E(ix_g,ix_s)(k,j,i) + dt * I.sc_E(ix_g,ix_s)(k,j,i)
      );
      const Real Nstar = (
        P.sc_nG(ix_g,ix_s)(k,j,i) + dt * I.sc_nG(ix_g,ix_s)(k,j,i)
      );

      const Real DE = C.sc_E(ix_g,ix_s)(k,j,i)  - Estar;
      const Real DN = C.sc_nG(ix_g,ix_s)(k,j,i) - Nstar;

      if (pm1->opt_solver.limit_src_radiation)
      {
        if (DE < 0)
        {
          theta__ = std::min(
            -par_src_lim * std::max(
              Estar, 0.0
            ) / DE, theta__
          );
        }

        if (DN < 0)
        {
          theta__ = std::min(
            -par_src_lim * std::max(
              Nstar, 0.0
            ) / DN, theta__
          );
        }
      }

#if FLUID_ENABLED
      // Cf. CoupleSourcesHydro, CoupleSourcesYe
      Dtau_sum -= DE;
      if (pm1->N_SPCS == 3)
        DDYe += mb_raw * DN * ( (ix_s == 1) - (ix_s == 0) );
#endif // FLUID_ENABLED
    }

    // limiting based on sums -------------------------------------------------
    // BD: TODO - double check factors in the following
#if FLUID_ENABLED
    if (!pm1->opt_solver.limit_src_fluid)
    {
      continue;
    }

    if (Dtau_sum < 0)
    {
      theta__ = std::min(
        -par_src_lim * std::max(
          tau, 0.0
        ) / Dtau_sum,
        theta__
      );
    }

    if (DDYe > 0)
    {
      theta__ = std::min(
        par_src_lim * std::max(
          D * (par_src_Ye_max - w_Y_e), 0.0
        ) / DDYe,
        theta__
      );
    }
    else if (DDYe < 0)
    {
      theta__ = std::min(
        par_src_lim * std::min(
          D * (par_src_Ye_min - w_Y_e), 0.0
        ) / DDYe,
        theta__
      );
    }

#endif // FLUID_ENABLED

  }

  // Finally enforce mask to be non-negative ----------------------------------
  M1_MLOOP3(k, j, i)
  if (pm1->MaskGet(k, j, i))
  // if (pm1->MaskGetHybridize(k,j,i))
  {
    theta(k, j, i) = std::max(0.0, theta(k, j, i));
  }
}

void Apply(
  M1 * pm1,
  const Real dt,
  AT_C_sca & theta,
  M1::vars_Lab & U_C,
  const M1::vars_Lab & U_P,
  const M1::vars_Lab & U_I,
  M1::vars_Source & U_S)
{
  using namespace Update;
  using namespace Closures;

  for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<pm1->N_SPCS; ++ix_s)
  {
    StateMetaVector C = ConstructStateMetaVector(*pm1, U_C, ix_g, ix_s);
    StateMetaVector P = ConstructStateMetaVector(
      *pm1, const_cast<M1::vars_Lab&>(U_P), ix_g, ix_s
    );
    StateMetaVector I = ConstructStateMetaVector(
      *pm1, const_cast<M1::vars_Lab&>(U_I), ix_g, ix_s
    );

    SourceMetaVector S = ConstructSourceMetaVector(*pm1, U_S, ix_g, ix_s);

    ClosureMetaVector CL_C = ConstructClosureMetaVector(*pm1, U_C, ix_g, ix_s);

    M1_MLOOP3(k, j, i)
    if (pm1->MaskGet(k, j, i))
    // if (pm1->MaskGetHybridize(k,j,i))
    {
      // Subtract off unlimited sources: C <- C - dt * S
      InPlaceScalarMulAdd_nG_E_F_d(-dt, C, S, k, j, i);

      // Limit: S <- theta * S for (nG, ) & (E, F_d) components
      InPlaceScalarMul_nG_E_F_d(theta(k, j, i), S, k, j, i);

      // Updated state with limited sources [C <- C + dt * theta * S]
      InPlaceScalarMulAdd_nG_E_F_d(dt, C, S, k, j, i);

      // Require states to be physical
      EnforcePhysical_E_F_d(*pm1, C, k, j, i);
      EnforcePhysical_nG(*pm1, C, k, j, i);

      // Compute (pm1 storage) (sp_P_dd, ...) based on (sc_E*, sp_F_d*)
      CL_C.Closure(k, j, i);

      // We now have nG*, it is useful to immediately construct n*
      Prepare_n_from_nG(*pm1, C, k, j, i);

      /*
      // Overwrite current state with previous state
      Copy_nG_E_F_d(C, P, k, j, i);

      // Add flux div. / gravitational sources
      InPlaceScalarMulAdd_E_F_d(dt, C, I, k, j, i);
      InPlaceScalarMulAdd_nG(dt, C, I, k, j, i);

      // Limit: S <- theta * S for (nG, ) & (E, F_d) components
      InPlaceScalarMul_nG_E_F_d(theta(k, j, i), S, k, j, i);

      // Updated state with limited sources [C <- C + dt * theta * S]
      InPlaceScalarMulAdd_nG_E_F_d(dt, C, S, k, j, i);

      // Require states to be physical
      EnforcePhysical_E_F_d(*pm1, C, k, j, i);
      EnforcePhysical_nG(*pm1, C, k, j, i);

      // Compute (pm1 storage) (sp_P_dd, ...) based on (sc_E*, sp_F_d*)
      CL_C.Closure(k, j, i);

      // We now have nG*, it is useful to immediately construct n*
      Prepare_n_from_nG(*pm1, C, k, j, i);
      */
    }
  }

}

void PrepareFull(
  M1 * pm1,
  const Real dt,
  AT_C_sca & theta,
  const M1::vars_Lab & U_C,
  const M1::vars_Lab & U_P,
  const M1::vars_Lab & U_I,
  const M1::vars_Source & U_S)
{
  // parameters & aliases------------------------------------------------------
  MeshBlock * pmb = pm1->pmy_block;
  const Real par_full_lim = pm1->opt_solver.full_lim;

  M1::vars_Lab & C = const_cast<M1::vars_Lab &>(U_C);
  M1::vars_Lab & P = const_cast<M1::vars_Lab &>(U_P);
  M1::vars_Lab & I = const_cast<M1::vars_Lab &>(U_I);

  theta.Fill(1.0);
  // --------------------------------------------------------------------------

  M1_MLOOP3(k, j, i)
  if (pm1->MaskGet(k, j, i))
  // if (pm1->MaskGetHybridize(k,j,i))
  {
    Real & theta__ = theta(k,j,i);

    for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
    for (int ix_s=0; ix_s<pm1->N_SPCS; ++ix_s)
    {
      // C = P + dt (I + S[C])
      // => C - P = dt (I + S[C])
      const Real DE = C.sc_E(ix_g,ix_s)(k,j,i)  - P.sc_E(ix_g,ix_s)(k,j,i);
      const Real DN = C.sc_nG(ix_g,ix_s)(k,j,i) - P.sc_nG(ix_g,ix_s)(k,j,i);

      if (pm1->opt_solver.limit_full_radiation)
      {
        if (DE < 0)
        {
          theta__ = std::min(
            -par_full_lim * std::max(
              P.sc_E(ix_g,ix_s)(k,j,i), 0.0
            ) / DE, theta__
          );
        }

        if (DN < 0)
        {
          theta__ = std::min(
            -par_full_lim * std::max(
              P.sc_nG(ix_g,ix_s)(k,j,i), 0.0
            ) / DN, theta__
          );
        }
      }
    }
  }

  // Finally enforce mask to be non-negative ----------------------------------
  M1_MLOOP3(k, j, i)
  if (pm1->MaskGet(k, j, i))
  // if (pm1->MaskGetHybridize(k,j,i))
  {
    theta(k, j, i) = std::max(0.0, theta(k, j, i));
  }
}

void ApplyFull(
  M1 * pm1,
  const Real dt,
  AT_C_sca & theta,
  M1::vars_Lab & U_C,
  const M1::vars_Lab & U_P,
  const M1::vars_Lab & U_I,
  M1::vars_Source & U_S)
{
  using namespace Update;
  using namespace Closures;

  for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<pm1->N_SPCS; ++ix_s)
  {
    StateMetaVector C = ConstructStateMetaVector(*pm1, U_C, ix_g, ix_s);
    StateMetaVector P = ConstructStateMetaVector(
      *pm1, const_cast<M1::vars_Lab&>(U_P), ix_g, ix_s
    );
    StateMetaVector I = ConstructStateMetaVector(
      *pm1, const_cast<M1::vars_Lab&>(U_I), ix_g, ix_s
    );

    SourceMetaVector S = ConstructSourceMetaVector(*pm1, U_S, ix_g, ix_s);

    ClosureMetaVector CL_C = ConstructClosureMetaVector(*pm1, U_C, ix_g, ix_s);

    M1_MLOOP3(k, j, i)
    if (pm1->MaskGet(k, j, i))
    // if (pm1->MaskGetHybridize(k,j,i))
    {
      // Subtract off unlimited sources: C <- C - dt * S
      InPlaceScalarMulAdd_nG_E_F_d(-dt, C, S, k, j, i);

      // Subtract off unlimited inhomogeneity: C <- C - dt * I
      InPlaceScalarMulAdd_nG_E_F_d(-dt, C, I, k, j, i);

      // Limit: S <- theta * S for (nG, ) & (E, F_d) components
      InPlaceScalarMul_nG_E_F_d(theta(k, j, i), S, k, j, i);
      // Same for I
      InPlaceScalarMul_nG_E_F_d(theta(k, j, i), I, k, j, i);

      // Updated state with limited sources [C <- C + dt * theta * S]
      InPlaceScalarMulAdd_nG_E_F_d(dt, C, S, k, j, i);
      InPlaceScalarMulAdd_nG_E_F_d(dt, C, I, k, j, i);

      // Require states to be physical
      EnforcePhysical_E_F_d(*pm1, C, k, j, i);
      EnforcePhysical_nG(*pm1, C, k, j, i);

      // Compute (pm1 storage) (sp_P_dd, ...) based on (sc_E*, sp_F_d*)
      CL_C.Closure(k, j, i);

      // We now have nG*, it is useful to immediately construct n*
      Prepare_n_from_nG(*pm1, C, k, j, i);
    }
  }

}

void CheckPhysicalFallback(
  M1 * pm1,
  const Real dt,
  const M1::vars_Source & U_S)
{
#if FLUID_ENABLED
  // parameters & aliases------------------------------------------------------
  MeshBlock * pmb = pm1->pmy_block;
  const Real mb_raw = pmb->peos->GetEOS().GetRawBaryonMass();

  const Real par_tau_min = pm1->opt_solver.flux_lo_fallback_tau_min;
  const Real par_Ye_min  = pm1->opt_solver.flux_lo_fallback_Ye_min;
  const Real par_Ye_max  = pm1->opt_solver.flux_lo_fallback_Ye_max;

  const bool check_tau = (
    pm1->opt.flux_lo_fallback_E &&
    (pm1->opt_solver.flux_lo_fallback_tau_min > -1)
  );

  const bool check_Ye = (
    pm1->opt.flux_lo_fallback_nG &&
    (pm1->opt_solver.flux_lo_fallback_Ye_min > -1) &&
    (pm1->opt_solver.flux_lo_fallback_Ye_max > -1)
  );

  M1::vars_Source & S = const_cast<M1::vars_Source &>(U_S);
  // --------------------------------------------------------------------------

  M1_MLOOP3(k, j, i)
  if (pm1->MaskGet(k, j, i))
  {
    if (pm1->ev_strat.masks.pp(k,j,i) == 1)
    {
      // already flagged
      continue;
    }

    const Real D = (
      pm1->hydro.sc_W(k,j,i) *
      pm1->hydro.sc_w_rho(k,j,i) *
      pm1->geom.sc_sqrt_det_g(k,j,i)
    );

    Real cons_IEN = pmb->phydro->u(IEN,k,j,i);
    Real s_Y_e    = pmb->pscalars->s(0,k,j,i);

    bool is_finite = true;
    bool need_fallback = false;

    for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
    {

      if (check_tau)
      {
        for (int ix_s=0; ix_s<pm1->N_SPCS; ++ix_s)
        {
          AT_C_sca & S_sc_E   = S.sc_E(ix_g,ix_s);
          AT_N_vec & S_sp_F_d = S.sp_F_d(ix_g,ix_s);

          cons_IEN -= dt * S_sc_E(k,j,i);

          for (int a=0; a<N; ++a)
          {
            is_finite = is_finite && std::isfinite(
              S_sp_F_d(a,k,j,i)
            );
          }
        }

        is_finite = is_finite && std::isfinite(cons_IEN);

        if (!is_finite || (cons_IEN < par_tau_min))
        {
          need_fallback = true;
        }
      }

      if (check_Ye)
      {
        AT_C_sca & S_sc_nG_nue = S.sc_nG(ix_g,0);
        AT_C_sca & S_sc_nG_nua = S.sc_nG(ix_g,1);

        s_Y_e += dt * mb_raw * (
          S_sc_nG_nua(k,j,i) - S_sc_nG_nue(k,j,i)
        );

        is_finite = is_finite && std::isfinite(s_Y_e);

        const Real w_Y_e = (D != 0.0) ? (s_Y_e / D) : 0.0;

        if (!is_finite ||
            (w_Y_e < par_Ye_min) ||
            (w_Y_e > par_Ye_max))
        {
          need_fallback = true;
        }
      }
    }

    // Update flux hybridization mask -----------------------------------------
    if (need_fallback)
    {
      pm1->ev_strat.masks.pp(k,j,i) = 1.0;
    }
  }
#endif
}

} // namespace M1::Sources::Limiter
// ============================================================================

// ============================================================================
} // namespace M1::Sources
// ============================================================================


//
// :D
//