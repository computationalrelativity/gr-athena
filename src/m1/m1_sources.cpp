// C++ standard headers
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
  ost st = pm1.ev_strat.masks.source_treatment(k,j,i);

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
  ost st = pm1.ev_strat.masks.source_treatment(k,j,i);

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
  const Real mb = pmb->peos->GetEOS().GetRawBaryonMass();
#endif

  const Real par_src_lim = pm1->opt_solver.src_lim;
  const Real par_src_Ye_min = pm1->opt_solver.src_lim_Ye_min;
  const Real par_src_Ye_max = pm1->opt_solver.src_lim_Ye_max;

  M1::vars_Lab & C = const_cast<M1::vars_Lab &>(U_C);
  M1::vars_Lab & P = const_cast<M1::vars_Lab &>(U_P);
  M1::vars_Lab & I = const_cast<M1::vars_Lab &>(U_I);

  theta.Fill(1.0);
  // --------------------------------------------------------------------------

  M1_ILOOP3(k, j, i)
  if (pm1->MaskGet(k, j, i))
  {
    Real & theta__ = theta(k,j,i);

#if FLUID_ENABLED
    const Real & tau = pmb->phydro->u(IEN, k, j, i);
    const Real & Y_e = pm1->hydro.sc_w_Ye(0,k,j,i);

    const Real D = (
      pm1->hydro.sc_W(k,j,i) *
      pm1->hydro.sc_w_rho(k,j,i) *
      pm1->geom.sc_sqrt_det_g(k,j,i)
    );

    Real Dtau_sum (0);
    Real DDxp_sum (0);
#endif // FLUID_ENABLED

    for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
    for (int ix_s=0; ix_s<pm1->N_SPCS; ++ix_s)
    {
      // these points have zero sources
      if (pm1->ev_strat.masks.solution_regime(ix_g,ix_s,k,j,i) ==
          M1::evolution_strategy::opt_solution_regime::equilibrium)
      {
        theta__ = 0;
        continue;
      }

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

#if FLUID_ENABLED
      // Cf. CoupleSourcesHydro, CoupleSourcesYe
      Dtau_sum -= DE;
      if (pm1->N_SPCS == 3)
        DDxp_sum += mb * DN * ( (ix_s == 1) - (ix_s == 0) );
#endif // FLUID_ENABLED
    }

    // limiting based on sums -------------------------------------------------
#if FLUID_ENABLED
    if (Dtau_sum < 0)
    {
      theta__ = std::min(
        -par_src_lim * std::max(
          tau, 0.0
        ) / Dtau_sum,
        theta__
      );
    }

    const Real DYe = DDxp_sum / D;
    if (DYe > 0)
    {
      theta__ = std::min(
        par_src_lim * std::max(
          par_src_Ye_max - Y_e, 0.0
        ) / DYe,
        theta__
      );
    }
    else if (DYe < 0)
    {
      theta__ = std::min(
        par_src_lim * std::min(
          par_src_Ye_min - Y_e, 0.0
        ) / DYe,
        theta__
      );
    }
#endif // FLUID_ENABLED

  }

  // Finally enforce mask to be non-negative ----------------------------------
  M1_ILOOP3(k, j, i)
  if (pm1->MaskGet(k, j, i))
  {
    theta(k, j, i) = std::max(0.0, theta(k, j, i));
  }
}


// old implementation
void _Prepare(
  M1 * pm1,
  const Real dt,
  AT_C_sca & theta,
  const M1::vars_Lab & U_C,
  const M1::vars_Lab & U_P,
  const M1::vars_Lab & U_I,
  const M1::vars_Source & U_S)
{
  using namespace Update;

  theta.Fill(1.0);

  const Real par_src_lim = pm1->opt_solver.src_lim;

#if FLUID_ENABLED
    MeshBlock * pmb = pm1->pmy_block;
    PassiveScalars * ps = pmb->pscalars;

    const Real mb = pmb->peos->GetEOS().GetRawBaryonMass();

    // Note _dense_ (not sliced) scratch array
    AT_C_sca & dt_S_tau = pm1->scratch.sc_A;
    dt_S_tau.Fill(0.0);

    AT_C_sca & dt_xp_sum = pm1->scratch.sc_B;
    if (pm1->N_SPCS > 1)
    {
      dt_xp_sum.Fill(0.0);
    }
#endif // FLUID_ENABLED

  for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<pm1->N_SPCS; ++ix_s)
  {
    StateMetaVector C = ConstructStateMetaVector(
      *pm1, const_cast<M1::vars_Lab&>(U_C), ix_g, ix_s
    );
    StateMetaVector P = ConstructStateMetaVector(
      *pm1, const_cast<M1::vars_Lab&>(U_P), ix_g, ix_s
    );
    StateMetaVector I = ConstructStateMetaVector(
      *pm1, const_cast<M1::vars_Lab&>(U_I), ix_g, ix_s
    );

    SourceMetaVector S = ConstructSourceMetaVector(
      *pm1, const_cast<M1::vars_Source&>(U_S), ix_g, ix_s
    );

    if (1)
    M1_ILOOP3(k, j, i)
    if (pm1->MaskGet(k, j, i))
    {
      // C = P + dt (I + S[C])
      // => dt S[C] = C - (P + dt I)
      //
      // N.B.
      // We take the difference rather than value directly from the sources;
      // This is because values may differ based on solution regime selected

      // Approximate update (without matter source)
      const Real appr_sc_E_star = P.sc_E(k,j,i) + dt * I.sc_E(k,j,i);
      const Real dt_S_sc_E = C.sc_E(k, j, i) - appr_sc_E_star;
      // const Real dt_S_sc_E = dt * S.sc_E(k,j,i);

      if (0)
      if (dt_S_sc_E < 0)
      {
        theta(k, j, i) = std::min(
          -par_src_lim * std::max(appr_sc_E_star, 0.0) / dt_S_sc_E,
          theta(k, j, i));
      }

#if FLUID_ENABLED
      dt_S_tau(k, j, i) -= dt_S_sc_E;
#endif // FLUID_ENABLED
    }

    // Now deal with Neutrino number densities (if we have multiple species)
    if (pm1->N_SPCS > 1)
    M1_ILOOP3(k, j, i)
    if (pm1->MaskGet(k, j, i))
    {
      // Approximate update (without matter source)
      const Real appr_sc_nG_star = P.sc_nG(k,j,i) + dt * I.sc_nG(k,j,i);
      const Real dt_S_sc_nG = C.sc_nG(k, j, i) - appr_sc_nG_star;
      // const Real dt_S_sc_nG = dt * S.sc_nG(k,j,i);

      if (dt_S_sc_nG < 0)
      {
        theta(k, j, i) = std::min(
          -par_src_lim *
          std::max(appr_sc_nG_star, 0.0) / dt_S_sc_nG,
          theta(k, j, i));
      }

#if FLUID_ENABLED
      // Fluid lepton sources
      const Real dt_xp = -mb * dt_S_sc_nG * ( + (ix_s == 0)
                                              - (ix_s == 1));

      dt_xp_sum(k, j, i) += dt_xp;
#endif // FLUID_ENABLED
    }
  }

#if FLUID_ENABLED
  M1_ILOOP3(k, j, i)
  if (pm1->MaskGet(k, j, i))
  {
    if (dt_S_tau(k, j, i) < 0)
    {
      // Note that tau is densitized, as is sc_E
      const Real tau = pmb->phydro->u(IEN, k, j, i);
      // const Real D = (
      //   pm1->hydro.sc_W(k,j,i) *
      //   pm1->hydro.sc_w_rho(k,j,i) *
      //   pm1->geom.sc_sqrt_det_g(k,j,i)
      // );

      theta(k, j, i) = std::min(
        -par_src_lim * std::max(
          0.0, tau
        ) / dt_S_tau(k, j, i), theta(k, j, i)
      );
    }
  }

  if (0)
  if (pm1->N_SPCS > 1)
  if ((pm1->opt_solver.src_lim_Ye_min >= 0) &&
      (pm1->opt_solver.src_lim_Ye_max >= 0))
  {
    AT_C_sca & sc_oo_sqrt_det_g = pm1->geom.sc_oo_sqrt_det_g;

    M1_ILOOP3(k, j, i)
    if (pm1->MaskGet(k, j, i))
    {
      /*
      // DY_e is sqrt(gamma) * D * Y_e
      const Real DY_e = ps->s(0, k, j, i);

      // Note that dt_xp_sum is sqrt(gamma) densitized, as is D
      const Real oo_sqrt_det_g = sc_oo_sqrt_det_g(k, j, i);
      const Real D = pmb->phydro->u(IDN, k, j, i);
      const Real dt_DY_e = oo_sqrt_det_g * D * dt_xp_sum(k, j, i);

      if (dt_DY_e > 0)
      {
        const Real par_src_lim_Ye_max = pm1->opt_solver.src_lim_Ye_max;

        theta(k, j, i) = std::min(
            par_src_lim *
            std::max(D * par_src_lim_Ye_max - DY_e,
                     0.0) / dt_DY_e,
            theta(k, j, i)
        );
      }
      else if (dt_DY_e < 0)
      {
        const Real par_src_lim_Ye_min = pm1->opt_solver.src_lim_Ye_min;

        theta(k, j, i) = std::min(
            par_src_lim *
            std::min(D * par_src_lim_Ye_min - DY_e,
                    0.0) / dt_DY_e,
            theta(k, j, i)
        );
      }
      */

      // DY_e is sqrt(gamma) * D * Y_e
      const Real w_Y_e = pm1->hydro.sc_w_Ye(0, k, j, i);
      const Real w_rho = pm1->hydro.sc_w_rho(k, j, i);

      const Real D = pmb->phydro->u(IDN, k, j, i);

      // Note that dt_xp_sum is sqrt(gamma) densitized, as is D
      const Real oo_sqrt_det_g = sc_oo_sqrt_det_g(k, j, i);
      const Real dt_Y_e = oo_sqrt_det_g * dt_xp_sum(k, j, i) / D;
      // const Real dt_Y_e = dt_xp_sum(k, j, i) / D;

      if (dt_Y_e > 0)
      {
        const Real par_src_lim_Ye_max = pm1->opt_solver.src_lim_Ye_max;

        theta(k, j, i) = std::min(
            par_src_lim *
            std::max(par_src_lim_Ye_max - w_Y_e,
                     0.0) / dt_Y_e,
            theta(k, j, i)
        );
      }
      else if (dt_Y_e < 0)
      {
        const Real par_src_lim_Ye_min = pm1->opt_solver.src_lim_Ye_min;

        theta(k, j, i) = std::min(
            par_src_lim *
            std::min(par_src_lim_Ye_min - w_Y_e,
                    0.0) / dt_Y_e,
            theta(k, j, i)
        );
      }


    }
  }

#endif // FLUID_ENABLED
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

    M1_ILOOP3(k, j, i)
    if (pm1->MaskGet(k, j, i))
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

} // namespace M1::Sources::Limiter
// ============================================================================

// ============================================================================


// ============================================================================
} // namespace M1::Sources
// ============================================================================


//
// :D
//