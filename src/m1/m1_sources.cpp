// C++ standard headers
#include <iostream>

// Athena++ headers
#include "m1_sources.hpp"
#include "m1_calc_closure.hpp"
#include "m1_calc_update.hpp"

#if FLUID_ENABLED
#include "../hydro/hydro.hpp"
#include "../eos/eos.hpp"
#include "../scalars/scalars.hpp"
#endif // FLUID_ENABLED

// External libraries
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_vector.h>


// ============================================================================
namespace M1::Sources {
// ============================================================================

void SetMatterSourceZero(
  Update::SourceMetaVector & S,
  const int k, const int j, const int i)
{
  S.sc_nG(k,j,i) = 0;
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
  const Real W    = pm1.fidu.sc_W(k,j,i);
  const Real W2   = SQR(W);

  const Real kap_as = V.sc_kap_a(k,j,i) + V.sc_kap_s(k,j,i);

  const Real dotFv = Assemble::sc_dot_dense_sp__(V.sp_F_d, pm1.fidu.sp_v_u,
                                                 k, j, i);

  const Real dotPvv = Assemble::sc_ddot_dense_sp__(pm1.fidu.sp_v_u, V.sp_P_dd,
                                                   k, j, i);

  Real S_fac (0);
  S_fac = pm1.geom.sc_sqrt_det_g(k,j,i) * V.sc_eta(k,j,i);
  S_fac += V.sc_kap_s(k,j,i) * W2 * (
    V.sc_E(k,j,i) - 2.0 * dotFv + dotPvv
  );

  S.sc_E(k,j,i) = pm1.geom.sc_alpha(k,j,i) * W * (
    kap_as * (dotFv - V.sc_E(k,j,i)) + S_fac
  );

  for (int a=0; a<N; ++a)
  {
    Real dotPv (0);
    for (int b=0; b<N; ++b)
    {
      dotPv += V.sp_P_dd(a,b,k,j,i) * pm1.fidu.sp_v_u(b,k,j,i);
    }

    S.sp_F_d(a,k,j,i) = pm1.geom.sc_alpha(k,j,i) * W * (
      kap_as * (dotPv - V.sp_F_d(a,k,j,i)) +
      S_fac * pm1.fidu.sp_v_d(a,k,j,i)
    );
  }

}

void PrepareMatterSource_nG(
  M1 & pm1,
  const Update::StateMetaVector & V,
  Update::SourceMetaVector & S,
  const int k, const int j, const int i)
{
  S.sc_nG(k,j,i) = pm1.geom.sc_alpha(k,j,i) * (
    pm1.geom.sc_sqrt_det_g(k,j,i) * V.sc_eta_0(k,j,i) -
    V.sc_kap_a_0(k,j,i) * V.sc_n(k,j,i)
  );
}

// Given nG and a StateMetaVector construct n [i.e. prepare Gam]
void Prepare_n_from_nG(
  M1 & pm1,
  const Update::StateMetaVector & V,
  const int k, const int j, const int i)
{
  const Real W  = pm1.fidu.sc_W(k,j,i);
  const Real W2 = SQR(W);

  const Real dotFv = Assemble::sc_dot_dense_sp__(V.sp_F_d, pm1.fidu.sp_v_u,
                                                 k, j, i);

  V.sc_J(k,j,i) = Assemble::sc_J__(
    W2, dotFv, V.sc_E, pm1.fidu.sp_v_u, V.sp_P_dd,
    k, j, i
  );

  const Real C_Gam = Assemble::sc_G__(
    W, V.sc_E(k,j,i), V.sc_J(k,j,i), dotFv,
    pm1.opt.fl_E, pm1.opt.fl_J, pm1.opt.eps_E
  );

  V.sc_n(k,j,i) = V.sc_nG(k,j,i) / C_Gam;
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
  const Real W = pm1.fidu.sc_W(k,j,i);
  const Real W2 = SQR(W);
  const Real W3 = W * W2;

  const Real kap_as = (C.sc_kap_a(k,j,i) + C.sc_kap_s(k,j,i));
  const Real kap_s = C.sc_kap_s(k,j,i);

  const Real alpha = pm1.geom.sc_alpha(k,j,i);

  // P_dd thick (tk) and thin (tn) factors
  const Real d_tk = 3.0 * 0.5 * (1.0 - C.sc_chi(k,j,i));
  const Real d_tn = 1.0 - d_tk;

  const Real dotFv = Assemble::sc_dot_dense_sp__(C.sp_F_d, pm1.fidu.sp_v_u,
                                                 k, j, i);

  const Real dotvv = Assemble::sc_dot_dense_sp__(pm1.fidu.sp_v_d,
                                                 pm1.fidu.sp_v_u,
                                                 k, j, i);

  Assemble::sp_d_to_u_(&pm1, pm1.scratch.sp_F_u_, C.sp_F_d, k, j, i, i);
  Real dotFF = 0;

  for (int a=0; a<N; ++a)
  {
    dotFF += pm1.scratch.sp_F_u_(a,i) * C.sp_F_d(a,k,j,i);
  }

  // J(I,J) ~ D[S_I,(E,F_d)_J]

  // sc_Stil_1 ----------------------------------------------------------------

  // D_E
  J(0,0) = alpha * W * (
    -kap_as + kap_s * W2
  );

  const Real fac_J_0a = alpha * W * (
    kap_as - 2.0 * kap_s * W2
  );

  // D_F_d
  for (int a=0; a<N; ++a)
  {
    J(0,1+a) = fac_J_0a * pm1.fidu.sp_v_u(a,k,j,i);
  }

  // sc_Stil_1pa --------------------------------------------------------------
  for (int a=0; a<N; ++a)
  {
    // D_E
    J(1+a,0) = alpha * kap_s * W3 * pm1.fidu.sp_v_d(a,k,j,i);

    // D_F_d
    for (int b=0; b<N; ++b)
    {
      J(1+a,1+b) = -alpha * W * (
        (a == b) * kap_as +
        2.0 * kap_s * W2 * pm1.fidu.sp_v_u(b,k,j,i) * pm1.fidu.sp_v_d(a,k,j,i)
      );
    }
  }

  // thin correction to Jacobian ----------------------------------------------
  if (dotFF > 0)
  {
    J(0,0) += d_tn * alpha * W3 * kap_s * SQR(dotFv) / dotFF;

    for (int b=0; b<N; ++b)
    {
      J(0,1+b) += d_tn * 2.0 * alpha * dotFv * C.sc_E(k,j,i) * kap_s * W3 * (
        -dotFv * pm1.scratch.sp_F_u_(b,i) +
        dotFF * pm1.fidu.sp_v_u(b,k,j,i)
      ) / SQR(dotFF);
    }

    for (int a=0; a<N; ++a)
    {
      J(1+a,0) += d_tn * alpha * dotFv * W * (
        C.sp_F_d(a,k,j,i) * kap_as +
        W2 * dotFv * kap_s * pm1.fidu.sp_v_d(a,k,j,i)
      ) / dotFF;

      for (int b=0; b<N; ++b)
      {
        J(1+a,1+b) += d_tn * alpha * C.sc_E(k,j,i) * W * (
          -2.0 * dotFv * pm1.scratch.sp_F_u_(b,i) *
          (C.sp_F_d(a,k,j,i) * kap_as +
           W2 * dotFv * kap_s * pm1.fidu.sp_v_d(a,k,j,i)) +
          dotFF * ((a==b) * dotFv * kap_as +
                   pm1.fidu.sp_v_u(b,k,j,i) * (
                    C.sp_F_d(a,k,j,i) * kap_as +
                    2.0 * W2 * dotFv * kap_s * pm1.fidu.sp_v_d(a,k,j,i)
                   ))
        ) / SQR(dotFF);
      }
    }
  }

  // thick correction to Jacobian ---------------------------------------------
  J(0,0) += d_tk * alpha * dotvv * kap_s * W3 * (
    -1.0 + (2.0 - 4.0 * dotvv) * W2
  ) / (1.0 + 2.0 * W2);

  for (int b=0; b<N; ++b)
  {
    J(0,1+b) += d_tk * 2.0 * alpha * dotvv * kap_s * W3 *
                pm1.fidu.sp_v_u(b,k,j,i) * (
                  1.0 + (1.0 + dotvv) * W2
                ) / (1.0 + 2.0 * W2);
  }

  for (int a=0; a<N; ++a)
  {
    J(1+a,0) += -d_tk * alpha * pm1.fidu.sp_v_d(a,k,j,i) * W * (
      1.0 + (-2.0 + 4.0 * dotvv) * W2
    ) * (
      kap_as + dotvv * kap_s * W2
    ) / (1.0 + 2.0 * W2);

    for (int b=0; b<N; ++b)
    {
      J(1+a,1+b) += d_tk * alpha * W * (
        (a == b) * dotvv * kap_as +
        pm1.fidu.sp_v_u(b,k,j,i) * pm1.fidu.sp_v_d(a,k,j,i) * (
          kap_as + 2.0 * dotvv * kap_as * W2 +
          2.0 * dotvv * kap_s * W2 * (1.0 + (1.0 + dotvv) * W2)
        ) / (1.0 + 2.0 * W2)
      );
    }
  }
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
      const Real dt_xp = -mb * dt_S_sc_nG * ( + (ix_g == 0)
                                              - (ix_g == 1));

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
      theta(k, j, i) = std::min(
        -par_src_lim * std::max(
          0.0, tau
        ) / dt_S_tau(k, j, i), theta(k, j, i)
      );
    }
  }

  if (pm1->N_SPCS > 1)
  if ((pm1->opt_solver.src_lim_Ye_min >= 0) &&
      (pm1->opt_solver.src_lim_Ye_max >= 0))
  M1_ILOOP3(k, j, i)
  if (pm1->MaskGet(k, j, i))
  {
    // Note that dt_xp_sum is densitized, as is D
    const Real DYe = dt_xp_sum(k, j, i) / pmb->phydro->u(IDN, k, j, i);

    if (DYe > 0)
    {
      const Real oo_sqrt_det_g = 1.0 / pm1->geom.sc_sqrt_det_g(k, j, i);
      const Real par_src_lim_Ye_max = pm1->opt_solver.src_lim_Ye_max;

      theta(k, j, i) = std::min(
          par_src_lim *
          std::max(par_src_lim_Ye_max - oo_sqrt_det_g * ps->s(0, k, j, i),
                   0.0) / DYe,
          theta(k, j, i)
      );
    }
    else if (DYe < 0)
    {
      const Real oo_sqrt_det_g = 1.0 / pm1->geom.sc_sqrt_det_g(k, j, i);
      const Real par_src_lim_Ye_min = pm1->opt_solver.src_lim_Ye_min;

      theta(k, j, i) = std::min(
          par_src_lim *
          std::min(par_src_lim_Ye_min - oo_sqrt_det_g * ps->s(0, k, j, i),
                   0.0) / DYe,
          theta(k, j, i)
      );
    }

  }
#endif // FLUID_ENABLED

  // Finally enforce mask to be non-negative
  M1_ILOOP3(k, j, i)
  if (pm1->MaskGet(k, j, i))
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

    M1_ILOOP3(k, j, i)
    if (pm1->MaskGet(k, j, i))
    {
      /*
      // construct limited sources & update solution
      const Real appr_sc_E_star = P.sc_E(k,j,i) + dt * I.sc_E(k,j,i);
      S.sc_E(k,j,i) = theta(k,j,i) * (
        C.sc_E(k,j,i) - appr_sc_E_star
      );
      C.sc_E(k,j,i) = appr_sc_E_star + dt * S.sc_E(k,j,i);

      for (int a=0; a<N; ++a)
      {
        const Real appr_sp_F_a_star = P.sp_F_d(a,k,j,i) + dt * I.sp_F_d(k,j,i);
        S.sp_F_d(a,k,j,i) = theta(k,j,i) * (
          C.sp_F_d(a,k,j,i) - appr_sp_F_a_star
        );

        C.sp_F_d(a,k,j,i) = appr_sp_F_a_star + dt * S.sp_F_d(a,k,j,i);
      }

      EnforcePhysical_E_F_d(*pm1, C, k, j, i);

      // Compute (pm1 storage) (sp_P_dd, ...) based on (sc_E*, sp_F_d*)
      CL_C.Closure(k, j, i);

      const Real appr_sc_nG_star = P.sc_nG(k,j,i) + dt * I.sc_nG(k,j,i);
      S.sc_nG(k,j,i) = theta(k,j,i) * (
        C.sc_nG(k,j,i) - appr_sc_nG_star
      );
      C.sc_nG(k,j,i) = appr_sc_nG_star + dt * S.sc_nG(k,j,i);

      // We now have nG*, it is useful to immediately construct n*
      Prepare_n_from_nG(*pm1, C, k, j, i);
      */

      // C <- C - dt * S
      InPlaceScalarMulAdd_nG_E_F_d(-dt, C, S, k, j, i);

      // Limit: S <- theta * S for (nG, ) & (E, F_d) components
      InPlaceScalarMul_nG_E_F_d(theta(k, j, i), S, k, j, i);

      // Updated state with limited sources [C <- C + dt * theta * S]
      InPlaceScalarMulAdd_nG_E_F_d(dt, C, S, k, j, i);

      // We now have nG*, it is useful to immediately construct n*
      Prepare_n_from_nG(*pm1, C, k, j, i);

      EnforcePhysical_E_F_d(*pm1, C, k, j, i);

      // Compute (pm1 storage) (sp_P_dd, ...) based on (sc_E*, sp_F_d*)
      CL_C.Closure(k, j, i);
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