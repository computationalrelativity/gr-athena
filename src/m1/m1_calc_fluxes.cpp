// c++
// ...

// Athena++ headers
#include "m1_calc_fluxes.hpp"
#include "m1_macro.hpp"
#include "m1_utils.hpp"
#include <limits>

// ============================================================================
namespace M1 {
// ============================================================================

// ----------------------------------------------------------------------------
// Interface for assembling flux vector components on cell-centers; these are
// subsequently reconstructed using a limiter.
//
// Method is 2nd order with high-Peclet limit fix
void M1::CalcFluxes(AthenaArray<Real> & u, const bool use_lo)
{
  // retain then overwrite setting if we want to force lo flux
  opt_flux_variety ofv = opt.flux_variety;
  if (use_lo)
  {
    opt.flux_variety = opt_flux_variety::LO;
  }

  vars_Lab U { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u, U);

  // (dense) scratch for flux vector component assembly (cell-centered samp.)
  AT_C_sca & F_sca = scratch.F_sca;
  AT_N_vec & F_vec = scratch.F_vec;

  AT_C_sca & lambda = scratch.lambda;

  // required geometric quantities
  AT_C_sca & sc_alpha  = geom.sc_alpha;
  AT_N_vec & sp_beta_u = geom.sp_beta_u;
  AT_N_sym & sp_g_uu   = geom.sp_g_uu;

  // required fiducial quantities
  AT_C_sca & sc_W   = fidu.sc_W;
  AT_N_vec & sp_v_u = fidu.sp_v_u;

  // point to scratches -------------------------------------------------------
  AT_N_sym & sp_P_dd_ = scratch.sp_P_dd_;

  // point to flux storage ----------------------------------------------------
  vars_Flux & flx = (use_lo) ? fluxes_lo : fluxes;

  // indicial ranges ----------------------------------------------------------
  const int il = pm1->mbi.il-M1_NGHOST_MIN;
  const int iu = pm1->mbi.iu+M1_NGHOST_MIN;

  const bool approximate_speeds = opt.characteristics_variety ==
        opt_characteristics_variety::approximate;

  for (int ix_d=0; ix_d<N; ++ix_d)
  {
    if (approximate_speeds)
      CalcCharacteristicSpeedApproximate(ix_d, lambda);

    for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
    for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
    {

      AT_C_sca & F_nG  = flx.sc_nG( ix_g,ix_s,ix_d);
      AT_C_sca & F_E   = flx.sc_E(  ix_g,ix_s,ix_d);
      AT_N_vec & F_F_d = flx.sp_F_d(ix_g,ix_s,ix_d);

      AT_C_sca & U_nG   = U.sc_nG( ix_g,ix_s);
      AT_N_vec & U_F_d  = U.sp_F_d(ix_g,ix_s);
      AT_C_sca & U_E    = U.sc_E(  ix_g,ix_s);

      AT_C_sca & sc_kap_a = radmat.sc_kap_a(ix_g,ix_s);
      AT_C_sca & sc_kap_s = radmat.sc_kap_s(ix_g,ix_s);

      AT_C_sca & sc_chi = lab_aux.sc_chi(ix_g,ix_s);
      AT_C_sca & sc_xi  = lab_aux.sc_xi(ix_g,ix_s);

      AT_C_sca & sc_J   = rad.sc_J(  ix_g,ix_s);
      AT_D_vec & st_H_u = rad.st_H_u(ix_g, ix_s);
      AT_C_sca & sc_n   = rad.sc_n(  ix_g,ix_s);

      if (!approximate_speeds)
      {
        CalcCharacteristicSpeed(ix_d, U_E, U_F_d, sc_chi, lambda);
      }

      // M1_FLOOP3(k,j,i)
      // {
      //   if (lambda(k,j,i) > 1)
      //   {
      //     lambda(k,j,i) = 1.0;
      //     std::printf("lam>1\n");
      //   }
      // }


      // Flux assembly and reconstruction =====================================
      // See Eq.(28) of [1] (note densitized)

      // nG -------------------------------------------------------------------
      M1_FLOOP2(k,j)
      {
        // Require fiducial frame to prepare flux
        M1_FLOOP1(i)
        {
          const Real sp_f_u__ = Assemble::Frames::sp_f_u__(
            *this, sc_J, st_H_u,
            ix_d, k, j, i
          );
          // sc_n needs to be computed in CalcFiducialFrame
          F_sca(k,j,i) = sc_alpha(k,j,i) * sc_n(k,j,i) * sp_f_u__;
        }
      }
      Fluxes::ReconstructLimitedFlux(this, ix_d, U_nG, F_sca,
                                     sc_xi,
                                     sc_kap_a, sc_kap_s, lambda, F_nG);

      // E --------------------------------------------------------------------
      M1_FLOOP2(k,j)
      {
        M1_FLOOP1(i)
        {
          F_sca(k,j,i) = -sp_beta_u(ix_d,k,j,i) * U_E(k,j,i);
        }

        for (int a=0; a<N; ++a)
        M1_FLOOP1(i)
        {
          F_sca(k,j,i) += sc_alpha(k,j,i) *
                          sp_g_uu(ix_d,a,k,j,i) *
                          U_F_d(a,k,j,i);
        }
      }

      Fluxes::ReconstructLimitedFlux(this, ix_d, U_E, F_sca,
                                     sc_xi,
                                     sc_kap_a, sc_kap_s, lambda, F_E);

      // F_k ------------------------------------------------------------------
      M1_FLOOP2(k,j)
      {
        Assemble::Frames::sp_P_dd_(
          *this, sp_P_dd_, sc_chi, U_E, U_F_d,
          k, j, il, iu
        );

        for (int a=0; a<N; ++a)
        {
          M1_FLOOP1(i)
          {
            F_vec(a,k,j,i) = -sp_beta_u(ix_d,k,j,i) * U_F_d(a,k,j,i);
          }

          for (int b=0; b<N; ++b)
          M1_FLOOP1(i)
          {
            F_vec(a,k,j,i) += sc_alpha(k,j,i) *
                              sp_g_uu(ix_d,b,k,j,i) *
                              sp_P_dd_(b,a,i);
          }
        }
      }
      Fluxes::ReconstructLimitedFlux(this, ix_d, U_F_d, F_vec,
                                     sc_xi,
                                     sc_kap_a, sc_kap_s, lambda, F_F_d);
    }
  }

  // revert if lo overwrite used
  if (use_lo)
    opt.flux_variety = ofv;
}

// ----------------------------------------------------------------------------
// Build flux limiter over multiple function components
void M1::CalcFluxLimiter(AthenaArray<Real> & u)
{
  using namespace Fluxes;

  vars_Lab U { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u, U);

  AA & flux_limiter = pm1->ev_strat.masks.flux_limiter;
  flux_limiter.Fill(0.0);


  AT_C_sca U_sl;

  // indicial ranges ----------------------------------------------------------
  for (int ix_d=0; ix_d<N; ++ix_d)
  {
    for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
    for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
    {
      AT_C_sca & U_nG   = U.sc_nG( ix_g,ix_s);
      AT_N_vec & U_F_d  = U.sp_F_d(ix_g,ix_s);
      AT_C_sca & U_E    = U.sc_E(  ix_g,ix_s);

      AT_C_sca & sc_kap_a = radmat.sc_kap_a(ix_g,ix_s);
      AT_C_sca & sc_kap_s = radmat.sc_kap_s(ix_g,ix_s);

      AT_C_sca & sc_xi = lab_aux.sc_xi(ix_g,ix_s);

      switch (ix_d)
      {
        case 0:
        {
          LimiterMaskX1(this, flux_limiter, U_E, sc_xi, sc_kap_a, sc_kap_s);
          LimiterMaskX1(this, flux_limiter, U_nG, sc_xi, sc_kap_a, sc_kap_s);

          for (int a=0; a<N; ++a)
          {
            U_F_d.slice(a, U_sl);
            LimiterMaskX1(this, flux_limiter, U_sl, sc_xi, sc_kap_a, sc_kap_s);
          }

          break;
        }
        case 1:
        {
          LimiterMaskX2(this, flux_limiter, U_E, sc_xi, sc_kap_a, sc_kap_s);
          LimiterMaskX2(this, flux_limiter, U_nG, sc_xi, sc_kap_a, sc_kap_s);

          for (int a=0; a<N; ++a)
          {
            U_F_d.slice(a, U_sl);
            LimiterMaskX2(this, flux_limiter, U_sl, sc_xi, sc_kap_a, sc_kap_s);
          }

          break;
        }
        case 2:
        {
          LimiterMaskX3(this, flux_limiter, U_E, sc_xi, sc_kap_a, sc_kap_s);
          LimiterMaskX3(this, flux_limiter, U_nG, sc_xi, sc_kap_a, sc_kap_s);

          for (int a=0; a<N; ++a)
          {
            U_F_d.slice(a, U_sl);
            LimiterMaskX3(this, flux_limiter, U_sl, sc_xi, sc_kap_a, sc_kap_s);
          }

          break;
        }
        default:
        {
          break;
        }
      }

    }
  }

}


// ----------------------------------------------------------------------------
// Characteristic speeds on CC
void M1::CalcCharacteristicSpeedApproximate(const int dir, AT_C_sca & lambda)
{
  auto AMAX = [&](const Real A, const Real B)
  {
    return std::max(std::abs(A), std::abs(B));
  };

  AT_C_sca & alpha  = geom.sc_alpha;
  AT_N_vec & beta_u = geom.sp_beta_u;
  AT_N_sym & g_uu   = geom.sp_g_uu;

  M1_FLOOP3(k,j,i)
  if (pm1->MaskGet(k,j,i))
  {
    // Cf. [1] we have ONE_3RD
    const Real A = alpha(k,j,i) * std::sqrt(g_uu(dir,dir,k,j,i));
    const Real B = beta_u(dir,k,j,i);
    lambda(k,j,i) = AMAX(A+B, A-B);
  }
}

void M1::CalcCharacteristicSpeed(const int dir,
                                 const AT_C_sca & sc_E,
                                 const AT_N_vec & sp_F_d,
                                 const AT_C_sca & sc_chi,
                                 AT_C_sca & lambda)
{
  AT_C_sca & alpha  = geom.sc_alpha;
  AT_N_vec & beta_d = geom.sp_beta_d;
  AT_N_vec & beta_u = geom.sp_beta_u;
  AT_N_sym & g_uu   = geom.sp_g_uu;

  AT_C_sca & W   = fidu.sc_W;
  AT_N_vec & v_d = fidu.sp_v_d;
  AT_N_vec & v_u = fidu.sp_v_u;

  auto AMAX = [&](const Real A, const Real B)
  {
    return std::max(std::abs(A), std::abs(B));
  };

  auto lam_mixed = [&](const int k, const int j, const int i)
  {
    // See [3]
    AT_C_sca & alpha  = geom.sc_alpha;
    AT_N_vec & beta_u = geom.sp_beta_u;
    AT_N_sym & g_uu   = geom.sp_g_uu;

    Real F_u (0.);
    Real nF2 (0.);
    for (int a=0; a<N; ++a)
    {
      F_u += g_uu(dir,a,k,j,i) * sp_F_d(a,k,j,i);

      for (int b=0; b<N; ++b)
      {
        nF2 += g_uu(a,b,k,j,i) * sp_F_d(a,k,j,i) * sp_F_d(b,k,j,i);
      }
    }

    Real A = alpha(k,j,i) * std::sqrt(g_uu(dir,dir,k,j,i) * ONE_3RD);
    Real B = beta_u(dir,k,j,i);
    const Real lam_thick = AMAX(A+B, A-B);

    const Real oo_nF2 = (nF2 > 0) ? OO(nF2) : 0.0;
    const Real oo_nF  = (nF2 > 0) ? std::sqrt(oo_nF2) : 0.0;

    A = alpha(k,j,i) * oo_nF * std::abs(F_u);
    const Real lam_thin = AMAX(A+B, A-B);

    return std::max(lam_thick, lam_thin);
  };

  auto lam_thin = [&](const int k, const int j, const int i)
  {
    /*
    Real normF2 (0.);
    for (int a=0; a<N; ++a)
    for (int b=0; b<N; ++b)
    {
      normF2 += g_uu(a,b,k,j,i) * sp_F_d(a,k,j,i) * sp_F_d(b,k,j,i);
    }

    const Real fac_a = (normF2 > 0)
      ? std::abs(sp_F_d(dir,k,j,i)) / std::sqrt(normF2)
      : 0;

    const Real fac_b = (normF2 > 0)
      ? std::abs(sp_F_d(dir,k,j,i)) / normF2
      : 0;

    const Real lam_a = std::max(
      std::abs(beta_d(dir,k,j,i)) + alpha(k,j,i) * fac_a,
      std::abs(beta_d(dir,k,j,i)) - alpha(k,j,i) * fac_a
    );

    const Real lam_b = (
      std::abs(beta_d(dir,k,j,i)) +
      alpha(k,j,i) * sc_E(k,j,i) / sc_sqrt_det_g(k,j,i) * fac_b
    );

    return std::max(lam_a, lam_b);
    */

    // Cf. Shibata '11
    Real F_u (0.);
    Real nF2 (0.);
    for (int a=0; a<N; ++a)
    {
      F_u += g_uu(dir,a,k,j,i) * sp_F_d(a,k,j,i);

      for (int b=0; b<N; ++b)
      {
        nF2 += g_uu(a,b,k,j,i) * sp_F_d(a,k,j,i) * sp_F_d(b,k,j,i);
      }
    }

    const Real oo_nF2 = (nF2 > 0) ? OO(nF2) : 0;
    const Real oo_nF  = std::sqrt(oo_nF2);

    const Real lfac_a = alpha(k,j,i) * F_u * oo_nF;
    const Real lam_a = AMAX(
      -beta_u(dir,k,j,i) - lfac_a,
      -beta_u(dir,k,j,i) + lfac_a
    );

    return AMAX(
      lam_a,
      -beta_u(dir,k,j,i) + alpha(k,j,i) * sc_E(k,j,i) * F_u * oo_nF2
    );
  };

  auto lam_thick = [&](const int k, const int j, const int i)
  {
    const Real W2 = SQR(W(k,j,i));

    const Real fac_sqrt = alpha(k,j,i) * std::sqrt(
      (2.0 * W2 + 1.0) * g_uu(dir,dir,k,j,i) -
      2.0 * SQR(v_u(dir,k,j,i))
    );

    const Real fac_A = 2.0 * W(k,j,i) * alpha(k,j,i) * v_u(dir,k,j,i);
    const Real fac_B = OO(2.0 * W2 + 1);

    const Real lam_m = -beta_u(dir,k,j,i) + fac_B * (fac_A - fac_sqrt);
    const Real lam_p = -beta_u(dir,k,j,i) + fac_B * (fac_A + fac_sqrt);

    return AMAX(lam_m, lam_p);
  };

  switch (opt.characteristics_variety)
  {
    case (opt_characteristics_variety::mixed):
    {
      // see e.g. [2].
      M1_FLOOP3(k,j,i)
      if (pm1->MaskGet(k,j,i))
      {
        lambda(k,j,i) = lam_mixed(k,j,i);
      }

      break;
    }
    case (opt_characteristics_variety::exact_thin):
    {
      // see e.g. [2].
      M1_FLOOP3(k,j,i)
      if (pm1->MaskGet(k,j,i))
      {
        lambda(k,j,i) = lam_thin(k,j,i);
      }

      break;
    }
    case (opt_characteristics_variety::exact_thick):
    {
      // see e.g. [2].
      M1_FLOOP3(k,j,i)
      if (pm1->MaskGet(k,j,i))
      {
        lambda(k,j,i) = lam_thick(k,j,i);
      }

      break;
    }
    case (opt_characteristics_variety::exact_closure):
    {
      // see e.g. [2].
      M1_FLOOP3(k,j,i)
      if (pm1->MaskGet(k,j,i))
      {
        lambda(k,j,i) = (
          Assemble::Frames::d_th(sc_chi, k, j, i) * lam_thin(k,j,i) +
          Assemble::Frames::d_tk(sc_chi, k, j, i) * lam_thick(k,j,i)
        );
      }

      break;
    }
    default:
    {
      assert(false);
      std::exit(0);
    }
  }

}

// ============================================================================
} // namespace M1
// ============================================================================


//
// :D
//