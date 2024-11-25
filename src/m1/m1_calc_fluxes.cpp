// c++
// ...

// Athena++ headers
#include "m1_calc_fluxes.hpp"
#include "m1_macro.hpp"
#include "m1_utils.hpp"

// ============================================================================
namespace M1 {
// ============================================================================

// ----------------------------------------------------------------------------
// Interface for assembling flux vector components on cell-centers; these are
// subsequently reconstructed using a limiter.
//
// Method is 2nd order with high-Peclet limit fix
void M1::CalcFluxes(AthenaArray<Real> & u)
{
  vars_Lab U { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u, U);

  // (dense) scratch for flux vector component assembly (cell-centered samp.)
  AT_C_sca & F_sca = scratch.F_sca;
  AT_N_vec & F_vec = scratch.F_vec;

  AT_C_sca & lambda = scratch.lambda;

  // required geometric quantities
  AT_C_sca & alpha  = geom.sc_alpha;
  AT_N_vec & beta_u = geom.sp_beta_u;
  AT_N_sym & g_uu   = geom.sp_g_uu;

  // point to scratches -------------------------------------------------------
  AT_C_sca & sc_norm_sp_H_ = scratch.sc_norm_sp_H_;
  AT_C_sca & sc_G_         = scratch.sc_G_;

  AT_N_vec & sp_H_u_       = scratch.sp_H_u_;
  AT_N_vec & sp_f_u_       = scratch.sp_f_u_;

  AT_D_vec & st_F_d_       = scratch.st_F_d_;

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

      AT_C_sca & F_nG  = fluxes.sc_nG( ix_g,ix_s,ix_d);
      AT_C_sca & F_E   = fluxes.sc_E(  ix_g,ix_s,ix_d);
      AT_N_vec & F_F_d = fluxes.sp_F_d(ix_g,ix_s,ix_d);

      AT_C_sca & U_nG   = U.sc_nG( ix_g,ix_s);
      AT_N_vec & U_F_d  = U.sp_F_d(ix_g,ix_s);
      AT_C_sca & U_E    = U.sc_E(  ix_g,ix_s);

      AT_N_sym & sp_P_dd = lab_aux.sp_P_dd(ix_g,ix_s);

      AT_C_sca & sc_n   = rad.sc_n(ix_g,ix_s);
      AT_C_sca & sc_J   = rad.sc_J(ix_g,ix_s);
      AT_N_vec & sp_H_d = rad.sp_H_d(ix_g,ix_s);

      AT_C_sca & sc_kap_a = radmat.sc_kap_a(ix_g,ix_s);
      AT_C_sca & sc_kap_s = radmat.sc_kap_s(ix_g,ix_s);

      AT_C_sca & sc_chi = lab_aux.sc_chi(ix_g,ix_s);

      if (!approximate_speeds)
        CalcCharacteristicSpeed(ix_d, U_E, U_F_d, sc_chi, lambda);

      // Flux assembly and reconstruction =====================================
      // See Eq.(28) of [1] (note densitized)

      // nG -------------------------------------------------------------------
      M1_FLOOP2(k,j)
      {
        Assemble::sp_d_to_u_(this, sp_H_u_, sp_H_d, k, j, il, iu);
        Assemble::sc_norm_sp_(this, sc_norm_sp_H_, sp_H_u_, sp_H_d,
                              k, j, il, iu);

        Assemble::sp_f_u_(
          this, sp_f_u_, sp_H_u_, sc_norm_sp_H_, sc_J,
          ix_d, k, j, il, iu
        );

        M1_FLOOP1(i)
        {
          const Real W  = pm1->fidu.sc_W(k,j,i);
          const Real W2 = SQR(W);

          const Real dotFv = Assemble::sc_dot_dense_sp__(
            U_F_d, pm1->fidu.sp_v_u, k, j, i);

          sc_G_(i) = Assemble::sc_G__(
            this->fidu.sc_W(k,j,i), U_E(k,j,i), sc_J(k,j,i), dotFv,
            this->opt.fl_E, this->opt.fl_J, this->opt.eps_E
          );
        }

        M1_FLOOP1(i)
        {
          sc_n(k,j,i) = U_nG(k,j,i) / sc_G_(i);
          F_sca(k,j,i) = alpha(k,j,i) * sc_n(k,j,i) * sp_f_u_(ix_d,i);
        }
      }
      Fluxes::ReconstructLimitedFlux(this, ix_d, U_nG, F_sca,
                                     sc_kap_a, sc_kap_s, lambda, F_nG);

      // E --------------------------------------------------------------------
      M1_FLOOP2(k,j)
      {
        M1_FLOOP1(i)
        {
          F_sca(k,j,i) = -beta_u(ix_d,k,j,i) * U_E(k,j,i);
        }

        for (int a=0; a<N; ++a)
        M1_FLOOP1(i)
        {
          F_sca(k,j,i) += alpha(k,j,i) * geom.sp_g_uu(ix_d,a,k,j,i) *
                          U_F_d(a,k,j,i);
        }

      }
      Fluxes::ReconstructLimitedFlux(this, ix_d, U_E, F_sca,
                                     sc_kap_a, sc_kap_s, lambda, F_E);

      // F_k ------------------------------------------------------------------
      M1_FLOOP2(k,j)
      {
        for (int a=0; a<N; ++a)
        {
          M1_FLOOP1(i)
          {
            F_vec(a,k,j,i) = -beta_u(ix_d,k,j,i) * U_F_d(a,k,j,i);
          }

          for (int b=0; b<N; ++b)
          M1_FLOOP1(i)
          {
            F_vec(a,k,j,i) += alpha(k,j,i) *
                              g_uu(ix_d,b,k,j,i) * sp_P_dd(b,a,k,j,i);
          }
        }
      }
      Fluxes::ReconstructLimitedFlux(this, ix_d, U_F_d, F_vec,
                                     sc_kap_a, sc_kap_s, lambda, F_F_d);
    }
  }

  return;
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

  AT_C_sca & sc_sqrt_det_g = geom.sc_sqrt_det_g;

  auto AMAX = [&](const Real A, const Real B)
  {
    return std::max(std::abs(A), std::abs(B));
  };

  auto lam_thin = [&](const int k, const int j, const int i)
  {
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
  };

  auto lam_thick = [&](const int k, const int j, const int i)
  {
    Real normv2 (0.);
    for (int a=0; a<N; ++a)
    {
      normv2 += v_d(a,k,j,i) * v_u(a,k,j,i);
    }

    const Real W2 = SQR(W(k,j,i));

    const Real fac_sqrt = std::sqrt(
      (2.0 * W2 + 1.0) * alpha(k,j,i) * g_uu(dir,dir,k,j,i) -
      2.0 * W2 * normv2
    ) / (2.0 * W2 + 1);

    const Real fac_A = 2.0 * W2 * std::abs(v_d(dir,k,j,i)) / (2.0 * W2 + 1);

    const Real lam_m = std::abs(beta_d(dir,k,j,i)) + fac_A - fac_sqrt;
    const Real lam_p = std::abs(beta_d(dir,k,j,i)) + fac_A + fac_sqrt;

    return AMAX(lam_m, lam_p);
  };

  switch (opt.characteristics_variety)
  {
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
    case (opt_characteristics_variety::exact_Minerbo):
    {
      // see e.g. [2].
      M1_FLOOP3(k,j,i)
      if (pm1->MaskGet(k,j,i))
      {
        lambda(k,j,i) = (
          0.5 * (3.0 * sc_chi(k,j,i) - 1.0) * lam_thin(k,j,i) +
          0.5 * 3.0 * (1.0 - sc_chi(k,j,i)) * lam_thick(k,j,i)
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