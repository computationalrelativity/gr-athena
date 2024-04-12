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
  const int iu = pm1->mbi.iu-M1_NGHOST_MIN;

  for (int ix_d=0; ix_d<N; ++ix_d)
  {
    if (opt.characteristics_variety ==
        opt_characteristics_variety::approximate)
      CalcCharacteristicSpeed(ix_d, lambda);

    for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
    for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
    {
      if (opt.characteristics_variety ==
          opt_characteristics_variety::exact)
        CalcCharacteristicSpeed(ix_d, lambda);

      AT_C_sca & F_nG  = fluxes.sc_nG( ix_g,ix_s,ix_d);
      AT_C_sca & F_E   = fluxes.sc_E(  ix_g,ix_s,ix_d);
      AT_N_vec & F_F_d = fluxes.sp_F_d(ix_g,ix_s,ix_d);

      AT_C_sca & U_nG   = U.sc_nG( ix_g,ix_s);
      AT_N_vec & U_F_d  = U.sp_F_d(ix_g,ix_s);
      AT_C_sca & U_E    = U.sc_E(  ix_g,ix_s);

      AT_N_sym & sp_P_dd = lab_aux.sp_P_dd(ix_g,ix_s);
      AT_C_sca & sc_n    = lab_aux.sc_n(   ix_g,ix_s);

      AT_C_sca & sc_J   = rad.sc_J(ix_g,ix_s);
      AT_N_vec & sp_H_d = rad.sp_H_d(ix_g,ix_s);

      AT_C_sca & sc_kap_a = radmat.sc_kap_a(ix_g,ix_s);
      AT_C_sca & sc_kap_s = radmat.sc_kap_s(ix_g,ix_s);

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

        Assemble::st_F_d_(this, st_F_d_, U_F_d, k, j, il, iu);
        // TODO: could reduce the internal contraction to only utilize sp_F_d
        Assemble::sc_G_(this, sc_G_, U_E, sc_J, st_F_d_, k, j, il, iu);

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
void M1::CalcCharacteristicSpeed(const int dir,
                                 AT_C_sca & lambda)
{
  switch (opt.characteristics_variety)
  {
    case (opt_characteristics_variety::approximate):
    {
      AT_C_sca & alpha  = geom.sc_alpha;
      AT_N_vec & beta_u = geom.sp_beta_u;
      AT_N_sym & g_uu   = geom.sp_g_uu;

      auto AMAX = [&](const Real A, const Real B){
        return std::max(std::abs(A), std::abs(B));
      };

      M1_FLOOP3(k,j,i)
      {
        const Real A = alpha(k,j,i) * std::sqrt(g_uu(dir,dir,k,j,i));
        const Real B = beta_u(dir,k,j,i);
        lambda(k,j,i) = AMAX(A+B, A-B);
      }
      break;
    }
    case (opt_characteristics_variety::exact):
    {
      std::stringstream msg;
      msg << "Exact M1::CalcCharacteristicSpeed not implemented" << std::endl;
      ATHENA_ERROR(msg);
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