// c++
#include <iostream>

// Athena++ headers
#include "m1.hpp"

// ============================================================================
namespace M1 {
// ============================================================================

void M1::HybridizeLOFlux(AA & u_cur)
{

  ::M1::M1::vars_Lab U_C { {pm1->N_GRPS,pm1->N_SPCS},
                            {pm1->N_GRPS,pm1->N_SPCS},
                            {pm1->N_GRPS,pm1->N_SPCS} };
  pm1->SetVarAliasesLab(pm1->storage.u, U_C);

  /*
  for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<pm1->N_SPCS; ++ix_s)
  M1_ILOOP3(k, j, i)
  if (pm1->MaskGet(k, j, i))
  {
    // if (!(U_C.sc_E(ix_g,ix_s)(k,j,i)  > pm1->opt.fl_E) ||
    //     (!(U_C.sc_nG(ix_g,ix_s)(k,j,i)  > pm1->opt.fl_nG )))
    if (!(U_C.sc_E(ix_g,ix_s)(k,j,i)  > pm1->opt.fl_E) )
    {
      // update CC mask
      pm1->ev_strat.masks.pp(k,j,i) = std::max(
        pm1->ev_strat.masks.pp(k,j,i),
        1.0
      );
    }
  }
  */

  // hybridize (into ho) pp corrected fluxes
  for (int ix_d=0; ix_d<N; ++ix_d)
  {
    const int IO = ix_d == 0;
    const int JO = ix_d == 1;
    const int KO = ix_d == 2;

    for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
    for (int ix_s=0; ix_s<pm1->N_SPCS; ++ix_s)
    {
      AT_C_sca & F_nG  = pm1->fluxes.sc_nG( ix_g,ix_s,ix_d);
      AT_C_sca & F_E   = pm1->fluxes.sc_E(  ix_g,ix_s,ix_d);
      AT_N_vec & F_f_d = pm1->fluxes.sp_F_d(ix_g,ix_s,ix_d);

      AT_C_sca & lo_F_nG  = pm1->fluxes_lo.sc_nG( ix_g,ix_s,ix_d);
      AT_C_sca & lo_F_E   = pm1->fluxes_lo.sc_E(  ix_g,ix_s,ix_d);
      AT_N_vec & lo_F_f_d = pm1->fluxes_lo.sp_F_d(ix_g,ix_s,ix_d);

      M1_ILOOP3(k, j, i)
      {
        const Real Theta = std::max(
          pm1->ev_strat.masks.pp(k,j,i),
          pm1->ev_strat.masks.pp(k-KO,j-JO,i-IO)
        );

        F_E(k,j,i) = F_E(k,j,i) - Theta * (F_E(k,j,i) - lo_F_E(k,j,i));
        F_nG(k,j,i) = F_nG(k,j,i) - Theta * (F_nG(k,j,i) - lo_F_nG(k,j,i));
        for (int a=0; a<N; ++a)
        {
          F_f_d(a,k,j,i) = F_f_d(a,k,j,i) - Theta * (
            F_f_d(a,k,j,i) - lo_F_f_d(a,k,j,i)
          );
        }
      }
    }

  }

}


// ============================================================================
} // namespace M1::Fluxes
// ============================================================================

//
// :D
//