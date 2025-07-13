// c++
#include <iostream>

// Athena++ headers
#include "m1.hpp"
#include "m1_macro.hpp"

// ============================================================================
namespace M1 {
// ============================================================================

void M1::HybridizeLOFlux(AA & u_cur)
{

  ::M1::M1::vars_Lab U_C { {pm1->N_GRPS,pm1->N_SPCS},
                            {pm1->N_GRPS,pm1->N_SPCS},
                            {pm1->N_GRPS,pm1->N_SPCS} };
  pm1->SetVarAliasesLab(pm1->storage.u, U_C);

  // hybridize (into ho) pp corrected fluxes
  AA & mask_pp = pm1->ev_strat.masks.pp;

  // no fallback if @ equilibrium (get unlimited fluxes)
  if (pm1->opt.flux_lo_fallback_eql_ho)
  {
    M1_GLOOP3(k, j, i)
    {
      if (pm1->IsEquilibrium(k,j,i))
      {
        mask_pp(k,j,i) = 0.0;
      }
    }
  }

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

      M1_MLOOP3(k, j, i)
      {
        const Real Theta = std::max(
          mask_pp(k,j,i),
          mask_pp(k-KO,j-JO,i-IO)
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

void M1::AdjustMaskPropertyPreservation()
{
  AA & mask_pp = pm1->ev_strat.masks.pp;
  M1_MLOOP3(k,j,i)
  {
    // Hybridization done, flip mask to execute on LO points
    mask_pp(k,j,i) = (mask_pp(k,j,i) < 1.0) ? 1.0 : 0.0;
  }

  if (!pmy_block->NeighborBlocksSameLevel())
  M1_MLOOP3(k,j,i)
  {
    // Execute on outermost physical cells to take into account flux corr.
    const bool boundary_cell = (
        (i==mbi.il || i==mbi.iu) ||
        (j==mbi.jl || j==mbi.ju) ||
        (k==mbi.kl || k==mbi.ku)
    );
    if (!boundary_cell)
    {
      continue;
    }

    mask_pp(k,j,i) = 0.0;
  }
}


// ============================================================================
} // namespace M1::Fluxes
// ============================================================================

//
// :D
//