// c++
#include <iostream>

// Athena++ headers
#include "m1.hpp"
#include "m1_macro.hpp"

// ============================================================================
namespace M1 {
// ============================================================================

void M1::HybridizeLOFlux(AA & mask_hyb,
                         vars_Flux & fluxes_ho,
                         vars_Flux & fluxes_lo)
{

  // hybridize (into ho) pp corrected fluxes

  // no fallback if @ equilibrium (get unlimited fluxes)
  // if (pm1->opt.flux_lo_fallback_eql_ho)
  // {
  //   M1_GLOOP3(k, j, i)
  //   {
  //     if (pm1->IsEquilibrium(k,j,i))
  //     {
  //       mask_pp(k,j,i) = 0.0;
  //     }
  //   }
  // }

  for (int ix_d=0; ix_d<N; ++ix_d)
  {
    const int IO = ix_d == 0;
    const int JO = ix_d == 1;
    const int KO = ix_d == 2;

    for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
    for (int ix_s=0; ix_s<pm1->N_SPCS; ++ix_s)
    {
      AT_C_sca & F_nG  = fluxes_ho.sc_nG( ix_g,ix_s,ix_d);
      AT_C_sca & F_E   = fluxes_ho.sc_E(  ix_g,ix_s,ix_d);
      AT_N_vec & F_f_d = fluxes_ho.sp_F_d(ix_g,ix_s,ix_d);

      AT_C_sca & lo_F_nG  = fluxes_lo.sc_nG( ix_g,ix_s,ix_d);
      AT_C_sca & lo_F_E   = fluxes_lo.sc_E(  ix_g,ix_s,ix_d);
      AT_N_vec & lo_F_f_d = fluxes_lo.sp_F_d(ix_g,ix_s,ix_d);

      // mask species index if using per-species
      const int ix_ms = (opt.flux_lo_fallback_species)
        ? ix_s
        : 0;

      M1_MLOOP3(k, j, i)
      {
        Real Theta = std::max(
          mask_hyb(ix_ms,k,j,i),
          mask_hyb(ix_ms,k-KO,j-JO,i-IO)
        );

        if (pm1->opt.flux_lo_fallback_eql_ho &&
            pm1->IsEquilibrium(ix_s,k,j,i) &&
            pm1->IsEquilibrium(ix_s,k-KO,j-JO,i-IO))
        {
          Theta = 0;
        }

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
  assert(!opt.flux_lo_fallback_species);  // B.D. needs fix for species

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