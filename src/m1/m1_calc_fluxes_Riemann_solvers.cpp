// c++
#include <iostream>

// Athena++ headers
#include "m1_calc_fluxes.hpp"
#include "../reconstruct/reconstruction.hpp"

// ============================================================================
namespace M1::Fluxes {
// ============================================================================

// N.B.
// Eigenvalues and closure are assembled at interfaces through nn average.

void RiemannHLLEmod(
  M1 * pm1,
  AA & u,
  const bool use_lo
)
{
  // DEBUG --------------------------------------------------------------------
  assert(false); // not implemented, probably never will be

  Reconstruction * pr = pm1->pmy_block->precon;
  typedef Reconstruction::ReconstructionVariant ReconstructionVariant;
  ReconstructionVariant rv = ReconstructionVariant::lin_vl;

  // point to flux storage ----------------------------------------------------
  M1::vars_Flux & flx = (use_lo) ? pm1->fluxes_lo : pm1->fluxes;

  // scratch for recon. -------------------------------------------------------
  AT_C_sca sc_E_l_;
  AT_C_sca sc_E_r_;

  AT_N_vec sp_F_d_l_;
  AT_N_vec sp_F_d_r_;

  AT_C_sca sc_chi_;
  AT_C_sca sc_lam_m_;
  AT_C_sca sc_lam_p_;

  int il, iu, jl, ju, kl, ku;
  int ix_d;

  // Cf. (M)HD sector ---------------------------------------------------------
  for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<pm1->N_SPCS; ++ix_s)
  {
    ix_d = 0; // along x

    pr->SetIndicialLimitsCalculateFluxes(IVX, il, iu, jl, ju, kl, ku);

    AT_C_sca & F_nG  = flx.sc_nG( ix_g,ix_s,ix_d);
    AT_C_sca & F_E   = flx.sc_E(  ix_g,ix_s,ix_d);
    AT_N_vec & F_F_d = flx.sp_F_d(ix_g,ix_s,ix_d);

  }
}

// ============================================================================
} // namespace M1::Fluxes
// ============================================================================

//
// :D
//