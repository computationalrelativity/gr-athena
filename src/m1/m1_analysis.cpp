// C++ standard headers

// Athena++ headers
#include "../mesh/mesh.hpp"

#if FLUID_ENABLED
#include "../hydro/hydro.hpp"
#include "../eos/eos.hpp"
#endif // FLUID_ENABLED

#include "m1_macro.hpp"
#include "m1_analysis.hpp"

// ============================================================================
namespace M1 {
// ============================================================================

void M1::PerformAnalysis()
{
  MeshBlock * pmb = pmy_block;
  Analysis::CalcRadFlux(pmb);
  Analysis::CalcNeutrinoDiagnostics(pmb);
}

// ============================================================================
namespace Analysis {
// ============================================================================

void CalcRadFlux(MeshBlock *pmb)
{
  // ...
}

void CalcNeutrinoDiagnostics(MeshBlock *pmb)
{
#if FLUID_ENABLED
  M1 * pm1 = pmb->pm1;
  EquationOfState * peos = pmb->peos;
  Hydro * ph = pmb->phydro;
  PassiveScalars * ps = pmb->pscalars;

  AT_C_sca & sc_w_rho = pm1->hydro.sc_w_rho;
  AT_C_sca & sc_sqrt_det_g = pm1->geom.sc_sqrt_det_g;

  if ((pm1->N_GRPS == 1) && pm1->N_SPCS == 3)
  {
    // BD: TODO - raw or not?
    const Real mb = peos->GetEOS().GetRawBaryonMass();

    AT_C_sca & sc_z_sum = pm1->scratch.sc_A;
    sc_z_sum.Fill(0);

    for (int ix_g=0; ix_g < pm1->N_GRPS; ++ix_g)
    for (int ix_s=0; ix_s < pm1->N_SPCS; ++ix_s)
    {
      AT_C_sca & sc_n = pm1->rad.sc_n(ix_g, ix_s);
      AT_C_sca & sc_J = pm1->rad.sc_J(ix_g, ix_s);

      AT_C_sca & sc_y = pm1->rdiag.sc_y(ix_g, ix_s);
      AT_C_sca & sc_z = pm1->rdiag.sc_z(ix_g, ix_s);

      M1_ILOOP3(k, j, i)
      {
        const Real oo_sc_sqrt_det_g = OO(sc_sqrt_det_g(k,j,i));
        const Real oo_nb = (sc_w_rho(k,j,i) > 0) ?  mb / sc_w_rho(k,j,i) : 0;

        sc_y(k,j,i) = oo_nb * sc_n(k,j,i) * oo_sc_sqrt_det_g;
        sc_z(k,j,i) = sc_J(k,j,i) * oo_sc_sqrt_det_g;

        sc_z_sum(k,j,i) += sc_z(k,j,i);
      }
    }

    // sc_z_sum as total including contribution from fluid
    M1_ILOOP3(k, j, i)
    {
      Real Y[NSCALARS] {0};
      for(int n=0; n<NSCALARS; n++)
      {
        Y[n] = ps->r(n,k,j,i);
      }

      Real E_tot = peos->GetEOS().GetEnergy(sc_w_rho(k,j,i) / mb,
                                            ph->temperature(k,j,i),
                                            Y);

      sc_z_sum(k,j,i) += E_tot;
    }

    // normalize energies by total
    for (int ix_g=0; ix_g < pm1->N_GRPS; ++ix_g)
    for (int ix_s=0; ix_s < pm1->N_SPCS; ++ix_s)
    {
      AT_C_sca & sc_z = pm1->rdiag.sc_z(ix_g, ix_s);

      M1_ILOOP3(k, j, i)
      {
        sc_z(k,j,i) = sc_z(k,j,i) / sc_z_sum(k,j,i);
      }
    }

  }
#endif // FLUID_ENABLED
}

// ============================================================================
} // namespace M1::Analysis
// ============================================================================

// ============================================================================
} // namespace M1
// ============================================================================


//
// :D
//