// C++ standard headers

// Athena++ headers
#include "../mesh/mesh.hpp"
#include "m1_containers.hpp"
#include "m1_utils.hpp"

#if FLUID_ENABLED
#include "../hydro/hydro.hpp"
#include "../eos/eos.hpp"
#endif // FLUID_ENABLED

#include "m1_macro.hpp"
#include "m1_analysis.hpp"
#include "m1_calc_closure.hpp"

// ============================================================================
namespace M1 {
// ============================================================================

void M1::PerformAnalysis()
{
  MeshBlock * pmb = pmy_block;
  Analysis::CalcEnergyAverages(pmb);
  Analysis::CalcRadFlux(pmb);
  Analysis::CalcNeutrinoDiagnostics(pmb);
}

// ============================================================================
namespace Analysis {
// ============================================================================

void CalcEnergyAverages(MeshBlock *pmb)
{
  using namespace Closures;

  M1 * pm1 = pmb->pm1;

  M1::vars_Lab U { {pm1->N_GRPS,pm1->N_SPCS},
                   {pm1->N_GRPS,pm1->N_SPCS},
                   {pm1->N_GRPS,pm1->N_SPCS} };
  pm1->SetVarAliasesLab(pm1->storage.u, U);

  // geometric quantities
  const AT_C_sca & sc_alpha = pm1->geom.sc_alpha;

  // required fiducial quantities
  AT_C_sca & sc_W = pm1->fidu.sc_W;

  // If reduction onto surfaces is required, then we need (n, J) on the whole
  // MeshBlock. Therefore we need to compute closures on the ghosts using the
  // communicated cons.

  for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<pm1->N_SPCS; ++ix_s)
  {
    ClosureMetaVector C = ConstructClosureMetaVector(*pm1, U, ix_g, ix_s);

    AT_C_sca & sc_E    = pm1->lab.sc_E(  ix_g,ix_s);
    AT_N_vec & sp_F_d  = pm1->lab.sp_F_d(ix_g,ix_s);
    AT_C_sca & sc_nG   = pm1->lab.sc_nG( ix_g,ix_s);

    AT_C_sca & sc_chi  = pm1->lab_aux.sc_chi(ix_g,ix_s);

    AT_C_sca & sc_n = pm1->rad.sc_n(ix_g,ix_s);
    AT_C_sca & sc_J = pm1->rad.sc_J(ix_g,ix_s);
    AT_D_vec & st_H_u = pm1->rad.st_H_u(ix_g, ix_s);

    AT_C_sca & sc_avg_nrg = pm1->radmat.sc_avg_nrg(ix_g, ix_s);

    M1_GLOOP3(k, j, i)
    if (pm1->MaskGet(k,j,i))
    if (!pmb->IsPhysicalIndex_cc(k, j, i))
    {
      C.Closure(k,j,i);
    }

    // Closures prepared, can map to fiducial frame globally ------------------
    M1_GLOOP3(k, j, i)
    if (pm1->MaskGet(k,j,i))
    {
      Assemble::Frames::ToFiducial(*pm1, sc_J, st_H_u, sc_chi, sc_E, sp_F_d,
                                   k, j, i, i);

      const Real J_0 = sc_J(k,j,i);
      // H^a = H_n n^a + SPATIAL where n^0 = 1 / alpha
      const Real H_n = st_H_u(0,k,j,i) * sc_alpha(k,j,i);
      const Real W = sc_W(k,j,i);

      const Real sc_G__ = (J_0 > 0)
        ? (W + H_n / J_0)
        : W;

      sc_n(k,j,i) = sc_nG(k,j,i) / sc_G__;

      sc_avg_nrg(k,j,i) = sc_J(k,j,i) / sc_n(k,j,i);
    }

  }
}

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

      // BD: this is energy density, not tot?
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
        sc_z(k,j,i) = sc_z(k,j,i); // / sc_z_sum(k,j,i);
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