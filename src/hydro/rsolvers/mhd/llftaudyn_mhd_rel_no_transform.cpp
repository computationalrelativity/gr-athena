// C++ headers
#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

// Athena++ headers
#include "../../hydro.hpp"
#include "../../../z4c/z4c.hpp"
#include "../../../utils/linear_algebra.hpp"
#include "../../../utils/interp_intergrid.hpp"
#include "../../../utils/floating_point.hpp"
#include "../../../athena_aliases.hpp"
#include "../../../coordinates/coordinates.hpp"
#include "../../../eos/eos.hpp"
#include "../../../mesh/mesh.hpp"

#include "../../../z4c/ahf.hpp"

//----------------------------------------------------------------------------------------
using namespace gra::aliases;
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
// Riemann solver
// Inputs:
//   kl,ku,jl,ju,il,iu: lower and upper x1-, x2-, and x3-indices
//   ivx: type of interface (IVX for x1, IVY for x2, IVZ for x3)
//   bb: 3D array of normal magnetic fields (not used)
//   prim_l,prim_r: 3D arrays of left and right primitive states
// Outputs:
//   flux: 3D array of hydrodynamical fluxes across interfaces
//   ey,ez: 3D arrays of magnetic fluxes (electric fields) across interfaces (not used)
// Notes:
//   implements LLF algorithm similar to that of fluxcalc() in step_ch.c in Harm
//   cf. LLFNonTransforming() in llf_rel.cpp
// Here we use the D, S, tau variable choice for conservatives, and assume a dynamically evolving spacetime
// so a factor of sqrt(detgamma) is included

void Hydro::RiemannSolver(
  const int ivx,
  const int k, const int j,
  const int il, const int iu,
  const AA &B,
  AA &prim_l_,
  AA &prim_r_,
  AA &pscalars_l_,
  AA &pscalars_r_,
  AA &aux_l_,
  AA &aux_r_,
  AT_N_sca & alpha_,
  AT_N_sca & oo_alpha_,
  AT_N_vec & beta_u_,
  AT_N_sym & gamma_dd_,
  AT_N_sca & detgamma_,
  AT_N_sca & oo_detgamma_,
  AT_N_sca & sqrt_detgamma_,
  AA &flux,
  AA &s_flux,
  AA &ey,
  AA &ez,
  AA &wct,
  const AA &dxw_,
  const Real lambda_rescaling)
{
  using namespace LinearAlgebra;
  using namespace FloatingPoint;

  MeshBlock * pmb = pmy_block;
  Mesh * pm = pmb->pmy_mesh;
  Hydro * ph = pmb->phydro;
  PassiveScalars * ps = pmb->pscalars;
  EquationOfState * peos = pmb->peos;
  Reconstruction * precon = pmb->precon;

  GRDynamical* pco_gr = static_cast<GRDynamical*>(pmb->pcoord);

  // Calculate cyclic permutations of indices
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;

  const int nn1 = pmy_block->nverts1;  // utilize the verts

  // time-step (needed for CT weight)
  const Real dt = pmb->pmy_mesh->dt;

  // Extract ratio of specific heats
#if USETM
  const Real mb = pmb->peos->GetEOS().GetBaryonMass();
#else
  const Real Gamma = pmb->peos->GetGamma();
  const Real Eos_Gamma_ratio = Gamma / (Gamma - 1.0);
#endif

  // 1d slices ----------------------------------------------------------------
  AT_N_sca w_rho_l_(prim_l_, IDN);
  AT_N_sca w_rho_r_(prim_r_, IDN);
  AT_N_sca w_p_l_(  prim_l_, IPR);
  AT_N_sca w_p_r_(  prim_r_, IPR);

  AT_N_vec w_util_u_l_(prim_l_, IVX);
  AT_N_vec w_util_u_r_(prim_r_, IVX);

  // reset values -------------------------------------------------------------
  Real T_min = 0;
  Real h_min = 0;

#if USETM
  T_min = peos->GetEOS().GetTemperatureFloor();
  h_min = peos->GetEOS().GetMinimumEnthalpy();
#endif

  // deal with excision -------------------------------------------------------
  auto excise = [&](const int i)
  {
    // Floor primitives during excision.
    peos->SetPrimAtmo(prim_l_, pscalars_l_, i);
    peos->SetPrimAtmo(prim_r_, pscalars_r_, i);

    aux_l_(IX_T,i) = T_min;
    aux_r_(IX_T,i) = T_min;

    aux_l_(IX_ETH,i) = h_min;
    aux_r_(IX_ETH,i) = h_min;

    aux_l_(IX_LOR,i) = 1.0;
    aux_r_(IX_LOR,i) = 1.0;
  };

  auto excise_with_factor = [&](Real excision_factor, const int i)
  {
    // Floor primitives during excision.
    for (int n=0; n<NHYDRO; ++n)
    {
      prim_l_(n,i) *= ph->excision_mask(k,j,i);
      prim_r_(n,i) *= ph->excision_mask(k,j,i);
    }

    aux_l_(IX_T,i) *= excision_factor;
    aux_r_(IX_T,i) *= excision_factor;

    aux_l_(IX_ETH,i) *= excision_factor;
    aux_r_(IX_ETH,i) *= excision_factor;

    aux_l_(IX_LOR,i) *= excision_factor;
    aux_r_(IX_LOR,i) *= excision_factor;
  };

  AA *x1, *x2, *x3;

  switch (ivx)
  {
    case IVX:
    {
      x1 = &pco_gr->x1f;
      x2 = &pco_gr->x2v;
      x3 = &pco_gr->x3v;
      break;
    }
    case IVY:
    {
      x1 = &pco_gr->x1v;
      x2 = &pco_gr->x2f;
      x3 = &pco_gr->x3v;
      break;
    }
    case IVZ:
    {
      x1 = &pco_gr->x1v;
      x2 = &pco_gr->x2v;
      x3 = &pco_gr->x3f;
      break;
    }
  }

  if (ph->opt_excision.excise_flux)
  {
    #pragma omp simd
    for (int i = il; i <= iu; ++i)
    {
      Real excision_factor = 1;
      const bool can_excise = peos->CanExcisePoint(
        excision_factor,
        true, alpha_, *x1, *x2, *x3, i, j, k);

      if (can_excise && !ph->opt_excision.excise_hydro_freeze_evo)
      {
        if (ph->opt_excision.use_taper)
        {
          excise_with_factor(excision_factor,i);
        }
        else
        {
          excise(i);
        }
      }
    }
  }
  // --------------------------------------------------------------------------

  // Continue with derived quantities -----------------------------------------
  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    detgamma_(i) = SQR(sqrt_detgamma_(i));
    oo_detgamma_(i)      = OO(detgamma_(i));
    oo_sqrt_detgamma_(i) = OO(sqrt_detgamma_(i));
  }

  Inv3Metric(oo_detgamma_, gamma_dd_, gamma_uu_, il, iu);

  // lower idx
  LinearAlgebra::SlicedVecMet3Contraction(
    w_util_d_l_, w_util_u_l_, gamma_dd_,
    il, iu
  );

  LinearAlgebra::SlicedVecMet3Contraction(
    w_util_d_r_, w_util_u_r_, gamma_dd_,
    il, iu
  );

  // Lorentz factors
  if (precon->xorder_use_aux_W)
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      W_l_(i) = aux_l_(IX_LOR,i);
      W_r_(i) = aux_r_(IX_LOR,i);
    }
  }
  else
  {
    for (int i=il; i<=iu; ++i)
    {
      const Real norm2_utilde_l = InnerProductSlicedVec3Metric(
        w_util_u_l_, gamma_dd_, i
      );

      const Real norm2_utilde_r = InnerProductSlicedVec3Metric(
        w_util_u_r_, gamma_dd_, i
      );

      // take abs for safety
      W_l_(i) = std::sqrt(1. + std::abs(norm2_utilde_l));
      W_r_(i) = std::sqrt(1. + std::abs(norm2_utilde_r));
    }
  }

  // need also reciprocals
  for (int i=il; i<=iu; ++i)
  {
    oo_W_l_(i) = OO(W_l_(i));
    oo_W_r_(i) = OO(W_r_(i));
  }

  // Eulerian vel.
  for (int a=0; a<NDIM; ++a)
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      w_v_u_l_(a,i) = w_util_u_l_(a,i) * oo_W_l_(i);
      w_v_u_r_(a,i) = w_util_u_r_(a,i) * oo_W_r_(i);

      alpha_w_vtil_u_l_(a,i) = alpha_(i) * w_v_u_l_(a, i) - beta_u_(a, i);
      alpha_w_vtil_u_r_(a,i) = alpha_(i) * w_v_u_r_(a, i) - beta_u_(a, i);
    }
  }

  InnerProductSlicedVec3Metric(w_norm2_v_l_, w_v_u_l_, gamma_dd_, il, iu);
  InnerProductSlicedVec3Metric(w_norm2_v_r_, w_v_u_r_, gamma_dd_, il, iu);

  // lower idx
  LinearAlgebra::SlicedVecMet3Contraction(
    beta_d_, beta_u_, gamma_dd_,
    il, iu
  );

  LinearAlgebra::SlicedVecMet3Contraction(
    w_v_d_l_, w_v_u_l_, gamma_dd_,
    il, iu
  );

  LinearAlgebra::SlicedVecMet3Contraction(
    w_v_d_r_, w_v_u_r_, gamma_dd_,
    il, iu
  );
  // =============================================================


  // assemble magnetic field components ---------------------------------------
  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    switch (ivx)
    {
      case IVX:
      {
        q_scB_u_l_(0,i) = oo_sqrt_detgamma_(i) * B(k,j,i);
        q_scB_u_l_(1,i) = oo_sqrt_detgamma_(i) * prim_l_(IBY,i);
        q_scB_u_l_(2,i) = oo_sqrt_detgamma_(i) * prim_l_(IBZ,i);
        q_scB_u_r_(0,i) = oo_sqrt_detgamma_(i) * B(k,j,i);
        q_scB_u_r_(1,i) = oo_sqrt_detgamma_(i) * prim_r_(IBY,i);
        q_scB_u_r_(2,i) = oo_sqrt_detgamma_(i) * prim_r_(IBZ,i);
        break;
      }
      case IVY:
      {
        q_scB_u_l_(1,i) = oo_sqrt_detgamma_(i) * B(k,j,i);
        q_scB_u_l_(2,i) = oo_sqrt_detgamma_(i) * prim_l_(IBY,i);
        q_scB_u_l_(0,i) = oo_sqrt_detgamma_(i) * prim_l_(IBZ,i);
        q_scB_u_r_(1,i) = oo_sqrt_detgamma_(i) * B(k,j,i);
        q_scB_u_r_(2,i) = oo_sqrt_detgamma_(i) * prim_r_(IBY,i);
        q_scB_u_r_(0,i) = oo_sqrt_detgamma_(i) * prim_r_(IBZ,i);
        break;
      }
      case IVZ:
      {
        q_scB_u_l_(2,i) = oo_sqrt_detgamma_(i) * B(k,j,i);
        q_scB_u_l_(0,i) = oo_sqrt_detgamma_(i) * prim_l_(IBY,i);
        q_scB_u_l_(1,i) = oo_sqrt_detgamma_(i) * prim_l_(IBZ,i);
        q_scB_u_r_(2,i) = oo_sqrt_detgamma_(i) * B(k,j,i);
        q_scB_u_r_(0,i) = oo_sqrt_detgamma_(i) * prim_r_(IBY,i);
        q_scB_u_r_(1,i) = oo_sqrt_detgamma_(i) * prim_r_(IBZ,i);
        break;
      }
      default:
      {
        assert(false);
      }
    }
  }


  b0_l_.ZeroClear();
  b0_r_.ZeroClear();

  for (int a = 0; a < NDIM; ++a)
  {
    #pragma omp simd
    for (int i = il; i <= iu; ++i)
    {
      b0_l_(i) += oo_alpha_(i) * W_l_(i) * q_scB_u_l_(a, i) * w_v_d_l_(a, i);
      b0_r_(i) += oo_alpha_(i) * W_r_(i) * q_scB_u_r_(a, i) * w_v_d_r_(a, i);
    }
  }

  for (int a = 0; a < NDIM; ++a)
  {
    #pragma omp simd
    for (int i = il; i <= iu; ++i)
    {
      bi_u_l_(a, i) =
        (oo_W_l_(i) * q_scB_u_l_(a, i) + b0_l_(i) * alpha_w_vtil_u_l_(a, i));
      bi_u_r_(a, i) =
        (oo_W_r_(i) * q_scB_u_r_(a, i) + b0_r_(i) * alpha_w_vtil_u_r_(a, i));
    }
  }


  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    b2_l_(i) = SQR(alpha_(i) * b0_l_(i) * oo_W_l_(i));
    b2_r_(i) = SQR(alpha_(i) * b0_r_(i) * oo_W_r_(i));
  }

  for (int a = 0; a < NDIM; ++a)
  {
    for (int b = 0; b < NDIM; ++b)
    {
      #pragma omp simd
      for (int i = il; i <= iu; ++i)
      {
        b2_l_(i) += SQR(oo_W_l_(i)) * q_scB_u_l_(a, i) * q_scB_u_l_(b, i) *
                    gamma_dd_(a, b, i);
        b2_r_(i) += SQR(oo_W_r_(i)) * q_scB_u_r_(a, i) * q_scB_u_r_(b, i) *
                    gamma_dd_(a, b, i);
      }
    }
  }

  for (int a = 0; a < NDIM; ++a)
  {
    #pragma omp simd
    for (int i = il; i <= iu; ++i)
    {
      bi_d_l_(a, i) = beta_d_(a, i) * b0_l_(i);
      bi_d_r_(a, i) = beta_d_(a, i) * b0_r_(i);
    }
    for (int b = 0; b < NDIM; ++b)
    {
      #pragma omp simd
      for (int i = il; i <= iu; ++i)
      {
        bi_d_l_(a, i) += gamma_dd_(a, b, i) * bi_u_l_(b, i);
        bi_d_r_(a, i) += gamma_dd_(a, b, i) * bi_u_r_(b, i);
      }
    }
  }
  // --------------------------------------------------------------------------


  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    // Calculate wavespeeds in left state NB EOS specific
#if USETM
    // If using the PrimitiveSolver framework, get the number density
    // and temperature to help calculate enthalpy.
    Real nl__ = w_rho_l_(i) / mb;
    Real nr__ = w_rho_r_(i) / mb;
    // FIXME: Generalize to work with EOSes accepting particle fractions.
    Real Yl__[MAX_SPECIES] = { 0.0 };
    Real Yr__[MAX_SPECIES] = { 0.0 };

    for (int n = 0; n < NSCALARS; n++)
    {
      Yl__[n] = pscalars_l_(n, i);
      Yr__[n] = pscalars_r_(n, i);
    }

    Real Tl__ = aux_l_(IX_T,i);
    Real Tr__ = aux_r_(IX_T,i);
    Real hl__ = aux_l_(IX_ETH,i);
    Real hr__ = aux_r_(IX_ETH,i);

    w_hrho_l_(i) = w_rho_l_(i) * hl__;
    w_hrho_r_(i) = w_rho_r_(i) * hr__;

    // Calculate the wave speeds
    if (precon->xorder_use_aux_cs2)
    {
      Real cs2l = aux_l_(IX_CS2,i);
      Real cs2r = aux_r_(IX_CS2,i);

      peos->FastMagnetosonicSpeedsGR(
        cs2l,
        nl__, Tl__,
        b2_l_(i), w_v_u_l_(ivx - 1, i), w_norm2_v_l_(i), alpha_(i),
        beta_u_(ivx - 1, i), gamma_uu_(ivx - 1, ivx - 1, i), &lambda_p_l(i),
        &lambda_m_l(i), Yl__);
      peos->FastMagnetosonicSpeedsGR(
        cs2r,
        nr__, Tr__,
        b2_r_(i), w_v_u_r_(ivx - 1, i), w_norm2_v_r_(i), alpha_(i),
        beta_u_(ivx - 1, i), gamma_uu_(ivx - 1, ivx - 1, i), &lambda_p_r(i),
        &lambda_m_r(i), Yr__);
    }
    else
    {
      peos->FastMagnetosonicSpeedsGR(
        nl__, Tl__,
        b2_l_(i), w_v_u_l_(ivx - 1, i), w_norm2_v_l_(i), alpha_(i),
        beta_u_(ivx - 1, i), gamma_uu_(ivx - 1, ivx - 1, i), &lambda_p_l(i),
        &lambda_m_l(i), Yl__);
      peos->FastMagnetosonicSpeedsGR(
        nr__, Tr__,
        b2_r_(i), w_v_u_r_(ivx - 1, i), w_norm2_v_r_(i), alpha_(i),
        beta_u_(ivx - 1, i), gamma_uu_(ivx - 1, ivx - 1, i), &lambda_p_r(i),
        &lambda_m_r(i), Yr__);
    }

#else
    w_hrho_l_(i) = w_rho_l_(i) + Eos_Gamma_ratio * w_p_l_(i);
    w_hrho_r_(i) = w_rho_r_(i) + Eos_Gamma_ratio * w_p_r_(i);

    peos->FastMagnetosonicSpeedsGR(
      w_hrho_l_(i), w_p_l_(i), b2_l_(i), w_v_u_l_(ivx - 1, i), w_norm2_v_l_(i),
      alpha_(i), beta_u_(ivx - 1, i), gamma_uu_(ivx - 1, ivx - 1, i),
      &lambda_p_l(i), &lambda_m_l(i));
    peos->FastMagnetosonicSpeedsGR(
      w_hrho_r_(i), w_p_r_(i), b2_r_(i), w_v_u_r_(ivx - 1, i), w_norm2_v_r_(i),
      alpha_(i), beta_u_(ivx - 1, i), gamma_uu_(ivx - 1, ivx - 1, i),
      &lambda_p_r(i), &lambda_m_r(i));
#endif
  }

  // Calculate extremal wavespeed
  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    Real lambda_l = std::min(lambda_m_l(i), lambda_m_r(i));
    Real lambda_r = std::max(lambda_p_l(i), lambda_p_r(i));
    lambda(i) = lambda_rescaling * std::max(lambda_r, -lambda_l);
  }

  // Calculate conserved quantities & fluxes
  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {

    // assemble conserved variables -------------------------------------------

    // l: D
    cons_l_(IDN, i) = sqrt_detgamma_(i) * w_rho_l_(i) * W_l_(i);

    // l: S_j
    for (int a=0; a<3; ++a)
    {
      cons_l_(IVX + a, i) =
        sqrt_detgamma_(i) *
        ((w_hrho_l_(i) + b2_l_(i)) * W_l_(i) * w_util_d_l_(a, i) -
         alpha_(i) * b0_l_(i) * bi_d_l_(a, i));
    }

    // l: tau
    cons_l_(IEN, i) =
      sqrt_detgamma_(i) *
      ((w_hrho_l_(i) + b2_l_(i)) * SQR(W_l_(i)) - w_rho_l_(i) * W_l_(i) -
       w_p_l_(i) - 0.5 * b2_l_(i) - SQR(alpha_(i) * b0_l_(i)));

    // l: B^k
    cons_l_(IBY, i) = sqrt_detgamma_(i) * q_scB_u_l_(ivy - 1, i);
    cons_l_(IBZ, i) = sqrt_detgamma_(i) * q_scB_u_l_(ivz - 1, i);

    // r: D
    cons_r_(IDN, i) = sqrt_detgamma_(i) * w_rho_r_(i) * W_r_(i);

    // r: S_j
    for (int a=0; a<3; ++a)
    {
      cons_r_(IVX + a, i) =
        sqrt_detgamma_(i) *
        ((w_hrho_r_(i) + b2_r_(i)) * W_r_(i) * w_util_d_r_(a, i) -
         alpha_(i) * b0_r_(i) * bi_d_r_(a, i));
    }

    // r: tau
    cons_r_(IEN, i) =
      sqrt_detgamma_(i) *
      ((w_hrho_r_(i) + b2_r_(i)) * SQR(W_r_(i)) - w_rho_r_(i) * W_r_(i) -
       w_p_r_(i) - 0.5 * b2_r_(i) - SQR(alpha_(i) * b0_r_(i)));

    // r: B^k
    cons_r_(IBY, i) = sqrt_detgamma_(i) * q_scB_u_r_(ivy - 1, i);
    cons_r_(IBZ, i) = sqrt_detgamma_(i) * q_scB_u_r_(ivz - 1, i);

    // ------------------------------------------------------------------------


    // assemble fluxes --------------------------------------------------------

    // l: D
    flux_l_(IDN, i) = cons_l_(IDN, i) * alpha_w_vtil_u_l_(ivx - 1, i);

    // l: S_j
    for (int a=0; a<3; ++a)
    {
      flux_l_(IVX + a, i) =
        cons_l_(IVX + a, i) * alpha_w_vtil_u_l_(ivx - 1, i) -
        alpha_(i) * sqrt_detgamma_(i) * bi_d_l_(a, i) *
          q_scB_u_l_(ivx - 1, i) * oo_W_l_(i);
    }

    flux_l_(ivx, i) +=
      (w_p_l_(i) + 0.5 * b2_l_(i)) * alpha_(i) * sqrt_detgamma_(i);

    // l: tau
    flux_l_(IEN, i) =
      cons_l_(IEN, i) * alpha_w_vtil_u_l_(ivx - 1, i) +
      alpha_(i) * sqrt_detgamma_(i) *
        ((w_p_l_(i) + 0.5 * b2_l_(i)) * w_v_u_l_(ivx - 1, i) -
         alpha_(i) * b0_l_(i) * q_scB_u_l_(ivx - 1, i) * oo_W_l_(i));

    // l: B^k
    flux_l_(IBY, i) = (cons_l_(IBY, i) * alpha_w_vtil_u_l_(ivx - 1, i) -
                       q_scB_u_l_(ivx - 1, i) * sqrt_detgamma_(i) *
                         alpha_w_vtil_u_l_(ivy - 1, i));
    flux_l_(IBZ, i) =
      (cons_l_(IBZ, i) * alpha_w_vtil_u_l_(ivx - 1, i) -
       q_scB_u_l_(ivx - 1, i) * sqrt_detgamma_(i) *
         alpha_w_vtil_u_l_(ivz - 1, i));  // check these indices

    // r: D
    flux_r_(IDN, i) = cons_r_(IDN, i) * alpha_w_vtil_u_r_(ivx - 1, i);

    // r: S_j
    for (int a=0; a<3; ++a)
    {
      flux_r_(IVX + a, i) =
        cons_r_(IVX + a, i) * alpha_w_vtil_u_r_(ivx - 1, i) -
        alpha_(i) * sqrt_detgamma_(i) * bi_d_r_(a, i) *
          q_scB_u_r_(ivx - 1, i) * oo_W_r_(i);
    }

    flux_r_(ivx, i) +=
      (w_p_r_(i) + 0.5 * b2_r_(i)) * alpha_(i) * sqrt_detgamma_(i);

    // r: tau
    flux_r_(IEN, i) =
      cons_r_(IEN, i) * alpha_w_vtil_u_r_(ivx - 1, i) +
      alpha_(i) * sqrt_detgamma_(i) *
        ((w_p_r_(i) + 0.5 * b2_r_(i)) * w_v_u_r_(ivx - 1, i) -
         alpha_(i) * b0_r_(i) * q_scB_u_r_(ivx - 1, i) * oo_W_r_(i));

    // r: B^k
    flux_r_(IBY, i) = (cons_r_(IBY, i) * alpha_w_vtil_u_r_(ivx - 1, i) -
                       q_scB_u_r_(ivx - 1, i) * sqrt_detgamma_(i) *
                         alpha_w_vtil_u_r_(ivy - 1, i));
    flux_r_(IBZ, i) =
      (cons_r_(IBZ, i) * alpha_w_vtil_u_r_(ivx - 1, i) -
       q_scB_u_r_(ivx - 1, i) * sqrt_detgamma_(i) *
         alpha_w_vtil_u_r_(ivz - 1, i));  // check these indices
  }

  // Set fluxes ---------------------------------------------------------------
  const bool use_hlle = pmy_block->precon->xorder_use_hlle;

  // probably cleaner to condense into single block, but verbose also works

  // hydro --------------------------------------------------------------------
  if (use_hlle)
  {
    for (int n=0; n<NHYDRO; ++n)
    {
      #pragma omp simd
      for (int i=il; i<=iu; ++i)
      {
        const Real lam_l__ = std::min(lambda_m_l(i), lambda_m_r(i));
        const Real lam_r__ = std::max(lambda_p_l(i), lambda_p_r(i));

        const Real flx_l__ = flux_l_(n,i);
        const Real flx_r__ = flux_r_(n,i);

        if (lam_l__ >= 0.0)
        {
          flux(n,k,j,i) = flx_l__;
        }
        else if (lam_r__ <= 0.0)
        {
          flux(n,k,j,i) = flx_r__;
        }
        else
        {
          flux(n,k,j,i) = (
            (lam_r__ * flx_l__ -  lam_l__ * flx_r__) +
            lam_l__ * lam_r__ * (cons_r_(n,i) - cons_l_(n,i))
          ) / (lam_r__ - lam_l__);
        }

        // probably better with a floor
        if (!std::isfinite(flux(n,k,j,i)))
        {
          flux(n,k,j,i) = 0.5 * (
            (flux_l_(n,i) + flux_r_(n,i)) -
            lambda(i) * (cons_r_(n,i) - cons_l_(n,i))
          );
        }
      }
    }
  }
  else
  {
    for (int n=0; n<NHYDRO; ++n)
    {
      #pragma omp simd
      for (int i=il; i<=iu; ++i)
      {
        flux(n,k,j,i) = 0.5 * (
          (flux_l_(n,i) + flux_r_(n,i)) -
          lambda(i) * (cons_r_(n,i) - cons_l_(n,i))
        );
      }
    }
  }

  // B_YZ ---------------------------------------------------------------------
  if (use_hlle)
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      const Real lam_l__ = std::min(lambda_m_l(i), lambda_m_r(i));
      const Real lam_r__ = std::max(lambda_p_l(i), lambda_p_r(i));

      const int N_BCPT = 2;
      Real flx__[N_BCPT];

      for (int I=0; I<N_BCPT; ++I)
      {

        if (lam_l__ >= 0.0)
        {
          flx__[I] = flux_l_(IBY+I,i);
        }
        else if (lam_r__ <= 0.0)
        {
          flx__[I] = flux_r_(IBY+I,i);
        }
        else
        {
          flx__[I] = (
            (lam_r__ * flux_l_(IBY+I,i) -
             lam_l__ * flux_r_(IBY+I,i)) +
            lam_l__ * lam_r__ * (cons_r_(IBY+I,i) -
                                 cons_l_(IBY+I,i))
          ) / (lam_r__ - lam_l__);
        }

        // LLF fallback - probably better with a floor
        if (!std::isfinite(flx__[I]))
        {
          flx__[I] = 0.5 * (
            (flux_l_(IBY+I,i) + flux_r_(IBY+I,i)) -
            lambda(i) * (cons_r_(IBY+I,i) - cons_l_(IBY+I,i))
          );
        }
      }

      // deal with CT
      ey(k, j, i) = -flx__[0];
      ez(k, j, i) = flx__[1];

      wct(k, j, i) = GetWeightForCT(flux(IDN, k, j, i), prim_l_(IDN, i),
                                    prim_r_(IDN, i), dxw_(i), dt);
    }
  }
  else
  {
    #pragma omp simd
    for (int i = il; i <= iu; ++i)
    {
      ey(k, j, i) =
        -0.5 * (flux_l_(IBY, i) + flux_r_(IBY, i) -
                lambda(i) * (cons_r_(IBY, i) - cons_l_(IBY, i)));
      ez(k, j, i) =
        0.5 * (flux_l_(IBZ, i) + flux_r_(IBZ, i) -
                lambda(i) * (cons_r_(IBZ, i) - cons_l_(IBZ, i)));

      wct(k, j, i) = GetWeightForCT(flux(IDN, k, j, i), prim_l_(IDN, i),
                                    prim_r_(IDN, i), dxw_(i), dt);
    }
  }

  // passive scalars ----------------------------------------------------------
  if (!pmy_block->precon->xorder_upwind_scalars)
  {
    if (use_hlle)
    {
      for (int n=0; n<NSCALARS; ++n)
      {
        #pragma omp simd
        for (int i=il; i<=iu; ++i)
        {
          const Real lam_l__ = std::min(lambda_m_l(i), lambda_m_r(i));
          const Real lam_r__ = std::max(lambda_p_l(i), lambda_p_r(i));

          const Real flx_l__ = flux_l_(IDN,i) * pscalars_l_(n,i);
          const Real flx_r__ = flux_r_(IDN,i) * pscalars_r_(n,i);

          if (lam_l__ >= 0.0)
          {
            s_flux(n,k,j,i) = flx_l__;
          }
          else if (lam_r__ <= 0.0)
          {
            s_flux(n,k,j,i) = flx_r__;
          }
          else
          {
            s_flux(n,k,j,i) = (
              (lam_r__ * flx_l__ -  lam_l__ * flx_r__) +
              lam_l__ * lam_r__ * (
                cons_r_(IDN,i) * pscalars_r_(n,i) -
                cons_l_(IDN,i) * pscalars_l_(n,i))
            ) / (lam_r__ - lam_l__);
          }

          if (!std::isfinite(s_flux(n,k,j,i)))
          {
             s_flux(n,k,j,i) = 0.5 * (
               (flux_l_(IDN,i) * pscalars_l_(n,i) +
                flux_r_(IDN,i) * pscalars_r_(n,i)) -
               lambda(i) * (cons_r_(IDN,i) * pscalars_r_(n,i) -
                            cons_l_(IDN,i) * pscalars_l_(n,i))
             );
          }
        }
      }
    }
    else
    {
      for (int n=0; n<NSCALARS; ++n)
      {
        #pragma omp simd
        for (int i=il; i<=iu; ++i)
        {
          s_flux(n,k,j,i) = 0.5 * (
            (flux_l_(IDN,i) * pscalars_l_(n,i) +
             flux_r_(IDN,i) * pscalars_r_(n,i)) -
            lambda(i) * (cons_r_(IDN,i) * pscalars_r_(n,i) -
                         cons_l_(IDN,i) * pscalars_l_(n,i))
          );
        }
      }
    }
  }
  else
  {
    for (int n=0; n<NSCALARS; ++n)
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      const Real mass_flx = flux(IDN,k,j,i);
      if (mass_flx >= 0.0)
      {
        s_flux(n,k,j,i) = mass_flx * pscalars_l_(n,i);
      }
      else
      {
        s_flux(n,k,j,i) = mass_flx * pscalars_r_(n,i);
      }
    }
  }

}