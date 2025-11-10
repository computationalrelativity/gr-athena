//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file llf_rel_no_transform.cpp
//  \brief Implements local Lax-Friedrichs Riemann solver for relativistic hydrodynamics
//  in pure GR.

// C++ headers
#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()
#include <iomanip>

// Athena++ headers
#include "../../hydro.hpp"
#include "../../../z4c/z4c.hpp"
#include "../../../utils/linear_algebra.hpp"
#include "../../../utils/interp_intergrid.hpp"
#include "../../../utils/floating_point.hpp"
#include "../../../athena_aliases.hpp"
#include "../../../coordinates/coordinates.hpp"  // Coordinates
#include "../../../eos/eos.hpp"                  // EquationOfState
#include "../../../mesh/mesh.hpp"                // MeshBlock

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
// compare with modification to add_flux_divergence_dyn, where factors of face area, cell volume etc are missing
// since they are included here.

void Hydro::RiemannSolver(
  const int ivx,
  const int k, const int j,
  const int il, const int iu,
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

  const int nn1 = pmb->nverts1;  // utilize the verts

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

  // Eulerian vel.
  for (int a=0; a<NDIM; ++a)
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      w_v_u_l_(a,i) = w_util_u_l_(a,i) / W_l_(i);
      w_v_u_r_(a,i) = w_util_u_r_(a,i) / W_r_(i);
    }
  }

  InnerProductSlicedVec3Metric(w_norm2_v_l_, w_v_u_l_, gamma_dd_, il, iu);
  InnerProductSlicedVec3Metric(w_norm2_v_r_, w_v_u_r_, gamma_dd_, il, iu);

  // #pragma omp simd
  // for (int i=il; i<=iu; ++i)
  // {
  //   w_norm2_v_l_(i) = 1.0 - OO(SQR(W_l_(i)));
  //   w_norm2_v_r_(i) = 1.0 - OO(SQR(W_r_(i)));
  // }


  // eigen-structure ----------------------------------------------------------
  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    // Calculate wavespeeds in left state NB EOS specific
#if USETM
    // If using the PrimitiveSolver framework, get the number density
    // and temperature to help calculate enthalpy.
    Real nl__ = w_rho_l_(i) / mb;
    Real nr__ = w_rho_r_(i) / mb;
    Real Yl__[MAX_SPECIES] = {0.0};
    Real Yr__[MAX_SPECIES] = {0.0};

    for (int n=0; n<NSCALARS; n++)
    {
      Yl__[n] = pscalars_l_(n,i);
      Yr__[n] = pscalars_r_(n,i);
    }

    Real Tl__ = aux_l_(IX_T,i);
    Real Tr__ = aux_r_(IX_T,i);
    Real hl__ = aux_l_(IX_ETH,i);
    Real hr__ = aux_r_(IX_ETH,i);

    w_hrho_l_(i) = w_rho_l_(i) * hl__;
    w_hrho_r_(i) = w_rho_r_(i) * hr__;

    // Calculate the sound speeds
    if (precon->xorder_use_aux_cs2)
    {
      Real cs2l = aux_l_(IX_CS2,i);
      Real cs2r = aux_r_(IX_CS2,i);

      peos->SoundSpeedsGR(cs2l, nl__, Tl__, w_v_u_l_(ivx-1,i),
                          w_norm2_v_l_(i),
                          alpha_(i), beta_u_(ivx-1,i),
                          gamma_uu_(ivx-1,ivx-1,i),
                          &lambda_p_l(i), &lambda_m_l(i), Yl__);
      peos->SoundSpeedsGR(cs2r, nr__, Tr__, w_v_u_r_(ivx-1,i),
                          w_norm2_v_r_(i),
                          alpha_(i), beta_u_(ivx-1,i),
                          gamma_uu_(ivx-1,ivx-1,i),
                          &lambda_p_r(i), &lambda_m_r(i), Yr__);
    }
    else
    {
      peos->SoundSpeedsGR(nl__, Tl__, w_v_u_l_(ivx-1,i), w_norm2_v_l_(i),
                          alpha_(i), beta_u_(ivx-1,i),
                          gamma_uu_(ivx-1,ivx-1,i),
                          &lambda_p_l(i), &lambda_m_l(i), Yl__);
      peos->SoundSpeedsGR(nr__, Tr__, w_v_u_r_(ivx-1,i), w_norm2_v_r_(i),
                          alpha_(i), beta_u_(ivx-1,i),
                          gamma_uu_(ivx-1,ivx-1,i),
                          &lambda_p_r(i), &lambda_m_r(i), Yr__);
    }
#else
    w_hrho_l_(i) = w_rho_l_(i) + Eos_Gamma_ratio * w_p_l_(i);
    w_hrho_r_(i) = w_rho_r_(i) + Eos_Gamma_ratio * w_p_r_(i);

    peos->SoundSpeedsGR(w_hrho_l_(i), w_p_l_(i), w_v_u_l_(ivx-1,i),
                        w_norm2_v_l_(i),
                        alpha_(i), beta_u_(ivx-1,i),
                        gamma_uu_(ivx-1,ivx-1,i),
                        &lambda_p_l(i), &lambda_m_l(i));
    peos->SoundSpeedsGR(w_hrho_r_(i), w_p_r_(i), w_v_u_r_(ivx-1,i),
                        w_norm2_v_r_(i),
                        alpha_(i), beta_u_(ivx-1,i),
                        gamma_uu_(ivx-1,ivx-1,i),
                        &lambda_p_r(i), &lambda_m_r(i));
#endif
}

  // Calculate extremal wavespeed
  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    const Real lambda_l = std::min(lambda_m_l(i), lambda_m_r(i));
    const Real lambda_r = std::max(lambda_p_l(i), lambda_p_r(i));
    lambda(i) = lambda_rescaling * std::max(lambda_r, -lambda_l);
  }

  // Calculate conserved quantities in L region incl. factor of sqrt(detgamma)

  // left ---------------------------------------------------------------------
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    // D = rho * gamma_lorentz
    cons_l_(IDN,i) = w_rho_l_(i) * W_l_(i) * sqrt_detgamma_(i);
    // tau = (rho * h = ) wgas * gamma_lorentz**2 - rho * gamma_lorentz - p
    cons_l_(IEN,i) = sqrt_detgamma_(i) * (
      w_hrho_l_(i) * SQR(W_l_(i)) - w_rho_l_(i)*W_l_(i) - w_p_l_(i)
    );
  }

  for (int a=0; a<NDIM; ++a)
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      // S_i = wgas * gamma_lorentz**2 * v_i = wgas * gamma_lorentz * u_i
      cons_l_(IVX+a,i) = sqrt_detgamma_(i) * (
        w_hrho_l_(i) * W_l_(i) * w_util_d_l_(a,i)
      );
    }
  }

  // Calculate fluxes in L region (rho u^i and T^i_\mu, where i = ivx)
  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    // D flux: D(v^i - beta^i/alpha)
    flux_l_(IDN,i) = cons_l_(IDN,i) * alpha_(i) * (
      w_v_u_l_(ivx-1,i) - beta_u_(ivx-1,i) * oo_alpha_(i)
    );

    // tau flux: alpha_(S^i - Dv^i) - beta^i tau
    flux_l_(IEN,i) = cons_l_(IEN,i) * alpha_(i) * (
      w_v_u_l_(ivx-1,i) - beta_u_(ivx-1,i) * oo_alpha_(i)
    ) + alpha_(i)*sqrt_detgamma_(i)*w_p_l_(i)*w_v_u_l_(ivx-1,i);
  }

  for (int a=0; a<NDIM; ++a)
  {
    #pragma omp simd
    for (int i = il; i <= iu; ++i)
    {
      //S_i flux alpha S^j_i - beta^j S_i
      flux_l_(IVX+a,i) = (cons_l_(IVX+a,i) * alpha_(i) *
                          (w_v_u_l_(ivx-1,i) -
                           beta_u_(ivx-1,i) * oo_alpha_(i)));

    }
  }

  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    flux_l_(ivx,i) += w_p_l_(i) * alpha_(i) * sqrt_detgamma_(i);
  }


  // right --------------------------------------------------------------------
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    // D = rho * gamma_lorentz
    cons_r_(IDN,i) = w_rho_r_(i) * W_r_(i) * sqrt_detgamma_(i);
    // tau = (rho * h = ) wgas * gamma_lorentz**2 - rho * gamma_lorentz - p
    cons_r_(IEN,i) = sqrt_detgamma_(i) * (
      w_hrho_r_(i) * SQR(W_r_(i)) - w_rho_r_(i)*W_r_(i) - w_p_r_(i)
    );
  }

  for (int a=0; a<NDIM; ++a)
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      // S_i = wgas * gamma_lorentz**2 * v_i = wgas * gamma_lorentz * u_i
      cons_r_(IVX+a,i) = sqrt_detgamma_(i) * (
        w_hrho_r_(i) * W_r_(i) * w_util_d_r_(a,i)
      );
    }
  }

  // Calculate fluxes in L region (rho u^i and T^i_\mu, where i = ivx)
  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    // D flux: D(v^i - beta^i/alpha)
    flux_r_(IDN,i) = cons_r_(IDN,i) * alpha_(i) * (
      w_v_u_r_(ivx-1,i) - beta_u_(ivx-1,i) * oo_alpha_(i)
    );

    // tau flux: alpha_(S^i - Dv^i) - beta^i tau
    flux_r_(IEN,i) = cons_r_(IEN,i) * alpha_(i) * (
      w_v_u_r_(ivx-1,i) - beta_u_(ivx-1,i) * oo_alpha_(i)
    ) + alpha_(i)*sqrt_detgamma_(i)*w_p_r_(i)*w_v_u_r_(ivx-1,i);
  }

  for (int a=0; a<NDIM; ++a)
  {
    #pragma omp simd
    for (int i = il; i <= iu; ++i)
    {
      //S_i flux alpha S^j_i - beta^j S_i
      flux_r_(IVX+a,i) = (cons_r_(IVX+a,i) * alpha_(i) *
                          (w_v_u_r_(ivx-1,i) -
                           beta_u_(ivx-1,i) * oo_alpha_(i)));

    }
  }

  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    flux_r_(ivx,i) += w_p_r_(i) * alpha_(i) * sqrt_detgamma_(i);
  }

  // Set fluxes ---------------------------------------------------------------
  const bool use_hll = pmy_block->precon->xorder_use_hll;

  if (use_hll)
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


  if (!pmy_block->precon->xorder_upwind_scalars)
  {
    if (use_hll)
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