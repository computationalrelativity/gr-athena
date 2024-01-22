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
#include "../../../athena.hpp"                   // enums, macros
#include "../../../athena_arrays.hpp"            // AthenaArray
#include "../../../coordinates/coordinates.hpp"  // Coordinates
#include "../../../eos/eos.hpp"                  // EquationOfState
#include "../../../mesh/mesh.hpp"                // MeshBlock


#include "../../../reconstruct/reconstruction.hpp"

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
  const int k, const int j,
  const int il, const int iu,
  const int ivx,
  AthenaArray<Real> &prim_l,
  AthenaArray<Real> &prim_r,
  AthenaArray<Real> &flux,
  const AthenaArray<Real> &dxw)
{
  using namespace LinearAlgebra;

  MeshBlock * pmb = pmy_block;

  if (0)
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      pmb->peos->ApplyPrimitiveFloors(prim_l,k,j,i);
      pmb->peos->ApplyPrimitiveFloors(prim_r,k,j,i);
    }
  }

  // for readability
  const int D = NDIM + 1;
  const int N = NDIM;

  typedef AthenaArray< Real>                         AA;
  typedef AthenaTensor<Real, TensorSymm::NONE, N, 0> AT_N_sca;
  typedef AthenaTensor<Real, TensorSymm::NONE, N, 1> AT_N_vec;
  typedef AthenaTensor<Real, TensorSymm::SYM2, N, 2> AT_N_sym;

  // For fluid-variable vector
  typedef AthenaTensor<Real, TensorSymm::NONE, NWAVE, 1> AT_F_vec;

  // Calculate cyclic permutations of indices
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;

  const int nn1 = pmy_block->nverts1;  // utilize the verts
  // Extract ratio of specific heats
#if USETM
  const Real mb = pmy_block->peos->GetEOS().GetBaryonMass();
#else
  const Real Gamma = pmy_block->peos->GetGamma();
  const Real Eos_Gamma_ratio = Gamma / (Gamma - 1.0);
#endif

  // perform variable resampling when required
  Z4c * pz4c = pmy_block->pz4c;

  // Slice 3d z4c metric quantities  (NDIM=3 in z4c.hpp) ----------------------
  AT_N_sym sl_adm_gamma_dd(pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sca sl_z4c_alpha(   pz4c->storage.u,   Z4c::I_Z4c_alpha);
  AT_N_vec sl_z4c_beta_u(  pz4c->storage.u,   Z4c::I_Z4c_betax);

  if(0)  // debug FC transfer
  {

    for (int k=0; k<pmb->ncells3; ++k)
    for (int j=0; j<pmb->ncells2; ++j)
    for (int i=0; i<pmb->ncells1; ++i)
    {
      sl_z4c_alpha(k,j,i) = 11 + std::sin((i+1)*(j+1)*(k+1));
      // sl_z4c_alpha(k,j,i) = SQR((i+1)*(j+1)*(k+1));
    }

  }


  // various scratches --------------------------------------------------------
  AT_N_sca sqrt_detgamma_(iu+1);
  AT_N_sca detgamma_(     iu+1);  // spatial met det
  AT_N_sca oo_detgamma_(  iu+1);  // 1 / spatial met det

  // AT_N_sca alpha_(   nn1);   // reconstruction performed on all possible i
  // AT_N_vec beta_u_(  nn1);   // so need nn1
  AT_N_sym gamma_dd_(nn1);
  AT_N_sym gamma_uu_(iu+1);

  // debug: reconstruction of variables
  AT_N_sca r_alpha_l_(nn1);
  AT_N_sca r_alpha_r_;

  AT_N_vec r_beta_u_l_(nn1);
  AT_N_vec r_beta_u_r_;

  AT_N_vec w_v_u_l_(iu+1);
  AT_N_vec w_v_u_r_(iu+1);

  AT_N_sca w_norm2_v_l(iu+1);
  AT_N_sca w_norm2_v_r(iu+1);

  AT_N_sca lambda_p_l(iu+1);
  AT_N_sca lambda_m_l(iu+1);
  AT_N_sca lambda_p_r(iu+1);
  AT_N_sca lambda_m_r(iu+1);
  AT_N_sca lambda(iu+1);

  // primitive vel. (covar.)
  AT_N_vec w_util_d_l_(iu+1);
  AT_N_vec w_util_d_r_(iu+1);

  // Lorentz factor
  AT_N_sca W_l_(iu+1);
  AT_N_sca W_r_(iu+1);

  // h * rho
  AT_N_sca w_hrho_l_(iu+1);
  AT_N_sca w_hrho_r_(iu+1);

  // prim / cons shaped scratches
  AT_F_vec cons_l_(iu+1);
  AT_F_vec cons_r_(iu+1);

  AT_F_vec flux_l_(iu+1);
  AT_F_vec flux_r_(iu+1);

  // 1d slices ----------------------------------------------------------------
  AT_N_sca w_rho_l_(prim_l, IDN);
  AT_N_sca w_rho_r_(prim_r, IDN);
  AT_N_sca w_p_l_(  prim_l, IPR);
  AT_N_sca w_p_r_(  prim_r, IPR);

  AT_N_vec w_util_u_l_(prim_l, IVX);
  AT_N_vec w_util_u_r_(prim_r, IVX);

  // Reconstruction to FC -----------------------------------------------------
  GRDynamical* pco_gr = static_cast<GRDynamical*>(pmy_block->pcoord);
  pco_gr->GetGeometricFieldFC(gamma_dd_,   sl_adm_gamma_dd, ivx-1, k, j);

#ifndef DBG_RECONSTRUCT_GEOM
  pco_gr->GetGeometricFieldFC(r_alpha_l_,  sl_z4c_alpha,    ivx-1, k, j);
  pco_gr->GetGeometricFieldFC(r_beta_u_l_, sl_z4c_beta_u,   ivx-1, k, j);

  r_alpha_r_.InitWithShallowSlice( r_alpha_l_.array(), 0);
  r_beta_u_r_.InitWithShallowSlice(r_beta_u_l_.array(), 0);
#else

  const bool enforce_floors = false;

  AthenaArray<Real> empty_array;
  r_alpha_r_.NewAthenaTensor( nn1);
  r_beta_u_r_.NewAthenaTensor(nn1);

  AT_N_sca r_alpha_lb_( nn1);
  AT_N_vec r_beta_u_lb_(nn1);

  if (ivx == IVX)
  {
    pmb->precon->WenoX1(k, j, pmb->is-1, pmb->ie+1,
                        sl_z4c_alpha.array(), empty_array,
                        r_alpha_l_.array(), r_alpha_r_.array(),
                        enforce_floors);

    pmb->precon->WenoX1(k, j, pmb->is-1, pmb->ie+1,
                        sl_z4c_beta_u.array(), empty_array,
                        r_beta_u_l_.array(), r_beta_u_r_.array(),
                        enforce_floors);
  }
/*  else if (ivx == IVY)
  {
    pmb->precon->WenoX2(k, j-1, pmb->is-1, pmb->ie+1,
                        sl_z4c_alpha.array(), empty_array,
                        r_alpha_l_.array(), r_alpha_r_.array(),
                        enforce_floors);

    pmb->precon->WenoX2(k, j, pmb->is-1, pmb->ie+1,
                        sl_z4c_alpha.array(), empty_array,
                        r_alpha_lb_.array(), r_alpha_r_.array(),
                        enforce_floors);

    // beta
    pmb->precon->WenoX2(k, j-1, pmb->is-1, pmb->ie+1,
                        sl_z4c_beta_u.array(), empty_array,
                        r_beta_u_l_.array(), r_beta_u_r_.array(),
                        enforce_floors);

    pmb->precon->WenoX2(k, j, pmb->is-1, pmb->ie+1,
                        sl_z4c_beta_u.array(), empty_array,
                        r_beta_u_lb_.array(), r_beta_u_r_.array(),
                        enforce_floors);

  }
  else if (ivx == IVZ)
  {
    pmb->precon->WenoX3(k-1, j, pmb->is-1, pmb->ie+1,
                        sl_z4c_alpha.array(), empty_array,
                        r_alpha_l_.array(), r_alpha_r_.array(),
                        enforce_floors);

    pmb->precon->WenoX3(k, j, pmb->is-1, pmb->ie+1,
                        sl_z4c_alpha.array(), empty_array,
                        r_alpha_lb_.array(), r_alpha_r_.array(),
                        enforce_floors);

    // beta
    pmb->precon->WenoX3(k-1, j, pmb->is-1, pmb->ie+1,
                        sl_z4c_beta_u.array(), empty_array,
                        r_beta_u_l_.array(), r_beta_u_r_.array(),
                        enforce_floors);

    pmb->precon->WenoX3(k, j, pmb->is-1, pmb->ie+1,
                        sl_z4c_beta_u.array(), empty_array,
                        r_beta_u_lb_.array(), r_beta_u_r_.array(),
                        enforce_floors);

  }*/
  else
  {
    pco_gr->GetGeometricFieldFC(r_alpha_l_,  sl_z4c_alpha,    ivx-1, k, j);
    pco_gr->GetGeometricFieldFC(r_beta_u_l_, sl_z4c_beta_u,   ivx-1, k, j);

    pco_gr->GetGeometricFieldFC(r_alpha_r_,  sl_z4c_alpha,    ivx-1, k, j);
    pco_gr->GetGeometricFieldFC(r_beta_u_r_, sl_z4c_beta_u,   ivx-1, k, j);
  }

#endif // DBG_RECONSTRUCT_GEOM



  if (0)   // debug FC transfer
  {
    if ((ivx==1) && (k==7) && (j==6))
    {
      /*
      std::cout << setprecision(16);
      std::cout << "k,j:" << k << "," << j << std::endl;
      std::cout << "prim_l:" << prim_l(0,8) << std::endl;
      std::cout << "prim_r:" << prim_r(0,8) << std::endl;

      std::cout << "alpha_l:"  << r_alpha_l_(0,8) << std::endl;
      std::cout << "alpha_r:"  << r_alpha_r_(0,8) << std::endl;

      // prim_l.print_all("%.4e");

      // std::cout << "prim_r" << std::endl;
      // prim_r.print_all("%.4e");
      std::exit(0);
      */
    }
  }

  // Prepare determinant-like
  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    detgamma_(i)      = Det3Metric(gamma_dd_, i);

    sqrt_detgamma_(i) = std::sqrt(detgamma_(i));
    oo_detgamma_(i)   = 1. / detgamma_(i);
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
  for (int i=il; i<=iu; ++i)
  {
    const Real norm2_utilde_l = InnerProductSlicedVec3Metric(
      w_util_u_l_, gamma_dd_, i
    );

    const Real norm2_utilde_r = InnerProductSlicedVec3Metric(
      w_util_u_r_, gamma_dd_, i
    );

    W_l_(i) = std::sqrt(1. + norm2_utilde_l);
    W_r_(i) = std::sqrt(1. + norm2_utilde_r);
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

  InnerProductSlicedVec3Metric(w_norm2_v_l, w_v_u_l_, gamma_dd_, il, iu);
  InnerProductSlicedVec3Metric(w_norm2_v_r, w_v_u_r_, gamma_dd_, il, iu);

  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    // Calculate wavespeeds in left state NB EOS specific
#if USETM
    // If using the PrimitiveSolver framework, get the number density
    // and temperature to help calculate enthalpy.
    Real nl = w_rho_l_(i)/mb;
    Real nr = w_rho_r_(i)/mb;
    // FIXME: Generalize to work with EOSes accepting particle fractions.
    Real Y[MAX_SPECIES] = {0.0};
    Real Tl = pmy_block->peos->GetEOS().GetTemperatureFromP(nl, w_p_l_(i), Y);
    Real Tr = pmy_block->peos->GetEOS().GetTemperatureFromP(nr, w_p_r_(i), Y);
    w_hrho_l_(i) = nl*pmy_block->peos->GetEOS().GetEnthalpy(nl, Tl, Y);
    w_hrho_r_(i) = nr*pmy_block->peos->GetEOS().GetEnthalpy(nr, Tr, Y);

    // Calculate the sound speeds
    pmy_block->peos->SoundSpeedsGR(nl, Tl, w_v_u_l_(ivx-1,i), w_norm2_v_l(i),
                                   r_alpha_l_(i), r_beta_u_l_(ivx-1,i),
                                   gamma_uu_(ivx-1,ivx-1,i),
                                   &lambda_p_l(i), &lambda_m_l(i));
    pmy_block->peos->SoundSpeedsGR(nr, Tr, w_v_u_r_(ivx-1,i), w_norm2_v_r(i),
                                   r_alpha_r_(i), r_beta_u_r_(ivx-1,i),
                                   gamma_uu_(ivx-1,ivx-1,i),
                                   &lambda_p_r(i), &lambda_m_r(i));
#else
    w_hrho_l_(i) = w_rho_l_(i) + Eos_Gamma_ratio * w_p_l_(i);
    w_hrho_r_(i) = w_rho_r_(i) + Eos_Gamma_ratio * w_p_r_(i);

    pmy_block->peos->SoundSpeedsGR(w_hrho_l_(i), w_p_l_(i), w_v_u_l_(ivx-1,i),
                                   w_norm2_v_l(i),
                                   r_alpha_l_(i), r_beta_u_l_(ivx-1,i),
                                   gamma_uu_(ivx-1,ivx-1,i),
                                   &lambda_p_l(i), &lambda_m_l(i));
    pmy_block->peos->SoundSpeedsGR(w_hrho_r_(i), w_p_r_(i), w_v_u_r_(ivx-1,i),
                                   w_norm2_v_r(i),
                                   r_alpha_r_(i), r_beta_u_r_(ivx-1,i),
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
    lambda(i) = std::max(lambda_r, -lambda_l);
  }

  /*
  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    // const Real lambda_l = std::min(lambda_m_l(i), lambda_m_r(i));
    // const Real lambda_r = std::max(lambda_p_l(i), lambda_p_r(i));
    // lambda(i) = std::max(lambda_r, -lambda_l);

    const Real lambda_l = std::min(lambda_m_l(i), lambda_p_l(i));
    const Real lambda_r = std::max(lambda_m_r(i), lambda_p_r(i));
    lambda(i) = std::max(std::abs(lambda_l), std::abs(lambda_r));
  }
  */

  /*
  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    const Real lambda_l = r_alpha_l_(i) * w_v_u_l_(ivx-1,i) - beta_u_(ivx-1,i);
    const Real lambda_r = r_alpha_r_(i) * w_v_u_r_(ivx-1,i) - beta_u_(ivx-1,i);

    const Real ma_lam_m = std::max(
      std::abs(lambda_m_l(i)), std::abs(lambda_m_r(i))
    );

    const Real ma_lam_p = std::max(
      std::abs(lambda_p_l(i)), std::abs(lambda_p_r(i))
    );

    lambda(i) = std::max(std::max(ma_lam_m, ma_lam_p),
                         std::max(std::abs(lambda_l), std::abs(lambda_r)));
  }
  */


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
    flux_l_(IDN,i) = cons_l_(IDN,i) * r_alpha_l_(i) * (
      w_v_u_l_(ivx-1,i) - r_beta_u_l_(ivx-1,i)/r_alpha_l_(i)
    );

    // tau flux: r_alpha_l_(S^i - Dv^i) - beta^i tau
    flux_l_(IEN,i) = cons_l_(IEN,i) * r_alpha_l_(i) * (
      w_v_u_l_(ivx-1,i) - r_beta_u_l_(ivx-1,i)/r_alpha_l_(i)
    ) + r_alpha_l_(i)*sqrt_detgamma_(i)*w_p_l_(i)*w_v_u_l_(ivx-1,i);
  }

  for (int a=0; a<NDIM; ++a)
  {
    #pragma omp simd
    for (int i = il; i <= iu; ++i)
    {
      //S_i flux alpha S^j_i - beta^j S_i
      flux_l_(IVX+a,i) = (cons_l_(IVX+a,i) * r_alpha_l_(i) *
                          (w_v_u_l_(ivx-1,i) -
                           r_beta_u_l_(ivx-1,i)/r_alpha_l_(i)));

    }
  }

  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    flux_l_(ivx,i) += w_p_l_(i) * r_alpha_l_(i) * sqrt_detgamma_(i);
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
    flux_r_(IDN,i) = cons_r_(IDN,i) * r_alpha_r_(i) * (
      w_v_u_r_(ivx-1,i) - r_beta_u_r_(ivx-1,i)/r_alpha_r_(i)
    );

    // tau flux: r_alpha_r_(S^i - Dv^i) - beta^i tau
    flux_r_(IEN,i) = cons_r_(IEN,i) * r_alpha_r_(i) * (
      w_v_u_r_(ivx-1,i) - r_beta_u_r_(ivx-1,i)/r_alpha_r_(i)
    ) + r_alpha_r_(i)*sqrt_detgamma_(i)*w_p_r_(i)*w_v_u_r_(ivx-1,i);
  }

  for (int a=0; a<NDIM; ++a)
  {
    #pragma omp simd
    for (int i = il; i <= iu; ++i)
    {
      //S_i flux alpha S^j_i - beta^j S_i
      flux_r_(IVX+a,i) = (cons_r_(IVX+a,i) * r_alpha_r_(i) *
                          (w_v_u_r_(ivx-1,i) -
                           r_beta_u_r_(ivx-1,i)/r_alpha_r_(i)));

    }
  }

  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    flux_r_(ivx,i) += w_p_r_(i) * r_alpha_r_(i) * sqrt_detgamma_(i);
  }

  // Set fluxes
  for (int n=0; n<NHYDRO; ++n)
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      flux(n,k,j,i) = 0.5 * (
        flux_l_(n,i) + flux_r_(n,i) - lambda(i) * (cons_r_(n,i) - cons_l_(n,i))
      );
    }
  }

  if (0)
  {
    for (int i = il; i <= iu; ++i)
    {
      if ((cons_l_(IDN,i) < 0) || (cons_r_(IDN,i) < 0))
      {
        std::cout << "ERR_0:" << "k,j,i" << k << "," << j << "," << i << " vals:" << cons_l_(IDN,i) << ", " << cons_r_(IDN,i) << std::endl;
      }

      if ((w_rho_l_(i) < 1e-20) || (w_rho_r_(i) < 1e-20))
      {
        std::cout << "ERR_1:" << "k,j,i" << k << "," << j << "," << i << " vals:" << w_rho_l_(i) << "," << w_rho_r_(i) << std::endl;
      }

      if (detgamma_(i) < 0)
      {
        std::cout << "ERR_2" << "k,j,i" << k << "," << j << "," << i << std::endl;
      }

    }
  }

  return;
}
