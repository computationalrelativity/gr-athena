// C++ headers
#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

// Athena++ headers
#include "../../hydro.hpp"
#include "../../../z4c/z4c.hpp"
#include "../../../utils/linear_algebra.hpp"
#include "../../../utils/interp_intergrid.hpp"
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
  const int k, const int j,
  const int il, const int iu,
  const int ivx,
  const AthenaArray<Real> &B,
  AthenaArray<Real> &prim_l,
  AthenaArray<Real> &prim_r,
  AthenaArray<Real> &flux,
  AthenaArray<Real> &ey,
  AthenaArray<Real> &ez,
  AthenaArray<Real> &wct,
  const AthenaArray<Real> &dxw)
{
  MeshBlock * pmb = pmy_block;

  // perform variable resampling when required
  Z4c * pz4c = pmb->pz4c;

  // Slice 3d z4c metric quantities  (NDIM=3 in z4c.hpp) ----------------------
  AT_N_sym sl_adm_gamma_dd(pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sca sl_adm_alpha(   pz4c->storage.adm, Z4c::I_ADM_alpha);
  AT_N_vec sl_adm_beta_u(  pz4c->storage.adm, Z4c::I_ADM_betax);

  // Reconstruction to FC -----------------------------------------------------
  GRDynamical* pco_gr = static_cast<GRDynamical*>(pmb->pcoord);
  pco_gr->GetGeometricFieldFC(gamma_dd_, sl_adm_gamma_dd, ivx-1, k, j);
  pco_gr->GetGeometricFieldFC(alpha_,    sl_adm_alpha,    ivx-1, k, j);
  pco_gr->GetGeometricFieldFC(beta_u_,   sl_adm_beta_u,   ivx-1, k, j);

#ifdef DBG_COMBINED_HYDPA
  AA & pscalars_l = pmy_block->pscalars->rl_;
  AA & pscalars_r = pmy_block->pscalars->rr_;

#else
  AA pscalars_l;
  AA pscalars_r;
#endif

  RiemannSolver(k, j, il, iu, ivx, B, prim_l, prim_r,
                pscalars_l, pscalars_r,
                alpha_, beta_u_, gamma_dd_,
                flux, ey, ez, wct, dxw);
}

void Hydro::RiemannSolver(
  const int k, const int j,
  const int il, const int iu,
  const int ivx,
  const AthenaArray<Real> &B,
  AthenaArray<Real> &prim_l,
  AthenaArray<Real> &prim_r,
  AthenaArray<Real> &pscalars_l,
  AthenaArray<Real> &pscalars_r,
  AT_N_sca & alpha_,
  AT_N_vec & beta_u_,
  AT_N_sym & gamma_dd_,
  AthenaArray<Real> &flux,
  AthenaArray<Real> &ey,
  AthenaArray<Real> &ez,
  AthenaArray<Real> &wct,
  const AthenaArray<Real> &dxw)
{
  using namespace LinearAlgebra;

  MeshBlock * pmb = pmy_block;
  EquationOfState * peos = pmb->peos;
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
  AT_N_sca w_rho_l_(prim_l, IDN);
  AT_N_sca w_rho_r_(prim_r, IDN);
  AT_N_sca w_p_l_(  prim_l, IPR);
  AT_N_sca w_p_r_(  prim_r, IPR);

  AT_N_vec w_util_u_l_(prim_l, IVX);
  AT_N_vec w_util_u_r_(prim_r, IVX);

  // =============================================================
  // Prepare determinant-like
  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    detgamma_(i)      = Det3Metric(gamma_dd_, i);
    sqrt_detgamma_(i) = std::sqrt(detgamma_(i));

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
  for (int i=il; i<=iu; ++i)
  {
    const Real norm2_utilde_l = InnerProductSlicedVec3Metric(
      w_util_d_l_, gamma_uu_, i
    );

    const Real norm2_utilde_r = InnerProductSlicedVec3Metric(
      w_util_d_r_, gamma_uu_, i
    );

    W_l_(i) = std::sqrt(1. + norm2_utilde_l);
    W_r_(i) = std::sqrt(1. + norm2_utilde_r);

    oo_W_l_(i) = OO(W_l_(i));
    oo_W_r_(i) = OO(W_r_(i));
  }

  // Eulerian vel.
  for (int a=0; a<NDIM; ++a)
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      w_v_u_l_(a,i) = w_util_u_l_(a,i) / W_l_(i);
      w_v_u_r_(a,i) = w_util_u_r_(a,i) / W_r_(i);

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


  // deal with excision -------------------------------------------------------
  auto excise = [&](const int i)
  {
    // set flat space if the interpolated det is negative and inside
    // horizon (either in ahf or lapse below excision value)

    // BD: TODO - This dump to output is stupid and use be addressed otherwise
    std::cout << "Set flat space" << "\n";
    for (int a=0; a<3; ++a)
    for (int b=a; b<3; ++b)
    {
      gamma_dd_(a,b,i) = (a==b);
      gamma_uu_(a,b,i) = (a==b);
    }
    detgamma_(i) = 1.0;
  };

  if (opt_excision.horizon_based)
  {
    #pragma omp simd
    for (int i = il; i <= iu; ++i)
    {
      // if ahf enabled set flat space if in horizon
      // TODO: read in centre of each horizon and shift origin - not needed for
      // now if collapse is at origin
      Real horizon_radius;
      for (auto pah_f : pmy_block->pmy_mesh->pah_finder)
      {
        if (not pah_f->ah_found)
          continue;
        horizon_radius = pah_f->rr_min;
        horizon_radius *= pmb->phydro->opt_excision.horizon_factor;
        Real R2;
        switch (ivx)
        {
          case IVX:
          {
            R2 = (
              SQR(pco_gr->x1f(i)) + SQR(pco_gr->x2v(j)) + SQR(pco_gr->x3v(k))
            );
            break;
          }
          case IVY:
          {
            R2 = (
              SQR(pco_gr->x1v(i)) + SQR(pco_gr->x2f(j)) + SQR(pco_gr->x3v(k))
            );
            break;
          }
          case IVZ:
          {
            R2 = (
              SQR(pco_gr->x1v(i)) + SQR(pco_gr->x2v(j)) + SQR(pco_gr->x3f(k))
            );
            break;
          }
          default:
          {
            assert(false);
          }
        }

        if ((R2 < SQR(horizon_radius)) ||
            (alpha_(i) < opt_excision.alpha_threshold))
        {
          excise(i);
        }
      }
    }
  }
  else if (opt_excision.alpha_threshold > 0)  // by default disabled (i.e. 0)
  {
    #pragma omp simd
    for (int i = il; i <= iu; ++i)
    {
      if (alpha_(i) < opt_excision.alpha_threshold)
      {
        excise(i);
      }
    }
  }
  // --------------------------------------------------------------------------


  // assemble magnetic field components ---------------------------------------
  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    switch (ivx)
    {
      case IVX:
      {
        q_scB_u_l_(0,i) = oo_sqrt_detgamma_(i) * B(k,j,i);
        q_scB_u_l_(1,i) = oo_sqrt_detgamma_(i) * prim_l(IBY,i);
        q_scB_u_l_(2,i) = oo_sqrt_detgamma_(i) * prim_l(IBZ,i);
        q_scB_u_r_(0,i) = oo_sqrt_detgamma_(i) * B(k,j,i);
        q_scB_u_r_(1,i) = oo_sqrt_detgamma_(i) * prim_r(IBY,i);
        q_scB_u_r_(2,i) = oo_sqrt_detgamma_(i) * prim_r(IBZ,i);
        break;
      }
      case IVY:
      {
        q_scB_u_l_(1,i) = oo_sqrt_detgamma_(i) * B(k,j,i);
        q_scB_u_l_(2,i) = oo_sqrt_detgamma_(i) * prim_l(IBY,i);
        q_scB_u_l_(0,i) = oo_sqrt_detgamma_(i) * prim_l(IBZ,i);
        q_scB_u_r_(1,i) = oo_sqrt_detgamma_(i) * B(k,j,i);
        q_scB_u_r_(2,i) = oo_sqrt_detgamma_(i) * prim_r(IBY,i);
        q_scB_u_r_(0,i) = oo_sqrt_detgamma_(i) * prim_r(IBZ,i);
        break;
      }
      case IVZ:
      {
        q_scB_u_l_(2,i) = oo_sqrt_detgamma_(i) * B(k,j,i);
        q_scB_u_l_(0,i) = oo_sqrt_detgamma_(i) * prim_l(IBY,i);
        q_scB_u_l_(1,i) = oo_sqrt_detgamma_(i) * prim_l(IBZ,i);
        q_scB_u_r_(2,i) = oo_sqrt_detgamma_(i) * B(k,j,i);
        q_scB_u_r_(0,i) = oo_sqrt_detgamma_(i) * prim_r(IBY,i);
        q_scB_u_r_(1,i) = oo_sqrt_detgamma_(i) * prim_r(IBZ,i);
        break;
      }
      default:
      {
        assert(false);
      }
    }
  }


  for (int a = 0; a < NDIM; ++a)
  {
    #pragma omp simd
    for (int i = il; i <= iu; ++i)
    {
      const Real oo_alpha = OO(alpha_(i));
      b0_l_(i) = oo_alpha * W_l_(i) * q_scB_u_l_(a, i) * w_v_d_l_(a, i);
      b0_r_(i) = oo_alpha * W_r_(i) * q_scB_u_r_(a, i) * w_v_d_r_(a, i);
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
    Real nl = w_rho_l_(i) / mb;
    Real nr = w_rho_r_(i) / mb;
    // FIXME: Generalize to work with EOSes accepting particle fractions.
    Real Yl[MAX_SPECIES] = { 0.0 };  // Should we worry about r vs l here?
    Real Yr[MAX_SPECIES] = { 0.0 };

    // BD: TODO - handle this better in the non-combined case
#ifdef DBG_COMBINED_HYDPA
    for (int n = 0; n < NSCALARS; n++)
    {
      Yl[n] = pscalars_l(n, i);
      Yr[n] = pscalars_r(n, i);
    }
#else
    for (int n = 0; n < NSCALARS; n++)
    {
      Yr[n] = pmy_block->pscalars->r(n, k, j, i);
    }
    switch (ivx)
    {
      case IVX:
      {
        for (int n = 0; n < NSCALARS; n++)
        {
          Yl[n] = pmy_block->pscalars->r(n, k, j, i - 1);
        }
        break;
      }
      case IVY:
      {
        for (int n = 0; n < NSCALARS; n++)
        {
          Yl[n] = pmy_block->pscalars->r(n, k, j - 1, i);
        }
        break;
      }
      case IVZ:
      {
        for (int n = 0; n < NSCALARS; n++)
        {
          Yl[n] = pmy_block->pscalars->r(n, k - 1, j, i);
        }
        break;
      }
    }
  #endif // DBG_COMBINED_HYDPA

    Real Tl, Tr;
    Real hl, hr;

    if (pmy_block->precon->xorder_use_aux_T)
    {
      Tl = pmy_block->phydro->al_(IX_T,i);
      Tr = pmy_block->phydro->ar_(IX_T,i);
    }
    else
    {
      Tl = pmy_block->peos->GetEOS().GetTemperatureFromP(nl, w_p_l_(i), Yl);
      Tr = pmy_block->peos->GetEOS().GetTemperatureFromP(nr, w_p_r_(i), Yr);
    }

    if (pmy_block->precon->xorder_use_aux_h)
    {
      hl = pmy_block->phydro->al_(IX_ETH,i);
      hr = pmy_block->phydro->ar_(IX_ETH,i);
    }
    else
    {
      hl = pmy_block->peos->GetEOS().GetEnthalpy(nl, Tl, Yl);
      hr = pmy_block->peos->GetEOS().GetEnthalpy(nr, Tr, Yr);
    }

    w_hrho_l_(i) = w_rho_l_(i) * hl;
    w_hrho_r_(i) = w_rho_r_(i) * hr;

    // Calculate the wave speeds
    pmy_block->peos->FastMagnetosonicSpeedsGR(
      nl, Tl, b2_l_(i), w_v_u_l_(ivx - 1, i), w_norm2_v_l_(i), alpha_(i),
      beta_u_(ivx - 1, i), gamma_uu_(ivx - 1, ivx - 1, i), &lambda_p_l(i),
      &lambda_m_l(i), Yl);
    pmy_block->peos->FastMagnetosonicSpeedsGR(
      nr, Tr, b2_r_(i), w_v_u_r_(ivx - 1, i), w_norm2_v_r_(i), alpha_(i),
      beta_u_(ivx - 1, i), gamma_uu_(ivx - 1, ivx - 1, i), &lambda_p_r(i),
      &lambda_m_r(i), Yr);
#else
    w_hrho_l_(i) = w_rho_l_(i) + Eos_Gamma_ratio * w_p_l_(i);
    w_hrho_r_(i) = w_rho_r_(i) + Eos_Gamma_ratio * w_p_r_(i);

    pmy_block->peos->FastMagnetosonicSpeedsGR(
      w_hrho_l_(i), w_p_l_(i), b2_l_(i), w_v_u_l_(ivx - 1, i), w_norm2_v_l_(i),
      alpha_(i), beta_u_(ivx - 1, i), gamma_uu_(ivx - 1, ivx - 1, i),
      &lambda_p_l(i), &lambda_m_l(i));
    pmy_block->peos->FastMagnetosonicSpeedsGR(
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
    lambda(i) = std::max(lambda_r, -lambda_l);
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


  for (int n = 0; n < NHYDRO; ++n)
  {
    #pragma omp simd
    for (int i = il; i <= iu; ++i)
    {
      flux(n, k, j, i) = 0.5 * (flux_l_(n, i) + flux_r_(n, i) -
                                lambda(i) * (cons_r_(n, i) - cons_l_(n, i)));
    }
  }

  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    ey(k, j, i) =
      -0.5 * (flux_l_(IBY, i) + flux_r_(IBY, i) -
              lambda(i) * (cons_r_(IBY, i) - cons_l_(IBY, i)));
    ez(k, j, i) =
      0.5 * (flux_l_(IBZ, i) + flux_r_(IBZ, i) -
              lambda(i) * (cons_r_(IBZ, i) - cons_l_(IBZ, i)));

    wct(k, j, i) = GetWeightForCT(flux(IDN, k, j, i), prim_l(IDN, i),
                                  prim_r(IDN, i), dxw(i), dt);
  }


}
