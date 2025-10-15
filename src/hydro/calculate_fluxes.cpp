//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file calculate_fluxes.cpp
//  \brief Calculate hydro/MHD fluxes

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../field/field_diffusion/field_diffusion.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "../scalars/scalars.hpp"
#include "hydro.hpp"
#include "hydro_diffusion/hydro_diffusion.hpp"

#include "../utils/floating_point.hpp"
#include "../utils/linear_algebra.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

// utilities ------------------------------------------------------------------
namespace {

// provide geometric quantities on FC grid
void InterpolateGeometry(
  MeshBlock * pmb,
  AT_N_sca & alpha_,
  AT_N_sca & oo_alpha_,
  AT_N_vec & beta_u_,
  AT_N_sym & gamma_dd_,
  AT_N_sym & gamma_uu_,
  AT_N_sca & chi_,
  AT_N_sca & oo_detgamma_,
  AT_N_sca & detgamma_,
  AT_N_sca & sqrt_detgamma_,
  const int ivx,
  const int k, const int j,
  const int il, const int iu
)
{
  using namespace LinearAlgebra;
  using namespace FloatingPoint;

  GRDynamical* pco_gr = static_cast<GRDynamical*>(pmb->pcoord);

  // perform variable resampling when required
  Z4c * pz4c = pmb->pz4c;

  // Slice 3d z4c metric quantities  (NDIM=3 in z4c.hpp) ----------------------
  AT_N_sym sl_adm_gamma_dd(pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sca sl_adm_alpha(   pz4c->storage.adm, Z4c::I_ADM_alpha);
  AT_N_vec sl_adm_beta_u(  pz4c->storage.adm, Z4c::I_ADM_betax);

  // AT_N_sca sl_adm_detgamma(pz4c->storage.aux_extended,
  //                          Z4c::I_AUX_EXTENDED_gs_sqrt_detgamma);

  AT_N_sca sl_z4c_chi(pz4c->storage.u, Z4c::I_Z4c_chi);

  // Reconstruction to FC -----------------------------------------------------
  pco_gr->GetGeometricFieldFC(gamma_dd_, sl_adm_gamma_dd, ivx-1, k, j);
  pco_gr->GetGeometricFieldFC(alpha_,    sl_adm_alpha,    ivx-1, k, j);
  pco_gr->GetGeometricFieldFC(beta_u_,   sl_adm_beta_u,   ivx-1, k, j);

  // interpolated conformal factor
  pco_gr->GetGeometricFieldFC(chi_, sl_z4c_chi, ivx-1, k, j);

  // chi -> det_gamma [ADM] power
  const Real chi_pow = 12.0 / pz4c->opt.chi_psi_power;

  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    const Real chi = std::abs(chi_(i));
    const Real chi_guarded = std::max(chi, pz4c->opt.chi_div_floor);
    sqrt_detgamma_(i) = std::pow(chi_guarded, chi_pow / 2.0);
  }

  // Metric derived quantities ------------------------------------------------
  // regularization factor
  const Real eps_alpha__ = pmb->pz4c->opt.eps_floor;

  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    detgamma_(i) = SQR(sqrt_detgamma_(i));
    oo_detgamma_(i) = 1. / detgamma_(i);
  }

  Inv3Metric(oo_detgamma_, gamma_dd_, gamma_uu_, il, iu);

  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    // regularize for reciprocal factor
    Real alpha__ = regularize_near_zero(alpha_(i), eps_alpha__);
    oo_alpha_(i) = OO(alpha__);
  }

}

void ReconstructFields(
  MeshBlock * pmb,
  Reconstruction::ReconstructionVariant rv_w,
  Reconstruction::ReconstructionVariant rv_r,
  Reconstruction::ReconstructionVariant rv_a,
  Reconstruction::ReconstructionVariant rv_b,
  AA & wl_, AA & wr_, // rec. primitive hydro
  AA & rl_, AA & rr_, // rec. passive scalars
  AA & al_, AA & ar_, // rec. auxiliary quantities
  AA & w,             // CC: primitive hydro
  AA & r,             // CC: passive scalars
  AA & bcc,           // CC: magnetic fields
  AA & aux,           // CC: auxiliary quantities
  const int ivx,
  const int k, const int j,
  const int il, const int iu
)
{
  Reconstruction * pr = pmb->precon;
  Hydro * ph = pmb->phydro;
  PassiveScalars * ps = pmb->pscalars;
  EquationOfState *peos = pmb->peos;

  const int os_il = (ivx == 1) ? 1 : 0; // l_ populated at i+1 on Recon. call

  // hydro primitives -------------------------------------
  for (int n=0; n<NHYDRO; ++n)
  {
    pr->ReconstructFieldXd(
      rv_w, w, wl_, wr_, ivx, n, n, k, j, il-os_il, iu
    );
  }

  // magnetic fields --------------------------------------
  if (MAGNETIC_FIELDS_ENABLED)
  {
    int ISA, ISB;

    switch (ivx)
    {
      case 1:
      {
        ISA = IB2;
        ISB = IB3;
        break;
      }
      case 2:
      {
        ISA = IB3;
        ISB = IB1;
        break;
      }
      case 3:
      {
        ISA = IB1;
        ISB = IB2;
        break;
      }
    }

    pr->ReconstructFieldXd(
      rv_b, bcc, wl_, wr_, ivx, IBY, ISA, k, j, il-os_il, iu
    );
    pr->ReconstructFieldXd(
      rv_b, bcc, wl_, wr_, ivx, IBZ, ISB, k, j, il-os_il, iu
    );
  }

  // passive scalars --------------------------------------
  for (int n=0; n<NSCALARS; ++n)
  {
    pr->ReconstructFieldXd(
      rv_r, r, rl_, rr_, ivx, n, n, k, j, il-os_il, iu
    );
  }

  // auxiliary quantities ---------------------------------
  if (pr->xorder_use_auxiliaries)
  {
    for (int n=0; n<NDRV_HYDRO; ++n)
    {
      if (((n == IX_T)  && pr->xorder_use_aux_T) ||
           (n == IX_ETH && pr->xorder_use_aux_h) ||
           (n == IX_LOR && pr->xorder_use_aux_W) ||
           (n == IX_CS2 && pr->xorder_use_aux_cs2))
      {
        pr->ReconstructFieldXd(
          rv_a, aux, al_, ar_, ivx, n, n, k, j, il-os_il, iu
        );
      }
    }
  }


  // impose whatever limits are required --------------------------------------
  // Only supporting PrimitiveSolver, proceed as follows:
  //
  // - Impose density limits
  // - Limit species first if they exist
  // - If we reconstructed T:
  //   - Limit
  //   - Recompute from P
  // - Apply primitive floors
  // - Limit W, h, cs2 if they are also reconstructed

#if !USETM
  // only support operation with PrimitiveSolver
  assert(false);
#endif


  Real mb = peos->GetEOS().GetBaryonMass();

  const Real min_ETH = peos->GetEOS().GetMinimumEnthalpy();

  Real Yl__[MAX_SPECIES] = {0.0};
  Real Yr__[MAX_SPECIES] = {0.0};

  Real Wvul__[NDIM] = {0.0};
  Real Wvur__[NDIM] = {0.0};

  Real nl__, nr__;

  for (int i=il-os_il; i<=iu; ++i)
  {
    nl__ = wl_(IDN,i) / mb;
    nr__ = wr_(IDN,i) / mb;

    peos->GetEOS().ApplyDensityLimits(nl__);
    peos->GetEOS().ApplyDensityLimits(nr__);

    for (int n=0; n<NDIM; ++n)
    {
      Wvul__[n] = wl_(IVX+n,i);
      Wvur__[n] = wr_(IVX+n,i);
    }

    for (int n=0; n<NSCALARS; ++n)
    {
      Yl__[n] = rl_(n,i);
      Yr__[n] = rr_(n,i);
    }

    if (NSCALARS > 0)
    {
      const bool ll__ = peos->GetEOS().ApplySpeciesLimits(Yl__);
      const bool lr__ = peos->GetEOS().ApplySpeciesLimits(Yr__);
    }

    if (!pr->xorder_use_aux_T || peos->recompute_temperature)
    {
      al_(IX_T,i) = peos->GetEOS().GetTemperatureFromP(nl__, wl_(IPR,i), Yl__);
      ar_(IX_T,i) = peos->GetEOS().GetTemperatureFromP(nr__, wr_(IPR,i), Yr__);
    }

    // now depending on settings unpack limited / floored
    if (pr->xorder_floor_primitives)
    {
      wl_(IDN,i) = mb * nl__;
      wr_(IDN,i) = mb * nr__;

      for (int n=0; n<NDIM; ++n)
      {
        wl_(IVX+n,i) = Wvul__[n];
        wr_(IVX+n,i) = Wvur__[n];
      }

      peos->GetEOS().ApplyTemperatureLimits(al_(IX_T,i));
      peos->GetEOS().ApplyTemperatureLimits(ar_(IX_T,i));

      const bool fll__ = peos->GetEOS().ApplyPrimitiveFloor(
        nl__, Wvul__, wl_(IPR,i), al_(IX_T,i), Yl__
      );
      const bool flr__ = peos->GetEOS().ApplyPrimitiveFloor(
        nr__, Wvur__, wr_(IPR,i), ar_(IX_T,i), Yr__
      );
    }

    if (!pr->xorder_use_aux_h)
    {
      al_(IX_ETH,i) = peos->GetEOS().GetEnthalpy(nl__, al_(IX_T,i), Yl__);
      ar_(IX_ETH,i) = peos->GetEOS().GetEnthalpy(nr__, ar_(IX_T,i), Yr__);
    }

    if (pr->xorder_limit_species)
    {
      for (int n=0; n<NSCALARS; ++n)
      {
        rl_(n,i) = Yl__[n];
        rr_(n,i) = Yr__[n];
      }
    }

    if (pr->xorder_floor_primitives)
    {
      al_(IX_ETH,i) = std::max(al_(IX_ETH,i), min_ETH);
      ar_(IX_ETH,i) = std::max(ar_(IX_ETH,i), min_ETH);
    }

    if (pr->xorder_use_aux_W)
    {
      al_(IX_LOR,i) = std::max(al_(IX_LOR,i), 1.0);
      ar_(IX_LOR,i) = std::max(ar_(IX_LOR,i), 1.0);
    }

    if (pr->xorder_use_aux_cs2)
    {
      const Real max_cs2 = peos->max_cs2;
      al_(IX_CS2,i) = std::max(0.0, std::min(al_(IX_CS2,i), max_cs2));
      ar_(IX_CS2,i) = std::max(0.0, std::min(ar_(IX_CS2,i), max_cs2));
    }
  }
}

void ReconstructSwap(
  MeshBlock * pmb,
  AA & wl_, AA & wlb_,
  AA & rl_, AA & rlb_,
  AA & al_, AA & alb_
)
{
  Reconstruction * pr = pmb->precon;
  wl_.SwapAthenaArray(wlb_);

  if (NSCALARS > 0)
    rl_.SwapAthenaArray(rlb_);

  al_.SwapAthenaArray(alb_);
}

}

// ----------------------------------------------------------------------------

void Hydro::CalculateFluxes(AA &w,
                            AA &r,
                            FaceField &b,
                            AA &bcc,
                            AA(& hflux)[3],
                            AA(& sflux)[3],
                            Reconstruction::ReconstructionVariant rv,
                            const int num_enlarge_layer)
{
  if (flux_reconstruction)
  {
    // Unsupported
    assert(false);
    // CalculateFluxes_FluxReconstruction(w, b, bcc, order);
    return;
  }

  CalculateFluxesCombined(w,r,b,bcc,hflux,sflux,rv,num_enlarge_layer);
  return;
}

void Hydro::CalculateFluxesCombined(AA &w,
                                    AA &r,
                                    FaceField &b,
                                    AA &bcc,
                                    AA(& hflux)[3],
                                    AA(& sflux)[3],
                                    Reconstruction::ReconstructionVariant rv,
                                    const int num_enlarge_layer)
{
  MeshBlock *pmb = pmy_block;

  Reconstruction *pr = pmb->precon;
  PassiveScalars *ps = pmb->pscalars;

  Reconstruction::ReconstructionVariant rv_w = rv;
  Reconstruction::ReconstructionVariant rv_r = rv;
  Reconstruction::ReconstructionVariant rv_a = rv;
  Reconstruction::ReconstructionVariant rv_b = rv;

  // For passive-scalar reconstruction
  AA mass_flux;

  int il, iu, jl, ju, kl, ku;

#if MAGNETIC_FIELDS_ENABLED
  // used only to pass to (up-to) 2x RiemannSolver() calls per dimension:
  // x1:
  AA &b1 = b.x1f, &w_x1f = pmb->pfield->wght.x1f,
                  &e3x1 = pmb->pfield->e3_x1f, &e2x1 = pmb->pfield->e2_x1f;
  // x2:
  AA &b2 = b.x2f, &w_x2f = pmb->pfield->wght.x2f,
                  &e1x2 = pmb->pfield->e1_x2f, &e3x2 = pmb->pfield->e3_x2f;
  // x3:
  AA &b3 = b.x3f, &w_x3f = pmb->pfield->wght.x3f,
                  &e1x3 = pmb->pfield->e1_x3f, &e2x3 = pmb->pfield->e2_x3f;
#endif

  const Real lambda_rescaling = 1.0;

  //---------------------------------------------------------------------------
  // i-direction
  AA &x1flux = hflux[X1DIR];
  AA s_x1flux;

  if (NSCALARS > 0)
  {
    mass_flux.InitWithShallowSlice(hflux[X1DIR], 4, IDN, 1);
    s_x1flux.InitWithShallowSlice(sflux[X1DIR], 4, 0, NSCALARS);
  }

  pr->SetIndicialLimitsCalculateFluxes(IVX, il, iu, jl, ju, kl, ku,
                                       num_enlarge_layer);

  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  {
    ReconstructFields(
      pmb,
      rv_w, rv_r, rv_a, rv_b,
      wl_, wr_,
      rl_, rr_,
      al_, ar_,
      w, r, bcc, derived_ms,
      IVX,
      k, j, il, iu
    );

    InterpolateGeometry(
      pmb,
      alpha_,
      oo_alpha_,
      beta_u_,
      gamma_dd_,
      gamma_uu_,
      chi_,
      oo_detgamma_,
      detgamma_,
      sqrt_detgamma_,
      IVX,
      k, j,
      il, iu
    );

    pmb->pcoord->CenterWidth1(k, j, il, iu, dxw_);

#if !MAGNETIC_FIELDS_ENABLED

    RiemannSolver(
      IVX, k, j, il, iu,
      wl_, wr_,
      rl_, rr_,
      al_, ar_,
      alpha_,
      oo_alpha_,
      beta_u_,
      gamma_dd_,
      detgamma_,
      oo_detgamma_,
      sqrt_detgamma_,
      x1flux, s_x1flux,
      dxw_, lambda_rescaling
    );

#else
    // x1flux(IBY) = (v1*b2 - v2*b1) = -EMFZ
    // x1flux(IBZ) = (v1*b3 - v3*b1) =  EMFY

    RiemannSolver(
      IVX, k, j, il, iu,
      b1,
      wl_, wr_,
      rl_, rr_,
      al_, ar_,
      alpha_,
      oo_alpha_,
      beta_u_,
      gamma_dd_,
      detgamma_,
      oo_detgamma_,
      sqrt_detgamma_,
      x1flux, s_x1flux,
      e3x1, e2x1, w_x1f,
      dxw_, lambda_rescaling
    );

#endif

  }
  //---------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  // j-direction
  if (pmb->pmy_mesh->f2)
  {
    AA &x2flux = hflux[X2DIR];
    AA s_x2flux;

    if (NSCALARS > 0)
    {
      mass_flux.InitWithShallowSlice(hflux[X2DIR], 4, IDN, 1);
      s_x2flux.InitWithShallowSlice(sflux[X2DIR], 4, 0, NSCALARS);
    }

    pr->SetIndicialLimitsCalculateFluxes(IVY, il, iu, jl, ju, kl, ku,
                                         num_enlarge_layer);

    for (int k=kl; k<=ku; ++k)
    {
      ReconstructFields(
        pmb,
        rv_w, rv_r, rv_a, rv_b,
        wl_, wr_,
        rl_, rr_,
        al_, ar_,
        w, r, bcc, derived_ms,
        IVY,
        k, jl-1, il, iu
      );

      for (int j=jl; j<=ju; ++j)
      {

        ReconstructFields(
          pmb,
          rv_w, rv_r, rv_a, rv_b,
          wlb_, wr_,
          rlb_, rr_,
          alb_, ar_,
          w, r, bcc, derived_ms,
          IVY,
          k, j, il, iu
        );

        InterpolateGeometry(
          pmb,
          alpha_,
          oo_alpha_,
          beta_u_,
          gamma_dd_,
          gamma_uu_,
          chi_,
          oo_detgamma_,
          detgamma_,
          sqrt_detgamma_,
          IVY,
          k, j,
          il, iu
        );

        pmb->pcoord->CenterWidth2(k, j, il, iu, dxw_);
#if !MAGNETIC_FIELDS_ENABLED

        RiemannSolver(
          IVY, k, j, il, iu,
          wl_, wr_,
          rl_, rr_,
          al_, ar_,
          alpha_,
          oo_alpha_,
          beta_u_,
          gamma_dd_,
          detgamma_,
          oo_detgamma_,
          sqrt_detgamma_,
          x2flux, s_x2flux,
          dxw_, lambda_rescaling
        );

#else
        // flx(IBY) = (v2*b3 - v3*b2) = -EMFX
        // flx(IBZ) = (v2*b1 - v1*b2) =  EMFZ

        RiemannSolver(
          IVY, k, j, il, iu,
          b2,
          wl_, wr_,
          rl_, rr_,
          al_, ar_,
          alpha_,
          oo_alpha_,
          beta_u_,
          gamma_dd_,
          detgamma_,
          oo_detgamma_,
          sqrt_detgamma_,
          x2flux, s_x2flux,
          e1x2, e3x2, w_x2f,
          dxw_, lambda_rescaling
        );
#endif

        // swap the arrays for the next step (l<->lb)
        ReconstructSwap(pmb, wl_, wlb_, rl_, rlb_, al_, alb_);
      }
    }
  }

  //---------------------------------------------------------------------------
  // k-direction
  if (pmb->pmy_mesh->f3)
  {
    AA &x3flux = hflux[X3DIR];
    AA s_x3flux;

    if (NSCALARS > 0)
    {
      mass_flux.InitWithShallowSlice(hflux[X3DIR], 4, IDN, 1);
      s_x3flux.InitWithShallowSlice(sflux[X3DIR], 4, 0, NSCALARS);
    }

    pr->SetIndicialLimitsCalculateFluxes(IVZ, il, iu, jl, ju, kl, ku,
                                         num_enlarge_layer);

    for (int j=jl; j<=ju; ++j)
    { // this loop ordering is intentional

      ReconstructFields(
        pmb,
        rv_w, rv_r, rv_a, rv_b,
        wl_, wr_,
        rl_, rr_,
        al_, ar_,
        w, r, bcc, derived_ms,
        IVZ,
        kl-1, j, il, iu
      );

      for (int k=kl; k<=ku; ++k)
      {
        ReconstructFields(
          pmb,
          rv_w, rv_r, rv_a, rv_b,
          wlb_, wr_,
          rlb_, rr_,
          alb_, ar_,
          w, r, bcc, derived_ms,
          IVZ,
          k, j, il, iu
        );

        InterpolateGeometry(
          pmb,
          alpha_,
          oo_alpha_,
          beta_u_,
          gamma_dd_,
          gamma_uu_,
          chi_,
          oo_detgamma_,
          detgamma_,
          sqrt_detgamma_,
          IVZ,
          k, j,
          il, iu
        );

        pmb->pcoord->CenterWidth3(k, j, il, iu, dxw_);

#if !MAGNETIC_FIELDS_ENABLED  // Hydro:

        RiemannSolver(
          IVZ, k, j, il, iu,
          wl_, wr_,
          rl_, rr_,
          al_, ar_,
          alpha_,
          oo_alpha_,
          beta_u_,
          gamma_dd_,
          detgamma_,
          oo_detgamma_,
          sqrt_detgamma_,
          x3flux, s_x3flux,
          dxw_, lambda_rescaling
        );
#else
        // flx(IBY) = (v3*b1 - v1*b3) = -EMFY
        // flx(IBZ) = (v3*b2 - v2*b3) =  EMFX

        RiemannSolver(
          IVZ, k, j, il, iu,
          b3,
          wl_, wr_,
          rl_, rr_,
          al_, ar_,
          alpha_,
          oo_alpha_,
          beta_u_,
          gamma_dd_,
          detgamma_,
          oo_detgamma_,
          sqrt_detgamma_,
          x3flux, s_x3flux,
          e2x3, e1x3, w_x3f,
          dxw_, lambda_rescaling
        );
#endif

        // swap the arrays for the next step (l<->lb)
        ReconstructSwap(pmb, wl_, wlb_, rl_, rlb_, al_, alb_);
      }
    }
  }

  return;

}

void Hydro::CalculateFluxes_FluxReconstruction(
  AA &w, FaceField &b,
  AA &bcc, const int order)
{
  using namespace fluxes;
  using namespace fluxes::grhd;
  using namespace characteristic::grhd;

  MeshBlock *pmb = pmy_block;
  Reconstruction * pr = pmb->precon;
  typedef Reconstruction::ReconstructionVariant ReconstructionVariant;
  ReconstructionVariant rv = pr->xorder_style;

  ReconstructionVariant r_rv = ReconstructionVariant::lin_vl;

  int il, iu, jl, ju, kl, ku;

  AA f(    NHYDRO,pmb->ncells3,pmb->ncells2,pmb->ncells1);

  AA sf_m( NHYDRO,pmb->ncells3,pmb->ncells2,pmb->ncells1);
  AA sf_p( NHYDRO,pmb->ncells3,pmb->ncells2,pmb->ncells1);

  // AA lam_( pmb->ncells1);

  // Shu convention
  AA fl_( NHYDRO,pmb->nverts1);
  AA flb_(NHYDRO,pmb->nverts1);
  AA fr_( NHYDRO,pmb->nverts1);

  AA eig_v(NHYDRO,pmb->ncells3,pmb->ncells2,pmb->ncells1);

  flux[X1DIR].ZeroClear();
  flux[X2DIR].ZeroClear();
  flux[X3DIR].ZeroClear();

  int ivx;

  //---------------------------------------------------------------------------
  // i-direction
  {
    ivx = 1;

    for (int k=0; k<pmb->ncells3; ++k)
    for (int j=0; j<pmb->ncells2; ++j)
    {
      AssembleFluxes(     pmb, k, j, 0, pmb->ncells1-1, ivx, f,  w, u);
      AssembleEigenvalues(pmb, k, j, 0, pmb->ncells1-1, ivx, eig_v, w, u);
      SplitFluxLLFMax(pmb, k, j, 0, pmb->ncells1-1, ivx, u, f,
                      eig_v, sf_m, sf_p);
    }

    AA &x1flux = flux[X1DIR];
    pr->SetIndicialLimitsCalculateFluxes(ivx, il, iu, jl, ju, kl, ku, 0);

    for (int k=kl; k<=ku; ++k)
    for (int j=jl; j<=ju; ++j)
    {
      for (int n=0; n<NHYDRO; ++n)
      {
        pr->ReconstructFieldX1(rv, sf_p, fl_, fr_, n, n, k, j, il-1, iu);
      }

      for (int n=0; n<NHYDRO; ++n)
      for (int i=il; i<=iu; ++i)
      {
        x1flux(n,k,j,i) += fl_(n,i);
      }

      for (int n=0; n<NHYDRO; ++n)
      {
        pr->ReconstructFieldX1(rv, sf_m, fl_, fr_, n, n, k, j, il-1, iu);
      }

      for (int n=0; n<NHYDRO; ++n)
      for (int i=il; i<=iu; ++i)
      {
        x1flux(n,k,j,i) += fr_(n,i);
      }
    }
  }

  //---------------------------------------------------------------------------
  // j-direction
  if (pmb->pmy_mesh->f2)
  {
    ivx = 2;

    for (int k=0; k<pmb->ncells3; ++k)
    for (int j=0; j<pmb->ncells2; ++j)
    {
      AssembleFluxes(     pmb, k, j, 0, pmb->ncells1-1, ivx, f,  w, u);
      AssembleEigenvalues(pmb, k, j, 0, pmb->ncells1-1, ivx, eig_v, w, u);
      SplitFluxLLFMax(pmb, k, j, 0, pmb->ncells1-1, ivx, u, f,
                      eig_v, sf_m, sf_p);
    }

    AA &x2flux = flux[X2DIR];
    pr->SetIndicialLimitsCalculateFluxes(ivx, il, iu, jl, ju, kl, ku, 0);

    for (int k=kl; k<=ku; ++k)
    {

      for (int n=0; n<NHYDRO; ++n)
      {
        pr->ReconstructFieldX2(rv, sf_p, fl_, fr_, n, n, k, jl-1, il, iu);
      }

      for (int j=jl; j<=ju; ++j)
      {

        for (int n=0; n<NHYDRO; ++n)
        {
          pr->ReconstructFieldX2(rv, sf_p, flb_, fr_, n, n, k, j, il, iu);
        }

        for (int n=0; n<NHYDRO; ++n)
        for (int i=il; i<=iu; ++i)
        {
          x2flux(n,k,j,i) += fl_(n,i);
        }

        // swap the arrays for the next step
        fl_.SwapAthenaArray(flb_);
      }


      for (int n=0; n<NHYDRO; ++n)
      {
        pr->ReconstructFieldX2(rv, sf_m, fl_, fr_, n, n, k, jl-1, il, iu);
      }

      for (int j=jl; j<=ju; ++j)
      {

        for (int n=0; n<NHYDRO; ++n)
        {
          pr->ReconstructFieldX2(rv, sf_m, flb_, fr_, n, n, k, j, il, iu);
        }

        for (int n=0; n<NHYDRO; ++n)
        for (int i=il; i<=iu; ++i)
        {
          x2flux(n,k,j,i) += fr_(n,i);
        }

        // swap the arrays for the next step
        fl_.SwapAthenaArray(flb_);
      }

    }
  }

  //---------------------------------------------------------------------------
  // k-direction
  if (pmb->pmy_mesh->f3)
  {
    ivx = 3;

    for (int k=0; k<pmb->ncells3; ++k)
    for (int j=0; j<pmb->ncells2; ++j)
    {
      AssembleFluxes(     pmb, k, j, 0, pmb->ncells1-1, ivx, f,  w, u);
      AssembleEigenvalues(pmb, k, j, 0, pmb->ncells1-1, ivx, eig_v, w, u);
      SplitFluxLLFMax(pmb, k, j, 0, pmb->ncells1-1, ivx, u, f,
                      eig_v, sf_m, sf_p);
    }

    AA &x3flux = flux[X3DIR];
    pr->SetIndicialLimitsCalculateFluxes(ivx, il, iu, jl, ju, kl, ku, 0);

    for (int j=jl; j<=ju; ++j)
    { // this loop ordering is intentional

      for (int n=0; n<NHYDRO; ++n)
      {
        pr->ReconstructFieldX3(rv, sf_p, fl_, fr_, n, n, kl-1, j, il, iu);
      }

      for (int k=kl; k<=ku; ++k)
      {

        for (int n=0; n<NHYDRO; ++n)
        {
          pr->ReconstructFieldX3(rv, sf_p, flb_, fr_, n, n, k, j, il, iu);
        }

        for (int n=0; n<NHYDRO; ++n)
        for (int i=il; i<=iu; ++i)
        {
          x3flux(n,k,j,i) += fl_(n,i);
        }

        // swap the arrays for the next step
        fl_.SwapAthenaArray(flb_);
      }


      for (int n=0; n<NHYDRO; ++n)
      {
        pr->ReconstructFieldX3(rv, sf_m, fl_, fr_, n, n, kl-1, j, il, iu);
      }

      for (int k=kl; k<=ku; ++k)
      {

        for (int n=0; n<NHYDRO; ++n)
        {
          pr->ReconstructFieldX3(rv, sf_m, flb_, fr_, n, n, k, j, il, iu);
        }

        for (int n=0; n<NHYDRO; ++n)
        for (int i=il; i<=iu; ++i)
        {
          x3flux(n,k,j,i) += fr_(n,i);
        }

        // swap the arrays for the next step
        fl_.SwapAthenaArray(flb_);
      }


    }
  }

}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::CalculateFluxes_STS
//  \brief Calculate Hydrodynamic Diffusion Fluxes for STS

void Hydro::CalculateFluxes_STS() {
  AddDiffusionFluxes();
}

void Hydro::AddDiffusionFluxes() {
  Field *pf = pmy_block->pfield;
  // add diffusion fluxes
  if (hdif.hydro_diffusion_defined) {
    if (hdif.nu_iso > 0.0 || hdif.nu_aniso > 0.0)
      hdif.AddDiffusionFlux(hdif.visflx,flux);
    if (NON_BAROTROPIC_EOS) {
      if (hdif.kappa_iso > 0.0 || hdif.kappa_aniso > 0.0)
        hdif.AddDiffusionEnergyFlux(hdif.cndflx,flux);
    }
  }
  if (MAGNETIC_FIELDS_ENABLED && NON_BAROTROPIC_EOS) {
    if (pf->fdif.field_diffusion_defined)
      pf->fdif.AddPoyntingFlux(pf->fdif.pflux);
  }
  return;
}
