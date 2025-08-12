// c++
#include <cmath>
#include <iostream>

// Athena++ headers
#include "m1_calc_fluxes.hpp"

// ============================================================================
namespace M1::Fluxes {
// ============================================================================

// scalar field flux reconstruction -------------------------------------------
void ReconstructLimitedFlux(M1 * pm1,
                            const int dir,
                            const AT_C_sca & q,
                            const AT_C_sca & F,
                            const AT_C_sca & xi,
                            const AT_C_sca & kap_a,
                            const AT_C_sca & kap_s,
                            const AT_C_sca & lambda,
                            AT_C_sca & Flux)
{
  switch (dir)
  {
    case 0:
    {
      AA & flux_limiter = pm1->ev_strat.masks.flux_limiter;

      MeshBlock * pmb = pm1->pmy_block;

      if (!pm1->opt.flux_limiter_multicomponent &&
          pm1->opt.flux_limiter_nn)
      {
        CC_GLOOP3(k, j, i)
        {
          flux_limiter(0,k,j,i) = 0.0;
        }

        LimiterMaskX1(pm1, flux_limiter, q, xi, kap_a, kap_s);
      }

      ReconstructLimitedFluxX1(pm1, q, F, xi, kap_a, kap_s, lambda,
                               flux_limiter, Flux);
      break;
    }
    case 1:
    {
      AA & flux_limiter = pm1->ev_strat.masks.flux_limiter;

      MeshBlock * pmb = pm1->pmy_block;

      if (!pm1->opt.flux_limiter_multicomponent &&
          pm1->opt.flux_limiter_nn)
      {
        CC_GLOOP3(k, j, i)
        {
          flux_limiter(1,k,j,i) = 0.0;
        }

        LimiterMaskX2(pm1, flux_limiter, q, xi, kap_a, kap_s);
      }

      ReconstructLimitedFluxX2(pm1, q, F, xi, kap_a, kap_s, lambda,
                               flux_limiter, Flux);
      break;
    }
    case 2:
    {
      AA & flux_limiter = pm1->ev_strat.masks.flux_limiter;

      MeshBlock * pmb = pm1->pmy_block;

      if (!pm1->opt.flux_limiter_multicomponent &&
          pm1->opt.flux_limiter_nn)
      {
        CC_GLOOP3(k, j, i)
        {
          flux_limiter(2,k,j,i) = 0.0;
        }

        LimiterMaskX3(pm1, flux_limiter, q, xi, kap_a, kap_s);
      }

      ReconstructLimitedFluxX3(pm1, q, F, xi, kap_a, kap_s, lambda,
                               flux_limiter, Flux);
      break;
    }
  }
}

// vector field flux reconstruction -------------------------------------------
void ReconstructLimitedFlux(M1 * pm1,
                            const int dir,
                            const AT_N_vec & q,
                            const AT_N_vec & F,
                            const AT_C_sca & xi,
                            const AT_C_sca & kap_a,
                            const AT_C_sca & kap_s,
                            const AT_C_sca & lambda,
                            AT_N_vec & Flux)
{
  AT_C_sca q_c;
  AT_C_sca F_c;
  AT_C_sca Flux_c;

  switch (dir)
  {
    case 0:
    {
      for (int a=0; a<N; ++a)
      {
        const_cast<AT_N_vec &>(q).slice(   a, q_c);
        const_cast<AT_N_vec &>(F).slice(   a, F_c);
        const_cast<AT_N_vec &>(Flux).slice(a, Flux_c);

        AA & flux_limiter = pm1->ev_strat.masks.flux_limiter;

        MeshBlock * pmb = pm1->pmy_block;

        if (!pm1->opt.flux_limiter_multicomponent &&
            pm1->opt.flux_limiter_nn)
        {
          CC_GLOOP3(k, j, i)
          {
            flux_limiter(0,k,j,i) = 0.0;
          }

          LimiterMaskX1(pm1, flux_limiter, q_c, xi, kap_a, kap_s);
        }

        ReconstructLimitedFluxX1(
          pm1,
          q_c,
          F_c,
          xi,
          kap_a,
          kap_s,
          lambda,
          flux_limiter,
          Flux_c);
      }
      break;
    }
    case 1:
    {
      for (int a=0; a<N; ++a)
      {
        const_cast<AT_N_vec &>(q).slice(   a, q_c);
        const_cast<AT_N_vec &>(F).slice(   a, F_c);
        const_cast<AT_N_vec &>(Flux).slice(a, Flux_c);

        AA & flux_limiter = pm1->ev_strat.masks.flux_limiter;

        MeshBlock * pmb = pm1->pmy_block;

        if (!pm1->opt.flux_limiter_multicomponent &&
            pm1->opt.flux_limiter_nn)
        {
          CC_GLOOP3(k, j, i)
          {
            flux_limiter(1,k,j,i) = 0.0;
          }

          LimiterMaskX2(pm1, flux_limiter, q_c, xi, kap_a, kap_s);
        }

        ReconstructLimitedFluxX2(
          pm1,
          q_c,
          F_c,
          xi,
          kap_a,
          kap_s,
          lambda,
          flux_limiter,
          Flux_c);
      }
      break;
    }
    case 2:
    {
      for (int a=0; a<N; ++a)
      {
        const_cast<AT_N_vec &>(q).slice(   a, q_c);
        const_cast<AT_N_vec &>(F).slice(   a, F_c);
        const_cast<AT_N_vec &>(Flux).slice(a, Flux_c);

        AA & flux_limiter = pm1->ev_strat.masks.flux_limiter;

        MeshBlock * pmb = pm1->pmy_block;

        if (!pm1->opt.flux_limiter_multicomponent &&
            pm1->opt.flux_limiter_nn)
        {
          CC_GLOOP3(k, j, i)
          {
            flux_limiter(2,k,j,i) = 0.0;
          }

          LimiterMaskX3(pm1, flux_limiter, q_c, xi, kap_a, kap_s);
        }

        ReconstructLimitedFluxX3(
          pm1,
          q_c,
          F_c,
          xi,
          kap_a,
          kap_s,
          lambda,
          flux_limiter,
          Flux_c);
      }
      break;
    }
  }
}

// implementation details =====================================================
namespace {

static const int Z = 0;
static const Real MINMOD_THETA = 1;

static const Real HMME_fac_A = 10;
static const Real HMME_fac_B = 10;

Real MinMod2(const Real A, const Real B)
{
  return std::min(1.0, MINMOD_THETA * std::min(A, B));
}

Real FluxLimiter(const M1::opt_flux_variety & flx_var,
                 const Real & dx,
                 const Real & xi_m1,
                 const Real & xi_0,
                 const Real & xi_p1,
                 const Real & kap_asm1,
                 const Real & kap_as0,
                 const Real & kap_asp1,
                 const Real & qm2,
                 const Real & qm1,
                 const Real & q0,
                 const Real & qp1,
                 const Real & qp2)
{
  Real hyb_fac = 1;

  switch (flx_var)
  {
    case (M1::opt_flux_variety::HO):
    {
      // F = F_HO - hyb_fac * (F_HO - F_LO)
      // Hybridize to high order flux
      return 0.0;
    }
    case (M1::opt_flux_variety::LO):
    {
      // F = F_HO - hyb_fac * (F_HO - F_LO)
      // Hybridize to low order flux
      return 1.0;
    }
    case (M1::opt_flux_variety::HybridizeMinMod):
    {
      const Real d_ql    = qm1 - qm2;
      const Real oo_d_qc = OO(q0  - qm1);
      const Real d_qr    = qp1 - q0;

      const Real d_qc = q0  - qm1;

      const Real d_qlc = d_ql * d_qc;
      const Real d_qcr = d_qc * d_qr;

      // const Real Phi = std::max(0.0, MinMod2(d_ql * oo_d_qc, d_qr * oo_d_qc));
      // return 1 - Phi;
      Real Phi = 0.0;
      if ((d_qlc > 0) && (d_qcr > 0))
      {
        Phi = MinMod2(d_ql * oo_d_qc, d_qr * oo_d_qc);
      }
      return 1 - Phi;
    }
    case (M1::opt_flux_variety::HybridizeMinModA):
    {
      const Real d_ql = qm1 - qm2;
      const Real d_qc = q0  - qm1;
      const Real d_qr = qp1 - q0;

      const Real d_qlc = d_ql * d_qc;
      const Real d_qcr = d_qc * d_qr;

      const Real oo_d_qc = OO(d_qc);

      const Real Phi = std::max(0.0, MinMod2(d_ql * oo_d_qc, d_qr * oo_d_qc));

      const Real kap_avg = 0.5 * (kap_asm1 + kap_as0);
      const Real kap_fac = (kap_avg > 0) ? kap_avg * dx : 1.0;

      const Real A = (((d_qlc < 0) && (d_qcr < 0)))
        ? 1.0
        : std::min(1.0, OO(kap_fac));

      return A * (1 - Phi);
    }
    case (M1::opt_flux_variety::HybridizeMinModB):
    {
      const Real d_ql = qm1 - qm2;
      const Real d_qc = q0  - qm1;
      const Real d_qr = qp1 - q0;

      const Real d_qlc = d_ql * d_qc;
      const Real d_qcr = d_qc * d_qr;

      const Real oo_d_qc = OO(d_qc);

      const Real Phi = std::max(0.0, MinMod2(d_ql * oo_d_qc, d_qr * oo_d_qc));

      const Real kap_avg = 0.5 * (kap_asm1 + kap_as0);
      const Real kap_fac = (kap_avg * dx > 1) ? kap_avg * dx : 1.0;

      Real A = std::tanh(OO(kap_fac));

      const Real g_m1 = q0 + qm2 - 2.0 * qm1;
      const Real g_0  = qp1 + qm1 - 2.0 * q0;
      const Real g_p1 = qp2 + q0 - 2.0 * qp1;

      A = ((g_0 * g_m1 < 0) && (g_0 * g_p1 < 0)) ? 1.0 : A;

      return A * (1 - Phi);
    }
    case (M1::opt_flux_variety::HybridizeMinModC):
    {
      const Real d_ql = qm1 - qm2;
      const Real d_qc = q0  - qm1;
      const Real d_qr = qp1 - q0;

      const Real d_qlc = d_ql * d_qc;
      const Real d_qcr = d_qc * d_qr;

      const Real oo_d_qc = OO(d_qc);

      const Real Phi = std::max(0.0, MinMod2(d_ql * oo_d_qc, d_qr * oo_d_qc));

      const Real xi_avg = std::min(1.0, 0.5 * (xi_m1 + xi_0));

      const Real A = (((d_qlc < 0) && (d_qcr < 0)))
        ? 1.0
        : xi_avg;

      return A * (1 - Phi);
    }
    case (M1::opt_flux_variety::HybridizeMinModD):
    {
      const Real d_ql = qm1 - qm2;
      const Real d_qc = q0  - qm1;
      const Real d_qr = qp1 - q0;

      const Real d_qlc = d_ql * d_qc;
      const Real d_qcr = d_qc * d_qr;

      const Real oo_d_qc = OO(d_qc);

      Real Phi = 0.0;
      bool sawtooth = false;
      if ((d_qlc > 0) && (d_qcr > 0))
      {
        Phi = MinMod2(d_ql * oo_d_qc, d_qr * oo_d_qc);
      }
      else if ((d_qlc < 0) && (d_qcr < 0))
      {
        sawtooth = true;
      }

      const Real kap_avg = 0.5 * (kap_asm1 + kap_as0);
      // const Real kap_min = std::min(kap_asm1, kap_as0);
      // const Real kap_svg = std::sqrt(kap_asm1 * kap_as0);
      const Real kap_fac = kap_avg * dx;

      Real A = 1.0;
      if (kap_fac > 1)
      {
        A = std::min(1.0, OO(kap_fac));
      }

      Real res = (sawtooth ? 1.0 : A) * (1 - Phi);
      return res;
    }
    case (M1::opt_flux_variety::HybridizeMinModE):
    {
      const Real d_ql = qm1 - qm2;
      const Real d_qc = q0  - qm1;
      const Real d_qr = qp1 - q0;

      const Real d_qlc = d_ql * d_qc;
      const Real d_qcr = d_qc * d_qr;

      const Real oo_d_qc = OO(d_qc);

      const Real Phi = std::max(0.0, MinMod2(d_ql * oo_d_qc, d_qr * oo_d_qc));

      const Real xi_avg = std::min(1.0, 0.5 * (xi_m1 + xi_0));

      // Smoothly interp instead of hard-cut
      // s_qlc simeq 1 when d_qlc < 0
      const Real s_qlc = 0.5 * (1.0 + std::tanh(-HMME_fac_A * d_qlc));
      const Real s_qcr = 0.5 * (1.0 + std::tanh(-HMME_fac_A * d_qcr));

      const Real fac_mon = s_qlc * s_qcr;
      const Real wei = OO(1.0 + std::exp(-HMME_fac_B * (-fac_mon)));

      const Real A = wei + (1.0-wei) * xi_avg;
      return A * (1 - Phi);
    }
    default:
    {
      assert(false);
    }
  }
}

}

void ReconstructLimitedFluxX1(M1 * pm1,
                              const AT_C_sca & q,
                              const AT_C_sca & F,
                              const AT_C_sca & xi,
                              const AT_C_sca & kap_a,
                              const AT_C_sca & kap_s,
                              const AT_C_sca & lambda,
                              AA & flux_limiter,
                              AT_C_sca & Flux)
{

  // flux(i) with i = pm1->mbi.il stores \hat{F}_{i-1/2}
  // Range required (for divergence along X1) is (il, ... iu+1)

  M1_RLOOP3_1(k,j,i)
  {
    const Real lam = std::max(lambda(k,j,i-1+Z), lambda(k,j,i+Z));

    const Real F_HO = 0.5 * (F(k,j,i-1+Z) + F(k,j,i+Z));
    const Real F_LO = F_HO - 0.5 * lam * (q(k,j,i+Z) - q(k,j,i-1+Z));

    Real th;
    if (!pm1->opt.flux_limiter_use_mask)
    {
      const Real hyb_fac = FluxLimiter(
        pm1->opt.flux_variety,
        pm1->mbi.dx1(i),
        xi(k,j,i-1+Z),
        xi(k,j,i+0+Z),
        xi(k,j,i+1+Z),
        (kap_a(k,j,i-1+Z) + kap_s(k,j,i-1+Z)),
        (kap_a(k,j,i+0+Z) + kap_s(k,j,i+0+Z)),
        (kap_a(k,j,i+1+Z) + kap_s(k,j,i+1+Z)),
        q(k,j,i-2+Z), q(k,j,i-1+Z), q(k,j,i+Z), q(k,j,i+1+Z), q(k,j,i+2+Z)
      );

      th = std::max(hyb_fac, pm1->opt.min_flux_Theta);
    }
    else
    {
      if (pm1->opt.flux_limiter_nn)
      {
      th = std::max(flux_limiter(0,k,j,i-1), flux_limiter(0,k,j,i));
      }
      else
      {
        th = flux_limiter(0,k,j,i);
      }
    }

    Flux(k,j,i) = F_HO - th * (F_HO - F_LO);
  }
}

void ReconstructLimitedFluxX2(M1 * pm1,
                              const AT_C_sca & q,
                              const AT_C_sca & F,
                              const AT_C_sca & xi,
                              const AT_C_sca & kap_a,
                              const AT_C_sca & kap_s,
                              const AT_C_sca & lambda,
                              AA & flux_limiter,
                              AT_C_sca & Flux)
{
  M1_RLOOP3_2(k,j,i)
  {
    const Real lam = std::max(lambda(k,j-1+Z,i), lambda(k,j+Z,i));

    const Real F_HO = 0.5 * (F(k,j-1+Z,i) + F(k,j+Z,i));
    const Real F_LO = F_HO - 0.5 * lam * (q(k,j+Z,i) - q(k,j-1+Z,i));

    Real th;
    if (!pm1->opt.flux_limiter_use_mask)
    {
      const Real hyb_fac = FluxLimiter(
        pm1->opt.flux_variety,
        pm1->mbi.dx2(j),
        xi(k,j-1+Z,i),
        xi(k,j+0+Z,i),
        xi(k,j+1+Z,i),
        (kap_a(k,j-1+Z,i) + kap_s(k,j-1+Z,i)),
        (kap_a(k,j+0+Z,i) + kap_s(k,j+0+Z,i)),
        (kap_a(k,j+1+Z,i) + kap_s(k,j+1+Z,i)),
        q(k,j-2+Z,i), q(k,j-1+Z,i), q(k,j+Z,i), q(k,j+1+Z,i), q(k,j+2+Z,i)
      );

      th = std::max(hyb_fac, pm1->opt.min_flux_Theta);
    }
    else
    {
      if (pm1->opt.flux_limiter_nn)
      {
      th = std::max(flux_limiter(1,k,j-1,i), flux_limiter(1,k,j,i));
      }
      else
      {
        th = flux_limiter(1,k,j,i);
      }
    }

    Flux(k,j,i) = F_HO -  th * (F_HO - F_LO);
  }
}

void ReconstructLimitedFluxX3(M1 * pm1,
                              const AT_C_sca & q,
                              const AT_C_sca & F,
                              const AT_C_sca & xi,
                              const AT_C_sca & kap_a,
                              const AT_C_sca & kap_s,
                              const AT_C_sca & lambda,
                              AA & flux_limiter,
                              AT_C_sca & Flux)
{
  M1_RLOOP3_3(k,j,i)
  {
    const Real lam = std::max(lambda(k-1+Z,j,i), lambda(k+Z,j,i));

    const Real F_HO = 0.5 * (F(k-1+Z,j,i) + F(k+Z,j,i));
    const Real F_LO = F_HO - 0.5 * lam * (q(k+Z,j,i) - q(k-1+Z,j,i));

    Real th;
    if (!pm1->opt.flux_limiter_use_mask)
    {
      const Real hyb_fac = FluxLimiter(
        pm1->opt.flux_variety,
        pm1->mbi.dx3(k),
        xi(k-1+Z,j,i),
        xi(k+0+Z,j,i),
        xi(k+1+Z,j,i),
        (kap_a(k-1+Z,j,i) + kap_s(k-1+Z,j,i)),
        (kap_a(k+0+Z,j,i) + kap_s(k+0+Z,j,i)),
        (kap_a(k+1+Z,j,i) + kap_s(k+1+Z,j,i)),
        q(k-2+Z,j,i), q(k-1+Z,j,i), q(k+Z,j,i), q(k+1+Z,j,i), q(k+2+Z,j,i)
      );

      th = std::max(hyb_fac, pm1->opt.min_flux_Theta);
    }
    else
    {
      if (pm1->opt.flux_limiter_nn)
      {
      th = std::max(flux_limiter(2,k-1,j,i), flux_limiter(2,k,j,i));
      }
      else
      {
        th = flux_limiter(2,k,j,i);
      }
    }

    Flux(k,j,i) = F_HO -  th * (F_HO - F_LO);
  }
}

void LimiterMaskX1(M1 * pm1,
                   AA & flux_limiter,
                   const AT_C_sca & q,
                   const AT_C_sca & xi,
                   const AT_C_sca & kap_a,
                   const AT_C_sca & kap_s)
{
  M1_LLOOP3_1(k,j,i)
  {
    const Real hyb_fac = FluxLimiter(
      pm1->opt.flux_variety,
      pm1->mbi.dx1(i),
      xi(k,j,i-1+Z),
      xi(k,j,i+0+Z),
      xi(k,j,i+1+Z),
      (kap_a(k,j,i-1+Z) + kap_s(k,j,i-1+Z)),
      (kap_a(k,j,i+0+Z) + kap_s(k,j,i+0+Z)),
      (kap_a(k,j,i+1+Z) + kap_s(k,j,i+1+Z)),
      q(k,j,i-2+Z), q(k,j,i-1+Z), q(k,j,i+Z), q(k,j,i+1+Z), q(k,j,i+2+Z)
    );

    const Real th_imh = std::max(hyb_fac, pm1->opt.min_flux_Theta);

    if (pm1->opt.flux_limiter_nn)
    {
      flux_limiter(0,k,j,i-1) = std::max(th_imh, flux_limiter(0,k,j,i-1));
    }
    flux_limiter(0,k,j,i+0) = std::max(th_imh, flux_limiter(0,k,j,i+0));
  }
}

void LimiterMaskX2(M1 * pm1,
                   AA & flux_limiter,
                   const AT_C_sca & q,
                   const AT_C_sca & xi,
                   const AT_C_sca & kap_a,
                   const AT_C_sca & kap_s)
{
  M1_LLOOP3_2(k,j,i)
  {
    const Real hyb_fac = FluxLimiter(
      pm1->opt.flux_variety,
      pm1->mbi.dx2(j),
      xi(k,j-1+Z,i),
      xi(k,j+0+Z,i),
      xi(k,j+1+Z,i),
      (kap_a(k,j-1+Z,i) + kap_s(k,j-1+Z,i)),
      (kap_a(k,j+0+Z,i) + kap_s(k,j+0+Z,i)),
      (kap_a(k,j+1+Z,i) + kap_s(k,j+1+Z,i)),
      q(k,j-2+Z,i), q(k,j-1+Z,i), q(k,j+Z,i), q(k,j+1+Z,i), q(k,j+2+Z,i)
    );

    const Real th_jmh = std::max(hyb_fac, pm1->opt.min_flux_Theta);

    if (pm1->opt.flux_limiter_nn)
    {
      flux_limiter(1,k,j-1,i) = std::max(th_jmh, flux_limiter(1,k,j-1,i));
    }
    flux_limiter(1,k,j+0,i) = std::max(th_jmh, flux_limiter(1,k,j+0,i));
  }
}

void LimiterMaskX3(M1 * pm1,
                   AA & flux_limiter,
                   const AT_C_sca & q,
                   const AT_C_sca & xi,
                   const AT_C_sca & kap_a,
                   const AT_C_sca & kap_s)
{
  M1_LLOOP3_3(k,j,i)
  {
    const Real hyb_fac = FluxLimiter(
      pm1->opt.flux_variety,
      pm1->mbi.dx3(k),
      xi(k-1+Z,j,i),
      xi(k+0+Z,j,i),
      xi(k+1+Z,j,i),
      (kap_a(k-1+Z,j,i) + kap_s(k-1+Z,j,i)),
      (kap_a(k+0+Z,j,i) + kap_s(k+0+Z,j,i)),
      (kap_a(k+1+Z,j,i) + kap_s(k+1+Z,j,i)),
      q(k-2+Z,j,i), q(k-1+Z,j,i), q(k+Z,j,i), q(k+1+Z,j,i), q(k+2+Z,j,i)
    );

    const Real th_kmh = std::max(hyb_fac, pm1->opt.min_flux_Theta);

    if (pm1->opt.flux_limiter_nn)
    {
      flux_limiter(2,k-1,j,i) = std::max(th_kmh, flux_limiter(2,k-1,j,i));
    }
    flux_limiter(2,k+0,j,i) = std::max(th_kmh, flux_limiter(2,k+0,j,i));
  }
}

// ============================================================================
} // namespace M1::Fluxes
// ============================================================================

//
// :D
//