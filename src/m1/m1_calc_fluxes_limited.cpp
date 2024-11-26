// c++
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
                            const AT_C_sca & kap_a,
                            const AT_C_sca & kap_s,
                            const AT_C_sca & lambda,
                            AT_C_sca & Flux)
{
  switch (dir)
  {
    case 0:
    {
      ReconstructLimitedFluxX1(pm1, q, F, kap_a, kap_s, lambda, Flux);
      break;
    }
    case 1:
    {
      ReconstructLimitedFluxX2(pm1, q, F, kap_a, kap_s, lambda, Flux);
      break;
    }
    case 2:
    {
      ReconstructLimitedFluxX3(pm1, q, F, kap_a, kap_s, lambda, Flux);
      break;
    }
  }
}

// vector field flux reconstruction -------------------------------------------
void ReconstructLimitedFlux(M1 * pm1,
                            const int dir,
                            const AT_N_vec & q,
                            const AT_N_vec & F,
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

        ReconstructLimitedFluxX1(
          pm1,
          q_c,
          F_c,
          kap_a,
          kap_s,
          lambda,
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

        ReconstructLimitedFluxX2(
          pm1,
          q_c,
          F_c,
          kap_a,
          kap_s,
          lambda,
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

        ReconstructLimitedFluxX3(
          pm1,
          q_c,
          F_c,
          kap_a,
          kap_s,
          lambda,
          Flux_c);
      }
      break;
    }
  }
}

// implementation details =====================================================
namespace {

static const Real MINMOD_THETA = 1;

Real MinMod2(const Real A, const Real B)
{
  return std::min(1.0, MINMOD_THETA * std::min(A, B));
}

Real FluxLimiter(const M1::opt_flux_variety & flx_var,
                 const Real & dx,
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

      const Real Phi = std::max(0.0, MinMod2(d_ql * oo_d_qc, d_qr * oo_d_qc));
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
      const Real kap_fac = (kap_avg > 0) ? kap_avg * dx : 1.0;

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

      const Real kap_min = std::min(kap_asm1, kap_as0);
      const Real kap_fac = (kap_min > 0) ? kap_min * dx : 1.0;

      const Real A = (((d_qlc < 0) && (d_qcr < 0)))
        ? 1.0
        : std::min(1.0, OO(kap_fac));

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
                              const AT_C_sca & kap_a,
                              const AT_C_sca & kap_s,
                              const AT_C_sca & lambda,
                              AT_C_sca & Flux)
{

  // flux(i) with i = pm1->mbi.il stores \hat{F}_{i-1/2}
  // Range required (for divergence along X1) is (il, ... iu+1)

  for (int k=pm1->mbi.kl; k<=pm1->mbi.ku; ++k)
  for (int j=pm1->mbi.jl; j<=pm1->mbi.ju; ++j)
  {
    #pragma omp simd
    for (int i=pm1->mbi.il; i<=pm1->mbi.iu+1; ++i)  // flux_idx : i-1/2
    {
      const Real lam = std::max(lambda(k,j,i-1), lambda(k,j,i));

      const Real F_HO = 0.5 * (F(k,j,i-1) + F(k,j,i));
      const Real F_LO = F_HO - 0.5 * lam * (q(k,j,i) - q(k,j,i-1));

      const Real hyb_fac = FluxLimiter(
        pm1->opt.flux_variety,
        pm1->mbi.dx1(i),
        kap_a(k,j,i-1) + kap_s(k,j,i-1),
        kap_a(k,j,i+0) + kap_s(k,j,i+0),
        kap_a(k,j,i+1) + kap_s(k,j,i+1),
        q(k,j,i-2), q(k,j,i-1), q(k,j,i), q(k,j,i+1), q(k,j,i+2)
      );

      const Real th = std::max(hyb_fac, pm1->opt.min_flux_Theta);
      Flux(k,j,i) = F_HO -  th * (F_HO - F_LO);
    }
  }
}

void ReconstructLimitedFluxX2(M1 * pm1,
                              const AT_C_sca & q,
                              const AT_C_sca & F,
                              const AT_C_sca & kap_a,
                              const AT_C_sca & kap_s,
                              const AT_C_sca & lambda,
                              AT_C_sca & Flux)
{
  for (int k=pm1->mbi.kl; k<=pm1->mbi.ku; ++k)
  for (int j=pm1->mbi.jl; j<=pm1->mbi.ju+1; ++j) // flux_idx : j-1/2
  {
    #pragma omp simd
    for (int i=pm1->mbi.il; i<=pm1->mbi.iu; ++i)
    {
      const Real lam = std::max(lambda(k,j-1,i), lambda(k,j,i));

      const Real F_HO = 0.5 * (F(k,j-1,i) + F(k,j,i));
      const Real F_LO = F_HO - 0.5 * lam * (q(k,j,i) - q(k,j-1,i));

      const Real hyb_fac = FluxLimiter(
        pm1->opt.flux_variety,
        pm1->mbi.dx2(j),
        kap_a(k,j-1,i) + kap_s(k,j-1,i),
        kap_a(k,j+0,i) + kap_s(k,j+0,i),
        kap_a(k,j+1,i) + kap_s(k,j+1,i),
        q(k,j-2,i), q(k,j-1,i), q(k,j,i), q(k,j+1,i), q(k,j+2,i)
      );

      const Real th = std::max(hyb_fac, pm1->opt.min_flux_Theta);
      Flux(k,j,i) = F_HO -  th * (F_HO - F_LO);
    }
  }
}

void ReconstructLimitedFluxX3(M1 * pm1,
                              const AT_C_sca & q,
                              const AT_C_sca & F,
                              const AT_C_sca & kap_a,
                              const AT_C_sca & kap_s,
                              const AT_C_sca & lambda,
                              AT_C_sca & Flux)
{
  for (int k=pm1->mbi.kl; k<=pm1->mbi.ku+1; ++k) // flux_idx : k-1/2
  for (int j=pm1->mbi.jl; j<=pm1->mbi.ju; ++j)
  {
    #pragma omp simd
    for (int i=pm1->mbi.il; i<=pm1->mbi.iu; ++i)
    {
      const Real lam = std::max(lambda(k-1,j,i), lambda(k,j,i));

      const Real F_HO = 0.5 * (F(k-1,j,i) + F(k,j,i));
      const Real F_LO = F_HO - 0.5 * lam * (q(k,j,i) - q(k-1,j,i));

      const Real hyb_fac = FluxLimiter(
        pm1->opt.flux_variety,
        pm1->mbi.dx3(k),
        kap_a(k-1,j,i) + kap_s(k-1,j,i),
        kap_a(k+0,j,i) + kap_s(k+0,j,i),
        kap_a(k+1,j,i) + kap_s(k+1,j,i),
        q(k-2,j,i), q(k-1,j,i), q(k,j,i), q(k+1,j,i), q(k+2,j,i)
      );

      const Real th = std::max(hyb_fac, pm1->opt.min_flux_Theta);
      Flux(k,j,i) = F_HO -  th * (F_HO - F_LO);
    }
  }
}

// ============================================================================
} // namespace M1::Fluxes
// ============================================================================

//
// :D
//