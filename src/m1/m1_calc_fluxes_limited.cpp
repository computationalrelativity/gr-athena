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

      const Real d_ql = q(k,j,i-1) - q(k,j,i-2);
      const Real d_qm = q(k,j,i  ) - q(k,j,i-1);
      const Real d_qr = q(k,j,i+1) - q(k,j,i  );

      const Real d_qlm = d_ql * d_qm;
      const Real d_qmr = d_qm * d_qr;

      Real kap_fac = 0.5 * (
        kap_a(k,j,i-1) + kap_a(k,j,i) +
        kap_s(k,j,i-1) + kap_s(k,j,i)
      );

      kap_fac = (kap_fac > 0) ? kap_fac * pm1->mbi.dx1(i) : 1.0;

      const Real A = ((d_qlm < 0) && (d_qmr < 0))
        ? 1.0
        : std::min(1.0, 1.0 / kap_fac);

      const Real phi = std::max(
        std::min(
          1.0, std::min(d_ql / d_qm, d_qr / d_qm)
        ), 0.0
      );

      const Real F_HO = 0.5 * (F(k,j,i-1) + F(k,j,i));
      const Real F_LO = F_HO - 0.5 * lam * d_qm;

      Flux(k,j,i) = F_HO - A * (1.0 - phi) * (F_HO - F_LO);
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

      const Real d_ql = q(k,j-1,i) - q(k,j-2,i);
      const Real d_qm = q(k,j,i  ) - q(k,j-1,i);
      const Real d_qr = q(k,j+1,i) - q(k,j,i  );

      const Real d_qlm = d_ql * d_qm;
      const Real d_qmr = d_qm * d_qr;

      Real kap_fac = 0.5 * (
        kap_a(k,j-1,i) + kap_a(k,j,i) +
        kap_s(k,j-1,i) + kap_s(k,j,i)
      );

      kap_fac = (kap_fac > 0) ? kap_fac * pm1->mbi.dx2(j) : 1.0;

      const Real A = ((d_qlm < 0) && (d_qmr < 0))
        ? 1.0
        : std::min(1.0, 1.0 / kap_fac);

      const Real phi = std::max(
        std::min(
          1.0, std::min(d_ql / d_qm, d_qr / d_qm)
        ), 0.0
      );

      const Real F_HO = 0.5 * (F(k,j-1,i) + F(k,j,i));
      const Real F_LO = F_HO - 0.5 * lam * d_qm;

      Flux(k,j,i) = F_HO - A * (1.0 - phi) * (F_HO - F_LO);
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

      const Real d_ql = q(k-1,j,i) - q(k-2,j,i);
      const Real d_qm = q(k,j,i  ) - q(k-1,j,i);
      const Real d_qr = q(k+1,j,i) - q(k,j,i  );

      const Real d_qlm = d_ql * d_qm;
      const Real d_qmr = d_qm * d_qr;

      Real kap_fac = 0.5 * (
        kap_a(k-1,j,i) + kap_a(k,j,i) +
        kap_s(k-1,j,i) + kap_s(k,j,i)
      );

      kap_fac = (kap_fac > 0) ? kap_fac * pm1->mbi.dx3(k) : 1.0;

      const Real A = ((d_qlm < 0) && (d_qmr < 0))
        ? 1.0
        : std::min(1.0, 1.0 / kap_fac);

      const Real phi = std::max(
        std::min(
          1.0, std::min(d_ql / d_qm, d_qr / d_qm)
        ), 0.0
      );

      const Real F_HO = 0.5 * (F(k-1,j,i) + F(k,j,i));
      const Real F_LO = F_HO - 0.5 * lam * d_qm;

      Flux(k,j,i) = F_HO - A * (1.0 - phi) * (F_HO - F_LO);
    }
  }
}

// ============================================================================
} // namespace M1::Fluxes
// ============================================================================

//
// :D
//