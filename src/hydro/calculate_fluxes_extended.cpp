#ifndef EIGENSTRUCTURE_GRHD_
#define EIGENSTRUCTURE_GRHD_

// C headers

// C++ headers

// Athena++ headers
#include "../athena_aliases.hpp"
#include "../utils/linear_algebra.hpp"
#include "hydro.hpp"

//----------------------------------------------------------------------------------------
using namespace gra::aliases;
//----------------------------------------------------------------------------------------

namespace fluxes {

void SplitFluxLLFMax(MeshBlock * pmb,
                     const int k, const int j,
                     const int il, const int iu,
                     const int ivx,
                     AthenaArray<Real> &u,
                     AthenaArray<Real> &flux,
                     AthenaArray<Real> &lambda,
                     AthenaArray<Real> &flux_m,
                     AthenaArray<Real> &flux_p)
{
  AthenaArray<Real> mal(iu+1);

  // use full MeshBlock
  for (int n=0; n<NHYDRO; ++n)
  switch (ivx)
  {
    case 1:
    {
      for (int i=il; i<=iu; ++i)
      {
        mal(i) = std::max(std::abs(lambda(0,       k,j,i)), mal(i));
        mal(i) = std::max(std::abs(lambda(NHYDRO-1,k,j,i)), mal(i));

        const int di = (i==pmb->ncells1-1) ? -1 : +1;
        mal(i) = std::max(std::abs(lambda(0,       k,j,i+di)), mal(i));
        mal(i) = std::max(std::abs(lambda(NHYDRO-1,k,j,i+di)), mal(i));
      }

      break;
    }
    case 2:
    {
      for (int i=il; i<=iu; ++i)
      {
        mal(i) = std::max(std::abs(lambda(0,       k,j,i)), mal(i));
        mal(i) = std::max(std::abs(lambda(NHYDRO-1,k,j,i)), mal(i));

        const int dj = (j==pmb->ncells2-1) ? -1 : +1;

        mal(i) = std::max(std::abs(lambda(0,       k,j+dj,i)), mal(i));
        mal(i) = std::max(std::abs(lambda(NHYDRO-1,k,j+dj,i)), mal(i));
      }
      break;
    }
    case 3:
    {
      for (int i=il; i<=iu; ++i)
      {
        mal(i) = std::max(std::abs(lambda(0,       k,j,i)), mal(i));
        mal(i) = std::max(std::abs(lambda(NHYDRO-1,k,j,i)), mal(i));

        const int dk = (k==pmb->ncells3-1) ? -1 : +1;

        mal(i) = std::max(std::abs(lambda(0,       k+dk,j,i)), mal(i));
        mal(i) = std::max(std::abs(lambda(NHYDRO-1,k+dk,j,i)), mal(i));
      }
      break;
    }
  }


  for (int n=0; n<NHYDRO; ++n)
  for (int i=il; i<=iu; ++i)
  {
    flux_m(n,k,j,i) = 0.5 * (flux(n,k,j,i) - mal(i) * u(n,k,j,i));
    flux_p(n,k,j,i) = 0.5 * (flux(n,k,j,i) + mal(i) * u(n,k,j,i));
  }

}

}  // namespace fluxes

namespace fluxes::grhd {

void AssembleFluxes(MeshBlock * pmb,
                    const int k, const int j,
                    const int il, const int iu,
                    const int ivx,
                    AthenaArray<Real> &f,
                    AthenaArray<Real> &w,
                    AthenaArray<Real> &u)
{
#if not MAGNETIC_FIELDS_ENABLED
  using namespace LinearAlgebra;

  const int nn1 = pmb->ncells1;  // utilize the cells

  // perform variable resampling when required
  Z4c * pz4c = pmb->pz4c;

  // Slice 3d z4c metric quantities  (NDIM=3 in z4c.hpp) ----------------------
  AT_N_sym gamma_dd(pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sca alpha(   pz4c->storage.adm, Z4c::I_ADM_alpha);
  AT_N_vec beta_u(  pz4c->storage.adm, Z4c::I_ADM_betax);

  AT_N_sca w_p(  w, IPR);
  AT_N_vec w_util_u(w, IVX);

  AT_N_sca sqrt_detgamma_(nn1);
  AT_N_sca detgamma_(     nn1);  // spatial met det
  AT_N_sca oo_detgamma_(  nn1);  // 1 / spatial met det
  AT_N_sym gamma_uu_(     nn1);

  AT_N_vec w_v_u_(nn1);

  // Lorentz factor
  AT_N_sca W_(nn1);

  // Prepare determinant-like
  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    detgamma_(i)      = Det3Metric(gamma_dd, k,j,i);
    sqrt_detgamma_(i) = std::sqrt(detgamma_(i));
  }

  // Lorentz factors
  for (int i=il; i<=iu; ++i)
  {
    W_(i) = std::sqrt(
      1. + InnerProductVec3Metric(w_util_u, gamma_dd, k, j, i)
    );
  }

  // Eulerian vel.
  for (int a=0; a<NDIM; ++a)
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      w_v_u_(a,i) = w_util_u(a,k,j,i) / W_(i);
    }
  }

  // calculate flux
  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    f(IDN,k,j,i) = u(IDN,k,j,i) * alpha(k,j,i) * (
      w_v_u_(ivx-1,i) - beta_u(ivx-1,k,j,i)/alpha(k,j,i)
    );
    f(IEN,k,j,i) = u(IEN,k,j,i) * alpha(k,j,i) * (
      w_v_u_(ivx-1,i) - beta_u(ivx-1,k,j,i)/alpha(k,j,i)
    ) + alpha(k,j,i)*sqrt_detgamma_(i)*w_p(k,j,i)*w_v_u_(ivx-1,i);
  }

  for (int a=0; a<NDIM; ++a)
  {
    #pragma omp simd
    for (int i = il; i <= iu; ++i)
    {
      f(IVX+a,k,j,i) = (
        u(IVX+a,k,j,i) * alpha(k,j,i) *
        (w_v_u_(ivx-1,i) -
         beta_u(ivx-1,k,j,i)/alpha(k,j,i))
      );

    }
  }

  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    f(ivx,k,j,i) += w_p(k,j,i) * alpha(k,j,i) * sqrt_detgamma_(i);
  }

#endif // MAGNETIC_FIELDS_ENABLED
}

}  // namespace fluxes::grhd

namespace characteristic::grhd {

void AssembleEigenvalues(MeshBlock * pmb,
                         const int k, const int j,
                         const int il, const int iu,
                         const int ivx,
                         AthenaArray<Real> &lambda,
                         AthenaArray<Real> &w,
                         AthenaArray<Real> &u)
{
#if not MAGNETIC_FIELDS_ENABLED
  using namespace LinearAlgebra;

  const int nn1 = pmb->ncells1;  // utilize the cells

  // perform variable resampling when required
  Z4c * pz4c = pmb->pz4c;

  // Slice 3d z4c metric quantities  (NDIM=3 in z4c.hpp) ----------------------
  AT_N_sym gamma_dd(pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sca alpha(   pz4c->storage.adm, Z4c::I_ADM_alpha);
  AT_N_vec beta_u(  pz4c->storage.adm, Z4c::I_ADM_betax);

  AT_N_sca w_rho(w, IDN);
  AT_N_sca w_p(  w, IPR);
  AT_N_vec w_util_u(w, IVX);


  AT_N_sca sqrt_detgamma_(nn1);
  AT_N_sca detgamma_(     nn1);  // spatial met det
  AT_N_sca oo_detgamma_(  nn1);  // 1 / spatial met det
  AT_N_sym gamma_uu_(     nn1);

  AT_N_vec w_util_u_(nn1);
  AT_N_vec w_v_u_(nn1);
  AT_N_sca w_norm2_v_(nn1);

  AT_N_sca lambda_p_(nn1);
  AT_N_sca lambda_m_(nn1);

  // primitive vel. (covar.)
  AT_N_vec w_util_d_(nn1);

  // Lorentz factor
  AT_N_sca W_(nn1);

  // h * rho
  AT_N_sca w_hrho_(nn1);

  // Prepare util slice
  for (int a=0; a<NDIM; ++a)
  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    w_util_u_(a,i) = w_util_u(a,k,j,i);
  }


  // Prepare determinant-like
  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    detgamma_(i)      = Det3Metric(gamma_dd, k,j,i);

    sqrt_detgamma_(i) = std::sqrt(detgamma_(i));
    oo_detgamma_(i)   = 1. / detgamma_(i);
  }

  Inv3Metric(oo_detgamma_, gamma_dd, gamma_uu_, k, j, il, iu);


  // Lorentz factors
  for (int i=il; i<=iu; ++i)
  {
    const Real norm2_utilde = InnerProductSlicedVec3Metric(
      w_util_u_, gamma_dd, k,j,i
    );

    W_(i) = std::sqrt(1. + norm2_utilde);
  }

  // Eulerian vel.
  for (int a=0; a<NDIM; ++a)
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      w_v_u_(a,i) = w_util_u_(a,i) / W_(i);
    }
  }


  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    w_norm2_v_(i) = InnerProductSlicedVec3Metric(w_v_u_, gamma_dd, k, j, i);
  }


#if USETM
  const Real mb = pmb->peos->GetEOS().GetBaryonMass();
  Real Y[MAX_SPECIES] = {0.0};

  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    const Real n = w_rho(k,j,i) / mb;

    for (int n=0; n<NSCALARS; n++)
    {
      Y[n] = pmb->pscalars->r(n,k,j,i);
    }
    const Real T = pmb->peos->GetEOS().GetTemperatureFromP(
      n, w_p(k,j,i), Y);

    w_hrho_(i) = w_rho(k,j,i)*pmb->peos->GetEOS().GetEnthalpy(n, T, Y);

    pmb->peos->SoundSpeedsGR(n, T,
                             w_v_u_(ivx-1,i),
                             w_norm2_v_(i),
                             alpha(k,j,i),
                             beta_u(ivx-1,k,j,i),
                             gamma_uu_(ivx-1,ivx-1,i),
                             &lambda_p_(i),
                             &lambda_m_(i),
                             Y);
  }
#else
  const Real Gamma = pmb->peos->GetGamma();
  const Real Eos_Gamma_ratio = Gamma / (Gamma - 1.0);

  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    w_hrho_(i) = w_rho(k,j,i) + Eos_Gamma_ratio * w_p(k,j,i);

    pmb->peos->SoundSpeedsGR(w_hrho_(i),
                             w_p(k,j,i),
                             w_v_u_(ivx-1,i),
                             w_norm2_v_(i),
                             alpha(k,j,i),
                             beta_u(ivx-1,k,j,i),
                             gamma_uu_(ivx-1,ivx-1,i),
                             &lambda_p_(i),
                             &lambda_m_(i));
  }
#endif

  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {

    lambda(0,k,j,i) = lambda_m_(i);

    // lambda_0 (three-fold degeneracy)
    lambda(1,k,j,i) = (alpha(k,j,i) * w_v_u_(ivx-1,i) - beta_u(ivx-1,k,j,i));
    lambda(2,k,j,i) = lambda(1,i);
    lambda(3,k,j,i) = lambda(1,i);

    lambda(4,k,j,i) = lambda_p_(i);
  }


#endif // MAGNETIC_FIELDS_ENABLED

}

}  // namespace characteristic::grhd

#endif  // EIGENSTRUCTURE_GRHD_
