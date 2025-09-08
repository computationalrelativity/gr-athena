//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file add_flux_divergence.cpp
//  \brief Computes divergence of the Hydro fluxes and adds that to a conserved variable
// register

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "hydro.hpp"
#include "../eos/eos.hpp"
#include <limits>

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::AddFluxDivergence
//  \brief Adds flux divergence to weighted average of conservative variables from
//  previous step(s) of time integrator algorithm

// TODO(felker): consider combining with PassiveScalars implementation + (see 57cfe28b)
// (may rename to AddPhysicalFluxDivergence or AddQuantityFluxDivergence to explicitly
// distinguish from CoordTerms)
// (may rename to AddHydroFluxDivergence and AddScalarsFluxDivergence, if
// the implementations remain completely independent / no inheritance is
// used)
void Hydro::AddFluxDivergence(const Real wght, AthenaArray<Real> &u_out)
{
  MeshBlock *pmb = pmy_block;
  AthenaArray<Real> &x1flux = flux[X1DIR];
  AthenaArray<Real> &x2flux = flux[X2DIR];
  AthenaArray<Real> &x3flux = flux[X3DIR];
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  AthenaArray<Real> &dflx = dflx_;

  for (int k=ks; k<=ke; ++k)
  for (int j=js; j<=je; ++j)
  {
    // calculate x1-flux divergence - fluxes now contain weighting by detgamma
    for (int n=0; n<NHYDRO; ++n)
    {
      #pragma omp simd
      for (int i=is; i<=ie; ++i)
      {
        dflx(n,i) = (x1flux(n,k,j,i+1) - x1flux(n,k,j,i))/pmb->pcoord->dx1f(i);
      }
    }

    // calculate x2-flux divergence
    if (pmb->block_size.nx2 > 1) {
      for (int n=0; n<NHYDRO; ++n) {
      #pragma omp simd
      for (int i=is; i<=ie; ++i)
      {
        dflx(n,i) += (x2flux(n,k,j+1,i) - x2flux(n,k,j,i))/pmb->pcoord->dx2f(j);
      }
      }
    }

    // calculate x3-flux divergence
    if (pmb->block_size.nx3 > 1)
    {
      for (int n=0; n<NHYDRO; ++n) {
      #pragma omp simd
      for (int i=is; i<=ie; ++i) {
        dflx(n,i) += (x3flux(n,k+1,j,i) - x3flux(n,k,j,i))/pmb->pcoord->dx3f(k);
      }
      }
    }

    // BD: TODO- probably remove
    if (opt_excision.use_taper && opt_excision.excise_hydro_freeze_evo)
    {
      for (int n=0; n<NHYDRO; ++n) {
        #pragma omp simd
        for (int i=is; i<=ie; ++i)
        {
          u_out(n,k,j,i) -= excision_mask(k,j,i) * wght*dflx(n,i);
        }
      }
    }
    else
    {
      for (int n=0; n<NHYDRO; ++n)
      {
        #pragma omp simd
        for (int i=is; i<=ie; ++i) {
          u_out(n,k,j,i) -= wght*dflx(n,i);
        }
      }
    }
  }
}

void Hydro::CheckStateWithFluxDivergence(
  const Real wght,
  AA &u,
  AA(& hflux)[3],
  AA(& sflux)[3],
  bool & all_valid,
  AA_B &mask,
  const int num_enlarge_layer
)
{
  MeshBlock *pmb = pmy_block;
  Reconstruction *pr = pmb->precon;

  // Undensitized conserved density floor
  const Real dfloor = pmb->peos->density_floor_;

  // point to adm sqrt gamma term
  AA & sqrt_detgamma = pmb->pz4c->aux_extended.ms_sqrt_detgamma.array();

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  // needs to be one cell larger than extremal flux idx.
  // this is because there is no nn flux corr, only fine->coarse
  // otherwise inconsitency between nn block faces would occur

  const int nel = num_enlarge_layer;
  for (int k=ks-nel; k<=ke+nel; ++k)
  for (int j=js-nel; j<=je+nel; ++j)
  for (int i=is-nel; i<=ie+nel; ++i) // avoid simd here
  {
    const Real oo_sqrt_detgamma = OO(sqrt_detgamma(k,j,i));

    const Real D     = u(IDN,k,j,i);
    const Real Dstar = D - wght * (
      (hflux[0](IDN,k,j,i+1) - hflux[0](IDN,k,j,i)) / pmb->pcoord->dx1f(i) +
      (hflux[1](IDN,k,j+1,i) - hflux[1](IDN,k,j,i)) / pmb->pcoord->dx2f(j) +
      (hflux[2](IDN,k+1,j,i) - hflux[2](IDN,k,j,i)) / pmb->pcoord->dx3f(k)
    );

    bool is_valid = (
      Dstar * oo_sqrt_detgamma >= pr->xorder_fb_dfloor_fac * dfloor
    );

    mask(k,j,i) = mask(k,j,i) && is_valid;
    all_valid = all_valid && is_valid;
  }
}

void Hydro::CheckStateWithFluxDivergenceDMP(
  const Real wght,
  AA &u,
  AA &u_old,
  AA &s,
  AA &s_old,
  AA(& hflux)[3],
  AA(& sflux)[3],
  bool &all_valid,
  AA_B &mask,
  const int num_enlarge_layer
)
{
  MeshBlock *pmb = pmy_block;
  Reconstruction *pr = pmb->precon;

  // DMP factors
  const Real fac_dmp_min = pr->xorder_dmp_min;
  const Real fac_dmp_max = pr->xorder_dmp_max;

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  // needs to be one cell larger than extremal flux idx.
  // this is because there is no nn flux corr, only fine->coarse
  // otherwise inconsitency between nn block faces would occur

  const int nel = num_enlarge_layer;
  for (int k=ks-nel; k<=ke+nel; ++k)
  for (int j=js-nel; j<=je+nel; ++j)
  for (int i=is-nel; i<=ie+nel; ++i) // avoid simd here
  {
    if (!mask(k,j,i))
      continue;

    const Real D     = u(IDN,k,j,i);
    const Real D_star = D - wght * (
      (hflux[0](IDN,k,j,i+1) - hflux[0](IDN,k,j,i)) / pmb->pcoord->dx1f(i) +
      (hflux[1](IDN,k,j+1,i) - hflux[1](IDN,k,j,i)) / pmb->pcoord->dx2f(j) +
      (hflux[2](IDN,k+1,j,i) - hflux[2](IDN,k,j,i)) / pmb->pcoord->dx3f(k)
    );

    const Real tau     = u(IEN,k,j,i);
    const Real tau_star = tau - wght * (
      (hflux[0](IEN,k,j,i+1) - hflux[0](IEN,k,j,i)) / pmb->pcoord->dx1f(i) +
      (hflux[1](IEN,k,j+1,i) - hflux[1](IEN,k,j,i)) / pmb->pcoord->dx2f(j) +
      (hflux[2](IEN,k+1,j,i) - hflux[2](IEN,k,j,i)) / pmb->pcoord->dx3f(k)
    );

    bool is_valid = true;

    Real D_min = +std::numeric_limits<Real>::infinity();
    Real D_max = -std::numeric_limits<Real>::infinity();

    Real tau_min = +std::numeric_limits<Real>::infinity();
    Real tau_max = -std::numeric_limits<Real>::infinity();

    for (int kk = k-1; kk <= k+1; ++kk)
    for (int jj = j-1; jj <= j+1; ++jj)
    for (int ii = i-1; ii <= i+1; ++ii)
    {
      const Real D_i = u_old(IDN,kk,jj,ii);
      D_min = fac_dmp_min * std::min(D_min, D_i);
      D_max = fac_dmp_max * std::max(D_max, D_i);

      const Real tau_i = u_old(IEN,kk,jj,ii);
      tau_min = fac_dmp_min * std::min(tau_min, tau_i);
      tau_max = fac_dmp_max * std::max(tau_max, tau_i);
    }

    if (is_valid && pr->xorder_use_dmp_scalars)
    {
      for (int n=0; n<NSCALARS; ++n)
      {
        const Real S     = pmb->pscalars->s(n,k,j,i);
        const Real S_star = S - wght * (
          (sflux[0](n,k,j,i+1) - sflux[0](n,k,j,i)) / pmb->pcoord->dx1f(i) +
          (sflux[1](n,k,j+1,i) - sflux[1](n,k,j,i)) / pmb->pcoord->dx2f(j) +
          (sflux[2](n,k+1,j,i) - sflux[2](n,k,j,i)) / pmb->pcoord->dx3f(k)
        );

        Real S_min = +std::numeric_limits<Real>::infinity();
        Real S_max = -std::numeric_limits<Real>::infinity();

        for (int kk = k-1; kk <= k+1; ++kk)
        for (int jj = j-1; jj <= j+1; ++jj)
        for (int ii = i-1; ii <= i+1; ++ii)
        {
          const Real S_i = pmb->pscalars->s2(n,kk,jj,ii);
          S_min = fac_dmp_min * std::min(S_min, S_i);
          S_max = fac_dmp_max * std::max(S_max, S_i);
        }

        if (S_star < S_min || S_star > S_max)
        {
          is_valid = false;
          break;
        }
      }
    }

    // Apply DMP: Dstar must remain within [Dmin, Dmax]
    if (D_star < D_min || D_star > D_max)
    {
      is_valid = false;
    }

    if (tau_star < tau_min || tau_star > tau_max)
    {
      is_valid = false;
    }

    mask(k,j,i) = mask(k,j,i) && is_valid;
    all_valid = all_valid && is_valid;
  }
}

//--------------------------------------------------------------------------------------
// Correct fluxes based on mask
void Hydro::HybridizeFluxes(
    AA (&hflux)[3], AA (&sflux)[3],
    AA (&lo_hflux)[3], AA (&lo_sflux)[3],
    const AA_B &mask)
{
  MeshBlock * pmb = pmy_block;

  int is = pmb->is, js = pmb->js, ks = pmb->ks;
  int ie = pmb->ie, je = pmb->je, ke = pmb->ke;

  for (int k = ks; k <= ke; ++k)
  for (int j = js; j <= je; ++j)
  for (int i = is; i <= ie+1; ++i)
  {
    // FC ix i->i-1/2
    // CC ix i->i
    // So for an FC i should check CC i-1 and i;
    // if both true then HO, else fallback
    if (!(mask(k,j,i-1) && mask(k,j,i)))
    {
      for (int n=0; n<NHYDRO; ++n)
      {
        hflux[0](n,k,j,i) = lo_hflux[0](n,k,j,i);
      }
      for (int n=0; n<NSCALARS; ++n)
      {
        sflux[0](n,k,j,i) = lo_sflux[0](n,k,j,i);
      }
    }
  }

  for (int k = ks; k <= ke; ++k)
  for (int j = js; j <= je+1; ++j)
  for (int i = is; i <= ie; ++i)
  {
    if (!(mask(k,j-1,i) && mask(k,j,i)))
    {
      for (int n=0; n<NHYDRO; ++n)
      {
        hflux[1](n,k,j,i) = lo_hflux[1](n,k,j,i);
      }
      for (int n=0; n<NSCALARS; ++n)
      {
        sflux[1](n,k,j,i) = lo_sflux[1](n,k,j,i);
      }
    }
  }

  for (int k = ks; k <= ke+1; ++k)
  for (int j = js; j <= je; ++j)
  for (int i = is; i <= ie; ++i)
  {
    if (!(mask(k-1,j,i) && mask(k,j,i)))
    {
      for (int n=0; n<NHYDRO; ++n)
      {
        hflux[2](n,k,j,i) = lo_hflux[2](n,k,j,i);
      }
      for (int n=0; n<NSCALARS; ++n)
      {
        sflux[2](n,k,j,i) = lo_sflux[2](n,k,j,i);
      }
    }
  }

}
