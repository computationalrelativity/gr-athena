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
  AthenaArray<bool> &mask,
  const int num_enlarge_layer
)
{

  MeshBlock *pmb = pmy_block;

  // Undensitized conserved density floor
  const Real dfloor = pmb->peos->density_floor_;

  // point to adm sqrt gamma term
  AA & sqrt_detgamma = pmb->pz4c->aux_extended.ms_sqrt_detgamma.array();

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  for (int k=ks; k<=ke; ++k)
  for (int j=js; j<=je; ++j)
  for (int i=is; i<=ie; ++i) // avoid simd here
  {
    const Real oo_sqrt_detgamma = OO(sqrt_detgamma(k,j,i));

    const Real D     = u(IDN,k,j,i);
    const Real Dstar = D - wght * (
      (hflux[0](IDN,k,j,i+1) - hflux[0](IDN,k,j,i)) / pmb->pcoord->dx1f(i) +
      (hflux[1](IDN,k,j+1,i) - hflux[1](IDN,k,j,i)) / pmb->pcoord->dx2f(j) +
      (hflux[2](IDN,k+1,j,i) - hflux[2](IDN,k,j,i)) / pmb->pcoord->dx3f(k)
    );

    const bool is_valid = Dstar * oo_sqrt_detgamma >= dfloor;
    mask(k,j,i) = is_valid;
    all_valid = all_valid && is_valid;
  }
}