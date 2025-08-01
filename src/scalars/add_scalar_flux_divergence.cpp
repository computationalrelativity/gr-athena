//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file add_scalar_flux_divergence.cpp
//  \brief Computes divergence of the passive scalar fluxes and adds that to a conserved
// variable register (passive scalar mass)

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "scalars.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//----------------------------------------------------------------------------------------
//! \fn  void PassiveScalars::AddFluxDivergence
//  \brief Adds flux divergence to weighted average of conservative variables from
//  previous step(s) of time integrator algorithm

// TODO(felker): after the removal of AddCoordTermsDivergence() fn call from
// Hydro::AddFluxDivergence(), the 2x fns could be trivially shared if:
// - flux/s_flux renamed to the same class member name
// - 7x below references of x1face_area_ ... dflx_ private class members (which are only
// ever used in this fn and are members to prevent de/allocating each fn call)
// - NHYDRO/NSCALARS is replaced with array_out.GetDim4()

// ----> Hydro should be derived from PassiveScalars

// TODO(felker): remove the following unnecessary private class member?
// field_diffusion.cpp:66:    cell_volume_.NewAthenaArray(nc1);

void PassiveScalars::AddFluxDivergence(const Real wght, AthenaArray<Real> &s_out)
{
  MeshBlock *pmb = pmy_block;
  AthenaArray<Real> &x1flux = s_flux[X1DIR];
  AthenaArray<Real> &x2flux = s_flux[X2DIR];
  AthenaArray<Real> &x3flux = s_flux[X3DIR];
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  AthenaArray<Real> &dflx = dflx_;

  for (int k=ks; k<=ke; ++k)
  for (int j=js; j<=je; ++j)
  {
    // calculate x1-flux divergence
    for (int n=0; n<NSCALARS; ++n)
    #pragma omp simd
    for (int i=is; i<=ie; ++i)
    {
      dflx(n,i) = (x1flux(n,k,j,i+1) - x1flux(n,k,j,i))/pmb->pcoord->dx1f(i);
    }

    // calculate x2-flux divergence
    if (pmb->block_size.nx2 > 1)
    for (int n=0; n<NSCALARS; ++n)
    #pragma omp simd
    for (int i=is; i<=ie; ++i)
    {
      dflx(n,i) += (x2flux(n,k,j+1,i) - x2flux(n,k,j,i))/pmb->pcoord->dx2f(j);
    }

    // calculate x3-flux divergence
    if (pmb->block_size.nx3 > 1)
    for (int n=0; n<NSCALARS; ++n)
    #pragma omp simd
    for (int i=is; i<=ie; ++i)
    {
      dflx(n,i) += (x3flux(n,k+1,j,i) - x3flux(n,k,j,i))/pmb->pcoord->dx3f(k);
    }

    // update conserved variables
    for (int n=0; n<NSCALARS; ++n)
    #pragma omp simd
    for (int i=is; i<=ie; ++i)
    {
      s_out(n,k,j,i) -= wght*dflx(n,i);
    }
  }
  return;
}
