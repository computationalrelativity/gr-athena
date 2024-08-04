//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hydro.cpp
//  \brief implementation of functions in class Hydro

// C headers

// C++ headers
#include <algorithm>
#include <string>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../mesh/mesh.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "hydro.hpp"
#include "hydro_diffusion/hydro_diffusion.hpp"
#include "srcterms/hydro_srcterms.hpp"

// constructor, initializes data structures and parameters

Hydro::Hydro(MeshBlock *pmb, ParameterInput *pin) :
    pmy_block(pmb),
    u(NHYDRO, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    w(NHYDRO, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    u1(NHYDRO, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    w1(NHYDRO, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    // C++11: nested brace-init-list in Hydro member initializer list = aggregate init. of
    // flux[3] array --> direct list init. of each array element --> direct init. via
    // constructor overload resolution of non-aggregate class type AthenaArray<Real>
    flux{ {NHYDRO, pmb->ncells3, pmb->ncells2, pmb->ncells1+1},
          {NHYDRO, pmb->ncells3, pmb->ncells2+1, pmb->ncells1,
           (pmb->pmy_mesh->f2 ? AthenaArray<Real>::DataStatus::allocated :
            AthenaArray<Real>::DataStatus::empty)},
          {NHYDRO, pmb->ncells3+1, pmb->ncells2, pmb->ncells1,
           (pmb->pmy_mesh->f3 ? AthenaArray<Real>::DataStatus::allocated :
            AthenaArray<Real>::DataStatus::empty)}
    },
    coarse_cons_(NHYDRO, pmb->ncc3, pmb->ncc2, pmb->ncc1,
                 (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
                  AthenaArray<Real>::DataStatus::empty)),
    coarse_prim_(NHYDRO, pmb->ncc3, pmb->ncc2, pmb->ncc1,
                 (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
                  AthenaArray<Real>::DataStatus::empty)),
    q_reset_mask(pmb->ncells3, pmb->ncells2, pmb->ncells1),
    hbvar(pmb, &u, &coarse_cons_, flux, HydroBoundaryQuantity::cons),
    hsrc(this, pin),
    hdif(this, pin)
{
  int nc1 = pmb->ncells1, nc2 = pmb->ncells2, nc3 = pmb->ncells3;
  Mesh *pm = pmy_block->pmy_mesh;

  pmb->RegisterMeshBlockDataCC(u);

  floor_both_states = pin->GetOrAddBoolean("time", "floor_both_states", false);
  flux_reconstruction = pin->GetOrAddBoolean(
    "hydro", "flux_reconstruction", false);

  // If user-requested time integrator is type 3S*, allocate additional memory registers
  std::string integrator = pin->GetOrAddString("time", "integrator", "vl2");
  if (integrator == "ssprk5_4" || STS_ENABLED) {
    // future extension may add "int nregister" to Hydro class
    u2.NewAthenaArray(NHYDRO, nc3, nc2, nc1);
  }

  // "Enroll" in S/AMR by adding to vector of tuples of pointers in MeshRefinement class
  if (pm->multilevel) {
    refinement_idx = pmy_block->pmr->AddToRefinementCC(&u, &coarse_cons_);
  }

  // enroll HydroBoundaryVariable object
  hbvar.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&hbvar);
  pmb->pbval->bvars_main_int.push_back(&hbvar);

  // Allocate memory for scratch arrays
  dt1_.NewAthenaArray(nc1);
  dt2_.NewAthenaArray(nc1);
  dt3_.NewAthenaArray(nc1);
  dxw_.NewAthenaArray(nc1);
  wl_.NewAthenaArray(NWAVE, nc1);
  wr_.NewAthenaArray(NWAVE, nc1);
  wlb_.NewAthenaArray(NWAVE, nc1);

  if (pmy_block->precon->xorder_use_fb)
  {
    r_wl_.NewAthenaArray(NWAVE, nc1);
    r_wr_.NewAthenaArray(NWAVE, nc1);
    r_wlb_.NewAthenaArray(NWAVE, nc1);
  }
  else
  {
#if USETM
    // Needed for PrimitiveSolver floors
    rl_.NewAthenaArray(NSCALARS, nc1);
    rr_.NewAthenaArray(NSCALARS, nc1);
    rlb_.NewAthenaArray(NSCALARS, nc1);
#endif
  }

  dflx_.NewAthenaArray(NHYDRO, nc1);

#if USETM
    temperature.NewAthenaArray(nc3, nc2, nc1);
#endif

  UserTimeStep_ = pmb->pmy_mesh->UserTimeStep_;
}

//----------------------------------------------------------------------------------------
//! \fn Real Hydro::GetWeightForCT(Real dflx, Real rhol, Real rhor, Real dx, Real dt)
//  \brief Calculate the weighting factor for the constrained transport method

Real Hydro::GetWeightForCT(Real dflx, Real rhol, Real rhor, Real dx, Real dt) {
  Real v_over_c = (1024.0)* dt * dflx / (dx * (rhol + rhor));
  Real tmp_min = std::min(static_cast<Real>(0.5), v_over_c);
  return 0.5 + std::max(static_cast<Real>(-0.5), tmp_min);
}
